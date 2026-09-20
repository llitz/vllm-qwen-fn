"""qwen3 reasoning-parser plugin: one-shot leading-newline strip (vLLM).

Deployment (this is the PREFERRED variant; the HTTP-middleware variant on
branch `master` is the fallback):

    1. bind-mount this file into the vllm container, e.g.
       /plugins/qwen_clean_reasoning.py (read-only);
    2. add `--reasoning-parser-plugin /plugins/qwen_clean_reasoning.py` to
       the `vllm serve` command line;
    3. set `reasoning-parser: qwen3_clean` in config.yaml
       (replacing `reasoning-parser: qwen3`);
    4. NEVER enable the ContentNewlineStripMiddleware line in config.yaml
       at the same time — the two strips would double-apply (see below).

What it does
------------
Strips the <= 2 leading newline units the qwen3 model emits at the head of
answer content after an ENGINE-CONFIRMED reasoning end. Registered as
`qwen3_clean`; the implementation subclass is
`Qwen3CleanReasoningAdapter(Qwen3ParserReasoningAdapter)` (dynamic built-in
class, vllm 0.27.x, `vllm/parser/engine/registered_adapters.py`).

Arming semantics (EXACT — version-pinned build 0.27.2rc1.dev77; verified
compatible with commit e9d1398d9, 2026-08-21 — see _Qwen3CleanEngine below
for the ParserManager.get_parser engine-collapse workaround):

    The strip arms (once, per parser instance) on an ENGINE-CONFIRMED
    REASONING END: `has_engine_confirmed_reasoning_end()`
    (vllm/parser/engine/adapters.py:110-111 — the same predicate
    DelegatingParser.parse_delta uses for the reasoning->content
    transition). That predicate is True in TWO cases for qwen3:

      (a) a genuine think-close tag: (REASONING, THINK_END) -> CONTENT
          with REASONING_END (qwen3.py:126-129);
      (b) an IMPLICIT end via tool-open: (REASONING, TOOL_START) and
          (CONTENT, TOOL_START) BOTH emit a cause-less REASONING_END
          (qwen3.py:137-145).

    The cause (think-close vs tool-open) is NOT observable via any
    non-private API in this build: REASONING_END is a bare EventType with
    no payload (engine/events.py), and the only public surface collapses
    both causes to the boolean `reasoning_ended`
    (parser_engine.py:173-175, adapters.py:110-111). Only the PRIVATE
    engine state differs afterwards — TOOL_PREAMBLE (tool-open) vs
    CONTENT (think-close), reachable solely via `_engine.state`
    (streaming_parser_engine.py _on_terminal) — and this plugin
    DELIBERATELY does not read it (private, rename-risky). Consequence:
    a tool-open WITHOUT a think-close can arm the strip. See
    "Accepted corner" below. This is an engine-confirmed reasoning END
    (both cases), NOT a guarantee of a genuine think-close tag.

    Covers the EMPTY-THINK case (think-close with zero reasoning text,
    which vLLM maps to reasoning=None): the engine confirms the close tag
    independently of the reasoning text, so arming does not depend on any
    reasoning content ever being emitted.

    The state machine (REUSED VERBATIM from the verified HTTP-middleware
    variant, master branch: code/vllm_defaults.py, spec docs/01, review
    docs/02):

        IDLE -> ARMED (budget = 2) -> RELEASED

      - arm once, per parser instance (instance lifetime = one completion
        request; see docs/08 for evidence), on the engine-confirmed
        reasoning end above;
      - strip AT MOST 2 leading newline units from content; CRLF counts
        as ONE unit; a bare CR is never consumed as a newline unit — a
        bare CR as the very last byte of an answer is dropped
        (documented 1-byte EOS edge, identical in the master-branch
        middleware; the streaming path has no end-of-stream hook to
        restore it);
      - content arriving before the engine confirms the reasoning end
        passes byte-exact (IDLE) — never-think answers are untouched;
      - ONE window at the answer head only: no re-arm on literal think
        tags inside content;
      - degenerate all-newline answer: withheld units are never emitted
        (dropped, same rule as the middleware, correction 4 / R3).

Accepted corner (KNOWN, ACCEPTED, TESTED — version-pinned container)
--------------------------------------------------------------------
Post-tool text is structurally LEGAL in the qwen3 config: TOOL_PREAMBLE
-> CONTENT on TOOL_END (qwen3.py:151-154) and TOOL_BETWEEN -> CONTENT on
TOOL_END (qwen3.py:180-183). Because a tool-open (case (b) above) can
arm the strip, content that follows a tool call and starts with <= 2
newline units LOSES up to 2 of them. The exposure is one-shot and
head-only (single budget, one window per request). This is accepted,
not a residual bug: the cause-discriminating guard is impossible on the
public API of this build (see above). Pinned by the TOOL_NO_CLOSE
regression test (evidence/plugin_tests/parse_delta_test2_v3.py) and
verified in-container on the pinned vllm build.

Where the artifact is emitted (SEAM-ONLY — vLLM internals byte-pristine)
------------------------------------------------------------------------
For engine-based parsers (qwen3), `DelegatingParser.parse_delta` NULLS
the transition delta's content and RE-FEEDS that text into the TOOL
phase: at the reasoning->content transition,
`delta_message.content = None` (when the tool parser is engine-based)
and the transition text (`current_text`) becomes the tool phase's first
delta (parser/abstract_parser.py, parse_delta). Consequence: EVERY
outgoing streaming content byte exits via the TOOL-phase message, which
the seam on the RETURN VALUE of parse_delta always sees — whether the
transition was caused by a think-close or by a tool-open, and whether
the bytes are the buffered post-transition text or later plain content.
Each byte crosses the strip exactly once.

Therefore the plugin installs ONE seam (narrow wrapper on
`DelegatingParser.parse_delta` at import time) and NOTHING else in the
streaming path: it does NOT override `extract_reasoning_streaming` or
`finish_streaming` — the base behavior is inherited byte-pristine. No
mutation inside the parser pipeline means no content/token-ID
desynchronization and no double-path state feeding (the two emission
points that the pre-redesign overrides patched — the reasoning-phase
delta and the finish_streaming transition flush — no longer exist as
separate plugin-visible paths; their bytes are the same tool-phase
bytes, seen once, at the seam).

The seam arms on `rp.has_engine_confirmed_reasoning_end()` and feeds
`msg.content` through the adapter instance's `_strip_state`. It is a
no-op (byte-exact) for any parser instance whose reasoning parser is not
`Qwen3CleanReasoningAdapter` (opt-in via the `qwen3_clean` registry
name), and it never runs for `parse()` (non-streaming), which is
covered by the adapter's `extract_reasoning` override with a FRESH
budget per call (chat non-stream, responses, anthropic non-stream use
`parse()` -> `extract_reasoning()`, never `parse_delta`, so the two
paths cannot both apply to one response).

Import context
--------------
Loaded at serve startup by `ReasoningParserManager.import_reasoning_parser`
from an ARBITRARY path (module name = file basename without .py, imported
via importlib spec_from_file_location — no package context, no sys.path
mutation). Therefore: absolute vllm imports only. Import side effects are
exactly two, both required: (a) the eager registration at the bottom, and
(b) the parse_delta seam installation. No prints, no other work (server
startup imports this file; an import error is logged by vLLM and then
fails the subsequent reasoning-parser name validation at serve startup).

Double-strip hard rule
----------------------
Running this plugin together with ContentNewlineStripMiddleware is
FORBIDDEN: the middleware (outermost, HTTP layer) would strip a second,
legitimate set of leading newline units after this plugin already stripped
the artifact — e.g. an answer that legitimately starts with "\n\n" after a
reasoning end loses one of its two newlines.
"""

from dataclasses import dataclass

from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import Qwen3ParserReasoningAdapter
from vllm.parser.qwen3 import Qwen3Parser
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

# --- one-shot head-strip state machine (VERBATIM from master
#     code/vllm_defaults.py; semantics per docs/01 + docs/02) ---------------

_BUDGET = 2                     # max newline units stripped per content stream


@dataclass
class _StripState:
    """Head-of-answer strip state; one instance per content stream (per
    choice index for OpenAI). Owned by _HeadStripper; lifetime = one HTTP
    response."""
    api: str                        # "openai" | "anthropic" | "responses"
    armed: bool = False             # ARMED once the engine confirmed the
                                    # reasoning end (_arm_strip)
    released: bool = False          # RELEASED: byte-exact forever
    budget: int = _BUDGET           # newline units left to drop (0..2)
    pending_cr: bool = False        # a trailing '\r' is held awaiting '\n'
    head_passed: bool = False       # a non-empty content fragment was consumed
    withheld: str = ""              # stripped newline units; DROPPED at finish()


def _is_newline_unit(ch: str, pending_cr: bool) -> bool:
    """True for a complete newline unit head char: '\n' (a lone LF, or the
    '\n' half of a CRLF when pending_cr is set). A '\r' is never classified
    here: a lone '\r' cannot be classified at sight (successor unknown); the
    hold logic in _feed_state() resolves it. Any other char is a
    NON-newline (releases)."""
    return ch == "\n"


def _feed_state(st: _StripState, frag: str) -> str:
    """Feed one content fragment in arrival order; return the output
    fragment per the state machine (step2 section 3 + corrections 2/3/4).
    O(len(frag)). A non-empty fragment always marks the head window as
    passed (a later arm on this key is then a no-op - correction 3)."""
    if not frag:
        return ""                   # empty fragment: no-op in every state
    st.head_passed = True
    if not st.armed:
        return frag                 # IDLE: byte-exact, holds nothing
    if st.released:
        return frag                 # RELEASED: byte-exact, no scan
    # ARMED: resolve a '\r' held from the previous fragment first.
    if st.pending_cr:
        if frag[0] == "\n" and st.budget > 0:
            # held '\r' + '\n' = one CRLF unit, exactly ONE budget decrement
            st.budget -= 1
            st.withheld += "\r\n"
            st.pending_cr = False
            frag = frag[1:]

        else:
            # held '\r' + non-'\n', or held '\r' + '\n' with budget
            # exhausted: emit held '\r' + entire fragment, RELEASED
            st.pending_cr = False
            st.released = True
            return "\r" + frag
    out: list[str] = []
    i, n = 0, len(frag)
    while i < n and not st.released:
        ch = frag[i]
        if _is_newline_unit(ch, st.pending_cr):
            # lone LF unit
            if st.budget > 0:
                st.budget -= 1
                st.withheld += "\n"
                i += 1
                continue
            st.released = True
            return "".join(out) + frag[i:]
        if ch == "\r":
            if i + 1 == n:
                # trailing '\r' at fragment end: successor unknown -> hold
                # it, emit nothing, stay ARMED (correction 2); the budget
                # is only decremented once the held CR is resolved by a
                # following '\n'
                st.pending_cr = True
                i += 1
                break
            if frag[i + 1] == "\n":
                # CRLF = one unit, one budget decrement
                if st.budget > 0:
                    st.budget -= 1
                    st.withheld += "\r\n"
                    i += 2
                    continue
            # bare '\r' (next != '\n') or budget exhausted: non-newline at
            # head -> emit it + entire rest, RELEASED
            st.released = True
            return "".join(out) + frag[i:]
        # any other char: non-newline at head -> emit it + entire rest
        st.released = True
        return "".join(out) + frag[i:]
    # i == n: still ARMED (possibly holding a trailing '\r')
    return "".join(out)


def _finish_state(st: _StripState) -> str:
    """End-of-stream (correction 4 / R3): DROP the withheld newline units
    and any still-held pending CR; emit NOTHING. A verbatim flush has no
    legal SSE placement (bytes after [DONE]/message_stop/response.completed
    are malformed; synthetic delta events would fabricate ids/indices).
    Documented 1-byte edge: an answer ending in a bare CR after <=2
    withheld units loses that CR. A never-armed stream is byte-exact and
    held nothing. Returns "" always."""
    st.pending_cr = False
    st.withheld = ""
    return ""


def _strip_head(text: str, api: str) -> str:
    """Pure head strip = the streaming machine over the whole text: arm,
    feed the full text, finish(). Result: min(2,k) leading newline units
    removed (CRLF = 1 unit; a bare CR is never consumed as a newline unit).
    FRESH budget per call, no shared state, so JSON, SSE deltas and the
    Responses final-state events cannot diverge (corrections 5/9)."""
    st = _StripState(api=api)
    st.armed = True
    return _feed_state(st, text) + _finish_state(st)


# --- shared arm + feed used by the seam (streaming) and extract_reasoning --


def _arm_strip(st: _StripState, confirmed: bool):
    """Arm (once) on an ENGINE-CONFIRMED REASONING END. `confirmed` is the
    adapter's `has_engine_confirmed_reasoning_end()`
    (vllm/parser/engine/adapters.py — the same predicate
    DelegatingParser.parse_delta uses for the reasoning->content
    transition, so this is the engine's own reasoning-end confirmation, not
    a proxy). The engine-confirmed end covers BOTH a genuine think-close
    and the IMPLICIT end emitted by a tool-open (qwen3.py:126-129,
    137-145); the cause is not observable via the public API of this
    build, so the strip cannot (and does not try to) distinguish them —
    see the module docstring, "Accepted corner". Unlike the old
    first-non-None-reasoning rule, an EMPTY think (think-close with zero
    reasoning text, which vLLM maps to reasoning=None) still arms, because
    the engine confirms the end independently of the reasoning text. The
    head_passed guard is correction 3: a late arm after any non-empty
    content fragment is a no-op (never strips mid-answer content).
    Idempotent."""
    if confirmed and not st.head_passed:
        st.armed = True


def _feed_strip(st: _StripState, content):
    """Feed `content` through the one-shot state; return the (possibly
    stripped) content, or `content` unchanged. Arming is the CALLER's
    responsibility (`_arm_strip` with the engine-confirmed
    reasoning-end); an un-armed (IDLE) state passes byte-exact and holds
    nothing. `content` may be None (reasoning-only delta) — passed through
    unchanged."""
    if isinstance(content, str) and content:
        out = _feed_state(st, content)
        if out != content:
            return out
    return content


# --- the plugin adapter -----------------------------------------------------

class _Qwen3CleanEngine(Qwen3Parser):
    """Identity-distinct engine class: forces DelegatingParser composition
    even when the tool parser resolves to the same underlying Qwen3Parser
    engine (engine-collapse in ParserManager.get_parser, commit ac2ae8798c)."""


class Qwen3CleanReasoningAdapter(Qwen3ParserReasoningAdapter):
    """Built-in qwen3 reasoning adapter + one-shot leading-newline strip.

    Strip state lives on the INSTANCE. Instance lifetime in this vLLM
    version is exactly one completion request (one instance per choice for
    OpenAI n>1): all three serving entrypoints (chat/anthropic/responses)
    hold the DelegatingParser CLASS and instantiate it per request, and
    `Parser.__init__` creates a fresh reasoning adapter per DelegatingParser
    (see docs/08 for file:line evidence). The base class defines no
    reset/copy hooks, so nothing else needs overriding.

    This class overrides ONLY `__init__` (per-instance `_strip_state`)
    and `extract_reasoning` (non-streaming). The streaming methods
    (`extract_reasoning_streaming`, `finish_streaming`) are inherited
    UNTOUCHED from the base — vLLM internals stay byte-pristine; the
    streaming strip happens exclusively in the parse_delta seam below,
    which consumes `_strip_state` (the DelegatingParser instance reaches
    it via `self._reasoning_parser`). One budget, one emission point
    (the tool-phase message), no double strip — see the module
    docstring, "Where the artifact is emitted".

    `api="openai"` in the state machine is a bookkeeping label only (the
    three entrypoints all converge on this one parser instance; the
    machine's behavior does not depend on it).
    """

    # Identity-distinct engine: defeats ParserManager.get_parser
    # engine-collapse — see _Qwen3CleanEngine above.
    _parser_engine_cls = _Qwen3CleanEngine

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._strip_state = _StripState(api="openai")

    def extract_reasoning(self, model_output, request):
        reasoning, content = super().extract_reasoning(model_output, request)
        # Non-stream: super() feeds the WHOLE model_output through the
        # parser engine (ParserEngine.extract_reasoning), so the
        # engine-confirmed reasoning end is observable on this instance
        # right after super() returns — verified in-container: empty-think
        # text (reasoning=None) sets it to True. Arm on THAT (covers the
        # empty-think case the old `isinstance(reasoning, str)` proxy
        # missed), then apply the PURE head strip with a FRESH budget over
        # the whole text, so JSON and SSE delta paths cannot diverge.
        # No head_passed guard applies here: the strip is one-shot over
        # the whole text and nothing has been emitted yet.
        if (isinstance(content, str) and content
                and self.has_engine_confirmed_reasoning_end()):
            stripped = _strip_head(content, "openai")
            if stripped != content:
                content = stripped
        return reasoning, content


# Eager registration: checked before the lazy built-in dict by
# ReasoningParserManager.get_reasoning_parser, so "qwen3_clean" resolves
# to this class. force=True is the default; nothing registers this name
# before us.
ReasoningParserManager.register_module(
    "qwen3_clean", module=Qwen3CleanReasoningAdapter
)

# --- parse_delta seam (THE streaming emission point) ------------------------
#
# For engine-based parsers (qwen3), DelegatingParser.parse_delta nulls the
# transition delta's content and re-feeds it into the tool phase, so EVERY
# outgoing streaming content byte exits via the tool-phase message:
# buffered post-transition text (e.g. the "\n\n" after a think-close, or
# the tool-open markup after an implicit end) and later plain content
# deltas alike. The seam on the returned DeltaMessage is therefore the
# single place every streaming content byte crosses — arm on the
# engine-confirmed reasoning end, feed content through the adapter
# instance's shared one-shot state. All three serving entrypoints consume
# the DeltaMessage RETURNED by parse_delta, so one seam covers
# chat/anthropic/responses with a single budget (verified in-container on
# the pinned build, including post-tool content: a tool-open arms, and a
# trailing "\n\nAnswer" after the tool XML is emitted as content).
#
# The seam is a no-op for any parser instance whose reasoning parser is not
# Qwen3CleanReasoningAdapter (opt-in via the "qwen3_clean" registry name),
# and it never runs for parse() (non-streaming — covered by the adapter's
# extract_reasoning override).

_ORIG_PARSE_DELTA = DelegatingParser.parse_delta


def _clean_parse_delta(self, *args, **kwargs):
    msg = _ORIG_PARSE_DELTA(self, *args, **kwargs)
    if msg is None:
        return msg
    rp = getattr(self, "_reasoning_parser", None)
    if not isinstance(rp, Qwen3CleanReasoningAdapter):
        return msg
    _arm_strip(rp._strip_state, rp.has_engine_confirmed_reasoning_end())
    msg.content = _feed_strip(rp._strip_state, msg.content)
    return msg


DelegatingParser.parse_delta = _clean_parse_delta
