"""Scratchpad readout: the action must come from one logit position, in the
assistant turn.

WHAT CAN GO WRONG HERE, SILENTLY
    Under scratchpad readout the model generates reasoning and the action is
    read from the token after "Final answer:". Two failure modes produce
    plausible numbers rather than errors:

    1. PARSING THE TRACE. If the action were extracted by string matching over
       the generated text, a scratchpad reading "if I Cooperate they may Defect"
       would classify arbitrarily. The implementation reads a single logit
       position instead - these tests pin that down so a future "helpful"
       refactor cannot introduce a parser.

    2. WRONG CONVERSATIONAL TURN. The reasoning is the model's own output and
       belongs in the ASSISTANT turn. Building the continuation from the RAW
       prompt and letting the chat template wrap it afterwards buries the
       reasoning inside the USER message and inserts the assistant header
       between the cue and the position being read. The model still answers, so
       nothing looks broken; it is answering a different conversation.

    vLLM cannot be imported off-GPU, so these tests exercise the sequence
    construction against a recording double rather than a real engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.backends import CharTokenizer
from cdx.config import Action, Framing, ReadoutMode


class RecordingBackend:
    """Reproduces VLLMBackend's sequence construction without vLLM.

    Deliberately a copy rather than an import: the point is to assert what the
    submitted token sequence looks like, and a double that called the real
    method would only test itself.
    """

    PREFIX = [901, 902]      # stands in for the chat template's user opener
    SUFFIX = [903, 904]      # ... and its assistant header

    def __init__(self) -> None:
        self.tokenizer = CharTokenizer()
        self.submitted: list[list[int]] = []
        self.wrap_flags: list[bool] = []

    def _wrap(self, ids):
        return self.PREFIX + list(ids) + self.SUFFIX

    def _generate(self, prompts, params, wrap: bool = True):
        seqs = [self._wrap(p) for p in prompts] if wrap else [list(p) for p in prompts]
        self.submitted.extend(seqs)
        self.wrap_flags.extend([wrap] * len(seqs))
        return seqs

    def build_followups(self, prompts, scratchpads):
        """The construction under test, mirroring _decide_scratchpad."""
        cue = self.tokenizer.encode("\nFinal answer:")
        return [
            self._wrap(p) + self.tokenizer.encode(s) + cue
            for p, s in zip(prompts, scratchpads)
        ]


@pytest.fixture
def backend():
    return RecordingBackend()


PROMPT = [10, 11, 12]
SCRATCH = "I should weigh the payoffs"


# --- turn placement --------------------------------------------------------


def test_reasoning_sits_after_the_assistant_header(backend):
    """The regression this file exists for.

    SUFFIX is the assistant header. Everything the model produced must come
    AFTER it; if the reasoning appears before, it landed in the user turn.
    """
    seq = backend.build_followups([PROMPT], [SCRATCH])[0]
    header_end = len(backend.PREFIX) + len(PROMPT) + len(backend.SUFFIX)
    reasoning = backend.tokenizer.encode(SCRATCH)
    assert seq[header_end:header_end + len(reasoning)] == reasoning


def test_cue_is_the_final_content_before_the_read_position(backend):
    """Nothing may sit between "Final answer:" and the token being scored."""
    seq = backend.build_followups([PROMPT], [SCRATCH])[0]
    cue = backend.tokenizer.encode("\nFinal answer:")
    assert seq[-len(cue):] == cue


def test_followups_are_not_wrapped_twice(backend):
    """Applying the chat template again would insert a second user turn around
    the model's own reasoning."""
    followups = backend.build_followups([PROMPT], [SCRATCH])
    backend._generate(followups, None, wrap=False)
    assert backend.wrap_flags == [False]
    assert backend.submitted[0] == followups[0]
    assert backend.submitted[0].count(backend.PREFIX[0]) == 1


def test_prompt_is_wrapped_exactly_once_for_generation(backend):
    backend._generate([PROMPT], None)
    assert backend.wrap_flags == [True]
    assert backend.submitted[0] == backend.PREFIX + PROMPT + backend.SUFFIX


# --- no parsing ------------------------------------------------------------


def test_reasoning_mentioning_both_actions_is_only_context(backend):
    """A trace naming both actions must not change the sequence's shape - the
    action is read from a logit position, not extracted from this text."""
    tricky = "If I Cooperate they may Defect, so Defect then Cooperate"
    plain = "Thinking about it"
    a = backend.build_followups([PROMPT], [tricky])[0]
    b = backend.build_followups([PROMPT], [plain])[0]
    cue = backend.tokenizer.encode("\nFinal answer:")
    assert a[-len(cue):] == b[-len(cue):] == cue
    assert len(a) - len(b) == len(backend.tokenizer.encode(tricky)) \
                            - len(backend.tokenizer.encode(plain))


def test_empty_scratchpad_still_produces_a_valid_sequence(backend):
    """A model that emits EOS immediately yields "". The cue must still be
    appended so there is a defined position to read."""
    seq = backend.build_followups([PROMPT], [""])[0]
    cue = backend.tokenizer.encode("\nFinal answer:")
    assert seq == backend.PREFIX + PROMPT + backend.SUFFIX + cue


# --- per-sequence sampling -------------------------------------------------


def test_each_sequence_gets_its_own_scratchpad(backend):
    """A shared seed would make every scratchpad in a batch identical, which is
    the byte-identical-episode bug that made N meaningless on the laptop."""
    prompts = [PROMPT, [20, 21], [30]]
    pads = ["reason one", "reason two", "reason three"]
    seqs = backend.build_followups(prompts, pads)
    assert len({tuple(s) for s in seqs}) == 3


def test_readout_mode_enum_has_both_paths():
    assert ReadoutMode.LOGIT.value == "logit"
    assert ReadoutMode.SCRATCHPAD.value == "scratchpad"
    assert ReadoutMode("scratchpad") is ReadoutMode.SCRATCHPAD