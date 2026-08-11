"""MLX backend — native Apple Silicon inference for the local pilot.

Why MLX rather than vLLM or Ollama on a Mac:

  vLLM   has no Metal backend. The macOS path is experimental CPU-only, so it
         would be slow AND unrepresentative of the production config.
  Ollama takes strings, not token IDs. Every prompt would be re-tokenised,
         destroying the token parity the whole causal claim rests on.
  MLX    runs natively on Metal, accepts token IDs, and exposes raw logits,
         which is exactly the readout method.

SCOPE. This backend exists to validate the code path and produce an early CPR
reading. Numbers from a 4-bit local run must NOT appear in the paper —
quantisation is a confound the design explicitly controls for. Production runs
bf16 on rented hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from .backends import ACTION_SURFACE_FORMS_BY_FRAMING, Decision
from .config import Action, Framing, ModelConfig, ReadoutMode

logger = logging.getLogger(__name__)

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class NoSpecialTokenizer:
    """Wraps a tokenizer so encode() NEVER prepends BOS or other special tokens.

    Mistral's SentencePiece tokenizer adds BOS on every encode() call, so "\\n"
    came back as 3 tokens and the single-token filler search failed outright.
    The deeper problem is worse than the crash: without this, every prompt
    SECTION and every scaffold block would carry its own embedded BOS, scattering
    special tokens through the middle of the prompt and inflating the token
    counts that parity is asserted on.

    Llama-3.1 and Qwen2.5 happened not to hit this, which is exactly why it went
    unnoticed until a third tokenizer was tried.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._supports_flag = self._probe_flag()

    def _probe_flag(self) -> bool:
        try:
            self._inner.encode("a", add_special_tokens=False)
            return True
        except TypeError:
            return False

    def encode(self, text: str) -> list[int]:
        if self._supports_flag:
            return self._inner.encode(text, add_special_tokens=False)
        ids = self._inner.encode(text)
        bos = getattr(self._inner, "bos_token_id", None)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        return ids

    def decode(self, ids) -> str:
        return self._inner.decode(list(ids))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MLXBackend:
    """Implements the LLMBackend protocol on top of mlx-lm."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        apply_chat_template: bool = True,
        swap_labels: bool = False,
    ) -> None:
        self.swap_labels = swap_labels
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MLX backend requires: pip install mlx mlx-lm\n"
                "Apple Silicon only. On the rented GPU use VLLMBackend instead."
            ) from exc

        self._mx = mx
        self._nn = nn
        self.model_config = model_config or ModelConfig(model_id=DEFAULT_MLX_MODEL)
        self.model, raw_tokenizer = load(self.model_config.model_id)
        # Wrap immediately: every downstream encode() must be free of
        # special tokens, or scaffold blocks carry embedded BOS and the
        # asserted token parity is measuring the wrong thing.
        self.tokenizer = NoSpecialTokenizer(raw_tokenizer)

        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        if apply_chat_template:
            self._prefix_ids, self._suffix_ids = self._chat_affixes()

        self._action_token_ids = self._resolve_action_tokens()

    # ---- setup -----------------------------------------------------------

    def _chat_affixes(self) -> tuple[list[int], list[int]]:
        """Split the chat template around a sentinel so the assembled body can be
        inserted as token IDs, untouched.

        Instruct models behave very differently without their template applied,
        so skipping it would measure the wrong thing. But re-encoding the whole
        prompt as text would let BPE merge across the scaffold boundary and
        silently break the token parity we just asserted. Splitting the template
        into fixed prefix/suffix ID lists avoids both problems, and because the
        affixes are identical across arms, parity is preserved.
        """
        sentinel = "<<<BODY>>>"
        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sentinel}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - template-less tokenizers
            logger.warning("no chat template available; running on raw text")
            return [], []

        if sentinel not in rendered:
            logger.warning("chat template did not preserve sentinel; running raw")
            return [], []

        head, tail = rendered.split(sentinel, 1)
        # The chat prefix legitimately needs its BOS, so take it from the raw
        # tokenizer; everything else must not have one.
        raw = self.tokenizer._inner
        try:
            head_ids = raw.encode(head)
        except Exception:
            head_ids = self.tokenizer.encode(head)
        return head_ids, self.tokenizer.encode(tail)

    def _resolve_action_tokens(self) -> dict[Framing, dict[Action, list[int]]]:
        """First-token IDs for every surface form, PER FRAMING.

        Framing-awareness is not optional. A semantic-framing prompt asks for
        "Cooperate"/"Defect" while the abstract one asks for "X"/"Y"; measuring
        mass on the wrong pair gives a 100% off-task rate and actions decided
        entirely by the tie-break.
        """
        resolved: dict[Framing, dict[Action, list[int]]] = {}
        for framing, forms_by_action in ACTION_SURFACE_FORMS_BY_FRAMING.items():
            if self.swap_labels:
                # Labels are inverted in the prompt, so the token that means
                # COOPERATE is the one that normally means DEFECT. Failing to
                # mirror this here would silently invert every recorded action.
                forms_by_action = {
                    Action.COOPERATE: forms_by_action[Action.DEFECT],
                    Action.DEFECT: forms_by_action[Action.COOPERATE],
                }
            per_action: dict[Action, list[int]] = {}
            for action, forms in forms_by_action.items():
                ids: list[int] = []
                for form in forms:
                    encoded = self.tokenizer.encode(form)
                    if encoded and encoded[0] not in ids:
                        ids.append(encoded[0])
                per_action[action] = ids
                logger.info("%s/%s -> token ids %s", framing.value, action.value, ids)
            resolved[framing] = per_action
        return resolved

    # ---- inference -------------------------------------------------------

    def _next_token_probs(self, token_ids: Sequence[int]):
        mx = self._mx
        wrapped = self._prefix_ids + list(token_ids) + self._suffix_ids
        logits = self.model(mx.array([wrapped]))
        return self._nn.softmax(logits[0, -1, :].astype(mx.float32))

    def decide(
        self,
        prompt_token_ids: Sequence[int],
        readout_mode: ReadoutMode,
        seed: int,
        framing: Framing = Framing.ABSTRACT,
    ) -> Decision:
        probs = self._next_token_probs(prompt_token_ids)

        mass = {
            action: float(sum(probs[i].item() for i in ids))
            for action, ids in self._action_token_ids[framing].items()
        }
        p_c = mass[Action.COOPERATE]
        p_d = mass[Action.DEFECT]
        total = p_c + p_d

        if readout_mode is ReadoutMode.SCRATCHPAD:
            scratchpad = self._generate(prompt_token_ids, self.model_config.max_scratchpad_tokens)
            action = _parse_action(scratchpad) or self._sample_action(p_c, p_d, seed)
        else:
            scratchpad = None
            action = self._sample_action(p_c, p_d, seed)

        top = self._top_tokens(probs, k=5)

        return Decision(
            action=action,
            logit_mass_cooperate=p_c,
            logit_mass_defect=p_d,
            action_mass_total=total,
            logit_gap=abs(p_d - p_c),
            top_tokens=top,
            scratchpad=scratchpad,
        )

    # Probe answers get more room than seems necessary. A 12-token budget was
    # enough for a bare "36" but not for "Your current total score is 36", and
    # instruct models add more preamble as the context grows. That produces a
    # CPR curve that looks like comprehension collapsing at a specific turn when
    # it is actually the answer being cut off before the number appears.
    # max_tokens is an implementation parameter, not part of the pre-registered
    # probe wording, so widening it before data collection is not a deviation.
    PROBE_MAX_TOKENS = 32

    @staticmethod
    def _sample_action(p_c: float, p_d: float, seed: int) -> Action:
        """Sample from the renormalised two-way action distribution.

        NOT argmax. Argmax plus a deterministic scripted opponent makes every
        episode in a cell byte-identical, so N episodes yield exactly one
        trajectory and every confidence interval computed over them is a
        fiction. Observed live: 15 episodes produced CPR 5/10 fifteen times.

        Sampling is seeded from the episode coordinates, so runs stay
        reproducible while episodes remain genuinely distinct. It also matches
        the configured temperature, which argmax silently ignored.
        """
        import random

        total = p_c + p_d
        if total <= 0:
            return Action.COOPERATE
        return (
            Action.DEFECT
            if random.Random(seed).random() < (p_d / total)
            else Action.COOPERATE
        )

    def probe(self, prompt_token_ids: Sequence[int], seed: int) -> str:
        return self._generate(prompt_token_ids, max_tokens=self.PROBE_MAX_TOKENS).strip()

    def _generate(self, prompt_token_ids: Sequence[int], max_tokens: int) -> str:
        from mlx_lm import generate

        text = self.tokenizer.decode(list(prompt_token_ids))
        try:
            return generate(
                self.model,
                self.tokenizer,
                prompt=text,
                max_tokens=max_tokens,
                verbose=False,
            )
        except TypeError:  # older mlx-lm signature
            return generate(self.model, self.tokenizer, text, max_tokens=max_tokens)

    def _top_tokens(self, probs, k: int = 5) -> tuple[tuple[str, float], ...]:
        mx = self._mx
        idx = mx.argsort(probs)[-k:].tolist()[::-1]
        return tuple(
            (self.tokenizer.decode([i]), round(float(probs[i].item()), 6)) for i in idx
        )


def _accepts_special(tokenizer) -> bool:
    try:
        tokenizer.encode("a", add_special_tokens=False)
        return True
    except TypeError:
        return False


def _parse_action(text: str) -> Action | None:
    lowered = text.strip().lower()
    for token, action in (
        ("cooperate", Action.COOPERATE),
        ("defect", Action.DEFECT),
    ):
        if token in lowered:
            return action
    for char, action in (("x", Action.COOPERATE), ("y", Action.DEFECT)):
        if lowered.startswith(char):
            return action
    return None
