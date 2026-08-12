"""vLLM backend — production inference, GPU only.

The critical difference from the MLX backend is BATCHING. vLLM's throughput comes
from filling the card with many sequences at once; calling it one decision at a
time wastes most of the GPU you are paying for. So this backend exposes
`decide_batch` and `probe_batch`, and the GPU runner advances many episodes in
lockstep to keep those batches full.

Rough scale: the MLX laptop path does ~0.4 decisions/s. A batched A100 does
1-2 orders of magnitude more. That difference is entirely batching, not the card.

Two invariants carried over from the laptop work, both learned the hard way:

  * Prompts are passed as TOKEN IDS, never re-encoded text. Re-encoding lets BPE
    merge across the scaffold boundary and silently breaks the token parity the
    causal claim rests on.
  * encode() must not add special tokens, or every prompt section carries its own
    embedded BOS. Mistral does this; Llama and Qwen do not, which is why it took
    a third tokenizer to notice.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from .backends import ACTION_SURFACE_FORMS_BY_FRAMING, Decision
from .config import Action, Framing, ModelConfig, ReadoutMode

logger = logging.getLogger(__name__)

# Number of top logprobs to request. Must comfortably exceed the number of action
# surface forms so the mass calculation is not truncated, and large enough that
# `top_tokens` shows what the model wanted when it goes off-task.
# Top-K logprobs requested per decision.
#
# This is a LOWER BOUND mechanism: if an action surface form falls outside
# the top-K, its probability reads as 0 and the turn is misreported as
# off-task. With 4 surface forms per action that is a real risk at small K,
# so K is set generously. `action_tokens_found` records how many were
# actually present, making truncation visible instead of silent.
# DEFAULT ONLY. Override per run with --logprobs-top-k; the effective value is
# written to run_meta via config_json.
#
# 20 because that is what exp3 ran, and changing it between experiments would
# alter measured action mass for no reason. It is also the cap enforced by some
# vLLM builds unless max_logprobs is raised in the engine constructor - vLLM
# 0.27 rejected a request for 60 outright, which killed a run mid-sweep.
#
# Raising it is safe where the engine allows it and reduces truncation risk when
# action tokens sit deep in the distribution - plausible under SCRATCHPAD, where
# the action follows generated reasoning. But raise it deliberately, per run,
# not by editing this line.
LOGPROBS_TOP_K = 20


class VLLMBackend:
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        swap_labels: bool = False,
        apply_chat_template: bool = True,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        logprobs_top_k: int = LOGPROBS_TOP_K,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - GPU only
            raise ImportError(
                "pip install vllm transformers  (GPU required; use MLXBackend on a Mac)"
            ) from exc

        self.model_config = model_config
        self.swap_labels = swap_labels
        self.logprobs_top_k = logprobs_top_k

        from .backends_mlx import NoSpecialTokenizer  # shared wrapper

        self.tokenizer = NoSpecialTokenizer(
            AutoTokenizer.from_pretrained(model_config.model_id)
        )
        self.llm = LLM(
            model=model_config.model_id,
            dtype=model_config.dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # vLLM caps per-request logprobs at 20 unless the engine is told
            # otherwise, and the cap is enforced at request time with a
            # validation error. Declaring it here is the only way to ask for
            # more than 20 later.
            max_logprobs=self.logprobs_top_k,
            enforce_eager=False,
        )

        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        if apply_chat_template:
            self._prefix_ids, self._suffix_ids = self._chat_affixes()

        self._action_token_ids = self._resolve_action_tokens()

    # ---- setup -----------------------------------------------------------

    def _chat_affixes(self) -> tuple[list[int], list[int]]:
        sentinel = "<<<BODY>>>"
        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sentinel}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.warning("no chat template; running raw")
            return [], []
        if sentinel not in rendered:
            logger.warning("chat template dropped sentinel; running raw")
            return [], []
        head, tail = rendered.split(sentinel, 1)
        return self.tokenizer._inner.encode(head), self.tokenizer.encode(tail)

    def _resolve_action_tokens(self) -> dict[Framing, dict[Action, list[int]]]:
        resolved: dict[Framing, dict[Action, list[int]]] = {}
        for framing, forms_by_action in ACTION_SURFACE_FORMS_BY_FRAMING.items():
            if self.swap_labels:
                forms_by_action = {
                    Action.COOPERATE: forms_by_action[Action.DEFECT],
                    Action.DEFECT: forms_by_action[Action.COOPERATE],
                }
            per_action: dict[Action, list[int]] = {}
            for action, forms in forms_by_action.items():
                ids: list[int] = []
                for form in forms:
                    enc = self.tokenizer.encode(form)
                    if enc and enc[0] not in ids:
                        ids.append(enc[0])
                per_action[action] = ids
            resolved[framing] = per_action
            logger.info(
                "%s action tokens: C=%s D=%s",
                framing.value,
                per_action[Action.COOPERATE],
                per_action[Action.DEFECT],
            )
        return resolved

    def _wrap(self, token_ids: Sequence[int]) -> list[int]:
        return self._prefix_ids + list(token_ids) + self._suffix_ids

    # ---- batched inference ----------------------------------------------

    def decide_batch(
        self,
        prompts: Sequence[Sequence[int]],
        readout_mode: ReadoutMode,
        seeds: Sequence[int],
        framing: Framing = Framing.ABSTRACT,
    ) -> list[Decision]:
        """One vLLM call for the whole batch. This is where the speed lives."""
        from vllm import SamplingParams

        if len(prompts) != len(seeds):
            raise ValueError(f"prompts/seeds length mismatch: {len(prompts)} vs {len(seeds)}")

        if readout_mode is ReadoutMode.SCRATCHPAD:
            return self._decide_scratchpad(prompts, seeds, framing)

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,          # sampling happens in _sample_action, seeded
            logprobs=self.logprobs_top_k,
        )
        outputs = self._generate(prompts, params)
        return [
            self._decision_from_logprobs(out, seed, framing, scratchpad=None)
            for out, seed in zip(outputs, seeds)
        ]

    def _decide_scratchpad(self, prompts, seeds, framing) -> list[Decision]:
        """Generate reasoning, then read the action from the continuation.

        THE ACTION IS READ FROM EXACTLY ONE LOGIT POSITION - the token after
        "Final answer:". The reasoning text is context, never parsed. So a
        scratchpad that says "if I Cooperate they may Defect" cannot contaminate
        the classification; there is no string matching anywhere in this path.

        TURN PLACEMENT MATTERS AND IS EASY TO GET WRONG.
            The reasoning is the model's own output and must sit in the
            ASSISTANT turn. An earlier version appended it to the raw prompt and
            let _generate apply the chat template afterwards, which placed the
            model's reasoning inside the USER message and put the template's
            assistant header between "Final answer:" and the position being
            read. The model would still answer, so nothing would look broken -
            it would just be answering a differently-structured conversation
            from the one intended, in every scratchpad cell.

            So: wrap first, then append. The continuation is built on the
            already-templated sequence.
        """
        from vllm import SamplingParams

        # One SamplingParams per sequence: a single shared seed would make every
        # scratchpad in the batch identical, recreating the byte-identical-episode
        # bug that made N meaningless on the laptop.
        gen = [
            SamplingParams(
                max_tokens=self.model_config.max_scratchpad_tokens,
                temperature=self.model_config.temperature,
                seed=int(s) % (2**31 - 1),
            )
            for s in seeds
        ]
        scratchpads = [o.outputs[0].text for o in self._generate(prompts, gen)]

        cue = self.tokenizer.encode("\nFinal answer:")
        followups = [
            self._wrap(p) + self.tokenizer.encode(s) + cue
            for p, s in zip(prompts, scratchpads)
        ]
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.logprobs_top_k)
        outputs = self._generate(followups, params, wrap=False)
        return [
            self._decision_from_logprobs(out, seed, framing, scratchpad=sp)
            for out, seed, sp in zip(outputs, seeds, scratchpads)
        ]

    def probe_batch(self, prompts: Sequence[Sequence[int]], seeds: Sequence[int]) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=32, temperature=0.0)
        return [o.outputs[0].text.strip() for o in self._generate(prompts, params)]

    def _generate(self, prompts: Sequence[Sequence[int]], params, wrap: bool = True):
        """Submit token IDs, never text.

        vLLM's API for this moved between versions, so both spellings are tried.
        Falling back to a text prompt would defeat token parity, so there is no
        text fallback - if both fail, that is a hard error worth surfacing.

        wrap=False is for sequences that already carry the chat template, which
        is the scratchpad continuation: applying the template twice would bury
        the reasoning inside a second user turn.
        """
        wrapped = [self._wrap(p) for p in prompts] if wrap else [list(p) for p in prompts]
        errors = []

        # Newer vLLM: TokensPrompt objects.
        try:
            from vllm import TokensPrompt

            return self.llm.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in wrapped], params
            )
        except Exception as exc:  # noqa: BLE001 - try the older spelling
            errors.append(f"TokensPrompt: {type(exc).__name__}: {exc}")

        # Older vLLM: dict prompts.
        try:
            return self.llm.generate(
                [{"prompt_token_ids": ids} for ids in wrapped], params
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dict prompt: {type(exc).__name__}: {exc}")

        # Oldest vLLM: keyword argument.
        try:
            return self.llm.generate(prompt_token_ids=wrapped, sampling_params=params)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kwarg: {type(exc).__name__}: {exc}")

        # There is deliberately NO text fallback. Re-encoding the prompt as a
        # string would let BPE merge across the scaffold boundary and destroy the
        # token parity the causal claim depends on. Better to stop.
        raise RuntimeError(
            "Could not submit token IDs to vLLM. Tried three API spellings:\n  "
            + "\n  ".join(errors)
            + "\nRefusing to fall back to text prompts: that would break token parity."
        )

    # ---- decoding --------------------------------------------------------

    def _decision_from_logprobs(self, output, seed: int, framing: Framing, scratchpad):
        logprobs = output.outputs[0].logprobs
        dist: dict[int, float] = {}
        if logprobs:
            for token_id, lp in logprobs[0].items():
                value = lp.logprob if hasattr(lp, "logprob") else float(lp)
                dist[int(token_id)] = math.exp(value)

        ids = self._action_token_ids[framing]
        wanted = list(ids[Action.COOPERATE]) + list(ids[Action.DEFECT])
        found = sum(1 for i in wanted if i in dist)
        if found == 0:
            logger.debug(
                'no action token in top-%d; mass reads as 0 (off-task)',
                self.logprobs_top_k
            )
        p_c = sum(dist.get(i, 0.0) for i in ids[Action.COOPERATE])
        p_d = sum(dist.get(i, 0.0) for i in ids[Action.DEFECT])

        top = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
        top_tokens = tuple(
            (self.tokenizer.decode([tid]), round(p, 6)) for tid, p in top
        )

        return Decision(
            action=_sample_action(p_c, p_d, seed),
            logit_mass_cooperate=p_c,
            logit_mass_defect=p_d,
            action_mass_total=p_c + p_d,
            logit_gap=abs(p_d - p_c),
            top_tokens=top_tokens,
            scratchpad=scratchpad,
        )

    # ---- single-call compatibility --------------------------------------

    def decide(self, prompt_token_ids, readout_mode, seed, framing=Framing.ABSTRACT):
        """Unbatched path. Correct, but wastes the GPU — use decide_batch."""
        return self.decide_batch([prompt_token_ids], readout_mode, [seed], framing)[0]

    def probe(self, prompt_token_ids, seed: int) -> str:
        return self.probe_batch([prompt_token_ids], [seed])[0]


def _sample_action(p_c: float, p_d: float, seed: int) -> Action:
    """Seeded sample from the renormalised two-way distribution.

    Not argmax: argmax against a deterministic opponent makes every episode in a
    cell byte-identical, so N episodes yield one trajectory and every interval
    computed over them is a fiction. Observed on the laptop before this was fixed.
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