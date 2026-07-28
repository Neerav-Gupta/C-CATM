"""
Base class shared by all multi-turn jailbreak attackers.

Handles the model-agnostic mechanics common to every attack strategy:
turn-by-turn tokenization, chat-template formatting (with special-casing
for Gemma models, whose tokenizer/template handling differs from the
standard HF interface), target-model generation, and hidden-state
extraction used downstream by the monitors. Subclasses (crescendo,
jigsaw, echo_chamber, safe_in_isolation) only need to implement the
turn-construction strategy and call into these shared helpers.
"""
import torch
from typing import Dict, Any
from ..utils.data_structures import AttackTrace


class MultiTurnAttacker:
    """Base class for multi-turn attackers targeting a chat model."""

    def __init__(self, target: Dict[str, Any], max_turns: int = 10):
        """
        Args:
            target: dict with "model", "tokenizer", and "repo_id" for the
                target model being attacked.
            max_turns: maximum number of attack turns to run.
        """
        self.model = target["model"]
        self.tokenizer = target["tokenizer"]
        self.repo_id = target["repo_id"]
        self.max_turns = max_turns

    def get_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return the mean-pooled last-layer hidden
        state for the given input, used as the per-turn representation
        fed to downstream monitors.
        """
        input_ids = input_ids.long()
        with torch.inference_mode():
            out = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )
        return out.hidden_states[-1].mean(dim=1).squeeze(0).cpu().float()

    def generate_response(
        self,
        conversation,
        max_new_tokens=256,
        temperature=0.0
    ):
        """
        Generate the target model's next response to `conversation`.

        Args:
            conversation: list of {"role", "content"} dicts (chat history).
            max_new_tokens: generation cap (reduced for Gemma below).
            temperature: 0.0 for greedy decoding, otherwise sampling.

        Returns:
            (response_text, input_ids) tuple — input_ids is returned so
            callers can compute hidden states for the prompt that was
            actually fed to the model.
        """
        is_gemma = "gemma" in self.repo_id.lower()

        if is_gemma:
            # Gemma3 has a much smaller effective context/generation budget
            # in our setup; cap generation length to avoid OOM/timeout.
            max_new_tokens = min(max_new_tokens, 128)

        # Get eos_token_id correctly for both model types
        if is_gemma:
            eos_token_id = self.tokenizer.tokenizer.eos_token_id
        else:
            eos_token_id = self.tokenizer.eos_token_id

        if is_gemma:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Some Gemma tokenizer/wrapper combinations don't ship a
                # usable chat template; fall back to manually constructing
                # the Gemma turn format below rather than failing the run.
                formatted = None

            if not formatted:
                formatted = ""
                for turn in conversation:
                    formatted += f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>\n"
                formatted += "<start_of_turn>model\n"

            inputs = self.tokenizer(
                text=formatted,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
        else:
            formatted = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)

        inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        with torch.inference_mode():
            if temperature == 0.0:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=eos_token_id
                )
            else:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=eos_token_id
                )

        response = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if not response:
            # Guard against empty generations (e.g. immediate EOS) so
            # downstream turn-building always has non-empty text to work with.
            response = "I cannot provide information on that topic."

        return response, inputs["input_ids"]

    def run(self, target_behavior: str) -> AttackTrace:
        """
        Run the full multi-turn attack against `target_behavior` and
        return the resulting AttackTrace. Must be implemented by subclasses,
        each of which encodes a different turn-construction strategy.

        Args:
            target_behavior: the harmful behavior string the attack is
                trying to elicit from the target model.

        Returns:
            AttackTrace populated with the conversation turns.
        """
        if not target_behavior or not target_behavior.strip():
            raise ValueError(f"Empty target behavior passed to attacker")
        raise NotImplementedError

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text correctly for both standard models and Gemma3.
        Returns input_ids tensor.
        """
        is_gemma = "gemma" in self.repo_id.lower()
        if is_gemma:
            inputs = self.tokenizer(text=text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True
            ).to(self.model.device)
        inputs["input_ids"] = inputs["input_ids"].long()
        return inputs["input_ids"]