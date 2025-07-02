from __future__ import annotations
"""A token‑level replay buffer that supports TD(0) sampling for Llama‑3 8B.

Revision 4 – merge‑friendly `load` + earlier tweaks
--------------------------------------------------
* **`load()` now *appends***: you can call ``buffer.load(dir)`` as many times
  to merge multiple conversation dumps.  It prefers a fast tensor snapshot
  (``tensors.pt``) and falls back to the old JSON pair.
* Removed the fragile *match* logic from the previous implementation.
* `clear()` unchanged; full reset is still `buffer.clear()`.
"""

from typing import List, Tuple
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase  # type: ignore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_non_none_attr(obj, name: str) -> bool:
    return getattr(obj, name, None) is not None


# ---------------------------------------------------------------------------
# Main buffer
# ---------------------------------------------------------------------------

class EfficientReplayBuffer(Dataset):
    """Token‑level replay buffer for TD(0)."""

    # .....................................................................
    # Construction
    # .....................................................................
    def __init__(
        self,
        seq_len: int = 2048,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__()
        if tokenizer is None:
            raise ValueError("A tokenizer must be provided.")
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Special tokens ---------------------------------------------------
        self.pad_token_id = (
            self.tokenizer.pad_token_id if _has_non_none_attr(self.tokenizer, "pad_token_id") else self.tokenizer.eos_token_id
        )
        self.eot_token_id = (
            self.tokenizer.eot_token_id  # type: ignore[attr-defined]
            if _has_non_none_attr(self.tokenizer, "eot_token_id")
            else self.tokenizer("<|eot|>", add_special_tokens=False)["input_ids"][0]
        )

        # Flat storage -----------------------------------------------------
        self.tokens: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.rewards: torch.Tensor = torch.empty(0, dtype=torch.float32)
        self.dones: torch.Tensor = torch.empty(0, dtype=torch.bool)

        self.action_indices: List[int] = []  # every assistant token
        self.transitions: List[Tuple[int, float, bool]] = []  # final‑token rewards
        self.messages: List[dict] = []  # optional, not used at runtime

    # .....................................................................
    # push / add
    # .....................................................................
    def push(self, message: dict, reward: float | None = None, done: bool = False):
        if not {"role", "content"}.issubset(message):
            raise ValueError("Message must contain 'role' and 'content'.")
        self.messages.append(message)

        # 1. Chat‑template → raw text -------------------------------------
        if _has_non_none_attr(self.tokenizer, "apply_chat_template") and _has_non_none_attr(self.tokenizer, "chat_template"):
            text = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
        else:
            START, END, EOT, BEGIN = "<|start_header_id|>", "<|end_header_id|>", "<|eot|>", "<|begin_of_text|>"
            text = f"{BEGIN}\n{START}{message['role']}{END}\n{message['content']}{EOT}\n"

        token_ids: List[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        num_new = len(token_ids)
        base_idx = len(self.tokens)

        # 2. Reward / done tensors ---------------------------------------
        reward_tensor = torch.zeros(num_new, dtype=torch.float32)
        done_tensor = torch.zeros(num_new, dtype=torch.bool)

        if message["role"] == "assistant":
            self.action_indices.extend(range(base_idx, base_idx + num_new))
            if reward is not None:
                reward_tensor[-1] = float(reward)
                self.transitions.append((base_idx + num_new - 1, float(reward), done))

        if done:
            done_tensor[-1] = True

        # 3. Append flat tensors ------------------------------------------
        new_tok = torch.tensor(token_ids, dtype=torch.long)
        self.tokens = torch.cat([self.tokens, new_tok]) if self.tokens.numel() else new_tok
        self.rewards = torch.cat([self.rewards, reward_tensor]) if self.rewards.numel() else reward_tensor
        self.dones = torch.cat([self.dones, done_tensor]) if self.dones.numel() else done_tensor

        # 4. Auto‑insert EOT if not present -------------------------------
        if done and (self.tokens[-1].item() != self.eot_token_id):
            self._insert_eot()

    add = push  # alias

    # .....................................................................
    # clear (unchanged from rev‑3)
    # .....................................................................
    def clear(self, n: int | None = None):
        if n is None:
            self.__init__(seq_len=self.seq_len, tokenizer=self.tokenizer)
            return
        if not 0 <= n <= len(self):
            raise ValueError("n must be between 0 and len(buffer)")
        if n == 0:
            return
        cut_token = self.action_indices[n - 1] + 1
        self.tokens = self.tokens[cut_token:]
        self.rewards = self.rewards[cut_token:]
        self.dones = self.dones[cut_token:]
        self.action_indices = [idx - cut_token for idx in self.action_indices[n:]]
        self.transitions = [(ti - cut_token, r, d) for (ti, r, d) in self.transitions if ti >= cut_token]
        self.messages = []

    # .....................................................................
    # Dataset interface
    # .....................................................................
    def __len__(self):
        return len(self.action_indices)

    def __getitem__(self, idx: int):
        token_idx = self.action_indices[idx]
        s_ta_tok, s_ta_mask = self._build_window(token_idx)
        s_tp1_tok, s_tp1_mask = self._build_window(token_idx + 1)
        return s_ta_tok, s_ta_mask, s_tp1_tok, s_tp1_mask, self.rewards[token_idx], self.dones[token_idx]

    # .....................................................................
    # Save + *new* merge‑friendly load
    # .....................................................................
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "messages.json"), "w") as f:
            json.dump(self.messages, f)
        with open(os.path.join(path, "transitions.json"), "w") as f:
            json.dump(self.transitions, f)
        torch.save(
            {
                "tokens": self.tokens,
                "rewards": self.rewards,
                "dones": self.dones,
                "action_indices": self.action_indices,
                "transitions": self.transitions,
            },
            os.path.join(path, "tensors.pt"),
        )

    def load(self, path: str):
        """Append a single conversation dump located at *path*.

        The method first looks for a *binary* snapshot (``tensors.pt``) for
        speed.  If absent, it falls back to the older
        ``messages.json``/``transitions.json`` pair and rebuilds via
        :py:meth:`push`.
        """
        offset = len(self.tokens)
        tensor_path = os.path.join(path, "tensors.pt")

        if os.path.exists(tensor_path):
            state = torch.load(tensor_path, map_location="cpu")
            # 1. Flat tensors ------------------------------------------------
            self.tokens = torch.cat([self.tokens, state["tokens"]])
            self.rewards = torch.cat([self.rewards, state["rewards"]])
            self.dones = torch.cat([self.dones, state["dones"]])
            # 2. Action & transition indices (shifted by offset) ------------
            self.action_indices.extend([idx + offset for idx in state.get("action_indices", [])])
            self.transitions.extend([(ti + offset, r, d) for (ti, r, d) in state.get("transitions", [])])
            # 3. Messages (optional, not required for sampling) -------------
            msg_path = os.path.join(path, "messages.json")
            if os.path.exists(msg_path):
                with open(msg_path, "r") as f:
                    self.messages.extend(json.load(f))
            return

        # --------- Legacy JSON path --------------------------------------
        msg_file = os.path.join(path, "messages.json")
        tran_file = os.path.join(path, "transitions.json")
        if not os.path.exists(msg_file):
            raise FileNotFoundError("No tensors.pt or messages.json in given path")

        with open(msg_file, "r") as f:
            messages = json.load(f)
        transitions = []
        if os.path.exists(tran_file):
            with open(tran_file, "r") as f:
                transitions = json.load(f)
        # Build set for quick lookup --------------------------------------
        trans_idx = {ti: (r, d) for ti, r, d in transitions}
        # Replay push ------------------------------------------------------
        cur_token_idx = offset
        for msg in messages:
            self.push(msg)  # tentative push (no reward/done yet)
            if self.action_indices and self.action_indices[-1] + 1 in trans_idx:
                # We just added an assistant message whose *final* token index
                # (global) should receive reward/done.
                r, d = trans_idx[self.action_indices[-1] + 1]
                self.rewards[-1] = r
                if d:
                    self.dones[-1] = True
            cur_token_idx = len(self.tokens)

    # .....................................................................
    # Internal helpers
    # .....................................................................
    def _insert_eot(self):
        if self.tokens.numel() and self.tokens[-1].item() == self.eot_token_id:
            return
        self.tokens = torch.cat([self.tokens, torch.tensor([self.eot_token_id], dtype=torch.long)])
        self.rewards = torch.cat([self.rewards, torch.tensor([0.0], dtype=torch.float32)])
        self.dones = torch.cat([self.dones, torch.tensor([False], dtype=torch.bool)])

    def _build_window(self, end_idx: int):
        if end_idx < 0:
            end_idx = -1
        max_idx = len(self.tokens) - 1
        if end_idx > max_idx:
            return (
                torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long),
                torch.zeros(self.seq_len, dtype=torch.bool),
            )
        start_idx = end_idx - self.seq_len + 1
        conv_start = (self.tokens[: end_idx + 1] == self.eot_token_id).nonzero()
        last_eot = conv_start[-1].item() + 1 if conv_start.numel() else 0
        real_start = max(start_idx, last_eot, 0)
        slice_tok = self.tokens[real_start : end_idx + 1]
        pad_len = self.seq_len - slice_tok.numel()
        pad_tok = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros(pad_len, dtype=torch.bool)
        slice_mask = torch.ones(slice_tok.numel(), dtype=torch.bool)
        return torch.cat([pad_tok, slice_tok]), torch.cat([pad_mask, slice_mask])
