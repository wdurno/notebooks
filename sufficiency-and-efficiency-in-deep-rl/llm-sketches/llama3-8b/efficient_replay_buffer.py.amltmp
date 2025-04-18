from torch.utils.data import Dataset
import torch
import os
import json

class EfficientReplayBuffer(Dataset):
    def __init__(self, max_size=10000, max_seq_len=2048, tokenizer=None):
        self.max_size = max_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.messages = []      # List of {"role": ..., "content": ...}
        self.transitions = []   # List of (t, reward, done)

    def push(self, message, reward=None, done=False):
        if len(self.messages) >= self.max_size:
            self.messages.pop(0)
            self.transitions = [(t - 1, r, d) for (t, r, d) in self.transitions if t > 0]

        self.messages.append(message)

        if message["role"] == "assistant" and reward is not None:
            self.transitions.append((len(self.messages) - 1, reward, done))

    def add(self, *args, **kwargs):
        'backwards compatible version of `push`'
        return self.push(*args, **kwargs)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx): 
        ## TODO consider attention masking to avoid self-learning 
        t, reward, done = self.transitions[idx]
        window_start = max(0, t - self.max_seq_len // 2)

        s_t_msgs = self.messages[window_start:t]
        a_t_msg = self.messages[t]
        s_tp1_msgs = self.messages[window_start:min(len(self.messages), t + 2)]

        s_t = self._tokenize(s_t_msgs)
        a_t = self._tokenize([a_t_msg])
        s_tp1 = self._tokenize(s_tp1_msgs)

        return s_t, a_t, torch.tensor(reward, dtype=torch.float32), s_tp1, torch.tensor(done, dtype=torch.bool)

    def _tokenize(self, messages):
        """
        Tokenizes a list of messages using the LLaMA chat format.
        Each message is a dict with 'role' and 'content'.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            # Manual fallback: Inject special tokens manually
            START = "<|start_header_id|>"
            END = "<|end_header_id|>"
            EOT = "<|eot_id|>"
            BEGIN = "<|begin_of_text|>"

            formatted = BEGIN + "\n"
            for msg in messages:
                role = msg['role']
                content = msg['content']
                formatted += f"{START}{role}{END}\n{content}{EOT}\n"
            text = formatted

        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids[0]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "messages.json"), "w") as f:
            json.dump(self.messages, f)
        with open(os.path.join(path, "transitions.json"), "w") as f:
            json.dump(self.transitions, f)

    def load(self, path):
        with open(os.path.join(path, "messages.json"), "r") as f:
            self.messages = json.load(f)
        with open(os.path.join(path, "transitions.json"), "r") as f:
            self.transitions = json.load(f)
