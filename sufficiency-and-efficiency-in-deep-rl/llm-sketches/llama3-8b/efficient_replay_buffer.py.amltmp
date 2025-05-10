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
        """
        Add a new message to the replay buffer.

        This function appends a message (from either the 'user' or 'assistant') to the internal
        buffer. When the message is from the assistant and a reward is provided, it marks that
        message as the end of a transition. The corresponding `(state, action, reward, next_state, done)`
        tuple can then be sampled later using `__getitem__`.

        Arguments:
            message (dict): A dictionary with at least keys `'role'` and `'content'`.
                            - 'role': either `'user'` or `'assistant'`
                            - 'content': string with the message text
            reward (float, optional): The scalar reward for the assistant's response.
                                    Should be provided only for 'assistant' messages.
            done (bool, optional): Whether this message terminates the episode. Defaults to False.

        Notes:
            - Only 'assistant' messages with an associated reward are treated as actionable transitions.
            - If `max_size` is exceeded, old messages and transitions are evicted.
            - The transition index stored is the index of the assistant message in `self.messages`.
        """
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
        '''
        Returns: 
        - `s_t` as the tokenized sequences prior to action `idx`. 
        - `a_t` as LLM's `idx`-th response, the tokenized content of its message. 
        - `reward` a `float32` scalar tensor for action `idx`. 
        - `s_tp1` as the tokenized sequence immediately after `s_t`, including all prior messages. 
        - `done` boolean scalar tensor, forced to True if `a_t` is the final sequence. 
        '''
        ## TODO consider attention masking to avoid self-learning 
        t, reward, done = self.transitions[idx] 
        window_start, window_end = self._get_window_start(t) 
        s_t_msgs = self.messages[window_start:window_end] 
        a_t_msg = self.messages[t] 
        s_tp1_msgs = self.messages[window_start:min(len(self.messages), t + 2)] 
        if t+1 >= len(self.messages): 
            done = True 
            pass 
        s_t = self._tokenize(s_t_msgs) 
        a_t = self._tokenize([a_t_msg]) 
        s_tp1 = self._tokenize(s_tp1_msgs) 

        return s_t, a_t, torch.tensor(reward, dtype=torch.float32), s_tp1, torch.tensor(done, dtype=torch.bool) 

    def _tokenize(self, messages): 
        """
        Tokenizes a list of messages using the LLaMA chat format.
        Each message is a dict with 'role' and 'content'.
        """
        if hasattr(self.tokenizer, "apply_chat_template") and has_non_none_attr(self.tokenizer, "chat_template"): 
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
        else: 
            ## TODO too much code duplication !!! 
            # Manual fallback: Inject special tokens manually 
            START = "<|start_header_id|>"
            END = "<|end_header_id|>"
            EOT = "<|eot|>"
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
    
    def _get_window_start(self, t): 
        '''Cycles backward through `messages` from index `[t]`. 
        Returns `window_start` such that `message[window_start:t]` content is at most, about `max_seq_len` tokens in length. 
        '''
        average_characters_per_token = 4 ## more-or-less constant in English 
        approximate_total_tokens = 0 
        window_start = t 
        window_end = t
        while approximate_total_tokens < self.max_seq_len and window_start > 0: 
            window_start -= 1 
            approximate_total_tokens += len(self.messages[window_start]) / average_characters_per_token 
            pass 
        return window_start, window_end 
    pass

def has_non_none_attr(obj, attr):
    if not hasattr(obj, attr): 
        return False 
    if getattr(obj, attr) is None: 
        return False
    return True 
