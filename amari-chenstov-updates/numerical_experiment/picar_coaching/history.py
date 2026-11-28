"""Causal operator events shared by the frozen teacher and the student."""

import copy
import threading

from numerical_experiment.picar.fix2_data import TASK, history_batch, observation_row


class History:
    def __init__(self, config, sim_config, state=None):
        self.config, self.sim_config = config, sim_config
        self.lock = threading.RLock()
        self.state = copy.deepcopy(state) if state else {
            "live": [], "deadline": None, "contexts": [{"text": TASK, "issued_at": 0.}],
            "statuses": [], "instruction_ids": []}

    def update(self, contexts, statuses, *, instruction_ids=()):
        with self.lock:
            self.state["contexts"] = [{"text": TASK, "issued_at": 0.}, *copy.deepcopy(contexts)]
            self.state["statuses"] = copy.deepcopy(statuses)
            self.state["instruction_ids"] = copy.deepcopy(list(instruction_ids))

    def context_at(self, now):
        with self.lock:
            return ([copy.deepcopy(c) for c in self.state["contexts"] if c["issued_at"] <= now],
                    [copy.deepcopy(s) for s in self.state["statuses"] if s["available_at"] <= now])

    def __call__(self, observation):
        with self.lock:
            if self.state["deadline"] is None:
                remaining = getattr(observation, "remaining_time_s", None)
                self.state["deadline"] = observation.capture_time_s + (
                    self.sim_config.deadline_s if remaining is None else remaining)
            live = self.state["live"]
            if not live or observation.capture_time_s > live[-1]["time"]:
                live.append(observation_row(observation, previous_time=live[-1]["time"] if live else None))
            live[:] = live[-self.config.model.context_length:]
            contexts, statuses = self.context_at(observation.capture_time_s)
            batch = history_batch(live, contexts=contexts, statuses=statuses,
                deadline=self.state["deadline"], length=self.config.model.context_length,
                width=self.sim_config.width, height=self.sim_config.height)
            active = {(c["text"], c["issued_at"]) for c in contexts
                      if c.get("completed_at") is None or c["completed_at"] > observation.capture_time_s}
            batch["coaching_instruction_ids"] = [[item["instruction_id"] for item in self.state.get("instruction_ids", [])
                if (item["text"], item["available_at_s"]) in active]]
            return batch

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)
