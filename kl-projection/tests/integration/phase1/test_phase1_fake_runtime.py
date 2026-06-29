import numpy as np
import pytest

from picar_kl.data.phase1 import load_phase1_run
from picar_kl.phase1.run import FixedActionVLMController, Phase1RunConfig, run_phase1


pytestmark = pytest.mark.integration


class FakeRobot:
    def capture_image(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def apply_vector(self, action_vector):
        return {"status": "ok", "vector": dict(action_vector)}


def test_fake_phase1_runtime_writes_loadable_run(tmp_path):
    summary = run_phase1(
        Phase1RunConfig(
            data_root=tmp_path,
            run_uuid="integration-run",
            max_steps=1,
        ),
        robot=FakeRobot(),
        vlm=FixedActionVLMController("look-left"),
    )

    records = load_phase1_run(summary.run_dir)

    assert len(records) == 1
    assert records[0].action is not None
    assert records[0].action.action_name == "look-left"
