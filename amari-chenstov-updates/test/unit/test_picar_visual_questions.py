import json

import numpy as np
from PIL import Image
import pytest
import torch

from numerical_experiment.picar.visual_questions import QUESTIONS, question_inputs, read_rgb, run


def test_question_input_is_one_fresh_image_without_task_history_or_labels():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return json.dumps(messages)
        def __call__(self, **kwargs):
            self.inputs = kwargs
            return {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    class Encoder:
        processor = Processor()
        max_tokens = 100
    encoder = Encoder()
    rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
    for question in QUESTIONS:
        question_inputs(encoder, rgb, question)
        messages = encoder.processor.messages
        assert len(messages) == 2
        assert messages[0]["content"][0]["text"] == "You are a helpful assistant."
        assert messages[1]["content"] == [{"type": "image"}, {"type": "text", "text": question}]
        kwargs = encoder.processor.inputs
        assert len(kwargs["images"]) == 1 and kwargs["max_pixels"] == 76800
        torch.testing.assert_close(kwargs["images"][0], torch.full((3, 24, 32), 128/255))
        assert kwargs["do_rescale"] is False


def test_rgb_loading_is_lossless_for_png_and_numpy_and_rejects_non_rgb(tmp_path):
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[2:8, 3:9, 0] = 210
    png, npy = tmp_path / "rgb.png", tmp_path / "rgb.npy"
    Image.fromarray(rgb).save(png)
    np.save(npy, rgb, allow_pickle=False)
    assert np.array_equal(read_rgb(png), rgb)
    assert np.array_equal(read_rgb(npy), rgb)
    np.save(npy, rgb.astype(float), allow_pickle=False)
    with pytest.raises(ValueError, match="uint8"):
        read_rgb(npy)


def test_duplicate_labels_rejected_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="uniquely labeled"):
        run(tmp_path / "out", [("a", tmp_path / "one"), ("a", tmp_path / "two")])
    assert not (tmp_path / "out").exists()
