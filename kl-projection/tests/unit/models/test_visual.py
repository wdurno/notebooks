from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from picar_kl.models.visual import QwenVisualTokenEncoder, VisualTokenEncoding


def test_visual_token_encoding_requires_rank_two_float_array():
    VisualTokenEncoding(tokens=np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="rank-2"):
        VisualTokenEncoding(tokens=np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="floating"):
        VisualTokenEncoding(tokens=np.zeros((2, 3), dtype=np.int64))


def test_qwen_visual_token_encoder_preserves_token_sequence():
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def __call__(self, *, images, return_tensors):
            assert len(images) == 1
            assert return_tensors == "pt"
            return {
                "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }

    class FakeInnerModel:
        def get_image_features(self, *, pixel_values, image_grid_thw, return_dict):
            assert pixel_values.shape == (1, 3, 4, 4)
            assert image_grid_thw.tolist() == [[1, 2, 2]]
            assert return_dict is True
            return SimpleNamespace(pooler_output=(torch.arange(12, dtype=torch.float32).reshape(3, 4),))

    class FakeModel:
        model = FakeInnerModel()

        def parameters(self):
            return iter(())

    class FakeRuntime:
        model = FakeModel()
        processor = FakeProcessor()

        @contextmanager
        def inference_context(self):
            yield

    encoder = QwenVisualTokenEncoder(FakeRuntime(), output_dtype="float32")
    encoding = encoder.encode_image(np.zeros((8, 8, 3), dtype=np.uint8))

    assert encoding.tokens.shape == (3, 4)
    assert encoding.metadata["source"] == "Qwen2_5_VLModel.get_image_features.pooler_output"
    assert encoding.metadata["image_grid_thw"] == [[1, 2, 2]]
