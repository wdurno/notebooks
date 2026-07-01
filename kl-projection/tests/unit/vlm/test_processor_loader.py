from picar_kl.vlm.processor_loader import Qwen2_5_VLImageOnlyProcessor


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeGrid:
    def __init__(self, value):
        self._value = value

    def prod(self):
        return FakeScalar(self._value)


class FakeImageProcessor:
    merge_size = 2

    def __call__(self, *, images, return_tensors):
        assert images == ["image"]
        assert return_tensors == "pt"
        return {"image_grid_thw": [FakeGrid(8)]}


class FakeTokenizer:
    image_token = "<|image_pad|>"
    image_token_id = 7

    def __init__(self):
        self.seen_text = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return f"messages={len(messages)} tokenize={tokenize} gen={add_generation_prompt}"

    def __call__(self, text, *, return_tensors, padding):
        self.seen_text = text
        assert return_tensors == "pt"
        assert padding is True
        return {"input_ids": [1, 2, 3]}

    def batch_decode(self, sequences, *, skip_special_tokens):
        assert skip_special_tokens is True
        return ["decoded"]


def test_qwen_image_only_processor_expands_image_tokens():
    tokenizer = FakeTokenizer()
    processor = Qwen2_5_VLImageOnlyProcessor(
        image_processor=FakeImageProcessor(),
        tokenizer=tokenizer,
    )

    payload = processor(
        text=["see <|image_pad|> now"],
        images=["image"],
        return_tensors="pt",
        padding=True,
    )

    assert payload["input_ids"] == [1, 2, 3]
    assert payload["image_grid_thw"][0].prod().item() == 8
    assert tokenizer.seen_text == ["see <|image_pad|><|image_pad|> now"]
    assert processor.batch_decode([[1]]) == ["decoded"]
    assert processor.apply_chat_template([{}], tokenize=False) == "messages=1 tokenize=False gen=True"
