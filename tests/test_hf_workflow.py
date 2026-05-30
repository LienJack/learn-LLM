import json
from pathlib import Path

import pytest
import torch

from src.finetune.hf_workflow import (
    GenerationSettings,
    HFLoadPlan,
    assert_causal_lm_logits_shape,
    causal_lm_data_collator,
    deterministic_split,
    format_messages_fallback,
    greedy_next_token,
    save_pretrained_artifacts,
    validate_tokenizer_batch,
    verify_pretrained_artifacts,
    write_generation_settings,
)


class FakePretrained:
    def __init__(self, filename: str, payload: dict[str, object]) -> None:
        self.filename = filename
        self.payload = payload

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir) / self.filename
        path.write_text(json.dumps(self.payload, sort_keys=True))


def test_load_plan_separates_tokenizer_and_model_kwargs() -> None:
    plan = HFLoadPlan(
        model_id="sshleifer/tiny-gpt2",
        revision="main",
        torch_dtype="float16",
        device_map="auto",
    )

    assert plan.tokenizer_kwargs() == {"revision": "main", "trust_remote_code": False}
    assert plan.model_kwargs() == {
        "revision": "main",
        "trust_remote_code": False,
        "torch_dtype": "float16",
        "device_map": "auto",
    }


def test_validate_tokenizer_batch_requires_input_ids_and_attention_mask() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    validate_tokenizer_batch(batch)


def test_validate_tokenizer_batch_rejects_shape_mismatch() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="same shape"):
        validate_tokenizer_batch(batch)


def test_causal_lm_data_collator_masks_padding_labels() -> None:
    batch = causal_lm_data_collator(
        [{"input_ids": [5, 6, 7]}, {"input_ids": [8]}],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[5, 6, 7], [8, -100, -100]]


def test_causal_lm_logits_shape_contract_matches_vocab_size() -> None:
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    logits = torch.randn(2, 3, 11)

    assert_causal_lm_logits_shape(logits, input_ids, expected_vocab_size=11)


def test_greedy_next_token_is_deterministic() -> None:
    logits = torch.tensor([[0.1, 2.0, 1.0], [3.0, 1.0, 2.0]])

    first = greedy_next_token(logits)
    second = greedy_next_token(logits)

    assert first.tolist() == [1, 0]
    assert torch.equal(first, second)


def test_save_pretrained_artifacts_and_verify_manifest(tmp_path) -> None:
    model = FakePretrained("config.json", {"model_type": "fake-causal-lm"})
    tokenizer = FakePretrained("tokenizer_config.json", {"model_max_length": 16})
    manifest = {"model_id": "fake", "dataset_version": "tiny-v1"}

    save_pretrained_artifacts(tmp_path, model, tokenizer, manifest)
    loaded_manifest = verify_pretrained_artifacts(tmp_path)

    assert loaded_manifest == manifest


def test_deterministic_split_is_reproducible() -> None:
    items = [{"id": str(index)} for index in range(10)]

    train_a, val_a = deterministic_split(items, val_ratio=0.2, seed=0)
    train_b, val_b = deterministic_split(items, val_ratio=0.2, seed=0)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 2


def test_format_messages_fallback_adds_generation_prompt() -> None:
    prompt = format_messages_fallback(
        [
            {"role": "system", "content": "你是助教"},
            {"role": "user", "content": "解释 causal mask"},
        ],
    )

    assert "<|system|>\n你是助教\n" in prompt
    assert prompt.endswith("<|assistant|>\n")


def test_generation_settings_can_be_written(tmp_path) -> None:
    path = tmp_path / "generation_config.json"

    write_generation_settings(path, GenerationSettings(max_new_tokens=8, do_sample=False))

    assert json.loads(path.read_text())["max_new_tokens"] == 8
