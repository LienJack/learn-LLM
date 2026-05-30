import torch

from src.data.text_datasets import (
    IGNORE_INDEX,
    ChatMessage,
    LanguageModelingTextDataset,
    build_sft_features,
)
from src.tokenizer.simple_tokenizer import CharacterTokenizer


def test_special_token_ids_are_fixed_and_unique() -> None:
    tokenizer = CharacterTokenizer.from_texts(["你好，LLM"])

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    }

    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert len(special_ids) == 4


def test_decode_encode_round_trip_for_known_characters() -> None:
    text = "中文 tokenizer"
    tokenizer = CharacterTokenizer.from_texts([text])

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_unknown_characters_use_unk_token() -> None:
    tokenizer = CharacterTokenizer.from_texts(["abc"])

    encoded = tokenizer.encode("a字", add_special_tokens=False)

    assert encoded == [tokenizer.token_to_id["a"], tokenizer.unk_token_id]


def test_batch_encode_pads_and_builds_attention_mask() -> None:
    tokenizer = CharacterTokenizer.from_texts(["短", "长一点"])

    batch = tokenizer.batch_encode(["短", "长一点"], max_length=6)

    assert batch.input_ids.shape == (2, 6)
    assert batch.attention_mask.shape == (2, 6)
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 0, 0, 0]
    assert batch.input_ids[0, 3:].tolist() == [tokenizer.pad_token_id] * 3
    assert batch.attention_mask[1].tolist() == [1, 1, 1, 1, 1, 0]


def test_batch_encode_refuses_overlong_sequence_without_truncation() -> None:
    tokenizer = CharacterTokenizer.from_texts(["abcdef"])

    try:
        tokenizer.batch_encode(["abcdef"], max_length=4, truncation=False)
    except ValueError as exc:
        assert "truncation=False" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overlong sequence.")


def test_language_modeling_dataset_returns_right_shifted_labels() -> None:
    token_ids = torch.arange(8, dtype=torch.long)
    dataset = LanguageModelingTextDataset(token_ids, block_size=3)

    input_ids, labels = dataset[2]

    assert len(dataset) == 5
    assert input_ids.tolist() == [2, 3, 4]
    assert labels.tolist() == [3, 4, 5]


def test_sft_features_mask_non_assistant_tokens() -> None:
    texts = [
        "<|system|>\n你是助教\n<|user|>\n解释 causal mask\n<|assistant|>\n只能看历史 token\n",
    ]
    tokenizer = CharacterTokenizer.from_texts(texts)
    features = build_sft_features(
        [
            ChatMessage(role="system", content="你是助教"),
            ChatMessage(role="user", content="解释 causal mask"),
            ChatMessage(role="assistant", content="只能看历史 token"),
        ],
        tokenizer=tokenizer,
        max_length=80,
    )

    assistant_ids = tokenizer.encode("只能看历史 token", add_special_tokens=False)
    supervised_ids = features.labels[features.labels != IGNORE_INDEX].tolist()

    assert features.input_ids.shape == (80,)
    assert features.attention_mask.shape == (80,)
    assert features.labels.shape == (80,)
    assert supervised_ids[: len(assistant_ids)] == assistant_ids
    assert tokenizer.encode("解释 causal mask", add_special_tokens=False)[0] not in supervised_ids


def test_sft_features_masks_padding_labels() -> None:
    tokenizer = CharacterTokenizer.from_texts(["<|assistant|>\n好\n"])

    features = build_sft_features(
        [ChatMessage(role="assistant", content="好")],
        tokenizer=tokenizer,
        max_length=24,
    )

    assert features.attention_mask[-1].item() == 0
    assert features.input_ids[-1].item() == tokenizer.pad_token_id
    assert features.labels[-1].item() == IGNORE_INDEX


def test_tokenizer_mismatch_changes_ids() -> None:
    text = "合同违约"
    original = CharacterTokenizer.from_texts([text])
    changed = CharacterTokenizer.from_texts(["违约合同"])

    assert original.encode(text, add_special_tokens=False) != changed.encode(
        text,
        add_special_tokens=False,
    )
