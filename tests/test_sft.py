import json

import pytest

from src.data.text_datasets import IGNORE_INDEX, ChatMessage
from src.finetune.sft import (
    SFTExample,
    assert_no_split_leakage,
    build_sft_batch_item,
    compare_behaviors,
    dump_sft_jsonl,
    load_sft_jsonl,
    render_training_text,
    split_by_source_group,
    supervised_label_text,
    write_behavior_report,
)
from src.tokenizer.simple_tokenizer import CharacterTokenizer


def make_example(example_id: str = "sft_001", source_group: str = "doc_a") -> SFTExample:
    return SFTExample(
        id=example_id,
        source="manual",
        source_group=source_group,
        risk_tags=["teaching"],
        messages=[
            ChatMessage(role="system", content="你是技术助教"),
            ChatMessage(role="user", content="解释 causal mask"),
            ChatMessage(role="assistant", content="只能看历史 token"),
        ],
    )


def test_sft_example_schema_requires_messages_and_assistant() -> None:
    with pytest.raises(ValueError, match="assistant"):
        SFTExample(
            id="bad",
            source="manual",
            source_group="doc",
            messages=[ChatMessage(role="user", content="问题")],
        )


def test_sft_example_from_dict_rejects_invalid_role() -> None:
    with pytest.raises(ValueError, match="role"):
        SFTExample.from_dict(
            {
                "id": "bad",
                "source": "manual",
                "source_group": "doc",
                "messages": [{"role": "developer", "content": "x"}],
            },
        )


def test_load_and_dump_sft_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    example = make_example()

    dump_sft_jsonl(path, [example])
    loaded = load_sft_jsonl(path)

    assert loaded == [example]
    assert json.loads(path.read_text().splitlines()[0])["id"] == "sft_001"


def test_render_training_text_contains_assistant_boundary() -> None:
    text = render_training_text(make_example())

    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "只能看历史 token" in text


def test_sft_labels_mask_system_user_and_padding() -> None:
    tokenizer = CharacterTokenizer.from_texts([render_training_text(make_example())])
    item = build_sft_batch_item(make_example(), tokenizer, max_length=80)
    supervised = supervised_label_text(item, tokenizer)

    assert supervised == "只能看历史 token"
    assert item.labels[-1].item() == IGNORE_INDEX


def test_split_by_source_group_prevents_group_leakage() -> None:
    examples = [
        make_example("a1", "doc_a"),
        make_example("a2", "doc_a"),
        make_example("b1", "doc_b"),
        make_example("c1", "doc_c"),
    ]

    train, val = split_by_source_group(examples, val_ratio=0.34, seed=0)
    train_groups = {example.source_group for example in train}
    val_groups = {example.source_group for example in val}

    assert train_groups.isdisjoint(val_groups)
    assert val


def test_assert_no_split_leakage_rejects_repeated_id_or_group() -> None:
    with pytest.raises(ValueError, match="ids"):
        assert_no_split_leakage([make_example("same", "a")], [make_example("same", "b")])

    with pytest.raises(ValueError, match="source groups"):
        assert_no_split_leakage([make_example("a", "same")], [make_example("b", "same")])


def test_compare_behaviors_and_write_report(tmp_path) -> None:
    comparisons = compare_behaviors(
        prompts=["解释 causal mask"],
        before_outputs=["继续写 causal mask"],
        after_outputs=["causal mask 只能看历史 token"],
    )
    report_path = tmp_path / "before_after.md"

    write_behavior_report(report_path, comparisons)

    assert comparisons[0].changed is True
    assert "SFT 行为对比报告" in report_path.read_text()
