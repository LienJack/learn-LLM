import torch
from torch import nn

from src.finetune.lora import (
    LoRAConfig,
    LoRALinear,
    QLoRAConfig,
    estimate_linear_parameter_count,
    estimate_lora_parameter_count,
    inject_lora_adapters,
    load_lora_adapter,
    qlora_memory_note,
    save_lora_adapter,
    trainable_parameter_summary,
    write_lora_manifest,
)


class TinyTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 6)
        self.v_proj = nn.Linear(4, 6)
        self.other = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(torch.relu(self.q_proj(x) + self.v_proj(x)))


def test_lora_linear_freezes_base_and_preserves_shape() -> None:
    base = nn.Linear(4, 6)
    layer = LoRALinear(base, LoRAConfig(r=2, alpha=4))
    x = torch.randn(3, 4)

    output = layer(x)

    assert output.shape == (3, 6)
    assert not layer.base.weight.requires_grad
    assert layer.lora_A.weight.requires_grad
    assert layer.lora_B.weight.requires_grad


def test_inject_lora_adapters_matches_target_modules() -> None:
    model = TinyTargetModel()
    matched = inject_lora_adapters(model, LoRAConfig(r=2, target_modules=("q_proj", "v_proj")))

    assert matched == ["q_proj", "v_proj"]
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.other, nn.Linear)


def test_inject_lora_adapters_fails_for_missing_target() -> None:
    model = TinyTargetModel()

    try:
        inject_lora_adapters(model, LoRAConfig(r=2, target_modules=("missing",)))
    except ValueError as exc:
        assert "no target modules" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing target module.")


def test_trainable_parameters_only_include_adapters_and_unfrozen_head() -> None:
    model = TinyTargetModel()
    inject_lora_adapters(model, LoRAConfig(r=2, target_modules=("q_proj", "v_proj")))
    summary = trainable_parameter_summary(model)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert "q_proj.base.weight" not in trainable_names
    assert "v_proj.base.weight" not in trainable_names
    assert any(name.endswith("lora_A.weight") for name in trainable_names)
    assert any(name.endswith("lora_B.weight") for name in trainable_names)
    assert 0 < summary.ratio < 1


def test_training_step_updates_adapter_but_not_base() -> None:
    torch.manual_seed(0)
    model = TinyTargetModel()
    inject_lora_adapters(model, LoRAConfig(r=2, alpha=4, target_modules=("q_proj",)))
    before_base = model.q_proj.base.weight.detach().clone()
    before_adapter = model.q_proj.lora_B.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(5, 4)
    target = torch.randn(5, 2)

    loss = nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.allclose(before_base, model.q_proj.base.weight.detach())
    assert not torch.allclose(before_adapter, model.q_proj.lora_B.weight.detach())


def test_save_and_load_lora_adapter_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    model = TinyTargetModel()
    config = LoRAConfig(r=2, alpha=4, target_modules=("q_proj", "v_proj"))
    inject_lora_adapters(model, config)
    path = tmp_path / "adapter.pt"
    x = torch.randn(2, 4)
    before = model(x).detach()

    save_lora_adapter(path, model, config, extra={"step": 7})
    fresh = TinyTargetModel()
    loaded_config, extra = load_lora_adapter(path, fresh)
    after = fresh(x).detach()

    assert loaded_config == config
    assert extra == {"step": 7}
    assert after.shape == before.shape


def test_parameter_count_estimates_show_lora_savings() -> None:
    full = estimate_linear_parameter_count(4096, 4096, bias=False)
    lora = estimate_lora_parameter_count(4096, 4096, rank=8)

    assert full == 16_777_216
    assert lora == 65_536
    assert lora < full


def test_qlora_note_and_manifest(tmp_path) -> None:
    note = qlora_memory_note(QLoRAConfig())
    path = tmp_path / "lora_manifest.json"
    model = TinyTargetModel()
    config = LoRAConfig(r=2, target_modules=("q_proj",))
    inject_lora_adapters(model, config)
    summary = trainable_parameter_summary(model)

    write_lora_manifest(path, config, summary)

    assert "load_in_4bit=True" in note
    assert "trainable_ratio" in path.read_text()
