"""Behavioral tests for the ordered compressor architecture (CPU, float32)."""

from __future__ import annotations

import copy

import pytest
import torch

from compression.model import ChannelStats, OrderedCompressor, export_compressor, load_compressor

DIM = 12
MODES = ("separate", "joint")


def make_model(mode: str, max_codes: int = 8, seed: int = 0) -> OrderedCompressor:
    torch.manual_seed(seed)
    model = OrderedCompressor(
        input_dim=DIM,
        width=32,
        max_codes=max_codes,
        encoder_layers=2,
        decoder_layers=2,
        heads=4,
        ff_multiplier=2,
        attention_mode=mode,
    )
    return model.eval()


def make_batch(lengths: list[int], seed: int = 0, garbage: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, longest = len(lengths), max(lengths)
    features = torch.randn(batch, longest, DIM, generator=generator) * 2.0
    positions = torch.rand(batch, longest, 3, generator=generator)
    token_types = torch.randint(0, 4, (batch, longest), generator=generator)
    valid = torch.zeros(batch, longest, dtype=torch.bool)
    for row, length in enumerate(lengths):
        valid[row, :length] = True
    if garbage:  # padding content must never matter, not even non-finite values
        features[~valid] = float("nan")
        features[~valid & (torch.arange(longest) % 2 == 0)] = 1e4
        positions[~valid] = float("inf")
        token_types[~valid] = 3
    return {"features": features, "positions": positions, "token_types": token_types, "valid_mask": valid}


def single(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    length = int(batch["valid_mask"][row].sum())
    return {key: value[row : row + 1, :length] for key, value in batch.items()}


@pytest.mark.parametrize("mode", MODES)
def test_encoding_with_k_codes_equals_prefix_of_full_encoding(mode):
    model = make_model(mode)
    batch = make_batch([5, 9, 3])
    with torch.no_grad():
        full = model.encode(**batch)
        assert full.shape == (3, 8, 32)
        for k in range(1, 9):
            prefix = model.encode(**batch, num_codes=k)
            assert prefix.shape == (3, k, 32)
            torch.testing.assert_close(prefix, full[:, :k], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_code_prefix_is_independent_of_later_code_queries(mode):
    model = make_model(mode)
    batch = make_batch([6, 4])
    altered = copy.deepcopy(model)
    k = 3
    with torch.no_grad():
        altered.code_queries[k:] += 5.0 * torch.randn_like(altered.code_queries[k:])
        original, changed = model.encode(**batch), altered.encode(**batch)
    torch.testing.assert_close(changed[:, :k], original[:, :k], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(changed[:, k:], original[:, k:], atol=1e-3)


@pytest.mark.parametrize("mode", MODES)
def test_short_prefix_loss_sends_no_gradient_to_suffix_queries(mode):
    model = make_model(mode).train()
    batch = make_batch([7, 5])
    k = 3
    codes = model.encode(**batch)
    reconstruction = model.decode(codes[:, :k], batch["positions"], batch["token_types"], batch["valid_mask"])
    target = torch.where(batch["valid_mask"][..., None], batch["features"], 0.0)
    reconstruction[batch["valid_mask"]].sub(target[batch["valid_mask"]]).square().mean().backward()
    grad = model.code_queries.grad
    assert torch.count_nonzero(grad[k:]) == 0
    assert torch.count_nonzero(grad[:k]) > 0


@pytest.mark.parametrize("mode", MODES)
def test_padding_never_changes_codes_or_reconstructions(mode):
    model = make_model(mode)
    batch = make_batch([9, 2, 6], garbage=True)
    with torch.no_grad():
        codes = model.encode(**batch)
        output = model.decode(codes[:, :5], batch["positions"], batch["token_types"], batch["valid_mask"])
        assert torch.isfinite(codes).all() and torch.isfinite(output).all()
        assert torch.count_nonzero(output[~batch["valid_mask"]]) == 0
        for row in range(3):
            alone = single(batch, row)
            alone_codes = model.encode(**alone)
            torch.testing.assert_close(codes[row : row + 1], alone_codes, atol=1e-5, rtol=1e-5)
            alone_output = model.decode(alone_codes[:, :5], alone["positions"], alone["token_types"],
                                        alone["valid_mask"])
            length = alone["valid_mask"].shape[1]
            torch.testing.assert_close(output[row : row + 1, :length], alone_output, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_decoder_queries_do_not_interact(mode):
    model = make_model(mode)
    batch = make_batch([8], garbage=False)
    with torch.no_grad():
        codes = model.encode(**batch)[:, :4]
        full = model.decode(codes, batch["positions"], batch["token_types"], batch["valid_mask"])
        subset = torch.tensor([6, 1, 3])
        partial = model.decode(codes, batch["positions"][:, subset], batch["token_types"][:, subset])
    torch.testing.assert_close(partial, full[:, subset], atol=1e-5, rtol=1e-5)


def test_invalid_budgets_and_empty_samples_are_rejected():
    model = make_model("separate", max_codes=4)
    batch = make_batch([3, 2])
    with pytest.raises(ValueError):
        model.encode(**batch, num_codes=0)
    with pytest.raises(ValueError):
        model.encode(**batch, num_codes=5)
    empty = dict(batch, valid_mask=torch.zeros_like(batch["valid_mask"]))
    with pytest.raises(ValueError):
        model.encode(**empty)
    with torch.no_grad():
        codes = model.encode(**batch)
    with pytest.raises(ValueError):
        model.decode(torch.cat([codes, codes], dim=1), batch["positions"], batch["token_types"])


@pytest.mark.parametrize("mode", MODES)
def test_exported_compressor_reloads_with_identical_native_outputs(tmp_path, mode):
    model = make_model(mode)
    generator = torch.Generator().manual_seed(3)
    stats = ChannelStats(
        mean=torch.randn(DIM, generator=generator, dtype=torch.float64),
        std=torch.rand(DIM, generator=generator, dtype=torch.float64) + 0.5,
        token_count=100,
        sample_count=10,
        split="train",
        cache_identity="abc",
        layer=None,
        std_floor=1e-6,
    )
    path = tmp_path / "model.pt"
    export_compressor(path, model, stats, {"stage": "prefix", "step": 7,
                                           "cache": {"identity": "abc", "layer": None}})
    bundle = load_compressor(path)
    assert bundle.model.config == model.config
    assert bundle.cache_identity == "abc" and bundle.metadata["step"] == 7
    assert bundle.stats.digest() == stats.digest()
    batch = make_batch([5, 3])
    native = batch["features"]
    with torch.no_grad():
        codes = bundle.encode_native(native, batch["positions"], batch["token_types"], batch["valid_mask"], 4)
        expected_codes = model.encode(stats.normalize(native), batch["positions"], batch["token_types"],
                                      batch["valid_mask"], 4)
        torch.testing.assert_close(codes, expected_codes)
        decoded = bundle.decode_native(codes, batch["positions"], batch["token_types"], batch["valid_mask"])
        expected = stats.denormalize(model.decode(expected_codes, batch["positions"], batch["token_types"],
                                                  batch["valid_mask"]))
    valid = batch["valid_mask"]
    torch.testing.assert_close(decoded[valid], expected[valid])
    assert torch.count_nonzero(decoded[~valid]) == 0
    torch.testing.assert_close(stats.denormalize(stats.normalize(native[valid])), native[valid])
