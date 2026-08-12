# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
import triton

import flag_attn


ATTNRES_EXTERNAL_BENCHMARK_SHAPES = [
    (2, 1),
    (5, 128),
    (5, 1024),
    (9, 128),
    (9, 8192),
]


def _load_fla_reference():
    try:
        from fla.ops.attnres import fused_attnres as fla_fused_attnres

        return fla_fused_attnres, None
    except Exception as exc:
        return None, str(exc)


FLA_FUSED_ATTNRES, FLA_IMPORT_ERROR = _load_fla_reference()


def _tle_available() -> bool:
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _require_fla_reference():
    if FLA_FUSED_ATTNRES is None:
        pytest.skip(f"external FLA AttnRes implementation is unavailable: {FLA_IMPORT_ERROR}")
    return FLA_FUSED_ATTNRES


@pytest.mark.parametrize(
    ("num_sources", "batch", "tokens", "hidden_size", "output_norm", "dtype"),
    [
        (1, 1, 3, 7168, False, torch.bfloat16),
        (5, 1, 3, 7168, False, torch.bfloat16),
        (9, 1, 3, 7168, True, torch.bfloat16),
        (5, 2, 7, 4096, True, torch.float16),
    ],
)
@torch.inference_mode()
def test_fused_attnres(
    num_sources: int,
    batch: int,
    tokens: int,
    hidden_size: int,
    output_norm: bool,
    dtype: torch.dtype,
):
    if not _tle_available():
        pytest.skip("fused_attnres requires CUDA and triton.experimental.tle")

    torch.manual_seed(42)
    residuals = [
        torch.randn(batch, tokens, hidden_size, device="cuda", dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=dtype)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device="cuda", dtype=dtype) if output_norm else None
    )

    expected, expected_weights = flag_attn.testing.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )
    actual, actual_weights = flag_attn.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_weights, expected_weights, atol=5e-5, rtol=5e-5)


@torch.inference_mode()
def test_fused_attnres_without_weights():
    if not _tle_available():
        pytest.skip("fused_attnres requires CUDA and triton.experimental.tle")

    hidden_size = 7168
    residuals = [
        torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    output = flag_attn.fused_attnres(query, residuals, rms_weight, scale=hidden_size**-0.5)
    assert isinstance(output, torch.Tensor)
    assert output.shape == residuals[0].shape


def test_fused_attnres_rejects_empty_residuals():
    query = torch.empty(128)
    rms_weight = torch.empty(128)
    with pytest.raises(ValueError, match="at least one"):
        flag_attn.fused_attnres(query, [], rms_weight)


@pytest.mark.skipif(
    not _tle_available(), reason="AttnRes external benchmark requires CUDA/TLE"
)
@pytest.mark.skipif(
    os.environ.get("FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS", "0") != "1",
    reason="set FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 to run AttnRes benchmarks",
)
@pytest.mark.parametrize("output_norm", [False, True])
@pytest.mark.parametrize(("num_sources", "num_rows"), ATTNRES_EXTERNAL_BENCHMARK_SHAPES)
@torch.inference_mode()
def test_fused_attnres_benchmark(num_sources, num_rows, output_norm, record_property):
    fla_fused_attnres = _require_fla_reference()
    torch.manual_seed(0)
    hidden_size = 7168
    dtype = torch.bfloat16
    residuals = [
        torch.randn(num_rows, hidden_size, device="cuda", dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=dtype)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device="cuda", dtype=dtype) if output_norm else None
    )
    scale = hidden_size**-0.5

    expected = fla_fused_attnres(
        query, residuals, rms_weight, output_rms_weight, scale=scale
    )
    actual = flag_attn.fused_attnres(
        query, residuals, rms_weight, output_rms_weight, scale=scale
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=1e-2)

    fla_ms = triton.testing.do_bench(
        lambda: fla_fused_attnres(
            query, residuals, rms_weight, output_rms_weight, scale=scale
        ),
        warmup=25,
        rep=100,
    )
    flag_attn_ms = triton.testing.do_bench(
        lambda: flag_attn.fused_attnres(
            query, residuals, rms_weight, output_rms_weight, scale=scale
        ),
        warmup=25,
        rep=100,
    )
    speedup = fla_ms / flag_attn_ms
    shape_name = f"L{num_sources}_N{num_rows}_D{hidden_size}"

    record_property("shape", shape_name)
    record_property("dtype", str(dtype))
    record_property("output_norm", output_norm)
    record_property("fla_ms", fla_ms)
    record_property("flag_attn_ms", flag_attn_ms)
    record_property("speedup_vs_fla", speedup)
    print(
        f"\n{shape_name} output_norm={output_norm}: FLA={fla_ms:.6f} ms, "
        f"FlagAttention={flag_attn_ms:.6f} ms, speedup={speedup:.3f}x"
    )
