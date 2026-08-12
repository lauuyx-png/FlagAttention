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

import argparse

import torch
import triton

import flag_attn

try:
    from fla.ops.attnres import fused_attnres as fla_fused_attnres

    HAS_FLA = True
except ImportError:
    fla_fused_attnres = None
    HAS_FLA = False


K3_SHAPES = [
    (2, 1),
    (5, 128),
    (5, 1024),
    (9, 128),
    (9, 8192),
]


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


@torch.inference_mode()
def run(output_norm: bool) -> None:
    torch.manual_seed(0)
    hidden_size = 7168
    dtype = torch.bfloat16
    query = torch.randn(hidden_size, device="cuda", dtype=dtype)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device="cuda", dtype=dtype) if output_norm else None
    )
    scale = hidden_size**-0.5

    providers = ["flag_attn", "torch"]
    if HAS_FLA:
        providers.insert(1, "fla")
    print(f"GPU={torch.cuda.get_device_name()} dtype={dtype} D={hidden_size} output_norm={output_norm}")
    print("L\tN\t" + "\t".join(f"{provider}_ms" for provider in providers))

    for num_sources, num_rows in K3_SHAPES:
        residuals = [
            torch.randn(num_rows, hidden_size, device="cuda", dtype=dtype)
            for _ in range(num_sources)
        ]
        timings = []
        for provider in providers:
            if provider == "flag_attn":
                fn = lambda: flag_attn.fused_attnres(
                    query,
                    residuals,
                    rms_weight,
                    output_rms_weight,
                    scale=scale,
                )
            elif provider == "fla":
                fn = lambda: fla_fused_attnres(
                    query,
                    residuals,
                    rms_weight,
                    output_rms_weight,
                    scale=scale,
                )
            else:
                fn = lambda: flag_attn.testing.fused_attnres(
                    query,
                    residuals,
                    rms_weight,
                    output_rms_weight,
                    scale=scale,
                )
            fn()
            torch.cuda.synchronize()
            timings.append(_bench(fn))
        print(f"{num_sources}\t{num_rows}\t" + "\t".join(f"{timing:.6f}" for timing in timings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-norm", action="store_true")
    args = parser.parse_args()
    run(output_norm=args.output_norm)
