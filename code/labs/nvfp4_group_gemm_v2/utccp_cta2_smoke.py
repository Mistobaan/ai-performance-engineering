"""UTCCP (cta_group::2) smoke test for SM100a.

This is a minimal isolation harness for the `tcgen05.cp.cta_group::2.*` path.
It launches a 2-CTA cluster and performs UTCCP shared->TMEM copies using the
same descriptor + participation contract as `tmem_sf_frg_probe.cu`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.nvfp4_group_gemm_v2.custom_cuda_submission import load_v2_custom_cuda_nvfp4_group_gemm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use64", type=int, default=0, help="Use 64x128b.warpx2 copies (else 32x128b.warpx4).")
    parser.add_argument("--schedule", type=int, default=0, help="For use64: 0=01_23, 1=02_13.")
    parser.add_argument("--pattern", choices=("row", "col"), default="row", help="Shared tile init pattern.")
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        help="0=return, 1=alloc/dealloc, 2=desc only, 3=UTCCP only, 4=UTCCP+wait+cluster.sync, 5=+dump",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeat launches (sanity).")
    parser.add_argument("--dump", action="store_true", help="Dump a TMEM window back to host (slow).")
    parser.add_argument("--verbose-build", action="store_true", help="Print extension build logs.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UTCCP smoke test")

    ext = load_v2_custom_cuda_nvfp4_group_gemm(verbose=args.verbose_build)
    if args.dump:
        dump_elems = 2 * 128 * 64  # 2 ranks * (dp=128) * (col=64)
        out = torch.empty((dump_elems,), device="cuda", dtype=torch.int32)
    else:
        out = torch.empty((0,), device="cuda", dtype=torch.int32)

    pattern_kind = 0 if args.pattern == "row" else 1
    stage = int(args.stage)
    for _ in range(int(args.repeats)):
        if args.dump:
            out.zero_()
        ext.nvfp4_group_gemm_v2_utccp_cta2_smoke_cuda(
            out, pattern_kind, int(args.use64), int(args.schedule), stage
        )
    torch.cuda.synchronize()

    print("UTCCP cta_group::2 smoke: ok")
    if args.dump:
        host = out.cpu()
        print("head(int32):", host[:16].tolist())


if __name__ == "__main__":
    main()
