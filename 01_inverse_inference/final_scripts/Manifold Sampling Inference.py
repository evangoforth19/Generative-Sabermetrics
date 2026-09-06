#!/usr/bin/env python3
"""Manifold Inference: sample x directly from admissible support."""

from __future__ import annotations

from pathlib import Path

from inference_core import build_arg_parser, build_context, run_manifold_sampler, write_outputs


def main() -> None:
    here = Path(__file__).resolve().parent
    incumbent = here.parent / "incumbent scripts"
    parser = build_arg_parser(default_root=here / "outputs_manifold")
    parser.set_defaults(
        source_script_path=incumbent / "run_mcmc_posterior_bank.py",
        production_script_path=incumbent / "run_inverse_collision_production.py",
    )
    args = parser.parse_args()
    ctx = build_context(args)
    draws, timing = run_manifold_sampler(ctx, args)
    write_outputs(draws, timing, Path(args.output_root), "manifold_inference", args)
    print("Manifold Inference complete.", len(draws), "draws across", len(timing), "events")


if __name__ == "__main__":
    main()
