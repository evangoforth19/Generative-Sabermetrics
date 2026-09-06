#!/usr/bin/env python3
"""Support Inference: broad-support x proposals with admissibility-gated MH."""

from __future__ import annotations

from pathlib import Path

from inference_core import build_arg_parser, build_context, run_support_sampler, write_outputs


def main() -> None:
    here = Path(__file__).resolve().parent
    incumbent = here.parent / "incumbent scripts"
    parser = build_arg_parser(default_root=here / "outputs_support")
    parser.set_defaults(
        source_script_path=incumbent / "run_mcmc_posterior_bank.py",
        production_script_path=incumbent / "run_inverse_collision_production.py",
    )
    args = parser.parse_args()
    ctx = build_context(args)
    draws, timing = run_support_sampler(ctx, args)
    write_outputs(draws, timing, Path(args.output_root), "support_inference", args)
    print("Support Inference complete.", len(draws), "draws across", len(timing), "events")


if __name__ == "__main__":
    main()
