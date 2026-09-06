#!/usr/bin/env python3
"""Manifold Compression Inference: deterministic canonical density on admissible intervals."""

from __future__ import annotations

from pathlib import Path

from inference_core import (
    build_arg_parser,
    build_context,
    run_manifold_compression_sampler,
    write_compression_outputs,
)


def main() -> None:
    here = Path(__file__).resolve().parent
    incumbent = here.parent / "incumbent scripts"
    parser = build_arg_parser(default_root=here / "outputs_manifold_compression")
    parser.set_defaults(
        source_script_path=incumbent / "run_mcmc_posterior_bank.py",
        production_script_path=incumbent / "run_inverse_collision_production.py",
        comparison_support_draws_path=here / "outputs_support_10events" / "support_inference_posterior_draws.parquet",
        comparison_manifold_draws_path=here / "outputs_manifold_10events" / "manifold_inference_posterior_draws.parquet",
    )
    args = parser.parse_args()
    ctx = build_context(args)
    density, timing, summary, comparison = run_manifold_compression_sampler(ctx, args)
    write_compression_outputs(
        density_df=density,
        timing_df=timing,
        summary_df=summary,
        comparison_df=comparison,
        out_root=Path(args.output_root),
        name="manifold_compression_inference",
        args=args,
    )
    print(
        "Manifold Compression Inference complete.",
        len(density),
        "density rows across",
        len(timing),
        "timed events",
    )


if __name__ == "__main__":
    main()
