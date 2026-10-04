"""Weekly TFT-style multi-horizon forecasts for the enriched master table."""

from __future__ import annotations

import argparse
from pathlib import Path

from multihorizon_sequence_forecast_common import add_common_arguments, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("outputs/tft")
    modes = ["absolute", "residual"] if args.target_mode == "both" else [args.target_mode]
    run_experiment("tft", args.frequency, modes, args)


if __name__ == "__main__":
    main()
