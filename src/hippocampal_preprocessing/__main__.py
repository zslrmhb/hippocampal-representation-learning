"""Run with PYTHONPATH=src python -m hippocampal_preprocessing."""
import argparse
from pathlib import Path
from .pipeline import Config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('data/raw_data'))
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--sessions', nargs='+', help='Default: all raw session directories')
    parser.add_argument('--window-ms', type=float, default=500)
    parser.add_argument('--sample-rate-hz', type=float, default=1000)
    parser.add_argument('--train-fraction', type=float, default=.7)
    parser.add_argument('--validation-fraction', type=float, default=.15)
    parser.add_argument('--artifact-z', type=float, default=20)
    parser.add_argument('--left-context', type=int, default=0)
    parser.add_argument('--right-context', type=int, default=0)
    parser.add_argument('--positive-offset', type=int, default=1)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        run(args.raw_root, args.output, args.sessions,
            Config(args.window_ms, args.sample_rate_hz, args.train_fraction,
                   args.validation_fraction, args.artifact_z, args.left_context,
                   args.right_context, args.positive_offset), not args.no_plots)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f'preprocessing: {error}\n')


if __name__ == '__main__':
    main()
