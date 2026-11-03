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
    parser.add_argument('--profile', choices=['baseline', 'inseq-decoding'], default='baseline',
                        help='inseq-decoding: shared notch-cleaned X with separate odor/position targets')
    parser.add_argument('--notch-hz', type=float, default=60)
    parser.add_argument('--notch-q', type=float, default=30)
    parser.add_argument('--filter-tolerance', type=float, default=1e-6,
                        help='Pole-envelope decay tolerance used to derive filter edge guards')
    parser.add_argument('--flatline-ms', type=float, default=100)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        run(args.raw_root, args.output, args.sessions,
            Config(window_ms=args.window_ms, sample_rate_hz=args.sample_rate_hz,
                   train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
                   artifact_z=args.artifact_z, left_context=args.left_context,
                   right_context=args.right_context, positive_offset=args.positive_offset,
                   profile=args.profile, notch_hz=args.notch_hz, notch_q=args.notch_q,
                   filter_tolerance=args.filter_tolerance, flatline_ms=args.flatline_ms), not args.no_plots)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f'preprocessing: {error}\n')


if __name__ == '__main__':
    main()
