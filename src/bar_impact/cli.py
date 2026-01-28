"""
Command-line interface for bar_impact package.

This module provides the main CLI entry point for the package.
"""

import argparse
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def _get_version() -> str:
    """Get package version from metadata."""
    try:
        return version("bar_impact")
    except PackageNotFoundError:
        return "0.1.0.dev0"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BAR_IMPACT: Baryon Impact Analysis for Cosmological Maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {_get_version()}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process L1 norms
    l1_parser = subparsers.add_parser('process-l1', help='Process L1 norms')
    l1_parser.add_argument('input', help='Input file or directory')
    l1_parser.add_argument('--output', '-o', help='Output file')
    
    # Process power spectrum
    ps_parser = subparsers.add_parser('process-ps', help='Process power spectra')
    ps_parser.add_argument('input', help='Input file or directory')
    ps_parser.add_argument('--output', '-o', help='Output file')
    
    # Run inference
    infer_parser = subparsers.add_parser('infer', help='Run NPE inference')
    infer_parser.add_argument('data', help='Data vectors file')
    infer_parser.add_argument('--params', help='Parameters file')
    infer_parser.add_argument('--output', '-o', help='Output directory')
    
    # Aggregate results
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate results')
    agg_parser.add_argument('pattern', help='File pattern to aggregate')
    agg_parser.add_argument('--output', '-o', help='Output file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Commands will be implemented in Step 3
    print(f"Command '{args.command}' will be implemented in the next step.")
    print("For now, use the scripts in the scripts/ directory.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
