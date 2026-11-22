"""Command-line interface for the three physical verification contracts."""
import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from decisions import run_decision_gates


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--fast',action='store_true',help='Use bounded grids (default)')
    modes.add_argument('--paper',action='store_true',help='Use the extended verification grids')
    parser.add_argument('--setup',choices=['2level','3level','multilevel','both'],default='2level')
    args=parser.parse_args(argv)
    verdict=run_decision_gates(fast=not args.paper,setup=args.setup)
    print(verdict.summary)
    return 0 if verdict.all_pass else 1


if __name__=='__main__':
    raise SystemExit(main())
