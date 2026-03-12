"""Console script entry point for `ignition` CLI command."""
import sys


def main() -> None:
    from ignition._ignition import _cli_main
    sys.exit(_cli_main())
