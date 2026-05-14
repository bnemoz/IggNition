"""Console script entry point for `iggnition` CLI command."""
import sys


def main() -> None:
    from iggnition._ignition import _cli_main
    sys.exit(_cli_main())
