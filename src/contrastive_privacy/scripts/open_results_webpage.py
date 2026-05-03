#!/usr/bin/env python3

from __future__ import annotations

from contrastive_privacy.scripts.generate_results_webpage import cli_main


def main() -> None:
    cli_main(default_open=True)


if __name__ == "__main__":
    main()