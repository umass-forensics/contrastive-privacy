#!/usr/bin/env python3
"""
Search Pushshift Reddit JSONL data for a keyword in the "body" field.

Reads the file line-by-line (streaming) so it can handle very large files.
Writes each matching body to a separate file in an output folder.

Example:
    search-reddit-body --keyword "wizard" -o ./matches /scratch3/gbiss/text/RC_2019-04
    search-reddit-body --keyword "wizard" --max-chars 1000000 -o ./matches /scratch3/gbiss/text/RC_2019-04
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Reddit JSONL for keyword in 'body', write each matching body to a file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to Pushshift Reddit JSONL file (e.g. RC_2019-04)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output folder; one file per matching body (named by comment id, or body_N.txt).",
    )
    parser.add_argument(
        "--keyword",
        "-k",
        required=True,
        help="Keyword to search for in the body field.",
    )
    parser.add_argument(
        "--max-chars",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Only scan the first N characters of the file. Omit to scan entire file.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match keyword case-sensitively (default is case-insensitive).",
    )
    args = parser.parse_args()

    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    keyword = args.keyword if args.case_sensitive else args.keyword.lower()

    chars_read = 0
    max_chars = args.max_chars
    count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_len = len(line)
            if max_chars is not None and chars_read + line_len > max_chars:
                break
            chars_read += line_len

            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = obj.get("body")
            if body is None:
                continue
            search_in = body if args.case_sensitive else body.lower()
            if keyword in search_in:
                count += 1
                comment_id = obj.get("id")
                out_name = f"{comment_id}.txt" if comment_id else f"body_{count:05d}.txt"
                out_path = out_dir / out_name
                out_path.write_text(body, encoding="utf-8")

    print(f"Wrote {count} body file(s) to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
