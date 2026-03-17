#!/usr/bin/env python3

import argparse
import csv
import io
import sys
import urllib.request

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a CSV file from a URL and plot its data."
    )
    parser.add_argument("url", help="Direct URL to a CSV file")
    parser.add_argument(
        "--x",
        help="Column name or zero-based index for x-axis. Defaults to row index.",
    )
    parser.add_argument(
        "--y",
        required=True,
        help="Column name or zero-based index for y-axis",
    )
    parser.add_argument(
        "--output",
        default="plot.png",
        help="Output image path (default: plot.png)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)",
    )
    parser.add_argument(
        "--title",
        default="Downloaded Data Plot",
        help="Plot title",
    )
    return parser.parse_args()


def download_text(url):
    with urllib.request.urlopen(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def looks_like_index(value):
    return value is not None and value.isdigit()


def get_series(rows, fieldnames, selector):
    if selector is None:
        return list(range(len(rows))), "index"

    if looks_like_index(selector):
        index = int(selector)
        if not rows:
            raise ValueError("CSV has no rows.")
        if index < 0 or index >= len(rows[0]):
            raise ValueError(f"Column index out of range: {index}")
        return [row[index] for row in rows], fieldnames[index] if fieldnames else str(index)

    if not fieldnames or selector not in fieldnames:
        raise ValueError(f"Column not found: {selector}")

    idx = fieldnames.index(selector)
    return [row[idx] for row in rows], selector


def to_float_series(values, label):
    converted = []
    for i, value in enumerate(values, start=1):
        try:
            converted.append(float(value))
        except ValueError as exc:
            raise ValueError(
                f'Non-numeric value in "{label}" at data row {i}: {value!r}'
            ) from exc
    return converted


def main():
    args = parse_args()

    try:
        text = download_text(args.url)
        reader = csv.reader(io.StringIO(text), delimiter=args.delimiter)
        all_rows = list(reader)

        if len(all_rows) < 2:
            raise ValueError("CSV must contain a header row and at least one data row.")

        fieldnames = all_rows[0]
        rows = all_rows[1:]

        x_raw, x_label = get_series(rows, fieldnames, args.x)
        y_raw, y_label = get_series(rows, fieldnames, args.y)

        x_values = to_float_series(x_raw, x_label) if args.x is not None else x_raw
        y_values = to_float_series(y_raw, y_label)

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker="o")
        plt.title(args.title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)

        print(f"Saved plot to {args.output}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
