#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download U.S. financial price data and plot it."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL, MSFT, SPY")
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        default=dt.date.today().isoformat(),
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1d", "1wk", "1mo"],
        help="Data interval (default: 1d)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <ticker>_price.png)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save downloaded data as CSV",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title",
    )
    return parser.parse_args()


def to_unix_timestamp(date_text):
    return int(dt.datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_price_history(ticker, start, end, interval):
    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
    params = {
        "period1": to_unix_timestamp(start),
        "period2": to_unix_timestamp(end) + 86400,
        "interval": interval,
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(error.get("description", "Unknown API error"))

    results = chart.get("result")
    if not results:
        raise ValueError("No data returned.")

    result = results[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])

    rows = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        date = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date().isoformat()
        rows.append((date, float(close)))

    if not rows:
        raise ValueError("No usable closing-price data returned.")

    return rows


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        writer.writerows(rows)


def plot_rows(rows, ticker, output_path, title=None):
    dates = [dt.datetime.strptime(date_text, "%Y-%m-%d").date() for date_text, _ in rows]
    closes = [close for _, close in rows]

    plt.figure(figsize=(11, 6))
    plt.plot(dates, closes, linewidth=2)
    plt.title(title or f"{ticker.upper()} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main():
    args = parse_args()

    output_path = args.output or f"{args.ticker.lower()}_price.png"

    try:
        rows = fetch_price_history(args.ticker, args.start, args.end, args.interval)
        plot_rows(rows, args.ticker, output_path, args.title)

        if args.csv:
            save_csv(rows, args.csv)

        print(f"Saved chart to {output_path}")
        if args.csv:
            print(f"Saved data to {args.csv}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
