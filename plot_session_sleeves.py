"""
Plot and export the 9 independent Session Edge Ensemble sleeve equity curves.

The curve calculation follows the user's requested pure-market logic:
    sleeve_returns = df["K_Return"] * mask.shift(1).fillna(False) * direction
    cumulative_return = (1 + sleeve_returns).cumprod() - 1

Transaction costs are intentionally excluded so each sleeve shows its raw
market capture ability.
"""
import argparse
import os

import pandas as pd

from download_kbar import load_5min_data


SLEEVE_RULES = {
    "Mon 18:10-22:10 Long": ("18:10", "22:10", 1, 0),
    "Tue 00:00-09:40 Long": ("00:00", "09:40", 1, 1),
    "Tue 09:40-13:40 Short": ("09:40", "13:40", -1, 1),
    "Tue 15:00-17:00 Long": ("15:00", "17:00", 1, 1),
    "Wed 08:50-10:25 Long": ("08:50", "10:25", 1, 2),
    "Wed 15:15-18:15 Long": ("15:15", "18:15", 1, 2),
    "Thu 02:45-04:15 Long": ("02:45", "04:15", 1, 3),
    "Fri 09:10-09:25 Short": ("09:10", "09:25", -1, 4),
    "Fri 09:30-13:30 Long": ("09:30", "13:30", 1, 4),
}


def build_sleeve_masks(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, int]]:
    sleeves = {}
    directions = {}

    for name, (start, end, direction, weekday) in SLEEVE_RULES.items():
        start_time = pd.Timestamp(start).time()
        end_time = pd.Timestamp(end).time()

        if start_time <= end_time:
            in_window = (df.index.time >= start_time) & (df.index.time < end_time)
        else:
            in_window = (df.index.time >= start_time) | (df.index.time < end_time)

        mask = pd.Series(in_window & (df.index.weekday == weekday), index=df.index)
        sleeves[name] = mask
        directions[name] = direction

    return sleeves, directions


def calculate_sleeve_equity_curves(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["K_Return"] = df["Close"].pct_change().fillna(0)

    sleeves, directions = build_sleeve_masks(df)
    equity_curves = {}

    for name, mask in sleeves.items():
        direction = directions[name]

        # Shift one bar to simulate trading the next K bar after the signal is known.
        sleeve_returns = df["K_Return"] * mask.shift(1).fillna(False) * direction

        # Pure market capture curve, excluding commission and slippage.
        cumulative_return = (1 + sleeve_returns).cumprod() - 1
        equity_curves[name] = cumulative_return

    return pd.DataFrame(equity_curves)


def summarize_curves(equity_curves: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for name in equity_curves.columns:
        curve = equity_curves[name].dropna()
        if curve.empty:
            continue

        peak = (1 + curve).cummax()
        drawdown = (1 + curve) / peak - 1

        summary_rows.append({
            "Sleeve": name,
            "Final Return (%)": curve.iloc[-1] * 100,
            "Max Drawdown (%)": drawdown.min() * 100,
            "Best Point (%)": curve.max() * 100,
            "Worst Point (%)": curve.min() * 100,
        })

    return pd.DataFrame(summary_rows).sort_values("Final Return (%)", ascending=False)


def plot_curves(equity_curves: pd.DataFrame, output_path: str, show: bool = False) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped PNG plotting.")
        return False

    plt.figure(figsize=(14, 8))

    for name in equity_curves.columns:
        cumulative_return = equity_curves[name]
        plt.plot(
            cumulative_return.index,
            cumulative_return.values,
            label=name,
            linewidth=1.5,
        )

    plt.title(
        "Session Edge Ensemble: 9 Sleeves Independent Equity Curves",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Sleeves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    return True


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        description="Export and plot Session Edge Ensemble sleeve equity curves."
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date, inclusive.")
    parser.add_argument("--end", default=None, help="End date, inclusive.")
    parser.add_argument("--output-dir", default=".", help="Directory for CSV/PNG outputs.")
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    args = parser.parse_args()

    df = load_5min_data()
    if df.empty:
        raise RuntimeError("No 5-minute data loaded.")

    df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]

    if df.empty:
        raise RuntimeError("No data remains after date filtering.")

    os.makedirs(args.output_dir, exist_ok=True)

    equity_curves = calculate_sleeve_equity_curves(df)
    summary = summarize_curves(equity_curves)

    curves_path = os.path.join(args.output_dir, "session_sleeve_equity_curves.csv")
    summary_path = os.path.join(args.output_dir, "session_sleeve_summary.csv")
    figure_path = os.path.join(args.output_dir, "session_sleeve_equity_curves.png")

    equity_curves.to_csv(curves_path, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plotted = plot_curves(equity_curves, figure_path, show=args.show)

    print("\nSession Edge Ensemble sleeve summary")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nEquity curve tail")
    print("=" * 80)
    print(equity_curves.tail().to_string(float_format=lambda x: f"{x:,.6f}"))

    print("\nOutputs")
    print("=" * 80)
    print(f"Curves CSV : {os.path.abspath(curves_path)}")
    print(f"Summary CSV: {os.path.abspath(summary_path)}")
    if plotted:
        print(f"Figure PNG : {os.path.abspath(figure_path)}")

    return equity_curves


if __name__ == "__main__":
    main()
