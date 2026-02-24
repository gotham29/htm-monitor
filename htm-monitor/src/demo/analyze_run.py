#demo/analyze_run.py

import argparse
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_hot(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return {}


def load_run(path: str):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["alert"] = df["alert"].astype(bool)
    df["hot_dict"] = df.get("hot_by_model", "").apply(parse_hot)
    return df


def summarize(df):
    one = df.drop_duplicates("t").copy()

    print("\n=== RUN SUMMARY ===")
    print("Timesteps:", one["t"].min(), "→", one["t"].max())
    print("Total alerts:", one["alert"].sum())

    alert_rows = one[one["alert"]]
    if len(alert_rows):
        print("\nAlert timesteps:")
        for _, r in alert_rows.iterrows():
            print(f"t={r['t']} ts={r['ts']} hot={r.get('hot_dict')}")
    else:
        print("\nNo alerts fired.")

    print("\n=== Per-model threshold crossings (p>=0.997) ===")
    piv = df.pivot_table(index="t", columns="model", values="p", aggfunc="first")
    for col in piv.columns:
        count = (piv[col] >= 0.997).sum()
        print(f"{col}: {count} crossings")


def plot_diagnostics(df, threshold=0.997):
    one = df.drop_duplicates("t").set_index("t")
    p_pivot = df.pivot_table(index="t", columns="model", values="p", aggfunc="first")

    plt.figure()
    for col in p_pivot.columns:
        plt.plot(p_pivot.index, p_pivot[col], label=col)
    plt.axhline(threshold, linestyle="--", color="black")
    plt.title("Anomaly Probability (p)")
    plt.xlabel("timestep")
    plt.ylabel("p")
    plt.legend()
    plt.show()

    hot_counts = one["hot_dict"].apply(
        lambda d: sum(1 for v in d.values() if v) if isinstance(d, dict) else 0
    )

    plt.figure()
    plt.plot(one.index, hot_counts, label="hot_count")
    plt.plot(one.index, one["alert"].astype(int), label="alert")
    plt.title("Decision Dynamics")
    plt.xlabel("timestep")
    plt.ylabel("count")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.997)
    args = ap.parse_args()

    df = load_run(args.csv)
    summarize(df)
    plot_diagnostics(df, threshold=args.threshold)


if __name__ == "__main__":
    main()