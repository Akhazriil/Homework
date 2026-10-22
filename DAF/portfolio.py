"""Оптимизация портфеля по Марковицу: минимальный риск, эффективная граница,
функция полезности, график коэффициента Шарпа и касательный портфель.

Ограничения: веса неотрицательны (без коротких позиций), сумма весов = 1.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

matplotlib.use("Agg")

TRADING_DAYS = 252
DEFAULT_TICKERS = ["SNGS", "PLZL", "AFLT", "RTKM", "FLOT"]
RISK_FREE = 0.14  # ключевая ставка ЦБ РФ на 21.10.2026, годовых
RISK_AVERSION = 3.0  # коэффициент неприятия риска лямбда в U = mu'w - lambda/2 * w'Sigma w
FRONTIER_POINTS = 40


def load_returns(path, tickers):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame[tickers].dropna()


def annual_stats(returns):
    mu = returns.mean().to_numpy() * TRADING_DAYS
    sigma = returns.cov().to_numpy() * TRADING_DAYS
    return mu, sigma


def port_vol(w, sigma):
    return float(np.sqrt(w @ sigma @ w))


def port_sharpe(w, mu, sigma, rf):
    return float((mu @ w - rf) / port_vol(w, sigma))


def _solve(objective, n, extra=(), x0=None):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra]
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        x0=x0,
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"оптимизация не сошлась: {result.message}")
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def min_variance(mu, sigma):
    return _solve(lambda w: w @ sigma @ w, len(mu))


def max_return_at_risk(mu, sigma, target_vol):
    constraint = {"type": "ineq", "fun": lambda w: target_vol**2 - w @ sigma @ w}
    # старт из минимального риска: он допустим при любом target_vol >= sigma_min
    start = min_variance(mu, sigma)
    return _solve(lambda w: -(mu @ w), len(mu), extra=(constraint,), x0=start)


def max_utility(mu, sigma, lam):
    return _solve(lambda w: -(mu @ w - lam / 2 * (w @ sigma @ w)), len(mu))


def tangency(mu, sigma, rf):
    return _solve(lambda w: -port_sharpe(w, mu, sigma, rf), len(mu))


def efficient_frontier(mu, sigma, points=FRONTIER_POINTS):
    vol_min = port_vol(min_variance(mu, sigma), sigma)
    vol_max = float(np.sqrt(np.max(np.diag(sigma))))
    rows = []
    for target in np.linspace(vol_min, vol_max, points):
        w = max_return_at_risk(mu, sigma, target)
        rows.append(
            {
                "target_vol": target,
                "return": float(mu @ w),
                "vol": port_vol(w, sigma),
                "sharpe": port_sharpe(w, mu, sigma, RISK_FREE),
                "weights": w,
            }
        )
    return pd.DataFrame(rows)


def run_all(returns, rf=RISK_FREE, lam=RISK_AVERSION, target_vol=None):
    tickers = list(returns.columns)
    mu, sigma = annual_stats(returns)
    n = len(tickers)
    equal = np.full(n, 1.0 / n)

    if target_vol is None:
        target_vol = port_vol(equal, sigma)

    portfolios = {
        "1a. минимальный риск": min_variance(mu, sigma),
        "1б. макс. доходность при заданном риске": max_return_at_risk(mu, sigma, target_vol),
        "1в. макс. функция полезности": max_utility(mu, sigma, lam),
        "3. касательный портфель": tangency(mu, sigma, rf),
        "для сравнения: равные веса": equal,
    }

    summary = []
    for name, w in portfolios.items():
        row = {"портфель": name}
        row.update({ticker: weight for ticker, weight in zip(tickers, w)})
        row["доходность, год"] = float(mu @ w)
        row["риск (sigma), год"] = port_vol(w, sigma)
        row["Шарп"] = port_sharpe(w, mu, sigma, rf)
        summary.append(row)
    summary = pd.DataFrame(summary)

    frontier = efficient_frontier(mu, sigma)
    lam_table = []
    for value in (1.0, 2.0, 3.0, 5.0, 10.0):
        w = max_utility(mu, sigma, value)
        lam_table.append(
            {
                "lambda": value,
                "U": float(mu @ w - value / 2 * (w @ sigma @ w)),
                "доходность, год": float(mu @ w),
                "риск (sigma), год": port_vol(w, sigma),
                **{ticker: weight for ticker, weight in zip(tickers, w)},
            }
        )

    return {
        "tickers": tickers,
        "mu": mu,
        "sigma": sigma,
        "rf": rf,
        "lam": lam,
        "target_vol": target_vol,
        "portfolios": portfolios,
        "summary": summary,
        "frontier": frontier,
        "lambda_table": pd.DataFrame(lam_table),
    }


def plot_frontier(result, path):
    mu, sigma, rf = result["mu"], result["sigma"], result["rf"]
    tickers = result["tickers"]
    frontier = result["frontier"]
    tangent = result["portfolios"]["3. касательный портфель"]
    tangent_vol = port_vol(tangent, sigma)
    tangent_ret = float(mu @ tangent)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(frontier["vol"], frontier["return"], "o-", color="steelblue", label="эффективная граница")

    asset_vol = np.sqrt(np.diag(sigma))
    axis.scatter(asset_vol, mu, color="gray", zorder=3, label="отдельные бумаги")
    for ticker, x, y in zip(tickers, asset_vol, mu):
        axis.annotate(ticker, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)

    grid = np.linspace(0, max(frontier["vol"].max(), tangent_vol) * 1.15, 100)
    slope = (tangent_ret - rf) / tangent_vol
    axis.plot(grid, rf + slope * grid, "--", color="darkgreen", label="касательная (CML)")

    markers = {
        "1a. минимальный риск": ("^", "purple"),
        "1в. макс. функция полезности": ("s", "darkorange"),
        "3. касательный портфель": ("*", "darkgreen"),
    }
    for name, (marker, color) in markers.items():
        w = result["portfolios"][name]
        axis.scatter(port_vol(w, sigma), mu @ w, marker=marker, s=140, color=color, zorder=4, label=name)
    axis.scatter(0, rf, color="black", zorder=4, label=f"безрисковая ставка {rf:.0%}")

    axis.set_xlabel("риск, годовой σ")
    axis.set_ylabel("ожидаемая доходность, годовая")
    axis.set_title("Эффективная граница и касательный портфель")
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_sharpe(result, path):
    mu, sigma, rf = result["mu"], result["sigma"], result["rf"]
    tickers = result["tickers"]
    frontier = result["frontier"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(frontier["vol"], frontier["sharpe"], "o-", color="steelblue")
    best = frontier["sharpe"].idxmax()
    axes[0].scatter(frontier.loc[best, "vol"], frontier.loc[best, "sharpe"], marker="*", s=200, color="darkgreen", zorder=3, label="максимум Шарпа на границе")
    axes[0].set_xlabel("риск, годовой σ")
    axes[0].set_ylabel("коэффициент Шарпа")
    axes[0].set_title("Шарп портфелей эффективной границы")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    asset_sharpe = [(mu[i] - rf) / np.sqrt(sigma[i, i]) for i in range(len(tickers))]
    axes[1].bar(tickers, asset_sharpe, color="gray")
    tangent_sharpe = port_sharpe(result["portfolios"]["3. касательный портфель"], mu, sigma, rf)
    axes[1].axhline(tangent_sharpe, color="darkgreen", ls="--", label=f"касательный портфель: {tangent_sharpe:.2f}")
    axes[1].set_title("Шарп отдельных бумаг")
    axes[1].set_ylabel("коэффициент Шарпа")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_outputs(result, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = results_dir / "plots_portfolio"
    plot_dir.mkdir(exist_ok=True)

    result["summary"].to_csv(results_dir / "portfolio_summary.csv", index=False)
    result["lambda_table"].to_csv(results_dir / "portfolio_utility_lambda.csv", index=False)
    frontier = result["frontier"].drop(columns=["weights"])
    frontier = pd.concat([frontier, pd.DataFrame(result["frontier"]["weights"].tolist(), columns=result["tickers"])], axis=1)
    frontier.to_csv(results_dir / "portfolio_frontier.csv", index=False)

    plot_frontier(result, plot_dir / "frontier.png")
    plot_sharpe(result, plot_dir / "sharpe.png")


def main():
    parser = argparse.ArgumentParser(description="Портфель Марковица на дневных доходностях.")
    parser.add_argument("--returns", type=Path, default=Path("results/returns_simple_daily.csv"))
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--rf", type=float, default=RISK_FREE)
    parser.add_argument("--lam", type=float, default=RISK_AVERSION)
    parser.add_argument("--target-vol", type=float, default=None, help="целевой риск для задачи 1б (по умолчанию риск равновесного портфеля)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    returns = load_returns(args.returns, args.tickers)
    print(f"Бумаги: {', '.join(args.tickers)}; наблюдений: {len(returns)}; rf={args.rf:.2%}; lambda={args.lam}")

    result = run_all(returns, rf=args.rf, lam=args.lam, target_vol=args.target_vol)
    save_outputs(result, args.results_dir)

    pd.set_option("display.width", 200)
    print(result["summary"].round(4).to_string(index=False))
    print(f"\nЦелевой риск для 1б: {result['target_vol']:.4f}")
    print("\nТаблица полезности по lambda:")
    print(result["lambda_table"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
