"""Отбор акций, дневная доходность которых согласуется с нормальным законом."""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib import pyplot as plt
from scipy import stats

matplotlib.use("Agg")

ISS_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/"
    "{board}/securities/{ticker}.json"
)

RU_TICKERS = (
    "SBER GAZP LKOH ROSN GMKN NVTK PLZL ALRO MGNT MTSS NLMK PHOR "
    "RTKM RUSN RAIL SNGS STSB TATN UNIP VKCO VSMO AFKS AFLT CBOM "
    "CHMK HYDR IRAO MAGN MELT MNCH NKHP TRNFP BASH BSPB LSNG MSRS "
    "OGKB FLOT GLTR AQUA DELI RENI"
).split()

GLOBAL_TICKERS = (
    "JNJ PG KO PEP MRK ABT TGT WMT COST CL PM MO GIS KMB HSY MDLZ "
    "NEE SO D DUK AEP SRE BMY CAH"
).split()

MIN_OBS = 200
TESTS = ("shapiro", "dagostino", "jarque")


def parse_args():
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    parser = argparse.ArgumentParser(
        description="Проверка нормальности дневных доходностей по ценам закрытия."
    )
    parser.add_argument("--start", default=start.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=end.strftime("%Y-%m-%d"))
    parser.add_argument("--source", choices=("moex", "yahoo", "csv"), default="moex")
    parser.add_argument("--tickers", nargs="+", help="свой список тикеров")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--kind", choices=("simple", "log"), default="simple")
    parser.add_argument("--freq", choices=("daily", "weekly"), default="daily")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        parser.error("alpha должен быть в диапазоне (0, 1)")
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        parser.error("начало окна должно быть раньше конца")
    return args


def resolve_tickers(args):
    if args.tickers:
        return args.tickers
    if args.source == "yahoo":
        return GLOBAL_TICKERS
    if args.source == "csv":
        return sorted(path.stem for path in (args.data_dir / "csv").glob("*.csv"))
    return RU_TICKERS


def adjust_splits(series):
    # MOEX отдает цены без корректировки на сплиты: цена падает в 10 или 100 раз за день.
    # Такие скачки делим предыдущие цены на целое отношение, пока ряд не станет непрерывным.
    values = series.to_numpy(dtype=float).copy()
    for i in range(1, len(values)):
        ratio = values[i - 1] / values[i]
        k = round(ratio)
        if k >= 2 and ratio > 1.8 and abs(ratio / k - 1) < 0.2:
            values[:i] /= k
    return pd.Series(values, index=series.index, name=series.name)


def clean_prices(series):
    series = series.astype(float)
    series.index = pd.to_datetime(series.index)
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series = series[series.index.notna()]
    series = series[np.isfinite(series) & (series > 0)]
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = adjust_splits(series)
    series.index.name = "date"
    return series


def _get_json(session, url, params):
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1.0 + attempt)


def _iss_table(payload):
    if "history" in payload and isinstance(payload["history"], dict):
        block = payload["history"]
        if "columns" in block and "data" in block:
            return pd.DataFrame(block["data"], columns=block["columns"])
    raise RuntimeError("в ответе ISS нет таблицы истории")


def fetch_moex_closes(session, ticker, start, end, board):
    url = ISS_URL.format(board=board, ticker=ticker)
    pages = []
    offset = 0
    max_pages = 100

    for _ in range(max_pages):
        payload = _get_json(
            session,
            url,
            {
                "from": start,
                "till": end,
                "start": offset,
                "iss.meta": "off",
            },
        )
        page = _iss_table(payload)
        # ISS отдает не более 100 строк за запрос, поэтому идем до пустой страницы
        if page.empty:
            break
        pages.append(page)
        offset += len(page)

    if not pages:
        raise RuntimeError("ISS не вернул историю цен")

    raw = pd.concat(pages, ignore_index=True)
    date_column = "TRADEDATE" if "TRADEDATE" in raw.columns else "BEGINDATE"
    price_column = "CLOSE" if "CLOSE" in raw.columns else "CLOSEPRICE"
    if date_column not in raw.columns or price_column not in raw.columns:
        raise RuntimeError("в ответе ISS нет дат или цен закрытия")

    prices = pd.Series(
        pd.to_numeric(raw[price_column], errors="coerce").to_numpy(),
        index=pd.to_datetime(raw[date_column], errors="coerce"),
        dtype="float64",
    )
    return clean_prices(prices)


def fetch_yahoo_closes(ticker, start, end):
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance не вернул данные по {ticker}")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return clean_prices(close)


def read_csv_closes(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    date_column = next(
        (c for c in ("date", "дата", "tradedate") if c in frame.columns), None
    )
    price_column = next(
        (
            c
            for c in ("adj close", "adj_close", "adjusted close", "close", "закрытие", "closeprice")
            if c in frame.columns
        ),
        None,
    )
    if date_column is None or price_column is None:
        raise ValueError(f"{path}: нужны колонки Date и Close (или Adj Close)")

    prices = pd.Series(
        pd.to_numeric(frame[price_column], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame[date_column], errors="coerce"),
        dtype="float64",
    )
    return clean_prices(prices)


def cached_series(cache_path, loader):
    if cache_path.exists():
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
        if len(frame) > 1:
            return frame.iloc[:, 0].astype(float)
    series = loader()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(cache_path)
    return series


def load_prices(ticker, args, session):
    if args.source == "csv":
        prices = read_csv_closes(args.data_dir / "csv" / f"{ticker}.csv")
    elif args.source == "yahoo":
        cache = args.data_dir / f"yahoo_{ticker}_{args.start}_{args.end}.csv"
        prices = cached_series(
            cache, lambda: fetch_yahoo_closes(ticker, args.start, args.end)
        )
    else:
        cache = args.data_dir / f"{args.board}_{ticker}_{args.start}_{args.end}.csv"
        prices = cached_series(
            cache,
            lambda: fetch_moex_closes(
                session, ticker, args.start, args.end, args.board
            ),
        )

    prices = prices.loc[(prices.index >= args.start) & (prices.index <= args.end)]
    if len(prices) < 2:
        raise RuntimeError("в окне меньше двух цен закрытия")
    return prices


def make_returns(prices, kind):
    prices = prices.sort_index()
    if kind == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return (prices / prices.shift(1) - 1.0).dropna()


def test_normality(returns):
    values = returns.to_numpy(dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "skew": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, bias=False)),
        "p_shapiro": float(stats.shapiro(values).pvalue),
        "p_dagostino": float(stats.normaltest(values).pvalue),
        "p_jarque": float(stats.jarque_bera(values).pvalue),
    }


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    count = len(values)
    adjusted = values[order] * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def plot_fit(returns, ticker, path):
    values = returns.to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(values, bins="auto", density=True, color="steelblue", alpha=0.6)
    grid = np.linspace(values.min(), values.max(), 400)
    axes[0].plot(
        grid,
        stats.norm.pdf(grid, values.mean(), values.std(ddof=1)),
        color="darkred",
        lw=2,
    )
    axes[0].set_title(f"{ticker}: распределение доходности")
    axes[0].set_xlabel("доходность за день")

    stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot против нормального закона")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def to_weekly(prices):
    # Недельная цена закрытия: последняя цена недели (пятница или последний торговый день)
    return prices.resample("W-FRI").last().dropna()


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tickers = resolve_tickers(args)
    if not tickers:
        raise SystemExit("Список бумаг пуст: передайте --tickers или положите CSV в data/csv")

    session = requests.Session()
    session.headers.update({"User-Agent": "return-normality-screen/1.0"})

    tested = []
    for ticker in tickers:
        try:
            prices = load_prices(ticker, args, session)
            if args.freq == "weekly":
                prices = to_weekly(prices)
            returns = make_returns(prices, args.kind)
            if len(returns) < MIN_OBS:
                print(f"{ticker}: пропуск, только {len(returns)} доходностей")
                continue
            row = {"ticker": ticker, **test_normality(returns)}
            row["min_p"] = min(row[f"p_{name}"] for name in TESTS)
            tested.append((row, returns))
        except (requests.RequestException, RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"{ticker}: пропуск, {exc}")
        time.sleep(0.1)

    if not tested:
        raise SystemExit("Нет ни одной пригодной истории; проверьте сеть и источник данных")

    report = pd.DataFrame([row for row, _ in tested])
    report = report.sort_values("min_p", ascending=False).reset_index(drop=True)
    for name in TESTS:
        report[f"q_{name}"] = benjamini_hochberg(report[f"p_{name}"].to_numpy())
    report["passed"] = (report[[f"p_{name}" for name in TESTS]] >= args.alpha).all(axis=1)

    confirmed = int(report["passed"].sum())
    selected = report.loc[report["passed"]].head(2) if confirmed >= 2 else report.head(2)

    print(f"\nОкно: {args.start} .. {args.end}; доходность: {args.kind}, частота: {args.freq}; проверено: {len(report)}")
    columns = (
        ["ticker", "n", "skew", "excess_kurtosis", "min_p", "passed"]
        + [f"p_{name}" for name in TESTS]
        + ["q_shapiro"]
    )
    preview = report[columns].copy()
    for name in [f"p_{name}" for name in TESTS] + ["q_shapiro", "min_p"]:
        preview[name] = preview[name].map(lambda value: f"{value:.3g}")
    print(preview.to_string(index=False))

    suffix = f"{args.kind}_{args.freq}"
    report.to_csv(args.results_dir / f"normality_report_{suffix}.csv", index=False)
    returns_matrix = pd.concat({row["ticker"]: returns for row, returns in tested}, axis=1)
    returns_matrix.index.name = "date"
    returns_matrix.to_csv(args.results_dir / f"returns_{suffix}.csv")
    print(f"\nПорог alpha={args.alpha:.2f}; бумаг, прошедших все три теста: {confirmed}")
    if confirmed < 2:
        print("Двух подтверждённых кандидатов нет. Ниже две ближайшие к нормали бумаги,")
        print("но это рейтинг близости, а не подтверждение нормальности.")

    plot_dir = args.results_dir / f"plots_{suffix}"
    plot_dir.mkdir(exist_ok=True)
    by_ticker = {row["ticker"]: returns for row, returns in tested}

    print("\nОтобранные бумаги:")
    for _, row in selected.iterrows():
        ticker = row["ticker"]
        print(
            f"{ticker}: n={int(row['n'])}, средняя={100 * row['mean']:.3f}% {'в неделю' if args.freq == 'weekly' else 'в день'}, "
            f"σ={100 * row['std']:.3f}%, асимметрия={row['skew']:.3f}, "
            f"эксцесс={row['excess_kurtosis']:.3f}, прошли тесты={bool(row['passed'])}"
        )
        if row["q_shapiro"] < args.alpha:
            print("  после FDR-поправки гипотеза уже отвергается: кандидат слабый")
        plot_path = plot_dir / f"{ticker}.png"
        plot_fit(by_ticker[ticker], ticker, plot_path)
        print(f"  график: {plot_path}")

    print("\nИтог: " + ", ".join(selected["ticker"]))


if __name__ == "__main__":
    main()