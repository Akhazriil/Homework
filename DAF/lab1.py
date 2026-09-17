import yfinance as yf
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
    'INTC', 'NFLX', 'DIS', 'BA', 'JPM', 'BAC', 'WMT', 'KO', 'PEP',
    'JNJ', 'PG', 'XOM', 'CVX', 'T', 'VZ', 'PFE', 'MRK', 'ABBV',
    'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV', 'TLT', 'HYG',
    '^GSPC', '^DJI', '^IXIC', '^RUT'
]

ALPHA = 0.05


def check_normality(data):
    data = data.dropna()

    if len(data) < 30:
        return False, None

    standardized = (data - data.mean()) / data.std()
    statistic, p_value = stats.kstest(standardized, 'norm')

    is_normal = p_value > ALPHA
    return is_normal, p_value


def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        if hist.empty or len(hist) < 50:
            return None

        hist['return'] = hist['Close'] / hist['Close'].shift(1)
        hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))

        close_prices = hist['Close']
        returns = hist['return'].dropna()
        log_returns = hist['log_return'].dropna()

        close_normal, close_p = check_normality(close_prices)
        return_normal, return_p = check_normality(returns)
        log_return_normal, log_return_p = check_normality(log_returns)

        return {
            'ticker': ticker,
            'close_normal': close_normal,
            'close_p': close_p,
            'return_normal': return_normal,
            'return_p': return_p,
            'log_return_normal': log_return_normal,
            'log_return_p': log_return_p,
        }
    except Exception:
        return None


def main():
    results = []

    for ticker in tickers:
        result = analyze_ticker(ticker)
        if result:
            results.append(result)

    close_normal_tickers = [r['ticker'] for r in results if r['close_normal']]
    return_normal_tickers = [r['ticker'] for r in results if r['return_normal']]
    log_return_normal_tickers = [r['ticker'] for r in results if r['log_return_normal']]

    print("Close prices with normal distribution:")
    if close_normal_tickers:
        for t in close_normal_tickers:
            print(t)
    else:
        print("none")

    print()
    print("Returns with normal distribution:")
    if return_normal_tickers:
        for t in return_normal_tickers:
            print(t)
    else:
        print("none")

    print()
    print("Log returns with normal distribution:")
    if log_return_normal_tickers:
        for t in log_return_normal_tickers:
            print(t)
    else:
        print("none")

    print()
    print(f"{'Ticker':<10}{'Close':<15}{'Return':<15}{'Log Return':<15}")
    print("-" * 55)
    for r in results:
        close_str = f"{r['close_p']:.4f}" if r['close_p'] is not None else "N/A"
        return_str = f"{r['return_p']:.4f}" if r['return_p'] is not None else "N/A"
        log_str = f"{r['log_return_p']:.4f}" if r['log_return_p'] is not None else "N/A"
        print(f"{r['ticker']:<10}{close_str:<15}{return_str:<15}{log_str:<15}")

# import requests_cache
# from datetime import timedelta

# # Настройка кэширования запросов на 4 часа
# session = requests_cache.CachedSession('yfinance.cache', expire_after=timedelta(hours=4))
# session.headers['User-agent'] = 'my-program/1.0'
import yfinance as yf
import numpy as np
# yf.set_yfin_session(session)

mrvl_dat = yf.Ticker("T")
hist_dat = mrvl_dat.history(period="2y")
#hist_dat = mrvl_dat.history(period="1y")
print(hist_dat.head())
hist_dat['return'] = hist_dat['Close'] / hist_dat['Close'].shift(1)
hist_dat['log_return'] = np.log(hist_dat['Close'] / hist_dat['Close'].shift(1))

import matplotlib.pyplot as plt
import scipy.stats as stats

hist_dat = hist_dat.dropna()

fig, axes = plt.subplots(3, 4, figsize=(18, 14))

# ---------- Строка 0: Close ----------
axes[0, 0].plot(hist_dat.index, hist_dat["Close"], color='blue', linewidth=1)
axes[0, 0].set_title('График цены закрытия')
axes[0, 0].set_xlabel('Дата')
axes[0, 0].set_ylabel('Цена ($)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].hist(hist_dat['Close'], color='blue', edgecolor='black', bins=30, alpha=0.7)
axes[0, 1].set_title('Гистограмма распределения цен')
axes[0, 1].set_xlabel('Цена')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].axvline(hist_dat['Close'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].boxplot(hist_dat['Close'], vert=True)
axes[0, 2].set_title('Boxplot для цены')
axes[0, 2].set_ylabel('Цена ($)')
axes[0, 2].grid(True, alpha=0.3)

stats.probplot(hist_dat['Close'], dist="norm", plot=axes[0, 3])
axes[0, 3].set_title('QQ-график цены закрытия')
axes[0, 3].set_xlabel('Теоретические квантили')
axes[0, 3].set_ylabel('Выборочные квантили')
axes[0, 3].grid(True, alpha=0.3)

# ---------- Строка 1: return ----------
axes[1, 0].plot(hist_dat.index, hist_dat["return"], color='green', linewidth=0.7)
axes[1, 0].axhline(hist_dat['return'].mean(), color='red', linestyle='-', linewidth=1, alpha=0.5)
axes[1, 0].set_title('График обычной доходности')
axes[1, 0].set_xlabel('Дата')
axes[1, 0].set_ylabel('Доходность')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].hist(hist_dat['return'], color='green', edgecolor='black', bins=30, alpha=0.7)
axes[1, 1].set_title('Гистограмма доходности')
axes[1, 1].set_xlabel('Доходность')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].axvline(hist_dat['return'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].boxplot(hist_dat['return'], vert=True)
axes[1, 2].set_title('Boxplot доходности')
axes[1, 2].set_ylabel('Доходность')
axes[1, 2].grid(True, alpha=0.3)

stats.probplot(hist_dat['return'], dist="norm", plot=axes[1, 3])
axes[1, 3].set_title('QQ-график обычной доходности')
axes[1, 3].set_xlabel('Теоретические квантили')
axes[1, 3].set_ylabel('Выборочные квантили')
axes[1, 3].grid(True, alpha=0.3)

# ---------- Строка 2: log_return ----------
axes[2, 0].plot(hist_dat.index, hist_dat["log_return"], color='purple', linewidth=0.7)
axes[2, 0].axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
axes[2, 0].set_title('График логарифмической доходности')
axes[2, 0].set_xlabel('Дата')
axes[2, 0].set_ylabel('Логарифмическая доходность')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].tick_params(axis='x', rotation=45)

axes[2, 1].hist(hist_dat['log_return'], color='purple', edgecolor='black', bins=30, alpha=0.7)
axes[2, 1].set_title('Гистограмма распределения логарифмической доходности')
axes[2, 1].set_xlabel('Логарифмическая доходность')
axes[2, 1].set_ylabel('Частота')
axes[2, 1].axvline(hist_dat['log_return'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

axes[2, 2].boxplot(hist_dat['log_return'], vert=True)
axes[2, 2].set_title('Boxplot логарифмической доходности')
axes[2, 2].set_ylabel('Логарифмическая доходность')
axes[2, 2].grid(True, alpha=0.3)

stats.probplot(hist_dat['log_return'], dist="norm", plot=axes[2, 3])
axes[2, 3].set_title('QQ-график логарифмической доходности')
axes[2, 3].set_xlabel('Теоретические квантили')
axes[2, 3].set_ylabel('Выборочные квантили')
axes[2, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lab1_T_3x4.png", dpi=150, bbox_inches='tight')
print("График сохранен: lab1_T_3x4.png")
# plt.show()  # заменено на savefig для отчета
plt.close()

print(hist_dat[["Close","return","log_return"]].describe())

from scipy import stats
import numpy as np
def kstest_normal(data):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    standardized = (data - data.mean()) / data.std()
    return stats.kstest(standardized, 'norm')

res_1 = kstest_normal(hist_dat["Close"])
res_2 = kstest_normal(hist_dat["return"].dropna())
res_3 = kstest_normal(hist_dat["log_return"].dropna())

print('Close:      ', 'нормальное' if res_1.pvalue > 0.05 else 'не нормальное')
print('Return:     ', 'нормальное' if res_2.pvalue > 0.05 else 'не нормальное')
print('Log return: ', 'нормальное' if res_3.pvalue > 0.05 else 'не нормальное')