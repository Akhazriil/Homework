import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


# --- одна загрузка на весь файл (исправлено) ---
TICKER = "T"  # AT&T как в лаб.1, период тот же 2y
hist_lab2 = yf.Ticker(TICKER).history(period="2y")
hist_dat = hist_lab2  # совместимость с остальным кодом

print(hist_dat.head())
close = hist_dat['Close']
sma_20 = close.rolling(window=20).mean()
sma_50 = close.rolling(window=50).mean()

# --- Поиск пересечений ---
diff = sma_20 - sma_50
sign = np.sign(diff)
cross = sign.diff()

golden = cross[cross == 2].index
death = cross[cross == -2].index

#print("Золотые кресты:", list(golden.date))
#print("Мёртвые кресты:", list(death.date))

# --- График ---
plt.figure(figsize=(13, 6))

plt.plot(close.index, close, label='Close', color='steelblue', linewidth=1, alpha=0.6)
plt.plot(sma_20.index, sma_20, label='SMA 20', color='orange', linewidth=2)
plt.plot(sma_50.index, sma_50, label='SMA 50', color='red', linewidth=2)


plt.scatter(golden, sma_20.loc[golden],
            marker='x', s=140, linewidths=3,
            color='gold', edgecolors='black', zorder=5,
            label='Golden Cross')

# Мёртвые кресты
plt.scatter(death, sma_20.loc[death],
            marker='x', s=140, linewidths=3,
            color='black', zorder=5,
            label='Death Cross')

plt.title('SMA 20 / SMA 50')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab2_SMA.png", dpi=150, bbox_inches='tight')
print("Сохранено lab2_SMA.png")
# plt.show()
plt.close()

window = 20
num_std = 2

mid   = close.rolling(window).mean()
std   = close.rolling(window).std()
upper = mid + num_std * std
lower = mid - num_std * std

plt.figure(figsize=(13, 6))

plt.plot(close.index, mid,   color='orange', linewidth=2, label='SMA 20')
plt.plot(close.index, upper, color='red',    linewidth=1.5, linestyle='--', label='(2σ)')
plt.plot(close.index, lower, color='red',    linewidth=1.5, linestyle='--', label='(-2σ)')

plt.plot(close.index, close, color='steelblue', linewidth=1, alpha=0.8, label='Close')

plt.title('Полосы Боллинджера (20, 2σ)')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab2_BB.png", dpi=150, bbox_inches='tight')
print("Сохранено lab2_BB.png")
# plt.show()
plt.close()

# hist_dat уже загружен выше, переиспользуем
close = hist_dat['Close'].copy()
n = len(close)

# RSI (Уайлдер, period=14)
period = 14
delta = close.diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
rsi = (100 - (100 / (1 + avg_gain / avg_loss))).fillna(50)

order = 5
price_highs = argrelextrema(close.values, np.greater, order=order)[0]
price_lows  = argrelextrema(close.values, np.less,    order=order)[0]

def find_divergences(price_ext, close, rsi,
                     mode='bearish',
                     min_gap=10, max_gap=80,
                     zone=55, max_skip=2):

    divs = []
    for i in range(len(price_ext)):
        for k in range(1, max_skip + 2):
            j = i + k
            if j >= len(price_ext):
                break

            p1, p2 = int(price_ext[i]), int(price_ext[j])
            gap = p2 - p1
            if gap < min_gap:
                continue
            if gap > max_gap:
                break

            pv1, pv2 = close[p1], close[p2]
            rv1, rv2 = rsi[p1],  rsi[p2]

            if mode == 'bearish':
                if pv2 > pv1 and rv2 < rv1 and max(rv1, rv2) > zone:
                    divs.append((p1, p2, 'bearish'))
            else:
                if pv2 < pv1 and rv2 > rv1 and min(rv1, rv2) < (100 - zone):
                    divs.append((p1, p2, 'bullish'))
    return divs


def keep_non_overlapping(divs, top_n=5):

    divs = sorted(divs, key=lambda d: d[1] - d[0], reverse=True)
    used = set()
    out = []
    for p1, p2, kind in divs:
        if p1 in used or p2 in used:
            continue
        out.append((p1, p2, kind))
        used.add(p1); used.add(p2)
        if len(out) >= top_n:
            break
    return out


bear_all = find_divergences(price_highs, close.values, rsi.values,
                            mode='bearish', min_gap=10, max_gap=80,
                            zone=55, max_skip=2)
bull_all = find_divergences(price_lows,  close.values, rsi.values,
                            mode='bullish', min_gap=10, max_gap=80,
                            zone=55, max_skip=2)



bear_div = keep_non_overlapping(bear_all, top_n=5)
bull_div = keep_non_overlapping(bull_all, top_n=5)

fig, (ax_p, ax_r) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

ax_p.plot(close.index, close, color='steelblue', linewidth=1, label='Close')
ax_p.set_title('RSI(14) и дивергенции', fontsize=13)
ax_p.set_ylabel('Цена')
ax_p.grid(True, alpha=0.3)

ax_r.plot(rsi.index, rsi, color='purple', linewidth=1.5, label='RSI(14)')
ax_r.axhline(70, color='red',   linestyle='--', linewidth=1, alpha=0.7)
ax_r.axhline(50, color='gray',  linestyle=':',  linewidth=1, alpha=0.5)
ax_r.axhline(30, color='green', linestyle='--', linewidth=1, alpha=0.7)
ax_r.fill_between(rsi.index, 70, 100, color='red',   alpha=0.06)
ax_r.fill_between(rsi.index, 0,  30, color='green', alpha=0.06)
ax_r.set_ylim(0, 100)
ax_r.set_ylabel('RSI')
ax_r.set_xlabel('Дата')
ax_r.grid(True, alpha=0.3)


def draw_divs(divs, color, label):
    for k, (p1, p2, kind) in enumerate(divs):
        t1, t2 = close.index[p1], close.index[p2]
        ax_p.plot([t1, t2], [close.values[p1], close.values[p2]],
                  color=color, linewidth=2.2, alpha=0.95,
                  label=label if k == 0 else None)
        ax_p.scatter([t1, t2], [close.values[p1], close.values[p2]],
                     color=color, s=55, zorder=5,
                     edgecolors='white', linewidths=1)
        ax_r.plot([t1, t2], [rsi.values[p1], rsi.values[p2]],
                  color=color, linewidth=2.2, alpha=0.95)
        ax_r.scatter([t1, t2], [rsi.values[p1], rsi.values[p2]],
                     color=color, s=55, zorder=5,
                     edgecolors='white', linewidths=1)


draw_divs(bear_div, color='red',   label='Медвежьих дивергенций')
draw_divs(bull_div, color='green', label='Бычьих дивергенций')

ax_p.legend(loc='upper left', fontsize=10)
ax_r.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("lab2_RSI.png", dpi=150, bbox_inches='tight')
print("Сохранено lab2_RSI.png")
# plt.show()
plt.close()

# hist_dat уже загружен выше, переиспользуем
close = hist_dat['Close'].copy()
high  = hist_dat['High'].copy()
low   = hist_dat['Low'].copy()


period = 14


argmax_high = high.rolling(period + 1).apply(np.argmax, raw=True)
argmin_low  = low.rolling(period + 1).apply(np.argmin, raw=True)

aroon_up   = 100 * argmax_high / period
aroon_down = 100 * argmin_low  / period
aroon_osc  = aroon_up - aroon_down

sign = np.sign(aroon_up - aroon_down)
cross = sign.diff()

golden = cross[cross ==  2].index
death  = cross[cross == -2].index



fig, (ax_p, ax_a) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True,
    gridspec_kw={'height_ratios': [3, 1.3]}
)


ax_p.plot(close.index, close, color='steelblue', linewidth=1.2, label='Close')


ax_p.scatter(golden, close.loc[golden],
             marker='^', s=120, color='limegreen',
             edgecolors='black', linewidths=1, zorder=5,
             label='Aroon: Бычий')
ax_p.scatter(death, close.loc[death],
             marker='v', s=120, color='red',
             edgecolors='black', linewidths=1, zorder=5,
             label='Aroon: Медвежий)')

ax_p.set_title('Aroon(14)', fontsize=13)
ax_p.set_ylabel('Цена')
ax_p.grid(True, alpha=0.3)
ax_p.legend(loc='upper left', fontsize=9)


ax_a.plot(aroon_up.index,   aroon_up,   color='green', linewidth=1.8, label='Aroon Up')
ax_a.plot(aroon_down.index, aroon_down, color='red',   linewidth=1.8, label='Aroon Down')
ax_a.fill_between(aroon_osc.index, 0, aroon_osc,
                  where=(aroon_osc >= 0), color='green', alpha=0.08)
ax_a.fill_between(aroon_osc.index, 0, aroon_osc,
                  where=(aroon_osc <  0), color='red',   alpha=0.08)


ax_a.axhline(70, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_a.axhline(30, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_a.axhline(50, color='gray', linestyle=':',  linewidth=0.8, alpha=0.4)


if len(golden): ax_a.axvline(x=golden[0], color='limegreen', alpha=0)
for t in golden:
    ax_a.axvline(t, color='limegreen', linestyle=':', linewidth=0.8, alpha=0.5)
for t in death:
    ax_a.axvline(t, color='red',       linestyle=':', linewidth=0.8, alpha=0.5)

ax_a.set_ylim(0, 100)
ax_a.set_ylabel('Aroon')
ax_a.set_xlabel('Дата')
ax_a.grid(True, alpha=0.3)
ax_a.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig("lab2_Aroon.png", dpi=150, bbox_inches='tight')
print("Сохранено lab2_Aroon.png")
# plt.show()
plt.close()