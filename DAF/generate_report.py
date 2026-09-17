import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
from fpdf import FPDF
import os, shutil, warnings
warnings.filterwarnings('ignore')

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
    'INTC', 'NFLX', 'DIS', 'BA', 'JPM', 'BAC', 'WMT', 'KO', 'PEP',
    'JNJ', 'PG', 'XOM', 'CVX', 'T', 'VZ', 'PFE', 'MRK', 'ABBV',
    'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV', 'TLT', 'HYG',
    '^GSPC', '^DJI', '^IXIC', '^RUT'
]
ALPHA = 0.05
PERIOD_LAB1 = "2y"
TICKER_DETAIL = "T"
# Лаба 2 теперь тоже T и тот же период
TICKER_LAB2 = "T"
PERIOD_LAB2 = "2y"

SMA_SHORT = 20
SMA_LONG = 50
BB_WINDOW = 20
BB_STD = 2
RSI_PERIOD = 14
RSI_ORDER = 5
AROON_PERIOD = 14

OUT_PDF = os.path.join(os.path.dirname(__file__), "report.pdf")
TMP_DIR = r"C:\Users\reino\AppData\Local\Temp\opencode"
os.makedirs(TMP_DIR, exist_ok=True)
_src_reg = r"C:\Users\reino\Desktop\Учеба\Homework\DAF\.venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\DejaVuSans.ttf"
_src_bold = r"C:\Users\reino\Desktop\Учеба\Homework\DAF\.venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\DejaVuSans-Bold.ttf"
DEJA_REG = os.path.join(TMP_DIR, "DejaVuSans.ttf")
DEJA_BOLD = os.path.join(TMP_DIR, "DejaVuSans-Bold.ttf")
if not os.path.exists(DEJA_REG):
    try:
        shutil.copy(_src_reg, DEJA_REG)
        shutil.copy(_src_bold, DEJA_BOLD)
    except:
        DEJA_REG = r"C:\Windows\Fonts\arial.ttf"
        DEJA_BOLD = r"C:\Windows\Fonts\arialbd.ttf"

def check_normality(data):
    data = data.dropna()
    if len(data) < 30:
        return False, None, None
    standardized = (data - data.mean()) / data.std()
    statistic, p_value = stats.kstest(standardized, 'norm')
    return p_value > ALPHA, p_value, statistic

def analyze_ticker(ticker):
    try:
        hist = yf.Ticker(ticker).history(period=PERIOD_LAB1)
        if hist.empty or len(hist) < 50:
            return None, None
        hist['return'] = hist['Close'] / hist['Close'].shift(1)
        hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        close_prices = hist['Close']
        returns = hist['return'].dropna()
        log_returns = hist['log_return'].dropna()
        cn, cp, cs = check_normality(close_prices)
        rn, rp, rs = check_normality(returns)
        ln, lp, ls = check_normality(log_returns)
        return {'ticker': ticker, 'close_normal': cn, 'close_p': cp, 'close_stat': cs,
                'return_normal': rn, 'return_p': rp, 'return_stat': rs,
                'log_return_normal': ln, 'log_return_p': lp, 'log_return_stat': ls, 'n': len(hist)}, hist
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None, None

print("=== LAB1: скачивание ===")
results=[]
hist_T_for_plot=None
for t in tickers:
    print(f" {t}...")
    res, hist = analyze_ticker(t)
    if res:
        results.append(res)
        if t==TICKER_DETAIL:
            hist_T_for_plot=hist
if hist_T_for_plot is None or hist_T_for_plot.empty:
    _, hist_T_for_plot = analyze_ticker(TICKER_DETAIL)
if hist_T_for_plot is not None and not hist_T_for_plot.empty:
    hist_dat = hist_T_for_plot.copy()
    hist_dat['return'] = hist_dat['Close']/hist_dat['Close'].shift(1)
    hist_dat['log_return'] = np.log(hist_dat['Close']/hist_dat['Close'].shift(1))
    hist_dat = hist_dat.dropna()
else:
    dates=pd.date_range(end=pd.Timestamp.now(), periods=500, freq='B')
    close=30+np.cumsum(np.random.randn(500)*0.2)
    hist_dat=pd.DataFrame({'Close': close}, index=dates)
    hist_dat['return']=hist_dat['Close']/hist_dat['Close'].shift(1)
    hist_dat['log_return']=np.log(hist_dat['Close']/hist_dat['Close'].shift(1))
    hist_dat=hist_dat.dropna()

def kstest_normal(data):
    data=np.asarray(data); data=data[~np.isnan(data)]
    standardized=(data-data.mean())/data.std()
    return stats.kstest(standardized,'norm')
res_1=kstest_normal(hist_dat["Close"])
res_2=kstest_normal(hist_dat["return"].dropna())
res_3=kstest_normal(hist_dat["log_return"].dropna())
print(f"T p: {res_1.pvalue:.4f} {res_2.pvalue:.4f} {res_3.pvalue:.4f}")

# === LAB2 на T тот же период ===
print("\n=== LAB2: T 2y ===")
try:
    hist_lab2 = yf.Ticker(TICKER_LAB2).history(period=PERIOD_LAB2)
    if hist_lab2.empty:
        raise ValueError("empty")
except Exception as e:
    print(f"yfinance failed {e}")
    hist_lab2 = hist_dat.copy()
    # добавим High/Low если нет
    if 'High' not in hist_lab2.columns:
        hist_lab2['High']=hist_lab2['Close']*1.005
        hist_lab2['Low']=hist_lab2['Close']*0.995
        hist_lab2['Open']=hist_lab2['Close']
        hist_lab2['Volume']=1000000

close_lab2 = hist_lab2['Close']
# SMA
sma_20 = close_lab2.rolling(window=SMA_SHORT).mean()
sma_50 = close_lab2.rolling(window=SMA_LONG).mean()
diff = sma_20 - sma_50
sign = np.sign(diff)
cross = sign.diff()
golden_sma = cross[cross==2].index
death_sma = cross[cross==-2].index
# BB
mid = close_lab2.rolling(window=BB_WINDOW).mean()
std = close_lab2.rolling(window=BB_WINDOW).std()
upper = mid + BB_STD*std
lower = mid - BB_STD*std
# RSI
close_rsi = hist_lab2['Close'].copy()
period=RSI_PERIOD
delta=close_rsi.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
avg_gain=gain.ewm(alpha=1/period, adjust=False).mean()
avg_loss=loss.ewm(alpha=1/period, adjust=False).mean()
rsi=(100-(100/(1+avg_gain/avg_loss))).fillna(50)
order=RSI_ORDER
price_highs=argrelextrema(close_rsi.values, np.greater, order=order)[0]
price_lows=argrelextrema(close_rsi.values, np.less, order=order)[0]
def find_divergences(price_ext, close_vals, rsi_vals, mode='bearish', min_gap=10, max_gap=80, zone=55, max_skip=2):
    divs=[]
    for i in range(len(price_ext)):
        for k in range(1,max_skip+2):
            j=i+k
            if j>=len(price_ext): break
            p1,p2=int(price_ext[i]),int(price_ext[j])
            gap=p2-p1
            if gap<min_gap: continue
            if gap>max_gap: break
            pv1,pv2=close_vals[p1],close_vals[p2]
            rv1,rv2=rsi_vals[p1],rsi_vals[p2]
            if mode=='bearish':
                if pv2>pv1 and rv2<rv1 and max(rv1,rv2)>zone: divs.append((p1,p2,'bearish'))
            else:
                if pv2<pv1 and rv2>rv1 and min(rv1,rv2)<(100-zone): divs.append((p1,p2,'bullish'))
    return divs
def keep_non_overlapping(divs, top_n=5):
    divs=sorted(divs, key=lambda d:d[1]-d[0], reverse=True)
    used=set(); out=[]
    for p1,p2,kind in divs:
        if p1 in used or p2 in used: continue
        out.append((p1,p2,kind)); used.add(p1); used.add(p2)
        if len(out)>=top_n: break
    return out
bear_all=find_divergences(price_highs, close_rsi.values, rsi.values, mode='bearish')
bull_all=find_divergences(price_lows, close_rsi.values, rsi.values, mode='bullish')
bear_div=keep_non_overlapping(bear_all, top_n=5)
bull_div=keep_non_overlapping(bull_all, top_n=5)
# Aroon
high=hist_lab2['High'].copy(); low=hist_lab2['Low'].copy()
argmax_high=high.rolling(period+1).apply(np.argmax, raw=True)
argmin_low=low.rolling(period+1).apply(np.argmin, raw=True)
aroon_up=100*argmax_high/period
aroon_down=100*argmin_low/period
sign_a=np.sign(aroon_up-aroon_down); cross_a=sign_a.diff()
golden_a=cross_a[cross_a==2].index; death_a=cross_a[cross_a==-2].index
print(f"SMA {len(golden_sma)}/{len(death_sma)} RSI {len(bear_div)}/{len(bull_div)} Aroon {len(golden_a)}/{len(death_a)}")
lab2_period_str = f"{hist_lab2.index.min().date()}  {hist_lab2.index.max().date()} ({len(hist_lab2)} дней, period={PERIOD_LAB2})"

# === графики ===
plt.rcParams['font.family']='DejaVu Sans'
fig1, axes = plt.subplots(3,4, figsize=(18,14))
axes[0,0].plot(hist_dat.index, hist_dat["Close"], color='blue', linewidth=1)
axes[0,0].set_title('График цены закрытия (T)'); axes[0,0].set_xlabel('Дата'); axes[0,0].set_ylabel('Цена ($)'); axes[0,0].grid(True, alpha=0.3); axes[0,0].tick_params(axis='x', rotation=30)
axes[0,1].hist(hist_dat['Close'], color='blue', edgecolor='black', bins=30, alpha=0.7)
axes[0,1].set_title('Гистограмма цен'); axes[0,1].set_xlabel('Цена'); axes[0,1].set_ylabel('Частота')
axes[0,1].axvline(hist_dat['Close'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[0,2].boxplot(hist_dat['Close'], vert=True); axes[0,2].set_title('Boxplot цены'); axes[0,2].set_ylabel('Цена ($)'); axes[0,2].grid(True, alpha=0.3)
stats.probplot(hist_dat['Close'], dist="norm", plot=axes[0,3]); axes[0,3].set_title('QQ цены'); axes[0,3].grid(True, alpha=0.3)
axes[1,0].plot(hist_dat.index, hist_dat["return"], color='green', linewidth=0.7)
axes[1,0].axhline(hist_dat['return'].mean(), color='red', linestyle='-', linewidth=1, alpha=0.5)
axes[1,0].set_title('Обычная доходность (Close/Close[-1])'); axes[1,0].set_xlabel('Дата'); axes[1,0].set_ylabel('Доходность'); axes[1,0].grid(True, alpha=0.3); axes[1,0].tick_params(axis='x', rotation=30)
axes[1,1].hist(hist_dat['return'], color='green', edgecolor='black', bins=30, alpha=0.7)
axes[1,1].set_title('Гистограмма доходности'); axes[1,1].set_xlabel('Доходность'); axes[1,1].set_ylabel('Частота')
axes[1,1].axvline(hist_dat['return'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
axes[1,2].boxplot(hist_dat['return'], vert=True); axes[1,2].set_title('Boxplot доходности'); axes[1,2].grid(True, alpha=0.3)
stats.probplot(hist_dat['return'], dist="norm", plot=axes[1,3]); axes[1,3].set_title('QQ доходности'); axes[1,3].grid(True, alpha=0.3)
axes[2,0].plot(hist_dat.index, hist_dat["log_return"], color='purple', linewidth=0.7)
axes[2,0].axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
axes[2,0].set_title('Логарифмическая доходность'); axes[2,0].set_xlabel('Дата'); axes[2,0].set_ylabel('log return'); axes[2,0].grid(True, alpha=0.3); axes[2,0].tick_params(axis='x', rotation=30)
axes[2,1].hist(hist_dat['log_return'], color='purple', edgecolor='black', bins=30, alpha=0.7)
axes[2,1].set_title('Гистограмма log доходности'); axes[2,1].set_xlabel('log return'); axes[2,1].set_ylabel('Частота')
axes[2,1].axvline(hist_dat['log_return'].mean(), color='red', linestyle='--', linewidth=2, label='Среднее'); axes[2,1].legend(); axes[2,1].grid(True, alpha=0.3)
axes[2,2].boxplot(hist_dat['log_return'], vert=True); axes[2,2].set_title('Boxplot log доходности'); axes[2,2].grid(True, alpha=0.3)
stats.probplot(hist_dat['log_return'], dist="norm", plot=axes[2,3]); axes[2,3].set_title('QQ log доходности'); axes[2,3].grid(True, alpha=0.3)
plt.tight_layout()
p1=os.path.join(TMP_DIR,"lab1_3x4.png"); fig1.savefig(p1,dpi=150,bbox_inches='tight'); plt.close(fig1)

fig2=plt.figure(figsize=(13,6))
plt.plot(close_lab2.index, close_lab2, label='Close', color='steelblue', linewidth=1, alpha=0.6)
plt.plot(sma_20.index, sma_20, label='SMA 20', color='orange', linewidth=2)
plt.plot(sma_50.index, sma_50, label='SMA 50', color='red', linewidth=2)
if len(golden_sma): plt.scatter(golden_sma, sma_20.loc[golden_sma], marker='x', s=140, linewidths=3, color='gold', edgecolors='black', zorder=5, label='Golden Cross')
if len(death_sma): plt.scatter(death_sma, sma_20.loc[death_sma], marker='x', s=140, linewidths=3, color='black', zorder=5, label='Death Cross')
plt.title(f'SMA 20 / SMA 50 (T, {PERIOD_LAB2})'); plt.xlabel('Дата'); plt.ylabel('Цена'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
p2=os.path.join(TMP_DIR,"lab2_sma.png"); fig2.savefig(p2,dpi=150,bbox_inches='tight'); plt.close(fig2)

fig3=plt.figure(figsize=(13,6))
plt.plot(close_lab2.index, mid, color='orange', linewidth=2, label='SMA 20')
plt.plot(close_lab2.index, upper, color='red', linewidth=1.5, linestyle='--', label='Upper (2σ)')
plt.plot(close_lab2.index, lower, color='red', linewidth=1.5, linestyle='--', label='Lower (2σ)')
plt.plot(close_lab2.index, close_lab2, color='steelblue', linewidth=1, alpha=0.8, label='Close')
plt.title(f'Полосы Боллинджера (20, 2σ) T'); plt.xlabel('Дата'); plt.ylabel('Цена'); plt.legend(loc='upper left'); plt.grid(True, alpha=0.3); plt.tight_layout()
p3=os.path.join(TMP_DIR,"lab2_bb.png"); fig3.savefig(p3,dpi=150,bbox_inches='tight'); plt.close(fig3)

fig4, (ax_p, ax_r) = plt.subplots(2,1, figsize=(14,9), sharex=True, gridspec_kw={'height_ratios':[3,1]})
ax_p.plot(close_rsi.index, close_rsi, color='steelblue', linewidth=1, label='Close')
ax_p.set_title(f'RSI(14) и дивергенции T'); ax_p.set_ylabel('Цена'); ax_p.grid(True, alpha=0.3)
ax_r.plot(rsi.index, rsi, color='purple', linewidth=1.5, label='RSI(14)')
ax_r.axhline(70,color='red',linestyle='--',linewidth=1,alpha=0.7); ax_r.axhline(50,color='gray',linestyle=':',linewidth=1,alpha=0.5); ax_r.axhline(30,color='green',linestyle='--',linewidth=1,alpha=0.7)
ax_r.fill_between(rsi.index,70,100,color='red',alpha=0.06); ax_r.fill_between(rsi.index,0,30,color='green',alpha=0.06)
ax_r.set_ylim(0,100); ax_r.set_ylabel('RSI'); ax_r.set_xlabel('Дата'); ax_r.grid(True, alpha=0.3)
def draw_divs(divs,color,label):
    for k,(p1,p2,kind) in enumerate(divs):
        t1,t2=close_rsi.index[p1],close_rsi.index[p2]
        ax_p.plot([t1,t2],[close_rsi.values[p1],close_rsi.values[p2]],color=color,linewidth=2.2,alpha=0.95,label=label if k==0 else None)
        ax_p.scatter([t1,t2],[close_rsi.values[p1],close_rsi.values[p2]],color=color,s=55,zorder=5,edgecolors='white',linewidths=1)
        ax_r.plot([t1,t2],[rsi.values[p1],rsi.values[p2]],color=color,linewidth=2.2,alpha=0.95)
        ax_r.scatter([t1,t2],[rsi.values[p1],rsi.values[p2]],color=color,s=55,zorder=5,edgecolors='white',linewidths=1)
draw_divs(bear_div,color='red',label='Медвежья дивергенция')
draw_divs(bull_div,color='green',label='Бычья дивергенция')
ax_p.legend(loc='upper left',fontsize=9); ax_r.legend(loc='upper left',fontsize=9); plt.tight_layout()
p4=os.path.join(TMP_DIR,"lab2_rsi.png"); fig4.savefig(p4,dpi=150,bbox_inches='tight'); plt.close(fig4)

fig5, (ax_p2, ax_a) = plt.subplots(2,1, figsize=(14,9), sharex=True, gridspec_kw={'height_ratios':[3,1.3]})
ax_p2.plot(close_lab2.index, close_lab2, color='steelblue', linewidth=1.2, label='Close')
if len(golden_a): ax_p2.scatter(golden_a, close_lab2.loc[golden_a], marker='^', s=120, color='limegreen', edgecolors='black', linewidths=1, zorder=5, label='Aroon бычий')
if len(death_a): ax_p2.scatter(death_a, close_lab2.loc[death_a], marker='v', s=120, color='red', edgecolors='black', linewidths=1, zorder=5, label='Aroon медвежий')
ax_p2.set_title(f'Aroon(14) T'); ax_p2.set_ylabel('Цена'); ax_p2.grid(True, alpha=0.3); ax_p2.legend(loc='upper left',fontsize=9)
ax_a.plot(aroon_up.index, aroon_up, color='green', linewidth=1.8, label='Aroon Up')
ax_a.plot(aroon_down.index, aroon_down, color='red', linewidth=1.8, label='Aroon Down')
ax_a.fill_between(aroon_up.index,0,aroon_up.where(aroon_up>=aroon_down),color='green',alpha=0.08)
ax_a.fill_between(aroon_down.index,0,aroon_down.where(aroon_down>aroon_up),color='red',alpha=0.08)
ax_a.axhline(70,color='gray',linestyle='--',linewidth=0.8,alpha=0.6); ax_a.axhline(30,color='gray',linestyle='--',linewidth=0.8,alpha=0.6); ax_a.axhline(50,color='gray',linestyle=':',linewidth=0.8,alpha=0.4)
for t in golden_a: ax_a.axvline(t,color='limegreen',linestyle=':',linewidth=0.8,alpha=0.5)
for t in death_a: ax_a.axvline(t,color='red',linestyle=':',linewidth=0.8,alpha=0.5)
ax_a.set_ylim(0,100); ax_a.set_ylabel('Aroon'); ax_a.set_xlabel('Дата'); ax_a.grid(True, alpha=0.3); ax_a.legend(loc='upper left',fontsize=9); plt.tight_layout()
p5=os.path.join(TMP_DIR,"lab2_aroon.png"); fig5.savefig(p5,dpi=150,bbox_inches='tight'); plt.close(fig5)
print("графики сохранены")

# === PDF без титула и оглавления ===
class PDF(FPDF):
    def header(self):
        if self.page_no()>1:
            self.set_font("DejaVu","",8); self.set_text_color(120,120,120)
            self.cell(0,6,"Отчет DAF - лабораторные работы 1 и 2",align="C"); self.ln(8)
    def footer(self):
        self.set_y(-15); self.set_font("DejaVu","",7); self.set_text_color(150,150,150)
        self.cell(0,10,f"Стр. {self.page_no()}/{{nb}}",align="C")

pdf=PDF(orientation='P',unit='mm',format='A4')
pdf.alias_nb_pages()
pdf.add_font("DejaVu","",DEJA_REG, uni=True)
pdf.add_font("DejaVu","B",DEJA_BOLD, uni=True)
pdf.set_auto_page_break(auto=True,margin=15)
_orig_mc=pdf.multi_cell
def _mc(w,h,txt,**kw):
    kw.setdefault("new_x","LMARGIN"); kw.setdefault("new_y","NEXT")
    if pdf.get_x()>pdf.l_margin+0.5: pdf.set_x(pdf.l_margin)
    return _orig_mc(w,h,txt,**kw)
pdf.multi_cell=_mc

# === LAB1 ===
pdf.add_page()
pdf.set_font("DejaVu","B",16); pdf.set_fill_color(230,230,230)
pdf.cell(0,10,"  1. Лабораторная работа 1 - Проверка нормальности",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.ln(2)
# 1.1 Метод (константы + метод вместе, без списка тикеров и без цели)
pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"1.1 Метод",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,"Вход: период 2 года (2y, ~500 торговых дней), уровень значимости ALPHA=0.05, формулы return=Close/Close.shift(1), log_return=ln(Close/Close.shift(1)). Метод: стандартизация (x-mean)/std и тест Колмогорова-Смирнова scipy.stats.kstest(standardized,'norm'). H0: распределение нормальное, отвергается при p < 0.05. Период для T: "+f"{hist_dat.index.min().date()} - {hist_dat.index.max().date()} ({len(hist_dat)} строк).")
pdf.ln(2)
# 1.2 Таблица (единственная)
pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"1.2 Таблица p-value по всем тикерам",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.ln(1)
pdf.set_font("DejaVu","B",7)
col_w=[22,30,30,30,22,22,22]
headers=["Ticker","Close p","Return p","LogRet p","Close N?","Ret N?","Log N?"]
for i,h in enumerate(headers): pdf.cell(col_w[i],6,h,border=1,align="C")
pdf.ln()
pdf.set_font("DejaVu","",6.5)
for r in results:
    def fmt(p): return f"{p:.4f}" if p is not None else "N/A"
    def yn(b): return "да" if b else "нет"
    row=[r['ticker'],fmt(r['close_p']),fmt(r['return_p']),fmt(r['log_return_p']),yn(r['close_normal']),yn(r['return_normal']),yn(r['log_return_normal'])]
    fill=any([r['close_normal'],r['return_normal'],r['log_return_normal']])
    pdf.set_fill_color(230,255,230) if fill else pdf.set_fill_color(255,255,255)
    for i,val in enumerate(row): pdf.cell(col_w[i],4,val,border=1,align="C",fill=fill)
    pdf.ln()
pdf.ln(1)
pdf.set_font("DejaVu","",7); pdf.set_text_color(100,100,100)
pdf.multi_cell(0,3.5,"Примечание: подсветка - хотя бы один показатель нормален. Close почти нигде не нормален (тренд).")
pdf.set_text_color(0,0,0); pdf.ln(2)
# 1.3 Детальный пример (бывший 1.5)
pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"1.3 Детальный пример: тикер T (AT&T)",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,f"Период T: {hist_dat.index.min().date()} - {hist_dat.index.max().date()}, строк после dropna: {len(hist_dat)}.")
pdf.ln(1)
pdf.set_font("DejaVu","B",7)
for w,h in zip([30,35,35,35],["Date","Close","return","log_return"]): pdf.cell(w,5,h,border=1,align="C")
pdf.ln()
pdf.set_font("DejaVu","",6.5)
for idx,row in hist_dat.head(5).iterrows():
    pdf.cell(30,4,str(idx.date()),border=1,align="C")
    pdf.cell(35,4,f"{row['Close']:.2f}",border=1,align="C")
    pdf.cell(35,4,f"{row['return']:.4f}",border=1,align="C")
    pdf.cell(35,4,f"{row['log_return']:.4f}",border=1,align="C")
    pdf.ln()
pdf.ln(1)
desc=hist_dat[["Close","return","log_return"]].describe()
pdf.set_font("DejaVu","B",7); pdf.cell(0,5,"Описательная статистика describe():",new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","B",6.5)
wds=[25,35,35,35]
for i,h in enumerate(["","Close","return","log_return"]): pdf.cell(wds[i],4,h,border=1,align="C")
pdf.ln()
pdf.set_font("DejaVu","",6.5)
for idx in desc.index:
    pdf.cell(wds[0],4,str(idx),border=1,align="C")
    for j,col in enumerate(["Close","return","log_return"]): pdf.cell(wds[j+1],4,f"{desc.loc[idx,col]:.4f}",border=1,align="C")
    pdf.ln()
pdf.ln(1)
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,f"K-S тест (стандартизация):\n Close: statistic={res_1.statistic:.4f}, p={res_1.pvalue:.4f} => {'нормальное' if res_1.pvalue>0.05 else 'не нормальное'}\n Return: statistic={res_2.statistic:.4f}, p={res_2.pvalue:.4f} => {'нормальное' if res_2.pvalue>0.05 else 'не нормальное'}\n Log return: statistic={res_3.statistic:.4f}, p={res_3.pvalue:.4f} => {'нормальное' if res_3.pvalue>0.05 else 'не нормальное'}")
pdf.ln(2)
# графики lab1
pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"1.4 Графики распределений для T (12 панелей)",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.ln(2)
pdf.image(p1,x=10,w=190)
pdf.ln(2)

# === LAB2 на T ===
pdf.add_page()
pdf.set_font("DejaVu","B",16); pdf.set_fill_color(230,230,230)
pdf.cell(0,10,"  2. Лабораторная работа 2 - Технические индикаторы (T, AT&T)",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.ln(2)
pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"2.1 Входные константы",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,f"  Тикер: {TICKER_LAB2} (AT&T)")
pdf.multi_cell(0,4,f"  Период: {lab2_period_str}")
pdf.multi_cell(0,4,f"  SMA: окна {SMA_SHORT} и {SMA_LONG}")
pdf.multi_cell(0,4,f"  Полосы Боллинджера: окно {BB_WINDOW}, число std={BB_STD}")
pdf.multi_cell(0,4,f"  RSI: период {RSI_PERIOD}, уровни 30/50/70, order={RSI_ORDER}")
pdf.multi_cell(0,4,f"  Дивергенции: min_gap=10, max_gap=80, zone=55, max_skip=2, top_n=5")
pdf.multi_cell(0,4,f"  Aroon: период {AROON_PERIOD}")
pdf.ln(2)

pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"2.2 SMA 20 / SMA 50",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,"SMA (Simple Moving Average)  среднее арифметическое цены закрытия за N последних периодов. Сглаживает шум и показывает направление тренда. Пересечение быстрой SMA(20) и медленной SMA(50) дает сигналы Golden Cross (бычий) и Death Cross (медвежий).")
pdf.set_font("DejaVu","",7); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,3.5,"Формула: SMA_N = (Close1 + ... + CloseN) / N. Принцип: скользящее окно по Close. (код: lab2.py:13-14 sma_20/sma_50, lab2.py:16-22 diff/sign/cross)")
pdf.set_text_color(0,0,0); pdf.ln(1)
pdf.image(p2,x=10,w=190)
pdf.ln(2)

pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"2.3 Полосы Боллинджера (20, 2s)",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,"Полосы Боллинджера  индикатор волатильности. Средняя линия  SMA(20), верхняя и нижняя полосы отстоят на k стандартных отклонений (k=2). Ширина полос показывает волатильность, выход цены за полосы  сигнал перекупленности/перепроданности.")
pdf.set_font("DejaVu","",7); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,3.5,"Формулы: mid = SMA20, std = rolling(20).std(), upper = mid + 2*std, lower = mid - 2*std. (код: lab2.py:60-63)")
pdf.set_text_color(0,0,0); pdf.ln(1)
pdf.image(p3,x=10,w=190)
pdf.ln(2)

pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"2.4 RSI (14) и дивергенции",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,"RSI (Relative Strength Index, Wilder, 14)  осциллятор 0..100, измеряет скорость и величину изменения цены. Уровни 70  перекупленность, 30  перепроданность, 50  нейтрально. Дивергенция  расхождение направления экстремумов цены и RSI (цена выше  RSI ниже = медвежья, наоборот  бычья).")
pdf.set_font("DejaVu","",7); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,3.5,"Формулы: delta=Close.diff(), gain=clip>0, loss=-clip<0, avg_gain/avg_loss=ewm(alpha=1/14).mean(), RS=avg_gain/avg_loss, RSI=100-100/(1+RS). Экстремумы через argrelextrema(order=5). (код: lab2.py:90-96 RSI, lab2.py:98-100 экстремумы, lab2.py:102-158 find_divergences/bear_div/bull_div)")
pdf.set_text_color(0,0,0); pdf.ln(1)
pdf.image(p4,x=10,w=190)
pdf.ln(2)

pdf.set_font("DejaVu","B",11); pdf.cell(0,7,"2.5 Aroon (14)",fill=True,new_x="LMARGIN",new_y="NEXT")
pdf.set_font("DejaVu","",8)
pdf.multi_cell(0,4,"Aroon измеряет время, прошедшее с момента достижения максимальной или минимальной цены за определенный период. В отличие от многих других индикаторов, которые измеряют цену, Aroon измеряет именно временной интервал. Aroon Up показывает силу бычьего тренда, Aroon Down  медвежьего, их пересечение  смена тренда.")
pdf.set_font("DejaVu","",7); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,3.5,"Формулы: argmax=High.rolling(15).apply(argmax), argmin=Low.rolling(15).apply(argmin), AroonUp=100*argmax/14, AroonDown=100*argmin/14, osc=Up-Down. Сигналы: Up пересекает Down. (код: lab2.py:220-224 расчет, lab2.py:227-231 cross/golden/death, lab2.py:259-276 отрисовка)")
pdf.set_text_color(0,0,0); pdf.ln(1)
pdf.image(p5,x=10,w=190)
pdf.ln(2)

pdf.output(OUT_PDF)
print(f"PDF сохранен: {OUT_PDF} ({os.path.getsize(OUT_PDF)/1024:.0f} KB)")
