"""Сборка PDF-отчета: нормальность двух акций и задания 1-3 (портфель Марковица)."""

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import lab3
import portfolio

RESULTS = Path("results")
PLOTS_DAILY = RESULTS / "plots_simple_daily"
PLOTS_PORTFOLIO = RESULTS / "plots_portfolio"
OUT = RESULTS / "report_portfolio.pdf"
NORMAL_PAIR = ["SNGS", "PLZL"]

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["Normal"], fontName="Arial", fontSize=10, leading=14, spaceAfter=6)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Arial-Bold", fontSize=16, leading=20, spaceAfter=12, alignment=1)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Arial-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Arial-Bold", fontSize=11, leading=15, spaceBefore=6, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))


WEIGHT_WIDTHS = [3.4 * cm] + [1.7 * cm] * 8


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def table(rows, col_widths=None):
    data = [[Paragraph(str(cell), ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10, spaceAfter=0)) for cell in row] for row in rows]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Arial"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def image(path, width_cm, ratio):
    return Image(str(path), width=width_cm * cm, height=width_cm * ratio * cm)


def weights_rows(result, names):
    tickers = result["tickers"]
    mu, sigma, rf = result["mu"], result["sigma"], result["rf"]
    header = ["Портфель", *tickers, "E(r), год", "σ, год", "Шарп"]
    rows = [header]
    for name in names:
        w = result["portfolios"][name]
        rows.append(
            [
                name,
                *[pct(x) for x in w],
                pct(mu @ w),
                pct(portfolio.port_vol(w, sigma)),
                f"{portfolio.port_sharpe(w, mu, sigma, rf):.3f}",
            ]
        )
    return rows


def normality_block(returns):
    story = [Paragraph("Нормальность двух акций", H1)]
    story.append(
        Paragraph(
            "Для проверки взяты дневные доходности (простые, по ценам закрытия, 2024-01-01 .. 2025-10-01). "
            "Из пяти бумаг портфеля две наиболее близкие к нормальности по критерию min p-value: "
            f"<b>{NORMAL_PAIR[0]}</b> и <b>{NORMAL_PAIR[1]}</b>. "
            "Использованы три теста: Шапиро-Уилка, Д'Агостино (асимметрия и эксцесс) и Жарке-Бера.",
            BODY,
        )
    )
    header = ["Бумага", "n", "Асимметрия", "Эксцесс", "p Шапиро", "p Д'Агостино", "p Жарке-Бера"]
    rows = [header]
    for ticker in NORMAL_PAIR:
        s = returns[ticker].dropna()
        stat = lab3.test_normality(s)
        rows.append(
            [
                ticker,
                stat["n"],
                f"{stat['skew']:.3f}",
                f"{stat['excess_kurtosis']:.3f}",
                f"{stat['p_shapiro']:.2g}",
                f"{stat['p_dagostino']:.2g}",
                f"{stat['p_jarque']:.2g}",
            ]
        )
    story.append(table(rows))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Вывод: гипотеза нормальности отвергается для обеих бумаг при alpha=0.05 по всем трем тестам "
            "(все p-value меньше 0.05). У SNGS асимметрия 0.26 и эксцесс 0.63, у PLZL асимметрия 0.10 и эксцесс 0.84. "
            "Это наименьшие отклонения среди пяти бумаг портфеля, поэтому SNGS и PLZL ближе всего к нормальности, "
            "но полностью нормальными их считать нельзя: тесты чувствительны к редким крупным движениям, которые "
            "в акциях встречаются регулярно. Для задач Марковица это не критично, так как оптимизация использует только "
            "средние доходности и ковариации, а нормальность важна для интерпретации вероятностей потерь.",
            BODY,
        )
    )
    for ticker in NORMAL_PAIR:
        story.append(Paragraph(f"График для {ticker}: гистограмма с наложенным нормальным законом и QQ-plot", H2))
        story.append(image(PLOTS_DAILY / f"{ticker}.png", 16, 4 / 11))
        story.append(Spacer(1, 4))
    return story


def main():
    tickers = portfolio.DEFAULT_TICKERS
    returns = portfolio.load_returns(RESULTS / "returns_simple_daily.csv", tickers)
    result = portfolio.run_all(returns)
    mu, sigma, rf = result["mu"], result["sigma"], result["rf"]
    frontier = result["frontier"]
    tangent_w = result["portfolios"]["3. касательный портфель"]
    min_w = result["portfolios"]["1a. минимальный риск"]
    target_w = result["portfolios"]["1б. макс. доходность при заданном риске"]
    util_w = result["portfolios"]["1в. макс. функция полезности"]

    story = []
    story.append(Paragraph("Оптимальный портфель акций по модели Марковица", TITLE))
    story.append(
        Paragraph(
            f"Бумаги: {', '.join(tickers)}. Период: 2024-01-01 .. 2025-10-01, дневные доходности "
            f"({len(returns)} общих торговых дней). Безрисковая ставка rf = {pct(rf, 2)} годовых "
            "(ключевая ставка Банка России на 21.10.2026). Годовые показатели: доходность = среднее × 252, "
            "ковариация = ковариация дневных × 252. Веса неотрицательны (короткие позиции запрещены), сумма весов равна 1.",
            BODY,
        )
    )

    story.append(Paragraph("Исходные характеристики бумаг", H2))
    rows = [["Бумага", "Средняя доходность, год", "Риск σ, год", "Шарп (отдельно)"]]
    for i, ticker in enumerate(tickers):
        rows.append([ticker, pct(mu[i]), pct(np.sqrt(sigma[i, i])), f"{(mu[i] - rf) / np.sqrt(sigma[i, i]):.3f}"])
    story.append(table(rows))
    story.append(Spacer(1, 6))
    story.extend(normality_block(returns))

    story.append(Paragraph("Задание 1. Эффективные портфели", H1))
    story.append(Paragraph("<b>1а. Портфель с минимальным риском.</b> Решается задача min w'Σw при Σw = 1, w ≥ 0.", BODY))
    story.append(Paragraph("<b>1б. Максимальная доходность при заданном риске.</b> Целевой риск выбран равным риску портфеля с равными весами, "
                           f"σ = {pct(portfolio.port_vol(np.full(len(tickers), 1 / len(tickers)), sigma))}. Решается max μ'w при w'Σw ≤ σ²<sub>target</sub>.", BODY))
    story.append(Paragraph("<b>1в. Максимум функции полезности Марковица.</b> U(w) = μ'w − (λ/2)·w'Σw, λ = 3 (умеренное неприятие риска).", BODY))
    story.append(table(weights_rows(result, [
        "1a. минимальный риск",
        "1б. макс. доходность при заданном риске",
        "1в. макс. функция полезности",
        "для сравнения: равные веса",
    ]), col_widths=WEIGHT_WIDTHS))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Портфель 1а имеет наименьший риск (σ = {pct(portfolio.port_vol(min_w, sigma))}), но и наименьшую доходность "
        f"({pct(mu @ min_w)}). Портфель 1б при том же риске, что у равных весов, дает доходность {pct(mu @ target_w)} "
        f"за счет концентрации в PLZL ({pct(target_w[tickers.index('PLZL')], 0)}). Портфель 1в "
        f"({pct(mu @ util_w)}, σ = {pct(portfolio.port_vol(util_w, sigma))}) при λ = 3 полностью состоит из PLZL: "
        "высокая историческая доходность PLZL перевешивает штраф за риск. При λ от 5 и выше в портфель добавляется AFLT.",
        BODY,
    ))

    story.append(Paragraph("Таблица полезности при разных λ", H2))
    lam_rows = [["λ", "U", "E(r), год", "σ, год", *tickers]]
    for _, row in result["lambda_table"].iterrows():
        lam_rows.append([
            f"{row['lambda']:.0f}",
            f"{row['U']:.4f}",
            pct(row["доходность, год"]),
            pct(row["риск (sigma), год"]),
            *[pct(row[t]) for t in tickers],
        ])
    story.append(table(lam_rows))
    story.append(Spacer(1, 6))
    story.append(image(PLOTS_PORTFOLIO / "frontier.png", 15, 5.5 / 8))
    story.append(Paragraph("Рисунок 1. Эффективная граница (синяя линия), отдельные бумаги и отмеченные портфели.", SMALL))

    story.append(Paragraph("Задание 2. Коэффициент Шарпа", H1))
    best = frontier["sharpe"].idxmax()
    story.append(Paragraph(
        f"Для каждого портфеля эффективной границы рассчитан коэффициент Шарпа S = (E(r) − rf)/σ. "
        f"Максимум на границе, S = {frontier.loc[best, 'sharpe']:.3f}, достигается при σ = {pct(frontier.loc[best, 'vol'])}. "
        "Большинство отдельных бумаг имеют отрицательный Шарп, а PLZL почти совпадает с касательным портфелем по этому критерию, "
        "поэтому прирост от диверсификации в данной выборке невелик.",
        BODY,
    ))
    story.append(image(PLOTS_PORTFOLIO / "sharpe.png", 16, 4.5 / 11))
    story.append(Paragraph("Рисунок 2. Слева: Шарп портфелей эффективной границы; справа: Шарп отдельных бумаг и касательного портфеля.", SMALL))

    story.append(Paragraph("Задание 3. Касательный портфель", H1))
    story.append(Paragraph(
        "Касательный портфель максимизирует коэффициент Шарпа: max (μ − rf·1)'w / sqrt(w'Σw) при Σw = 1, w ≥ 0. "
        "Это точка касания эффективной границы прямой, проходящей через (0, rf) (CML, рисунок 1).",
        BODY,
    ))
    story.append(table(weights_rows(result, ["3. касательный портфель"]), col_widths=WEIGHT_WIDTHS))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Выводы", H1))
    tangent_sharpe = portfolio.port_sharpe(tangent_w, mu, sigma, rf)
    conclusions = [
        f"Нормальность не подтверждена ни для одной бумаги. SNGS и PLZL ближе всего к нормальности, но тесты отвергают гипотезу; "
        "для портфеля это ограничение учтено через использование только средних и ковариаций.",
        f"Минимальный риск (σ = {pct(portfolio.port_vol(min_w, sigma))}) дает доходность {pct(mu @ min_w)}; "
        f"портфель с концентрацией в PLZL при σ = {pct(portfolio.port_vol(util_w, sigma))} дает {pct(mu @ util_w)}. "
        "Минимум риска и максимум доходности лежат в разных точках эффективной границы, выбор между ними определяется "
        "отношением инвестора к риску (задача 1в).",
        f"Касательный портфель: PLZL {pct(tangent_w[tickers.index('PLZL')], 0)}, AFLT {pct(tangent_w[tickers.index('AFLT')], 0)}, "
        f"Шарп {tangent_sharpe:.3f} при rf = {pct(rf, 0)}. Это наибольший Шарп на границе, но его превышение над одной PLZL "
        "незначительно, поэтому весь результат в основном определяется историческими данными PLZL.",
        "Результаты чувствительны к оценке средних: высокая доходность PLZL за 2024–2025 годы сильно влияет на решение. "
        "Вывод о том, что PLZL лучшая бумага, на одной двухлетней выборке ненадежен. Для выводов о будущей доходности нужны "
        "более длинные истории и более устойчивые оценки средних (например, shrinkage).",
        "Ограничение long-only (короткие позиции запрещены) - допущение модели. Без него оптимум мог бы отличаться.",
    ]
    for text in conclusions:
        story.append(Paragraph(f"• {text}", BODY))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Оптимальный портфель акций по модели Марковица")
    doc.build(story)
    print(f"Отчет сохранен: {OUT}")


if __name__ == "__main__":
    main()
