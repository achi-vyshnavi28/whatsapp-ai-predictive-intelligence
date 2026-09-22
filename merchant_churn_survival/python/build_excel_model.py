"""
Builds excel/churn_retention_model.xlsx from the account_scores.csv produced
by eda_analysis.py (run that first).

Produces a workbook with real Excel formulas (not pasted static values):
  - Account Scores: every account's RFM/health-score/churn-probability row
  - Segment Summary: rollup by RFM segment with live AVERAGEIF/SUMIF formulas
  - Retention Opportunity What-If: revenue-saved calculator with adjustable
    input cells (% of at-risk accounts saved by an intervention)
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = ROOT / "reports" / "account_scores.csv"
OUT_PATH = ROOT / "excel" / "churn_retention_model.xlsx"

HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
INPUT_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")


def style_header(ws, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


def build_scores_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.active
    ws.title = "Account Scores"
    cols = ["customer_id", "segment", "health_score", "churn_probability", "churned",
            "recency_days", "frequency", "monetary", "rfm_score"]
    ws.append(cols)
    style_header(ws, 1, len(cols))
    for _, r in df.iterrows():
        ws.append([r[c] for c in cols])
    widths = {"A": 14, "B": 20, "C": 14, "D": 18, "E": 10, "F": 14, "G": 12, "H": 14, "I": 12}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in range(2, ws.max_row + 1):
        ws.cell(row=row, column=4).number_format = "0.0%"


def build_segment_summary(wb: Workbook, segments: list) -> None:
    ws = wb.create_sheet("Segment Summary")
    headers = ["Segment", "Accounts", "Avg Health Score", "Avg Churn Probability", "Total Monetary Value"]
    ws.append(headers)
    style_header(ws, 1, len(headers))

    src = "'Account Scores'!"
    n = 5942  # generous upper bound on data rows referenced by the formulas below
    for i, seg in enumerate(segments):
        r = i + 2
        ws.cell(row=r, column=1, value=seg)
        ws.cell(row=r, column=2, value=f'=COUNTIF({src}B2:B{n},A{r})')
        ws.cell(row=r, column=3, value=f'=ROUND(AVERAGEIF({src}B2:B{n},A{r},{src}C2:C{n}),1)')
        ws.cell(row=r, column=4, value=f'=ROUND(AVERAGEIF({src}B2:B{n},A{r},{src}D2:D{n}),3)')
        ws.cell(row=r, column=5, value=f'=ROUND(SUMIF({src}B2:B{n},A{r},{src}H2:H{n}),2)')
        ws.cell(row=r, column=4).number_format = "0.0%"
        ws.cell(row=r, column=5).number_format = "#,##0.00"

    ws.column_dimensions["A"].width = 22
    for col in "BCDE":
        ws.column_dimensions[col].width = 18

    chart = BarChart()
    chart.title = "Total Monetary Value at Risk by Segment"
    chart.y_axis.title = "Value"
    data = Reference(ws, min_col=5, min_row=1, max_row=len(segments) + 1)
    cats = Reference(ws, min_col=1, min_row=2, max_row=len(segments) + 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.width = 20
    chart.height = 10
    ws.add_chart(chart, "G2")


def build_whatif_sheet(wb: Workbook, at_risk_count: int, at_risk_value: float) -> None:
    ws = wb.create_sheet("Retention Opportunity")
    ws["A1"] = "Churn-Prevention Revenue Opportunity Model"
    ws["A1"].font = Font(bold=True, size=13)

    rows = [
        ("At-risk accounts (health score < 30)", at_risk_count),
        ("Total monetary value of at-risk accounts (GBP)", round(at_risk_value, 2)),
        ("Avg value per at-risk account (GBP)", None),
        ("", None),
        ("--- Adjustable Inputs ---", None),
        ("Target save rate from intervention", 0.25),
        ("Avg cost per outreach/intervention (GBP)", 15.0),
    ]
    r = 3
    for label, val in rows:
        ws.cell(row=r, column=1, value=label)
        if val is not None:
            cell = ws.cell(row=r, column=2, value=val)
            if "rate" in label.lower():
                cell.number_format = "0.0%"
                cell.fill = INPUT_FILL
            elif "cost" in label.lower():
                cell.fill = INPUT_FILL
        r += 1

    ws["B5"] = "=B4/B3"
    ws["B5"].number_format = "#,##0.00"

    r += 1
    ws.cell(row=r, column=1, value="--- Model Output ---").font = Font(bold=True)
    out_start = r + 1
    outputs = [
        ("Accounts saved by the intervention", "=ROUND(B3*B8,0)"),
        ("Revenue retained (GBP)", f"=B{out_start}*B5"),
        ("Total intervention cost (GBP)", "=B3*B9"),
        ("Net revenue impact (GBP)", f"=B{out_start+1}-B{out_start+2}"),
    ]
    r = out_start
    for label, formula in outputs:
        ws.cell(row=r, column=1, value=label)
        cell = ws.cell(row=r, column=2, value=formula)
        cell.number_format = "#,##0" if "Accounts" in label else "#,##0.00"
        r += 1

    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 16

    note_row = r + 2
    ws.cell(row=note_row, column=1,
            value=("Model logic: accounts saved = at-risk accounts x target save rate; revenue retained = "
                   "accounts saved x avg value per at-risk account, net of outreach cost. Change the yellow "
                   "input cells to stress-test the ROI of a retention campaign before running it."))
    ws.cell(row=note_row, column=1).alignment = Alignment(wrap_text=True)
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row + 2, end_column=5)


def main() -> None:
    df = pd.read_csv(SCORES_CSV)
    segments = sorted(df["segment"].unique())
    at_risk = df[df["health_score"] < 30]

    wb = Workbook()
    build_scores_sheet(wb, df)
    build_segment_summary(wb, segments)
    build_whatif_sheet(wb, len(at_risk), at_risk["monetary"].sum())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_PATH)
    print(f"Workbook written to {OUT_PATH}")


if __name__ == "__main__":
    main()
