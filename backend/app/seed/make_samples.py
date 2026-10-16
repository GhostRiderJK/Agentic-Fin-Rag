"""Generate sample financial documents (TXT) and a gold Q&A suite.

Run:  python -m app.seed.make_samples --out ./data/samples

Produces:
  data/samples/acme_2025_annual_report.txt
  data/samples/acme_2025_q4_earnings_call.txt
  data/samples/gold_qa.jsonl

The text is deliberately structured so the chunker's section headers fire
and the table extractor picks up the financial summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


ANNUAL_REPORT = """\
ACME Corporation
Annual Report 2025
Form 10-K

1. BUSINESS OVERVIEW

ACME Corporation ("ACME", "the Company") is a global technology company
headquartered in Cupertino, California. The Company designs, manufactures,
and markets consumer electronics, services, and accessories. In fiscal year
2025 ACME shipped a record number of devices across its flagship product
lines.

Total net sales reached $120 billion in fiscal 2025, up from $110 billion
in fiscal 2024. The Company ended the year with $25 billion in cash and
short-term investments.

2. RISK FACTORS

The Company is subject to risks including but not limited to:
- Increased competition in the smartphone and personal computer markets
- Supply chain concentration in a small number of contract manufacturers
- Foreign exchange volatility affecting international revenue
- Regulatory scrutiny of the Company's App Store business model

A meaningful decline in consumer demand could materially reduce revenue.
Tariffs and trade restrictions may increase cost of goods sold.

3. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue grew approximately 9% year over year, driven primarily by higher
average selling prices and growth in the Services segment. Gross margin
declined from 46% to 43%, mainly due to higher component costs and
unfavorable foreign exchange. Operating margin remained stable at 30%.

The Services segment reached $25 billion, growing 18% year over year,
becoming the second-largest revenue contributor. Management expects
Services growth to continue outpacing product revenue.

4. CONSOLIDATED FINANCIAL HIGHLIGHTS

(In millions, except per-share data)

                                  2025      2024
Net sales                       120,000   110,000
Cost of goods sold               68,400    59,400
Gross profit                     51,600    50,600
Operating income                 36,000    35,200
Net income                       28,000    27,500
Diluted earnings per share        5.20      4.95
Operating cash flow              32,000    30,000
Capital expenditure               4,000     3,500
Free cash flow                   28,000    26,500
Total debt                       12,000    13,000
Dividend per share                1.40      1.20

5. OUTLOOK

For fiscal 2026 management expects revenue growth of 5% to 8%, with
Services continuing to grow faster than products. Gross margin is expected
to remain in the 43% to 44% range. Capital expenditure is planned at
approximately $4.5 billion.

6. SUSTAINABILITY

The Company has committed to a carbon-neutral supply chain by 2030.
During 2025, 60% of supplier facilities committed to using renewable
energy for ACME-related production.
"""


EARNINGS_CALL = """\
ACME Corporation
Q4 2025 Earnings Call — Prepared Remarks and Q&A

PREPARED REMARKS

Good afternoon. Welcome to ACME Corporation's fourth quarter 2025 earnings
call. I'm the CFO. Joining me today is our CEO.

For Q4 2025 we reported revenue of $32 billion, up 11% year over year. EPS
for the quarter was $1.45. Operating cash flow reached $9 billion and we
returned $5 billion to shareholders through dividends and buybacks.

Our Services business grew 20% in Q4 to $7 billion. This was the strongest
quarter of Services growth in two years, reflecting continued momentum in
subscriptions and advertising.

QUESTION AND ANSWER

Operator: Your first question comes from the line of Morgan Stanley.

Q: Can you talk about the gross margin trajectory into 2026?
A: We expect gross margin to stabilize in the 43% to 44% range. Component
costs have started to ease, which is positive. However, we are also
investing in new product categories that carry lower initial margins.

Q: What is driving the Services growth?
A: Two main drivers. First, our installed base reached a new all-time high.
Second, attach rates for paid subscriptions continue to climb, particularly
in emerging markets.

Q: Any commentary on AI investments?
A: Yes. We are investing meaningfully in AI capabilities across our
product lineup. We expect these investments to be a meaningful contributor
to Services revenue starting in 2026.
"""


GOLD_QA = [
    {
        "question": "What was ACME's net income in fiscal year 2025?",
        "expected": "ACME's net income in fiscal year 2025 was $28,000 million ($28 billion).",
        "expected_numbers": ["28,000", "28000"],
        "expected_keywords": ["net", "income", "acme"],
    },
    {
        "question": "How much did ACME's gross margin change from 2024 to 2025?",
        "expected": "Gross margin declined from 46% to 43% (about 3 percentage points).",
        "expected_numbers": ["46", "43"],
        "expected_keywords": ["gross", "margin", "declined"],
    },
    {
        "question": "What was the diluted EPS for ACME in 2025?",
        "expected": "Diluted earnings per share was $5.20 in 2025.",
        "expected_numbers": ["5.20"],
        "expected_keywords": ["eps", "earnings", "per", "share"],
    },
    {
        "question": "What are the main risks ACME faces?",
        "expected": "Risks include competition, supply chain concentration, FX volatility, and App Store regulation.",
        "expected_numbers": [],
        "expected_keywords": ["competition", "supply", "regulation", "risk"],
    },
    {
        "question": "What is ACME's Q4 2025 revenue?",
        "expected": "Q4 2025 revenue was $32 billion.",
        "expected_numbers": ["32"],
        "expected_keywords": ["revenue", "q4"],
    },
    {
        "question": "What is driving ACME Services growth?",
        "expected": "Services growth is driven by a record installed base and rising subscription attach rates.",
        "expected_numbers": [],
        "expected_keywords": ["services", "installed", "subscriptions"],
    },
    {
        "question": "What is ACME's 2026 revenue growth guidance?",
        "expected": "ACME expects 5% to 8% revenue growth in fiscal 2026.",
        "expected_numbers": ["5", "8"],
        "expected_keywords": ["revenue", "growth", "guidance"],
    },
]


@app.callback(invoke_without_command=True)
def main(
    out: str = typer.Option("./data/samples", help="Output directory"),
):
    """Generate sample documents + gold QA."""
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "acme_2025_annual_report.txt").write_text(ANNUAL_REPORT,
                                                        encoding="utf-8")
    (outdir / "acme_2025_q4_earnings_call.txt").write_text(EARNINGS_CALL,
                                                           encoding="utf-8")
    with (outdir / "gold_qa.jsonl").open("w", encoding="utf-8") as f:
        for item in GOLD_QA:
            f.write(json.dumps(item) + "\n")
    console.print(f"[green]✓[/green] wrote {len(GOLD_QA)} gold items + 2 sample docs to {outdir}")


if __name__ == "__main__":
    app()
