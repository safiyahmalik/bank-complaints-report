import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DATABASE_NAME = "practice_db"
COLLECTION_NAME = "bank_complaints"
MODEL = "openai/gpt-oss-120b"

# extracting from mongodb
print("--- Connecting to MongoDB... ---")
mongo = MongoClient(MONGO_URI)
documents = list(mongo[DATABASE_NAME][COLLECTION_NAME].find({}, {"_id": 0}))
print(f"Pulled {len(documents)} report documents")

# flattening the nested structure into dataframe
rows = []
for doc in documents:
    meta = doc.get("meta_data") or {}
    for item in doc.get("data") or []:
        rows.append({
            "report_id": meta.get("report_id"),
            "reporting_period": meta.get("reporting_period"),
            "institution_code": meta.get("institution_code"),
            "worksheet": meta.get("worksheet"),
            "status": meta.get("status"),
            "index_no": str(item.get("index_no", "")).strip(),
            "indicator": str(item.get("indicator_name", "")).strip(),
            "value_raw": item.get("value"),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No indicator rows found. Check the collection name and structure.")

# cleaning data 
df["value"] = pd.to_numeric(
    df["value_raw"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce",
)
df["period"] = df["reporting_period"].astype(str).apply(
    lambda p: f"{p[:4]}-{p[4:]}" if len(p) == 6 else p
)

key = ["institution_code", "period", "worksheet", "index_no"]
total_rows = len(df)
non_numeric = int(df["value"].isna().sum())
duplicates = int(df.duplicated(subset=key, keep="first").sum())

valid = df.dropna(subset=["value"]).drop_duplicates(subset=key, keep="first")
print(f"Total indicator rows: {total_rows} | valid: {len(valid)} | "
      f"non-numeric: {non_numeric} | duplicates: {duplicates}")

# pandas formatting functions
def fmt(x):
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"

def md_table(frame):
    frame = frame.copy()
    for c in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = frame[c].map(fmt)
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in frame.itertuples(index=False)]
    return "\n".join([head, sep] + body)

periods = sorted(df["period"].dropna().unique())
n_institutions = df["institution_code"].nunique()
worksheets = ", ".join(map(str, df["worksheet"].dropna().unique()))

# top indicators by total value
ind = (valid.groupby(["index_no", "indicator"])["value"]
       .agg(total="sum", reports="count")
       .reset_index()
       .sort_values("total", ascending=False))
top_ind = ind.head(12)[["indicator", "total", "reports"]]
top_ind.columns = ["Indicator", "Total", "Reports"]

# indicators where every reported value is zero
zero_ind = ind[ind["total"] == 0]["indicator"].tolist()

# total complaints per institution and period
tot = valid[valid["indicator"].str.contains("total number of complaints",
                                            case=False, na=False)]
tot_tbl = (tot.groupby(["institution_code", "period", "worksheet"])["value"]
           .sum().reset_index())
tot_tbl.columns = ["Institution", "Period", "Worksheet", "Total complaints"]

# trend table across periods (only if there is more than one period)
trend_md = ""
if len(periods) > 1:
    top_names = ind.head(8)["indicator"]
    trend = (valid[valid["indicator"].isin(top_names)]
             .pivot_table(index="indicator", columns="period",
                          values="value", aggfunc="sum", fill_value=0)
             .reset_index())
    trend_md = md_table(trend)

# information summary for the LLM
facts = (
    f"Reports: {len(documents)} | Institutions: {n_institutions} | "
    f"Periods: {', '.join(periods)} | Worksheets: {worksheets}\n"
    f"Valid indicator values: {len(valid)} of {total_rows} "
    f"(non-numeric: {non_numeric}, duplicates removed: {duplicates})\n\n"
    f"Top indicators by total value:\n{md_table(top_ind)}\n\n"
    f"Indicators reporting zero in every report: "
    f"{', '.join(zero_ind[:10]) if zero_ind else 'none'}\n\n"
    f"Total complaints by institution/period:\n"
    f"{md_table(tot_tbl.head(15)) if not tot_tbl.empty else 'not available'}\n"
)
if trend_md:
    facts += f"\nTrend by period:\n{trend_md}\n"

# langChain prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior regulatory analyst writing for a bank's management "
     "and compliance committee. Be precise and formal."),
    ("human",
     "Data:\n{facts}\n\n"
     "Write the narrative part of a complaints report in Markdown with "
     "exactly these three sections:\n"
     "## Executive Summary (one paragraph)\n"
     "## Key Findings (3 to 5 bullet points, each citing specific numbers)\n"
     "## Observations and Recommended Actions (3 bullet points)\n\n"
     "Rules: use only the numbers above, do not invent causes, do not call "
     "anything even or balanced unless the numbers are equal, and keep it "
     "under 350 words."),
])

llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=2000,
)

chain = prompt | llm | StrOutputParser()

print("--- Running LangChain chain with Groq... ---")
narrative = chain.invoke({"facts": facts})

# report assembling 
today = datetime.now().strftime("%d %B %Y")
report = f"""# Bank Complaints Analysis Report

**Generated:** {today}
**Reporting periods covered:** {', '.join(periods)}
**Source:** MongoDB `{DATABASE_NAME}.{COLLECTION_NAME}`

---

{narrative}

---

## Appendix A: Top Indicators by Total Value

{md_table(top_ind)}

## Appendix B: Total Complaints by Institution and Period

{md_table(tot_tbl) if not tot_tbl.empty else "_No 'Total Number of complaints' indicator found._"}

{"## Appendix C: Trend by Period" + chr(10) + chr(10) + trend_md if trend_md else ""}

## Data Quality Notes

- Reports analysed: {len(documents)} from {n_institutions} institution(s)
- Indicator values received: {total_rows}
- Valid numeric values used: {len(valid)}
- Non-numeric or missing values excluded: {non_numeric}
- Duplicate entries removed: {duplicates}
"""

# saving as text file
plain = report
plain = re.sub(r"^#{1,6}\s*", "", plain, flags=re.MULTILINE)
plain = plain.replace("**", "")
plain = re.sub(r"^\|?\s*-{3}.*$", "", plain, flags=re.MULTILINE)
plain = plain.replace("|", "  ")
plain = re.sub(r"\n{3,}", "\n\n", plain)

with open("bank_complaints_report1.txt", "w", encoding="utf-8") as f:
    f.write(plain)

print("\n=== Saved to bank_complaints_report1.txt ===")
print(plain)