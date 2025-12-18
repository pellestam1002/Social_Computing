import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
# 1) Zet hier het pad naar jouw grote feather file (op je laptop)
NOS_PATH = r"/Users/p.stam/Desktop/NOS_NL_articles_2015_jul_2025.feather"  # <-- aanpassen

# 2) Dit staat wél in je projectmap

TRENDS_PATH = "multiTimeline.csv"
EVENTS_PATH = "Protest_regel_data.txt"

IMAGES_DIR = "images"

KEYWORDS = [
    "stikstof",
    "stikstofcrisis",
    "stikstofbeleid",
    "stikstofprobleem",
    "boerenprotest",
]

TEXT_COLS = ["title", "keywords", "section", "description", "content"]

os.makedirs(IMAGES_DIR, exist_ok=True)

# =========================================================
# 1) LOAD + PREPROCESS NOS
# =========================================================
df = pd.read_feather(NOS_PATH)

# alleen NOS channel (geen Nieuwsuur)
if "channel" in df.columns:
    df = df[df["channel"].astype(str).str.lower() == "nos"]

df["published_time"] = pd.to_datetime(df["published_time"], errors="coerce")
df = df.dropna(subset=["published_time"])

# filter: match als keyword voorkomt in 1 van de text columns
pattern = "|".join(KEYWORDS)
mask = False
for col in TEXT_COLS:
    if col in df.columns:
        mask |= df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)

df_kw = df[mask].copy()

# monthly counts (elk artikel telt max 1 keer)
df_kw["year_month"] = df_kw["published_time"].dt.to_period("M")
nos_monthly = (
    df_kw.groupby("year_month")
    .size()
    .reset_index(name="NOS_article_count")
)
nos_monthly["date"] = nos_monthly["year_month"].dt.to_timestamp()
nos_monthly = nos_monthly.sort_values("date")

# =========================================================
# 2) LOAD + PREPROCESS GOOGLE TRENDS
# =========================================================
df_trends_raw = pd.read_csv(TRENDS_PATH, skiprows=1)

date_col = df_trends_raw.columns[0]
term_cols = list(df_trends_raw.columns[1:])

df_trends = df_trends_raw[[date_col] + term_cols].copy()
df_trends = df_trends.rename(columns={date_col: "date"})
df_trends["date"] = pd.to_datetime(df_trends["date"], errors="coerce")
df_trends = df_trends.dropna(subset=["date"])

for c in term_cols:
    df_trends[c] = df_trends[c].astype(str).str.replace("<1", "0", regex=False)
    df_trends[c] = pd.to_numeric(df_trends[c], errors="coerce")

df_trends = df_trends.sort_values("date")

# =========================================================
# 3) FIGURE 1 — NOS COVERAGE
# =========================================================
# --- load events (optional: if file exists) ---
events_in_range = None
if os.path.exists(EVENTS_PATH):
    events = pd.read_csv(EVENTS_PATH, sep=";", encoding="utf-8-sig")
    if "Date" in events.columns:
        events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
        events = events.dropna(subset=["Date"])

        # keep only events within the NOS timeline range
        if len(nos_monthly) > 0:
            start = nos_monthly["date"].min()
            end = nos_monthly["date"].max()
            events_in_range = events[(events["Date"] >= start) & (events["Date"] <= end)].sort_values("Date")
else:
    print(f"Warning: events file not found at '{EVENTS_PATH}' (skipping event markers).")

plt.figure(figsize=(12, 5))

# NOS line
plt.plot(nos_monthly["date"], nos_monthly["NOS_article_count"], linewidth=2)

# Event markers (vertical dashed lines)
if events_in_range is not None and len(events_in_range) > 0:
    for d in events_in_range["Date"]:
        plt.axvline(d, linestyle="--", alpha=0.25)

plt.title("Monthly NOS Coverage of Nitrogen-Related Issues with Major Events")
plt.xlabel("Time (monthly)")
plt.ylabel("Number of NOS articles")
plt.grid(True)
plt.tight_layout()

out1 = os.path.join(IMAGES_DIR, "figure_1_nos_coverage.png")
plt.savefig(out1, dpi=300)
plt.close()

# =========================================================
# 4) FIGURE 2 — GOOGLE TRENDS
# =========================================================
plt.figure(figsize=(12, 5))
for c in term_cols:
    if df_trends[c].notna().any():
        clean_label = c.split(":")[0]
        clean_label = clean_label.replace("Netherlands", "").replace("Nederland", "").strip()
        plt.plot(
            df_trends["date"],
            df_trends[c],
            linewidth=1.8,
            label=clean_label
        )

plt.title("Monthly Google Trends (selected search terms)")
plt.xlabel("Time (monthly)")
plt.ylabel("Google Trends index (normalized)")
plt.grid(True)
plt.legend(fontsize=9, loc="upper left")
plt.tight_layout()

out2 = os.path.join(IMAGES_DIR, "figure_2_google_trends.png")
plt.savefig(out2, dpi=300)
plt.close()

print("Saved:")
print(" -", out1)
print(" -", out2)
print("Matched NOS articles:", len(df_kw))