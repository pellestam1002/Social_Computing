import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Matplotlib styling (bigger text for paper-ready figures)
# =========================================================
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# Optional: only needed for correlation tests
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

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
# Helpers
# =========================================================
def normalize_colname(s: str) -> str:
    """Normalize Google Trends column names to something stable for matching."""
    if s is None:
        return ""
    s = str(s)
    # keep only the part before ':' (e.g., "stikstof: (Netherlands)")
    s = s.split(":")[0]
    s = s.lower().strip()
    # remove non-letters/numbers/spaces
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_trends_col(df_tr: pd.DataFrame, target: str) -> str | None:
    """Find the column in df_trends corresponding to a target term like 'stikstof'."""
    t = normalize_colname(target)
    best = None
    for c in df_tr.columns:
        if c == "date":
            continue
        if normalize_colname(c) == t:
            best = c
            break
    return best

# =========================================================
# 2b) Events -> monthly protest counts (for correlation test)
# =========================================================
protests_monthly = None
if os.path.exists(EVENTS_PATH):
    events_all = pd.read_csv(EVENTS_PATH, sep=";", encoding="utf-8-sig")
    if "Date" in events_all.columns:
        events_all["Date"] = pd.to_datetime(events_all["Date"], errors="coerce")
        events_all = events_all.dropna(subset=["Date"])
        # If the file has a 'Type' column, count only protests; otherwise count all events
        if "Type" in events_all.columns:
            ev = events_all[events_all["Type"].astype(str).str.lower().eq("protest")].copy()
        else:
            ev = events_all.copy()

        ev["year_month"] = ev["Date"].dt.to_period("M")
        protests_monthly = (
            ev.groupby("year_month")
            .size()
            .reset_index(name="Monthly_protests")
        )
        protests_monthly["date"] = protests_monthly["year_month"].dt.to_timestamp()
        protests_monthly = protests_monthly[["date", "Monthly_protests"]].sort_values("date")
else:
    print(f"Warning: events file not found at '{EVENTS_PATH}' (cannot compute Monthly_protests).")

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

plt.xlabel("Time (monthly)")
plt.ylabel("Number of NOS articles")
plt.xticks(rotation=0)
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

plt.xlabel("Time (monthly)")
plt.ylabel("Google Trends index (normalized)")
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()

out2 = os.path.join(IMAGES_DIR, "figure_2_google_trends.png")
plt.savefig(out2, dpi=300)
plt.close()

# =========================================================
# 5) Exploratory correlation tests (Spearman)
#    1) NOS ↔ stikstof
#    2) NOS ↔ stikstofcrisis
#    3) NOS ↔ stikstofbeleid
#    4) NOS ↔ stikstofprobleem
#    5) NOS ↔ boerenprotest
#    6) NOS ↔ Monthly_protests
# =========================================================
if spearmanr is None:
    print("\n[Correlation] scipy not available (install with: pip install scipy). Skipping correlation tests.")
else:
    # Prepare monthly master table
    master = nos_monthly[["date", "NOS_article_count"]].copy()

    # Pick the correct Trends columns (robust to ': Netherlands' etc.)
    stikstof_col = find_trends_col(df_trends, "stikstof")
    stikstofcrisis_col = find_trends_col(df_trends, "stikstofcrisis")
    stikstofbeleid_col = find_trends_col(df_trends, "stikstofbeleid")
    stikstofprobleem_col = find_trends_col(df_trends, "stikstofprobleem")
    boerenprotest_col = find_trends_col(df_trends, "boerenprotest")

    if stikstof_col is None:
        print("\n[Correlation] Could not find a Google Trends column for 'stikstof'.")
    else:
        master = master.merge(df_trends[["date", stikstof_col]].rename(columns={stikstof_col: "stikstof"}), on="date", how="inner")

    if stikstofcrisis_col is None:
        print("[Correlation] Could not find a Google Trends column for 'stikstofcrisis'.")
    else:
        master = master.merge(df_trends[["date", stikstofcrisis_col]].rename(columns={stikstofcrisis_col: "stikstofcrisis"}), on="date", how="inner")

    if stikstofbeleid_col is None:
        print("[Correlation] Could not find a Google Trends column for 'stikstofbeleid'.")
    else:
        master = master.merge(
            df_trends[["date", stikstofbeleid_col]].rename(columns={stikstofbeleid_col: "stikstofbeleid"}),
            on="date",
            how="inner"
        )

    if stikstofprobleem_col is None:
        print("[Correlation] Could not find a Google Trends column for 'stikstofprobleem'.")
    else:
        master = master.merge(
            df_trends[["date", stikstofprobleem_col]].rename(columns={stikstofprobleem_col: "stikstofprobleem"}),
            on="date",
            how="inner"
        )

    if boerenprotest_col is None:
        print("[Correlation] Could not find a Google Trends column for 'boerenprotest'.")
    else:
        master = master.merge(
            df_trends[["date", boerenprotest_col]].rename(columns={boerenprotest_col: "boerenprotest"}),
            on="date",
            how="inner"
        )

    if protests_monthly is not None and len(protests_monthly) > 0:
        master = master.merge(protests_monthly, on="date", how="left")
    else:
        master["Monthly_protests"] = pd.NA

    # Make sure numeric
    for c in [
        "NOS_article_count",
        "stikstof",
        "stikstofcrisis",
        "stikstofbeleid",
        "stikstofprobleem",
        "boerenprotest",
        "Monthly_protests",
    ]:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")

    print("\n=== Exploratory Spearman correlations (monthly) ===")
    print("Note: exploratory only; does not imply causality.")
    print("N months (after merges, may differ per test):", len(master))

    def run_test(x_name: str, y_name: str):
        if x_name not in master.columns or y_name not in master.columns:
            print(f" - Skipping {x_name} ↔ {y_name} (missing column)")
            return
        df_xy = master[[x_name, y_name]].dropna()
        if len(df_xy) < 6:
            print(f" - Skipping {x_name} ↔ {y_name} (too few data points: {len(df_xy)})")
            return
        rho, p = spearmanr(df_xy[x_name], df_xy[y_name])
        print(f" - {x_name} ↔ {y_name}: rho={rho:.3f}, p={p:.4f}, N={len(df_xy)}")

    run_test("NOS_article_count", "stikstof")
    run_test("NOS_article_count", "stikstofcrisis")
    run_test("NOS_article_count", "stikstofbeleid")
    run_test("NOS_article_count", "stikstofprobleem")
    run_test("NOS_article_count", "boerenprotest")
    run_test("NOS_article_count", "Monthly_protests")

print("Saved:")
print(" -", out1)
print(" -", out2)
print("Matched NOS articles:", len(df_kw))
print("Done.")