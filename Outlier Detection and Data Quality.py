"""
Outlier Detection Script for the enriched CRMLS Sold and Listing datasets.
 
Uses the IQR (Interquartile Range) method to FLAG statistical outliers.
Records are never deleted from the primary dataset, only marked. 
A second, filtered dataset (outliers removed) is saved separately for analysis.
 
Columns checked:
  Sold:    ClosePrice, LivingArea, DaysOnMarket
  Listing: ListPrice, LivingArea, DaysOnMarket  (ListPrice instead of
           ClosePrice, since most Listing rows are still-Active and don't
           have a reliable ClosePrice)
 
IQR METHOD (applied independently to each target column):
  Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
  lower bound = Q1 - 1.5 * IQR
  upper bound = Q3 + 1.5 * IQR
  A value outside [lower, upper] is flagged True. Missing values are NEVER
  flagged as outliers -- a missing value isn't "extreme", it's just absent,
  and the comparison against the bounds naturally evaluates to False for NaN.
 
NOTE: the surrounding context mentioned price-per-square-foot and the
close-to-list ratio as places extreme values also show up, but the explicit
column list given only covers ClosePrice/LivingArea/DaysOnMarket (Sold) and
ListPrice/LivingArea/DaysOnMarket (Listing) -- PricePerSqFt and
CloseToOriginalListRatio/PriceRatio are NOT flagged here. Flag those too if
that was meant to be included.
 
OUTPUTS (four files):
  Sold_OutlierFlagged_*.csv : every Sold row kept, flag columns added
  Sold_Filtered_NoOutliers_*.csv : Sold rows with Outlier_Any_Flag removed
  Listing_OutlierFlagged_*.csv : every Listing row kept, flag columns added
  Listing_Filtered_NoOutliers_*.csv : Listing rows with Outlier_Any_Flag removed
"""

# ===================== Install Packages =====================
import os
import pandas as pd

# Config
DATA_DIR = r"D:\Meng\document\AU\Career\2026 Intern\IDX\2. Data Analyst summer 2026\csv"
START_MONTH = 202401
END_MONTH = 202606

# Inputs: the feature-engineered datasets from engineer_market_metrics.py
SOLD_INPUT_PATH = os.path.join(DATA_DIR, f"Sold_Enriched_{START_MONTH}_{END_MONTH}.csv")
LISTING_INPUT_PATH = os.path.join(DATA_DIR, f"Listing_Enriched_{START_MONTH}_{END_MONTH}.csv")

# Outputs: one flagged (all rows) + one filtered (outliers removed) per dataset
SOLD_FLAGGED_OUTPUT = os.path.join(DATA_DIR, f"Sold_OutlierFlagged_{START_MONTH}_{END_MONTH}.csv")
SOLD_FILTERED_OUTPUT = os.path.join(DATA_DIR, f"Sold_Filtered_NoOutliers_{START_MONTH}_{END_MONTH}.csv")
LISTING_FLAGGED_OUTPUT = os.path.join(DATA_DIR, f"Listing_OutlierFlagged_{START_MONTH}_{END_MONTH}.csv")
LISTING_FILTERED_OUTPUT = os.path.join(DATA_DIR, f"Listing_Filtered_NoOutliers_{START_MONTH}_{END_MONTH}.csv")

# Which columns get an IQR outlier check, per dataset (see docstring for why
# these differ between Sold and Listing).
SOLD_OUTLIER_COLUMNS = ["ClosePrice", "LivingArea", "DaysOnMarket"]
LISTING_OUTLIER_COLUMNS = ["ListPrice", "LivingArea", "DaysOnMarket"]

def section(title):
    """Print a visual divider in the console so each step's output is easy to scan."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def flag_iqr_outliers(df, columns, label):
    """
    For each column in `columns`: compute Q1 (25th percentile) and Q3 (75th
    percentile) on that column's own values, derive the IQR-based lower/upper
    bounds, and add a boolean flag column named '{column}_OutlierFlag' that's
    True for any row whose value falls outside [lower, upper].
 
    Also adds a combined 'Outlier_Any_Flag' column that's True if the row was
    flagged on ANY of the checked columns -- this is what the filtered
    (no-outliers) dataset will remove.
    """
    df = df.copy() # never mutate the caller's dataframe in place
    combined_flag = pd.Series(False, index=df.index)  # accumulates "flagged by any column" as we loop
 
    print(f"[{label}] IQR bounds per column:")
    for col in columns:
        if col not in df.columns: # guard: skip gracefully if a column isn't present
            print(f"    - {col}: not found in this dataset -- skipped.")
            continue
 
        Q1 = df[col].quantile(0.25) # 25th percentile of this column (NaN values are ignored automatically)
        Q3 = df[col].quantile(0.75) # 75th percentile
        IQR = Q3 - Q1 # interquartile range -- the spread of the "typical" middle 50% of values
        lower = Q1 - 1.5 * IQR # anything below this is unusually low
        upper = Q3 + 1.5 * IQR # anything above this is unusually high
 
        flag_col = f"{col}_OutlierFlag"
        # A value is an outlier if it's outside [lower, upper]
        # Comparisons against NaN are always False, so missing values are never flagged
        df[flag_col] = (df[col] < lower) | (df[col] > upper)
 
        n_flagged = int(df[flag_col].sum())
        print(f"    - {col}: Q1={Q1:,.2f}, Q3={Q3:,.2f}, IQR={IQR:,.2f}, "
              f"bounds=[{lower:,.2f}, {upper:,.2f}], flagged={n_flagged} rows "
              f"({n_flagged / len(df) * 100:.2f}%)")
 
        combined_flag = combined_flag | df[flag_col] # OR this column's flag into the running "any" flag
 
    df["Outlier_Any_Flag"] = combined_flag # True if flagged by ClosePrice/ListPrice, LivingArea, OR DaysOnMarket
    print(f"    - Outlier_Any_Flag (flagged by at least one column): {int(combined_flag.sum())} rows "
          f"({combined_flag.mean() * 100:.2f}%)")
    return df


def compare_before_after(df, columns, label):
    """
    Print row-count and median comparisons between the full (flagged) dataset
    and the filtered (outliers removed) version, then return the filtered
    dataframe.
    """
    print(f"\n[{label}] Before vs After Filtering Comparison")
    filtered = df[~df["Outlier_Any_Flag"]].reset_index(drop=True) # keep only rows NOT flagged as an outlier
 
    removed = len(df) - len(filtered)
    print(f"    Row count: before={len(df)}, after={len(filtered)}, "
          f"removed={removed} ({removed / len(df) * 100:.2f}%)")
 
    for col in columns: # report median shift for each checked column
        if col not in df.columns:
            continue
        median_before = df[col].median() # median computed on the full, unfiltered data
        median_after = filtered[col].median() # median computed after outliers are removed
        print(f"    {col} median: before={median_before:,.2f}, after={median_after:,.2f} "
              f"(change={median_after - median_before:+,.2f})")
 
    return filtered


# ===================== Load Enriched Data =====================
section("Load Enriched Data")
sold = pd.read_csv(SOLD_INPUT_PATH, low_memory=False) # output of engineer_market_metrics.py
listing = pd.read_csv(LISTING_INPUT_PATH, low_memory=False) # output of engineer_market_metrics.py
print(f"Sold: {sold.shape[0]} rows, {sold.shape[1]} columns")
print(f"Listing: {listing.shape[0]} rows, {listing.shape[1]} columns")


# ===================== Flag Outliers (IQR method) =====================
section("IQR Outlier Flagging")
sold_flagged = flag_iqr_outliers(sold, SOLD_OUTLIER_COLUMNS, "Sold")
listing_flagged = flag_iqr_outliers(listing, LISTING_OUTLIER_COLUMNS, "Listing")
 
 
# ===================== Compare Before/After and Build Filtered Datasets =====================
section("Before/After Comparison")
sold_filtered = compare_before_after(sold_flagged, SOLD_OUTLIER_COLUMNS, "Sold")
listing_filtered = compare_before_after(listing_flagged, LISTING_OUTLIER_COLUMNS, "Listing")
 
 
# ===================== Save Outputs =====================
section("Save Outputs")
sold_flagged.to_csv(SOLD_FLAGGED_OUTPUT, index=False) # full Sold dataset, flags added, no rows removed
sold_filtered.to_csv(SOLD_FILTERED_OUTPUT, index=False) # Sold dataset with outlier rows removed
listing_flagged.to_csv(LISTING_FLAGGED_OUTPUT, index=False) # full Listing dataset, flags added, no rows removed
listing_filtered.to_csv(LISTING_FILTERED_OUTPUT, index=False) # Listing dataset with outlier rows removed
 
print(f"Saved: {SOLD_FLAGGED_OUTPUT}")
print(f"Saved: {SOLD_FILTERED_OUTPUT}")
print(f"Saved: {LISTING_FLAGGED_OUTPUT}")
print(f"Saved: {LISTING_FILTERED_OUTPUT}")