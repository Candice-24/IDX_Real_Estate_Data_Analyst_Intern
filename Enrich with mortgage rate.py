"""
Enrich the combined CRMLS Sold and Listing datasets with the national
30-year fixed mortgage rate (FRED series MORTGAGE30US).
 
  1. Fetch MORTGAGE30US (weekly) directly from FRED -- no API key required
  2. Resample weekly rates to monthly averages
  3. Build a year_month key on both MLS datasets from their transaction dates
     (Sold -> CloseDate, Listing -> ListingContractDate)
  4. Merge the monthly rate onto both datasets
  5. Validate the merge (null rate check) and save enriched CSVs
"""

# ===================== Install Package =====================
 
import os
import pandas as pd

# --------------------------- Config -------------------------
DATA_DIR = r"D:\Meng\document\AU\Career\2026 Intern\IDX\2. Data Analyst summer 2026\csv"
START_MONTH = 202401
END_MONTH = 202606
 
# Set False to enrich the full merged datasets instead of the
# Residential-filtered ones.
USE_RESIDENTIAL_ONLY = True
 
if USE_RESIDENTIAL_ONLY:
    SOLD_INPUT_PATH = os.path.join(DATA_DIR, f"Sold_Residential_{START_MONTH}_{END_MONTH}.csv")
    LISTING_INPUT_PATH = os.path.join(DATA_DIR, f"Listing_Residential_{START_MONTH}_{END_MONTH}.csv")
else:
    SOLD_INPUT_PATH = os.path.join(DATA_DIR, f"Sold_Merged_{START_MONTH}_{END_MONTH}.csv")
    LISTING_INPUT_PATH = os.path.join(DATA_DIR, f"Listing_Merged_{START_MONTH}_{END_MONTH}.csv")
 
SOLD_OUTPUT_PATH = os.path.join(DATA_DIR, f"Sold_with_MortgageRate_{START_MONTH}_{END_MONTH}.csv")
LISTING_OUTPUT_PATH = os.path.join(DATA_DIR, f"Listing_with_MortgageRate_{START_MONTH}_{END_MONTH}.csv")
 
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

# ===================== Step 1: Fetch Mortgage Rate Data =====================
print("Fetching MORTGAGE30US from FRED...")
mortgage = pd.read_csv(FRED_URL)
 
# FRED's column headers have changed format over the years (e.g. 'DATE' vs
# 'observation_date'). Rename by position instead of by name so this keeps
# working regardless of the exact header FRED currently uses.
if mortgage.shape[1] != 2:
    raise SystemExit(f"Expected 2 columns from FRED, got {mortgage.shape[1]}: {list(mortgage.columns)}")
mortgage.columns = ["date", "rate_30yr_fixed"]
 
mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["rate_30yr_fixed"], errors="coerce")
 
bad_dates = mortgage["date"].isna().sum()
bad_rates = mortgage["rate_30yr_fixed"].isna().sum()
if bad_dates or bad_rates:
    print(f"  Warning: {bad_dates} unparseable dates, {bad_rates} missing/non-numeric rates "
          f"in the raw FRED download (dropping those rows).")
mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])
 
print(f"  Fetched {len(mortgage)} weekly observations "
      f"({mortgage['date'].min().date()} to {mortgage['date'].max().date()})")

# ===================== Step 2: Resample Weekly to Monthly =====================
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)
print(f"  Resampled to {len(mortgage_monthly)} monthly averages "
      f"({mortgage_monthly['year_month'].min()} to {mortgage_monthly['year_month'].max()})")

# ===================== Step 3: Read MLS Datasets and Create year_month Key =====================
print("\nReading combined MLS datasets...")
sold = pd.read_csv(SOLD_INPUT_PATH, low_memory=False)
listing = pd.read_csv(LISTING_INPUT_PATH, low_memory=False)
print(f"  Sold: {sold.shape[0]} rows, {sold.shape[1]} columns")
print(f"  Listing: {listing.shape[0]} rows, {listing.shape[1]} columns")
 
sold["year_month"] = pd.to_datetime(sold["CloseDate"], errors="coerce").dt.to_period("M")
listing["year_month"] = pd.to_datetime(listing["ListingContractDate"], errors="coerce").dt.to_period("M")
 
sold_missing_date = sold["year_month"].isna().sum()
listing_missing_date = listing["year_month"].isna().sum()
print(f"\n  Sold rows with missing/unparseable CloseDate: {sold_missing_date}")
print(f"  Listing rows with missing/unparseable ListingContractDate: {listing_missing_date}")

# ===================== Step 4: Merge =====================
sold_rows_before_merge = len(sold)
listing_rows_before_merge = len(listing)
 
sold_with_rates = sold.merge(mortgage_monthly, on="year_month", how="left")
listing_with_rates = listing.merge(mortgage_monthly, on="year_month", how="left")
 
print(f"\nSold rows before merge: {sold_rows_before_merge}, after merge: {len(sold_with_rates)}")
print(f"Listing rows before merge: {listing_rows_before_merge}, after merge: {len(listing_with_rates)}")
 
assert len(sold_with_rates) == sold_rows_before_merge, "Row count changed during Sold merge -- check for duplicate year_month keys in mortgage_monthly!"
assert len(listing_with_rates) == listing_rows_before_merge, "Row count changed during Listing merge -- check for duplicate year_month keys in mortgage_monthly!"

# ===================== Step 5: Validate the Merge =====================
def validate_merge(df, label, date_col, missing_date_count):
    null_rate = df["rate_30yr_fixed"].isnull().sum()
    print(f"\n--- [{label}] Merge Validation ---")
    print(f"  Rows with null rate_30yr_fixed: {null_rate}")
 
    if null_rate > 0:
        # Split nulls into "no transaction date to key off of" vs
        # "had a valid year_month but FRED has no rate for that month"
        no_date = df["year_month"].isna().sum()
        has_date_no_rate = null_rate - no_date
        print(f"    -> {no_date} due to missing/unparseable {date_col}")
        print(f"    -> {has_date_no_rate} due to year_month outside FRED's fetched range "
              f"({mortgage_monthly['year_month'].min()} to {mortgage_monthly['year_month'].max()})")
        if has_date_no_rate > 0:
            gap_months = sorted(df.loc[df["rate_30yr_fixed"].isnull() & df["year_month"].notna(), "year_month"].unique())
            print(f"    Unmatched year_months: {gap_months}")
    else:
        print("  All rows with a valid transaction date received a mortgage rate.")
 
 
validate_merge(sold_with_rates, "Sold", "CloseDate", sold_missing_date)
validate_merge(listing_with_rates, "Listing", "ListingContractDate", listing_missing_date)
 
print("\nPreview (Sold):")
print(sold_with_rates[["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]].head())

# ===================== Save Enriched Datasets =====================
sold_with_rates.to_csv(SOLD_OUTPUT_PATH, index=False)
listing_with_rates.to_csv(LISTING_OUTPUT_PATH, index=False)
print(f"\nSaved: {SOLD_OUTPUT_PATH}")
print(f"Saved: {LISTING_OUTPUT_PATH}")