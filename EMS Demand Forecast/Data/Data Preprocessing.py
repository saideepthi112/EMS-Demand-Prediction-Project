import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Batch format EMS .part files to match a template CSV (column order/types), processing N files at a time.")
    ap.add_argument("--input_dir", required=True, help="Directory containing *.part files")
    ap.add_argument("--template_csv", required=True, help="Path to Formatted_EMS_Dataset.csv (used for exact columns/order)")
    ap.add_argument("--output_dir", required=True, help="Directory to write formatted outputs")
    ap.add_argument("--batch_size", type=int, default=10, help="Number of .part files to process per batch")
    ap.add_argument("--topN_calltypes", type=int, default=0, help="Optional: keep only top-N calltype columns present in template (0 = keep all in template)")
    return ap.parse_args()

def ensure_binary(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(0).astype(int)
    s = s.clip(0, 1)
    return s

def derive_date_hour(df):
    if "INCIDENT_DATETIME" in df.columns:
        ts = pd.to_datetime(df["INCIDENT_DATETIME"], errors="coerce")
        return ts.dt.date, ts.dt.hour
    if "Date" in df.columns and "Hour" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce").dt.date
        h = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int)
        return d, h
    if "INCIDENT_DATE" in df.columns and "INCIDENT_TIME" in df.columns:
        ts = pd.to_datetime(df["INCIDENT_DATE"].astype(str) + " " + df["INCIDENT_TIME"].astype(str), errors="coerce")
        return ts.dt.date, ts.dt.hour
    if "CREATED_DATE" in df.columns:
        ts = pd.to_datetime(df["CREATED_DATE"], errors="coerce")
        return ts.dt.date, ts.dt.hour
    return pd.NaT, pd.Series([np.nan]*len(df), index=df.index)

def compute_is_covid(date_series):
    dt = pd.to_datetime(date_series, errors="coerce")
    return ((dt >= pd.Timestamp("2020-03-01")) & (dt <= pd.Timestamp("2022-12-31"))).astype(int)

def mode_or_nan(x):
    m = x.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = pd.read_csv(args.template_csv)
    template_cols = list(template.columns)
    template_calltype_cols = [c for c in template_cols if c.endswith("_count")]

    part_files = sorted(input_dir.glob("*.part"))
    if not part_files:
        print(f"No .part files found in {input_dir}")
        return

    batch_size = args.batch_size
    for start in range(0, len(part_files), batch_size):
        batch = part_files[start:start+batch_size]
        batch_frames = []
        print(f"\\n=== Processing batch {start//batch_size + 1} ({len(batch)} files) ===")
        for fp in batch:
            print(f"  -> {fp.name}")
            df = pd.read_csv(fp)

            if "ZIP_CODE" in df.columns:
                zip_col = "ZIP_CODE"
            elif "ZIPCODE" in df.columns:
                df["ZIP_CODE"] = df["ZIPCODE"]
                zip_col = "ZIP_CODE"
            else:
                candidates = [c for c in df.columns if c.lower() in {"zip", "zip_code", "zipcode"}]
                if candidates:
                    df["ZIP_CODE"] = df[candidates[0]]
                    zip_col = "ZIP_CODE"
                else:
                    print(f"     !! Skipping (no ZIP column): {fp.name}")
                    continue

            Date, Hour = derive_date_hour(df)
            df["Date"] = Date
            df["Hour"] = Hour
            df["is_covid"] = compute_is_covid(df["Date"])

            numeric_median_cols = [
                "INITIAL_SEVERITY_LEVEL_CODE", "FINAL_SEVERITY_LEVEL_CODE",
                "DISPATCH_RESPONSE_SECONDS_QY", "INCIDENT_RESPONSE_SECONDS_QY", "INCIDENT_TRAVEL_TM_SECONDS_QY"
            ]
            for c in numeric_median_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            indicator_cols = [
                "VALID_DISPATCH_RSPNS_TIME_INDC", "VALID_INCIDENT_RSPNS_TIME_INDC",
                "HELD_INDICATOR", "REOPEN_INDICATOR", "SPECIAL_EVENT_INDICATOR",
                "STANDBY_INDICATOR", "TRANSFER_INDICATOR"
            ]

            if "incident_class" in df.columns:
                calltype_dummies = pd.get_dummies(df["incident_class"])
            else:
                ic_alt = None
                for alt in ["INCIDENT_CLASS", "incident_class_f", "INITIAL_CALL_TYPE", "FINAL_CALL_TYPE"]:
                    if alt in df.columns:
                        ic_alt = alt
                        break
                if ic_alt is not None:
                    calltype_dummies = pd.get_dummies(df[ic_alt])
                else:
                    calltype_dummies = pd.DataFrame(index=df.index)

            calltype_dummies = calltype_dummies.add_suffix("_count")
            # Ensure Call_Count exists for aggregation
            if "Call_Count" not in df.columns:
                df["Call_Count"] = 1
            if template_calltype_cols:
                calltype_dummies = calltype_dummies.reindex(columns=template_calltype_cols, fill_value=0)

            agg = {
                "INITIAL_SEVERITY_LEVEL_CODE": "median",
                "FINAL_SEVERITY_LEVEL_CODE": "median",
                "DISPATCH_RESPONSE_SECONDS_QY": "median",
                "INCIDENT_RESPONSE_SECONDS_QY": "median",
                "INCIDENT_TRAVEL_TM_SECONDS_QY": "median",
                **{c: mode_or_nan for c in indicator_cols if c in df.columns},
                "POPULATION": "first" if "POPULATION" in df.columns else (lambda x: np.nan),
                "BOROUGH": "first" if "BOROUGH" in df.columns else (lambda x: np.nan),
                "Call_Count": "sum",
                "is_covid": "first",
            }

            df_ct = pd.concat([df, calltype_dummies], axis=1)
            for c in calltype_dummies.columns:
                agg[c] = "sum"

            group_keys = ["ZIP_CODE", "Hour", "Date"]
            grouped = df_ct.groupby(group_keys).agg(agg).reset_index()

            for c in indicator_cols:
                if c in grouped.columns:
                    grouped[c] = ensure_binary(grouped[c])

            for col in template_cols:
                if col not in grouped.columns:
                    if col in template_calltype_cols:
                        grouped[col] = 0
                    elif col in ["BOROUGH"]:
                        grouped[col] = np.nan
                    else:
                        grouped[col] = np.nan

            grouped = grouped.reindex(columns=template_cols)

            batch_frames.append(grouped)

        if not batch_frames:
            print("No frames produced in this batch.")
            continue

        batch_df = pd.concat(batch_frames, ignore_index=True)
        batch_idx = start // batch_size + 1
        out_path = output_dir / f"all_formatted_batches_dedup.csv"
        batch_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}  rows={len(batch_df)}, cols={len(batch_df.columns)}")

def fix_missing_hours(df, count_col="Call_Count"):
    """
    Ensure each ZIP_CODE + Date has exactly 24 rows (0-23 hours).
    Missing rows are filled with forward/backward filled features and incident count = 0.
    """
    fixed_dfs = []

    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Loop over each group (ZIP_CODE, Date)
    for (zip_code, date), group in df.groupby(['ZIP_CODE', 'Date']):
        group = group.sort_values('Hour')

        # Create full set of hours
        full_hours = pd.DataFrame({'Hour': range(24)})
        merged = full_hours.merge(group, on='Hour', how='left')

        # Add ZIP_CODE and Date
        merged['ZIP_CODE'] = zip_code
        merged['Date'] = date

        # Forward fill/backward fill for non-count columns
        for col in df.columns:
            if col not in ['Hour', count_col]:
                merged[col] = merged[col].ffill().bfill()

        # Incident count = 0 where missing
        merged[count_col] = merged[count_col].fillna(0)

        fixed_dfs.append(merged)

    # Combine everything
    fixed_df = pd.concat(fixed_dfs, ignore_index=True)
    fixed_df = fixed_df.sort_values(['ZIP_CODE', 'Date', 'Hour']).reset_index(drop=True)

    return fixed_df


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("all_formatted_batches_dedup.csv")

    # Identify the total calls column (last column in your description)
    # For safety, create a single "Call_Count" column if not already present
    if "Call_Count" not in df.columns:
        # Assuming incident count = sum of individual type counts
        df["Call_Count"] = (
            df["Medical Emergencies_count"] +
            df["Environmental and Poisoning Emergencies_count"] +
            df["Other_count"] +
            df["Trauma-Related Incidents_count"]
            # add other count columns if present
        )

    # Fix dataset
    fixed_df = fix_missing_hours(df, count_col="Call_Count")

    # Save cleaned dataset
    fixed_df.to_csv("preprocessed_data.csv", index=False)


if __name__ == "__main__":
    main()
