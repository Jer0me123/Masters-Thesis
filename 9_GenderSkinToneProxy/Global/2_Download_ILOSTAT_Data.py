"""
ILOSTAT Complete Occupation Data Downloader (Robust Version)

- Discovers available ILOSTAT dataflows
- Determines ALL available countries from the API
- Downloads occupation-related datasets
- Country-by-country, year-by-year (NO ALL keyword)
- Supports:
    • Interactive UI
    • Non-interactive CLI (--dataflow)
- Produces ONE combined CSV output
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

BASE_URL = "https://sdmx.ilo.org/rest"

# ============================================================
# DISCOVERY
# ============================================================

def discover_dataflows():
    print("🔍 Discovering available dataflows from ILOSTAT...")
    r = requests.get(f"{BASE_URL}/dataflow/ILO", timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    ns = {
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }

    dataflows = []
    for df in root.findall(".//str:Dataflow", ns):
        dataflows.append({
            "id": df.get("id"),
            "name": (df.find(".//com:Name", ns).text or "").strip(),
        })

    print(f"✅ Found {len(dataflows)} dataflows")
    return dataflows


def find_occupation_dataflows(dataflows):
    keywords = ["OCU", "ISCO", "OCCUPATION"]
    occ = [df for df in dataflows if any(k in df["id"] for k in keywords)]
    print(f"\n📊 Found {len(occ)} occupation-related dataflows")
    return occ


# ============================================================
# COUNTRY DISCOVERY (CRITICAL)
# ============================================================

def discover_all_countries():
    """
    Discover all REF_AREA country codes supported by ILOSTAT
    using the generic CL_AREA codelist.
    """
    print("Discovering all available countries from ILOSTAT...")
    url = f"{BASE_URL}/codelist/ILO/CL_AREA"

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    ns = {
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    }

    countries = []
    for code in root.findall(".//str:Code", ns):
        cid = code.get("id")
        # Keep ISO-like country codes only (skip aggregates)
        if cid.isalpha() and len(cid) == 3:
            countries.append(cid)

    countries = sorted(set(countries))
    print(f"Discovered {len(countries)} countries")
    return countries


# ============================================================
# DOWNLOAD LOGIC (ROBUST)
# ============================================================

def download_country_year(
    dataflow_id,
    country,
    year
):
    url = f"{BASE_URL}/data/ILO,{dataflow_id}/{country}.A..."
    params = {
        "format": "jsondata",
        "startPeriod": str(year),
        "endPeriod": str(year),
    }

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return parse_sdmx_json(r.json())


def download_complete_occupation_data(
    dataflow_id,
    output_dir="occupation_data",
    start_year="2015",
    end_year="2024"
):
    Path(output_dir).mkdir(exist_ok=True)

    start_year = int(start_year)
    end_year = int(end_year)

    print(f"\nDownloading {dataflow_id}")
    print(f"   Countries: ALL (enumerated)")
    print(f"   Years: {start_year} → {end_year}")

    countries = discover_all_countries()
    all_dfs = []

    for year in range(start_year, end_year + 1):
        print(f"\n📅 Year {year}")
        for country in countries:
            try:
                df = download_country_year(dataflow_id, country, year)
                if not df.empty:
                    all_dfs.append(df)
                    print(f"{country}")
                else:
                    print(f"{country} (no data)")
            except Exception as e:
                print(f"{country}: {str(e)[:60]}")

            time.sleep(0.2)  # be nice to the API

    if not all_dfs:
        print("No data retrieved")
        return None, None

    print("\nCombining results...")
    final_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"{dataflow_id}_{start_year}_{end_year}_{ts}.csv"
    final_df.to_csv(out_path, index=False)

    print("\nDownload complete")
    print(f"   Observations: {len(final_df):,}")
    if "REF_AREA" in final_df.columns:
        print(f"   Countries: {final_df['REF_AREA'].nunique()}")
    print(f"   File: {out_path}")

    return final_df, out_path


# ============================================================
# SDMX PARSER
# ============================================================

def parse_sdmx_json(json_data):
    try:
        structure = json_data["data"].get("structures", [None])[0]
        dataset = json_data["data"]["dataSets"][0]

        series_dims = structure["dimensions"]["series"]
        obs_dims = structure["dimensions"]["observation"]

        dim_names = [d["id"] for d in series_dims]
        dim_values = {d["id"]: d["values"] for d in series_dims}

        labels = {
            d["id"]: {v["id"]: v.get("name", v["id"]) for v in d["values"]}
            for d in series_dims
        }

        time_dim = obs_dims[0]["id"]
        time_vals = obs_dims[0]["values"]

        rows = []
        for key, series in dataset["series"].items():
            idxs = list(map(int, key.split(":")))
            base = {}

            for i, dim in enumerate(dim_names):
                code = dim_values[dim][idxs[i]]["id"]
                base[dim] = code
                base[f"{dim}_LABEL"] = labels[dim].get(code, code)

            for obs_idx, obs_val in series["observations"].items():
                row = base.copy()
                row[time_dim] = time_vals[int(obs_idx)]["id"]
                row["VALUE"] = obs_val[0]
                rows.append(row)

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ILOSTAT occupation data downloader")
    parser.add_argument("--dataflow", type=str, help="Directly download a dataflow")
    parser.add_argument("--start-year", type=str, default="2015")
    parser.add_argument("--end-year", type=str, default="2024")
    args = parser.parse_args()

    if args.dataflow:
        download_complete_occupation_data(
            args.dataflow,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        return

    print("=" * 80)
    print("ILOSTAT OCCUPATION DATA DISCOVERY")
    print("=" * 80)

    dataflows = discover_dataflows()
    occ = find_occupation_dataflows(dataflows)

    print("\nAvailable occupation-related dataflows:\n")
    for i, df in enumerate(occ, 1):
        print(f"{i}. {df['id']}")

    choice = input("\nSelect dataflow number: ").strip()
    try:
        df = occ[int(choice) - 1]
    except Exception:
        print("Invalid selection")
        return

    start = input("Start year [2015]: ").strip() or "2015"
    end = input("End year   [2024]: ").strip() or "2024"

    download_complete_occupation_data(df["id"], start_year=start, end_year=end)
    print("\n🎉 DONE!")


if __name__ == "__main__":
    main()