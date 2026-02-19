# #!/usr/bin/env python3
# """
# ILOSTAT Complete Data Discovery and Download
# Discovers all available dataflows and downloads the most detailed occupation data
# """

# import requests
# import pandas as pd
# import json
# from datetime import datetime
# import time
# import xml.etree.ElementTree as ET
# from pathlib import Path

# BASE_URL = "https://sdmx.ilo.org/rest"

# def discover_dataflows():
#     """Discover all available dataflows from ILOSTAT"""
    
#     print("🔍 Discovering available dataflows from ILOSTAT...")
#     print(f"   URL: {BASE_URL}/dataflow/ILO")
    
#     try:
#         response = requests.get(f"{BASE_URL}/dataflow/ILO", timeout=30)
#         response.raise_for_status()
        
#         # Parse XML to find dataflows
#         root = ET.fromstring(response.content)
        
#         # Find all dataflow elements (namespace handling)
#         namespaces = {
#             'str': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
#             'com': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
#         }
        
#         dataflows = []
#         for dataflow in root.findall('.//str:Dataflow', namespaces):
#             df_id = dataflow.get('id')
            
#             # Get name
#             name_elem = dataflow.find('.//com:Name', namespaces)
#             df_name = name_elem.text if name_elem is not None else "Unknown"
            
#             # Get description if available
#             desc_elem = dataflow.find('.//com:Description', namespaces)
#             df_desc = desc_elem.text if desc_elem is not None else ""
            
#             dataflows.append({
#                 'id': df_id,
#                 'name': df_name,
#                 'description': df_desc[:100] if df_desc else ""
#             })
        
#         print(f"✅ Found {len(dataflows)} dataflows")
#         return dataflows
        
#     except Exception as e:
#         print(f"❌ Error discovering dataflows: {e}")
#         return []


# def find_occupation_dataflows(dataflows):
#     """Find all dataflows related to occupation (OCU, ISCO)"""
    
#     occupation_keywords = ['OCU', 'ISCO', 'OCCUPATION']
    
#     occ_dataflows = []
#     for df in dataflows:
#         df_id = df['id'].upper()
#         df_name = df['name'].upper()
        
#         # Check if dataflow contains occupation-related keywords
#         if any(keyword in df_id or keyword in df_name for keyword in occupation_keywords):
#             occ_dataflows.append(df)
    
#     print(f"\n📊 Found {len(occ_dataflows)} occupation-related dataflows:")
#     for df in occ_dataflows:
#         print(f"   • {df['id']}: {df['name']}")
    
#     return occ_dataflows


# def get_dataflow_structure(dataflow_id):
#     """Get the detailed structure of a dataflow to see available dimensions"""
    
#     print(f"\n🔍 Examining structure of {dataflow_id}...")
    
#     url = f"{BASE_URL}/dataflow/ILO/{dataflow_id}?references=descendants"
    
#     try:
#         response = requests.get(url, timeout=60)
#         response.raise_for_status()
        
#         # Parse XML
#         root = ET.fromstring(response.content)
        
#         # Find code lists
#         namespaces = {
#             'str': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
#             'com': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
#         }
        
#         # Look for OCU/ISCO related code lists
#         codelists = []
#         for codelist in root.findall('.//str:Codelist', namespaces):
#             cl_id = codelist.get('id')
#             if 'OCU' in cl_id or 'ISCO' in cl_id:
#                 name_elem = codelist.find('.//com:Name', namespaces)
#                 cl_name = name_elem.text if name_elem is not None else "Unknown"
                
#                 # Count codes
#                 codes = codelist.findall('.//str:Code', namespaces)
                
#                 codelists.append({
#                     'id': cl_id,
#                     'name': cl_name,
#                     'num_codes': len(codes)
#                 })
        
#         if codelists:
#             print(f"   Found {len(codelists)} occupation-related code lists:")
#             for cl in codelists:
#                 print(f"      • {cl['id']}: {cl['name']} ({cl['num_codes']} codes)")
        
#         return codelists
        
#     except Exception as e:
#         print(f"   ⚠️  Could not examine structure: {e}")
#         return []


# def test_dataflow_data_availability(dataflow_id, test_country="USA", test_year="2023"):
#     """Test if a dataflow has data available"""
    
#     print(f"   Testing data availability for {dataflow_id}...")
    
#     # Try to get a small sample
#     url = f"{BASE_URL}/data/ILO,{dataflow_id}/{test_country}.A..."
#     params = {
#         'format': 'jsondata',
#         'startPeriod': test_year,
#         'endPeriod': test_year,
#         'detail': 'serieskeysonly'  # Only get series keys, not data (faster)
#     }
    
#     try:
#         response = requests.get(url, params=params, timeout=30)
#         response.raise_for_status()
        
#         data = response.json()
        
#         # Check if data exists
#         if 'data' in data and 'dataSets' in data['data']:
#             dataset = data['data']['dataSets'][0]
#             if 'series' in dataset and len(dataset['series']) > 0:
#                 print(f"      ✅ Data available ({len(dataset['series'])} series)")
#                 return True
        
#         print(f"      ⚠️  No data available")
#         return False
        
#     except Exception as e:
#         print(f"      ❌ Error testing: {str(e)[:50]}")
#         return False


# def download_complete_occupation_data(dataflow_id, output_dir="occupation_data"):
#     """Download complete occupation data for all countries from a specific dataflow"""
    
#     Path(output_dir).mkdir(exist_ok=True)
    
#     print(f"\n📥 Downloading complete data from {dataflow_id}...")
#     print(f"   This may take several minutes...")
    
#     # Download all countries, all years
#     url = f"{BASE_URL}/data/ILO,{dataflow_id}/ALL.A..."
#     params = {
#         'format': 'jsondata',
#         'startPeriod': '2015',  # Last ~10 years
#         'endPeriod': '2024'
#     }
    
#     print(f"   URL: {url}")
#     print(f"   Params: {params}")
    
#     try:
#         print(f"   ⏳ Making request (this may take 2-5 minutes for large datasets)...")
#         response = requests.get(url, params=params, timeout=600)  # 10 minute timeout
#         response.raise_for_status()
        
#         print(f"   ✅ Data retrieved! Size: {len(response.content) / 1024 / 1024:.1f} MB")
        
#         # Save raw JSON
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         json_file = f"{output_dir}/{dataflow_id}_raw_{timestamp}.json"
        
#         with open(json_file, 'w') as f:
#             json.dump(response.json(), f)
#         print(f"   💾 Raw JSON saved: {json_file}")
        
#         # Parse to CSV
#         print(f"   🔄 Parsing to CSV...")
#         df = parse_sdmx_json(response.json())
        
#         if not df.empty:
#             csv_file = f"{output_dir}/{dataflow_id}_complete_{timestamp}.csv"
#             df.to_csv(csv_file, index=False)
#             print(f"   💾 CSV saved: {csv_file}")
#             print(f"   📊 Data summary:")
#             print(f"      • Total observations: {len(df):,}")
#             print(f"      • Columns: {list(df.columns)}")
#             print(f"      • Date range: {df['TIME_PERIOD'].min()} to {df['TIME_PERIOD'].max()}")
#             if 'REF_AREA' in df.columns:
#                 print(f"      • Countries: {df['REF_AREA'].nunique()}")
#             if 'OCU' in df.columns:
#                 print(f"      • Occupation codes: {df['OCU'].nunique()}")
            
#             return df, csv_file
#         else:
#             print(f"   ⚠️  No data could be parsed")
#             return None, None
            
#     except requests.exceptions.Timeout:
#         print(f"   ❌ Request timed out. Dataset too large.")
#         print(f"   💡 Try downloading by region or year range instead.")
#         return None, None
#     except Exception as e:
#         print(f"   ❌ Error: {e}")
#         return None, None


# def download_by_country_chunks(dataflow_id, output_dir="occupation_data", chunk_size=10):
#     """Download data in chunks by country to avoid timeouts"""
    
#     Path(output_dir).mkdir(exist_ok=True)
    
#     print(f"\n📥 Downloading {dataflow_id} in country chunks...")
    
#     # Common countries list (can be expanded)
#     countries = [
#         # Europe
#         "DEU", "FRA", "GBR", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "SWE", 
#         "NOR", "DNK", "FIN", "POL", "CZE", "HUN", "GRC", "PRT", "IRL", "ROU",
#         # Americas
#         "USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER", "URY",
#         # Asia-Pacific
#         "JPN", "CHN", "KOR", "IND", "IDN", "THA", "MYS", "SGP", "PHL", "VNM",
#         "AUS", "NZL",
#         # Middle East & Africa
#         "ISR", "TUR", "ZAF", "EGY", "KEN", "NGA",
#     ]
    
#     all_data = []
#     successful = 0
#     failed = 0
    
#     # Process in chunks
#     for i in range(0, len(countries), chunk_size):
#         chunk = countries[i:i+chunk_size]
#         chunk_str = "+".join(chunk)
        
#         print(f"\n   📦 Chunk {i//chunk_size + 1}/{(len(countries)-1)//chunk_size + 1}: {', '.join(chunk)}")
        
#         url = f"{BASE_URL}/data/ILO,{dataflow_id}/{chunk_str}.A..."
#         params = {
#             'format': 'jsondata',
#             'startPeriod': '2015',
#             'endPeriod': '2024'
#         }
        
#         try:
#             response = requests.get(url, params=params, timeout=300)
#             response.raise_for_status()
            
#             df = parse_sdmx_json(response.json())
            
#             if not df.empty:
#                 all_data.append(df)
#                 successful += len(chunk)
#                 print(f"      ✅ Retrieved {len(df):,} observations")
#             else:
#                 failed += len(chunk)
#                 print(f"      ⚠️  No data")
                
#         except Exception as e:
#             failed += len(chunk)
#             print(f"      ❌ Error: {str(e)[:50]}")
        
#         # Rate limiting - be nice to the server
#         time.sleep(2)
    
#     # Combine all chunks
#     if all_data:
#         print(f"\n   🔄 Combining {len(all_data)} chunks...")
#         combined_df = pd.concat(all_data, ignore_index=True)
        
#         # Remove duplicates
#         combined_df = combined_df.drop_duplicates()
        
#         csv_file = f"{output_dir}/{dataflow_id}.csv"
#         combined_df.to_csv(csv_file, index=False)
        
#         print(f"\n   ✅ Download complete!")
#         print(f"   📊 Summary:")
#         print(f"      • Successful countries: {successful}")
#         print(f"      • Failed countries: {failed}")
#         print(f"      • Total observations: {len(combined_df):,}")
#         print(f"      • File: {csv_file}")
        
#         return combined_df, csv_file
#     else:
#         print(f"\n   ❌ No data retrieved")
#         return None, None


# def parse_sdmx_json(json_data):
#     """Parse SDMX JSON format to DataFrame with enriched labels"""
    
#     try:
#         if 'data' not in json_data:
#             return pd.DataFrame()
        
#         # ILOSTAT returns structures as an array
#         if 'structures' in json_data['data']:
#             structure = json_data['data']['structures'][0]
#         elif 'structure' in json_data['data']:
#             structure = json_data['data']['structure']
#         else:
#             return pd.DataFrame()
            
#         dataset = json_data['data']['dataSets'][0]
        
#         # Get dimension info
#         series_dims = structure['dimensions']['series']
#         obs_dims = structure['dimensions']['observation']
        
#         dim_names = [d['id'] for d in series_dims]
#         dim_values = {d['id']: d['values'] for d in series_dims}
        
#         # Create label mappings
#         label_mappings = {}
#         for dim in series_dims:
#             dim_id = dim['id']
#             label_mappings[dim_id] = {
#                 val['id']: val.get('name', val['id']) 
#                 for val in dim['values']
#             }
        
#         time_values = obs_dims[0]['values']
#         time_name = obs_dims[0]['id']
        
#         # Parse series
#         rows = []
#         for series_key, series_info in dataset['series'].items():
#             indices = [int(i) for i in series_key.split(':')]
            
#             # Build dimension values (both codes and labels)
#             row_dims = {}
#             for i, dim_name in enumerate(dim_names):
#                 idx = indices[i]
#                 code = dim_values[dim_name][idx]['id']
#                 row_dims[dim_name] = code
#                 row_dims[f"{dim_name}_LABEL"] = label_mappings[dim_name].get(code, code)
            
#             # Add observations
#             for obs_idx, obs_value in series_info['observations'].items():
#                 row = row_dims.copy()
#                 row[time_name] = time_values[int(obs_idx)]['id']
#                 row['VALUE'] = obs_value[0]
#                 rows.append(row)
        
#         df = pd.DataFrame(rows)
        
#         # Reorder columns
#         ordered_cols = []
#         for dim_name in dim_names:
#             ordered_cols.append(dim_name)
#             if f"{dim_name}_LABEL" in df.columns:
#                 ordered_cols.append(f"{dim_name}_LABEL")
#         ordered_cols.extend([time_name, 'VALUE'])
        
#         df = df[ordered_cols]
        
#         return df
        
#     except Exception as e:
#         print(f"⚠️  Parse error: {e}")
#         return pd.DataFrame()


# def main():
#     print("="*80)
#     print("ILOSTAT COMPLETE OCCUPATION DATA DISCOVERY & DOWNLOAD")
#     print("="*80)
    
#     # Step 1: Discover all dataflows
#     dataflows = discover_dataflows()
    
#     if not dataflows:
#         print("❌ Could not discover dataflows")
#         return
    
#     # Step 2: Find occupation-related dataflows
#     occ_dataflows = find_occupation_dataflows(dataflows)
    
#     if not occ_dataflows:
#         print("❌ No occupation dataflows found")
#         return
    
#     # Step 3: Examine each dataflow's structure
#     print("\n" + "="*80)
#     print("EXAMINING DATAFLOW STRUCTURES")
#     print("="*80)
    
#     dataflow_info = []
#     for df in occ_dataflows:
#         codelists = get_dataflow_structure(df['id'])
#         has_data = test_dataflow_data_availability(df['id'])
        
#         dataflow_info.append({
#             'id': df['id'],
#             'name': df['name'],
#             'has_data': has_data,
#             'codelists': codelists
#         })
        
#         time.sleep(1)  # Rate limiting
    
#     # Step 4: Present options to user
#     print("\n" + "="*80)
#     print("AVAILABLE DATAFLOWS FOR DOWNLOAD")
#     print("="*80)
    
#     available = [info for info in dataflow_info if info['has_data']]
    
#     if not available:
#         print("❌ No dataflows with available data found")
#         return
    
#     print(f"\nFound {len(available)} dataflows with data:\n")
#     for i, info in enumerate(available, 1):
#         print(f"{i}. {info['id']}")
#         print(f"   Name: {info['name']}")
#         if info['codelists']:
#             for cl in info['codelists']:
#                 print(f"   → {cl['id']}: {cl['num_codes']} occupation codes")
#         print()
    
#     # Let user choose
#     print("Options:")
#     print("1. Download ALL available dataflows (comprehensive)")
#     print("2. Select specific dataflow(s)")
#     print("3. Download most detailed dataflow only")
    
#     choice = input("\nSelect option (1-3) [default: 3]: ").strip() or "3"
    
#     if choice == "1":
#         # Download all
#         for info in available:
#             download_by_country_chunks(info['id'])
#             time.sleep(5)
    
#     elif choice == "2":
#         # Select specific
#         selection = input(f"Enter dataflow numbers (comma-separated, 1-{len(available)}): ").strip()
#         try:
#             indices = [int(x.strip())-1 for x in selection.split(",")]
#             for idx in indices:
#                 if 0 <= idx < len(available):
#                     download_by_country_chunks(available[idx]['id'])
#                     time.sleep(5)
#         except:
#             print("Invalid selection")
    
#     elif choice == "3":
#         # Find most detailed (most occupation codes)
#         most_detailed = max(available, 
#                           key=lambda x: max([cl['num_codes'] for cl in x['codelists']] or [0]))
        
#         print(f"\n📥 Downloading most detailed dataflow: {most_detailed['id']}")
#         download_by_country_chunks(most_detailed['id'])
    
#     print("\n✅ Download process complete!")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
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
    print("🌍 Discovering all available countries from ILOSTAT...")
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
    print(f"✅ Discovered {len(countries)} countries")
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

    print(f"\n📥 Downloading {dataflow_id}")
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
                    print(f"   ✅ {country}")
                else:
                    print(f"   ⚠️  {country} (no data)")
            except Exception as e:
                print(f"   ❌ {country}: {str(e)[:60]}")

            time.sleep(0.2)  # be nice to the API

    if not all_dfs:
        print("❌ No data retrieved")
        return None, None

    print("\n🔄 Combining results...")
    final_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"{dataflow_id}_{start_year}_{end_year}_{ts}.csv"
    final_df.to_csv(out_path, index=False)

    print("\n✅ Download complete")
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


# python 2_Download_ILOSTAT_Data.py --dataflow DF_EES_TEES_SEX_OC2_NB --start-year 2015 --end-year 2026

# python 2_Download_ILOSTAT_Data.py -> 2 -> DF_EES_TEES_SEX_OC2_NB (5)