# #!/usr/bin/env python3
# """
# Quick Start Script - Get ILOSTAT Gender-Occupation Data Fast
# This script downloads gender breakdown by occupation data with minimal configuration

# UPDATED: Fixed DNS resolution issues and added better error handling
# """

# import requests
# import pandas as pd
# import json
# from datetime import datetime
# import socket
# import time

# # Base URL options - try both if one fails
# BASE_URLS = [
#     "https://sdmx.ilo.org/rest",      # New URL (as of 2024)
#     "https://www.ilo.org/sdmx/rest"    # Legacy URL (may still work)
# ]

# def test_dns_and_connection():
#     """Test DNS resolution and connection to ILOSTAT servers"""
#     print("\n🔍 Testing connectivity to ILOSTAT servers...")
    
#     for base_url in BASE_URLS:
#         hostname = base_url.replace("https://", "").split("/")[0]
#         print(f"\n   Testing: {hostname}")
        
#         # Test DNS resolution
#         try:
#             ip = socket.gethostbyname(hostname)
#             print(f"   ✅ DNS resolved: {hostname} -> {ip}")
            
#             # Test HTTPS connection
#             try:
#                 response = requests.get(f"{base_url}/dataflow/ILO", timeout=10)
#                 print(f"   ✅ Connection successful (Status: {response.status_code})")
#                 return base_url  # Return the working URL
#             except Exception as e:
#                 print(f"   ⚠️  Connection failed: {e}")
                
#         except socket.gaierror:
#             print(f"   ❌ DNS resolution failed for {hostname}")
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
    
#     return None


# def quick_query(countries="USA+GBR+DEU+FRA+JPN", start_year="2020", end_year="2023", 
#                 base_url=None, use_alternative_dataflow=False):
#     """
#     Quick query to get employment data by gender and occupation
    
#     Args:
#         countries: Plus-separated country codes (e.g., "USA+GBR+DEU")
#         start_year: Start year (e.g., "2020")
#         end_year: End year (e.g., "2023")
#         base_url: Base URL to use (if None, will auto-detect)
#         use_alternative_dataflow: If True, try alternative dataflows
#     """
    
#     # Auto-detect working base URL if not provided
#     if base_url is None:
#         base_url = test_dns_and_connection()
#         if base_url is None:
#             print("\n❌ Cannot connect to any ILOSTAT servers.")
#             print("   This could be due to:")
#             print("   1. Network/firewall blocking HTTPS connections")
#             print("   2. DNS resolution issues")
#             print("   3. ILOSTAT servers temporarily unavailable")
#             print("\n💡 Troubleshooting steps:")
#             print("   1. Check your internet connection")
#             print("   2. Try disabling VPN if you're using one")
#             print("   3. Check if your firewall/antivirus is blocking Python")
#             print("   4. Try running: ping sdmx.ilo.org")
#             print("   5. Try accessing https://sdmx.ilo.org/rest/dataflow/ILO in your browser")
#             return None
    
#     # Primary dataflow: Employment by Sex and Occupation
#     dataflows = [
#         "DF_EMP_TEMP_SEX_OCU_NB",      # Primary: Employment by Sex and Occupation
#         "DF_EMP_TEMP_SEX_AGE_NB",      # Alternative: Employment by Sex and Age
#     ]
    
#     dataflow_id = dataflows[1] if use_alternative_dataflow else dataflows[0]
    
#     # Build URL
#     filters = f"{countries}.A..."  # Annual data, all other dimensions wildcarded
    
#     url = f"{base_url}/data/ILO,{dataflow_id}/{filters}"
#     params = {
#         'format': 'jsondata',
#         'startPeriod': start_year,
#         'endPeriod': end_year
#     }
    
#     print(f"\n🌍 Querying ILOSTAT for gender-occupation data...")
#     print(f"   Base URL: {base_url}")
#     print(f"   Dataflow: {dataflow_id}")
#     print(f"   Countries: {countries}")
#     print(f"   Period: {start_year}-{end_year}")
#     print(f"   Full URL: {url}")
#     print(f"   Params: {params}")
    
#     try:
#         print(f"\n⏳ Making request (this may take 30-120 seconds)...")
#         response = requests.get(url, params=params, timeout=120)
        
#         # Check for common error codes
#         if response.status_code == 404:
#             print(f"\n⚠️  Dataflow not found (404). Trying alternative dataflow...")
#             return quick_query(countries, start_year, end_year, base_url, True)
        
#         response.raise_for_status()
        
#         data = response.json()
#         print(f"✅ Data retrieved successfully!")
#         print(data)
        
#         # Parse to DataFrame
#         df = parse_sdmx_json(data)
        
#         if not df.empty:
#             print(f"\n📊 Data Summary:")
#             print(f"   Total observations: {len(df)}")
#             print(f"   Columns: {list(df.columns)}")
#             print(f"   Unique countries: {df['REF_AREA'].nunique() if 'REF_AREA' in df.columns else 'N/A'}")
#             print(f"   Date range: {df['TIME_PERIOD'].min() if 'TIME_PERIOD' in df.columns else 'N/A'} to {df['TIME_PERIOD'].max() if 'TIME_PERIOD' in df.columns else 'N/A'}")
            
#             # Save to CSV
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"gender_occupation_data_{timestamp}.csv"
#             df.to_csv(filename, index=False)
#             print(f"\n💾 Saved to: {filename}")
            
#             # Display sample
#             print(f"\n📋 Sample data (first 10 rows):")
#             print(df.head(10).to_string())
            
#             return df
#         else:
#             print("⚠️  No data found in response")
#             print("   The query executed successfully but returned no data.")
#             print("   This might mean:")
#             print("   - No data available for these countries/years")
#             print("   - The dataflow doesn't contain the requested dimensions")
#             return None
            
#     except requests.exceptions.Timeout:
#         print("❌ Request timed out after 120 seconds.")
#         print("   Try:")
#         print("   - Fewer countries")
#         print("   - Shorter time period")
#         print("   - Running the query again (server might be slow)")
#         return None
#     except requests.exceptions.ConnectionError as e:
#         print(f"❌ Connection Error: {e}")
#         print("   The connection to the server failed.")
#         print("   This could be due to network issues or the server being down.")
#         return None
#     except requests.exceptions.HTTPError as e:
#         print(f"❌ HTTP Error: {e}")
#         print(f"   Status Code: {response.status_code}")
#         if response.status_code == 500:
#             print("   Server error. The ILOSTAT server encountered an error.")
#             print("   Try again later or with different parameters.")
#         return None
#     except json.JSONDecodeError as e:
#         print(f"❌ JSON Decode Error: {e}")
#         print("   The server returned invalid JSON.")
#         print(f"   Response preview: {response.text[:500]}")
#         return None
#     except Exception as e:
#         print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
#         return None


# def parse_sdmx_json(json_data):
#     """Parse SDMX JSON format to DataFrame"""
    
#     try:
#         # Check if data structure exists
#         if 'data' not in json_data:
#             print("⚠️  Response doesn't contain 'data' field")
#             print(f"   Available keys: {list(json_data.keys())}")
#             return pd.DataFrame()
            
#         structure = json_data['data']['structure']
#         dataset = json_data['data']['dataSets'][0]
        
#         # Get dimension info
#         series_dims = structure['dimensions']['series']
#         obs_dims = structure['dimensions']['observation']
        
#         dim_names = [d['id'] for d in series_dims]
#         dim_values = {d['id']: d['values'] for d in series_dims}
        
#         time_values = obs_dims[0]['values']
#         time_name = obs_dims[0]['id']
        
#         # Parse series
#         rows = []
#         for series_key, series_info in dataset['series'].items():
#             indices = [int(i) for i in series_key.split(':')]
            
#             # Build dimension values
#             row_dims = {}
#             for i, dim_name in enumerate(dim_names):
#                 idx = indices[i]
#                 row_dims[dim_name] = dim_values[dim_name][idx]['id']
            
#             # Add observations
#             for obs_idx, obs_value in series_info['observations'].items():
#                 row = row_dims.copy()
#                 row[time_name] = time_values[int(obs_idx)]['id']
#                 row['VALUE'] = obs_value[0]
#                 rows.append(row)
        
#         df = pd.DataFrame(rows)
#         print(f"   Successfully parsed {len(rows)} observations")
#         return df
        
#     except KeyError as e:
#         print(f"⚠️  Error parsing data - missing key: {e}")
#         print("   The response structure might be different than expected")
#         return pd.DataFrame()
#     except Exception as e:
#         print(f"⚠️  Error parsing data: {type(e).__name__}: {e}")
#         return pd.DataFrame()


# def get_all_countries_data(start_year="2022", end_year="2023"):
#     """
#     Get data for ALL countries (warning: large dataset)
#     """
#     print("⚠️  WARNING: This will download a large dataset for all countries")
#     print("   This may take several minutes and could fail due to size.")
#     confirm = input("Continue? (yes/no): ")
    
#     if confirm.lower() != 'yes':
#         print("Cancelled.")
#         return None
    
#     return quick_query(countries="ALL", start_year=start_year, end_year=end_year)


# def get_regional_data(region_code="X01_COU", start_year="2020", end_year="2023"):
#     """
#     Get data for a specific region
    
#     Region codes:
#     - X01_COU: World (all countries expanded)
#     - X85_COU: BRICS countries
#     - Use country group codes with _COU suffix
#     """
#     print(f"🌍 Querying region: {region_code}")
#     return quick_query(countries=region_code, start_year=start_year, end_year=end_year)


# def list_available_dataflows(base_url=None):
#     """List all available dataflows from ILOSTAT"""
#     if base_url is None:
#         base_url = test_dns_and_connection()
#         if base_url is None:
#             return None
    
#     print(f"\n📋 Fetching available dataflows from {base_url}...")
    
#     try:
#         url = f"{base_url}/dataflow/ILO"
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
        
#         print("✅ Successfully retrieved dataflow list")
#         print(f"   Response size: {len(response.text)} bytes")
        
#         # Save to file for inspection
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"ilostat_dataflows_{timestamp}.xml"
#         with open(filename, 'w', encoding='utf-8') as f:
#             f.write(response.text)
#         print(f"   Saved to: {filename}")
        
#         return response.text
        
#     except Exception as e:
#         print(f"❌ Error fetching dataflows: {e}")
#         return None


# if __name__ == "__main__":
#     print("=" * 80)
#     print("ILOSTAT QUICK START - Gender by Occupation Data")
#     print("=" * 80)
    
#     print("\nOptions:")
#     print("1. Sample countries (USA, GBR, DEU, FRA, JPN) - 2020-2023")
#     print("2. Custom countries")
#     print("3. All countries (large download)")
#     print("4. BRICS countries")
#     print("5. Test connectivity & list available dataflows")
    
#     choice = input("\nSelect option (1-5) [default: 1]: ").strip() or "1"
    
#     if choice == "1":
#         df = quick_query()
#     elif choice == "2":
#         countries = input("Enter country codes (e.g., USA+GBR+DEU): ").strip()
#         start = input("Start year [2020]: ").strip() or "2020"
#         end = input("End year [2023]: ").strip() or "2023"
#         df = quick_query(countries=countries, start_year=start, end_year=end)
#     elif choice == "3":
#         df = get_all_countries_data()
#     elif choice == "4":
#         df = get_regional_data("X85_COU")
#     elif choice == "5":
#         list_available_dataflows()
#         df = None
#     else:
#         print("Invalid choice")
#         df = None
    
#     if df is not None and not df.empty:
#         print("\n✅ Complete! Your data is ready for analysis.")
#         print(f"\n📁 Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
#     elif df is not None and df.empty:
#         print("\n⚠️  Query completed but no data was returned.")
#     else:
#         print("\n❌ Query failed or was cancelled.")


#!/usr/bin/env python3
"""
Quick Start Script - Get ILOSTAT Gender-Occupation Data Fast
This script downloads gender breakdown by occupation data with minimal configuration

UPDATED: Fixed DNS resolution issues and added better error handling
"""

import requests
import pandas as pd
import json
from datetime import datetime
import socket
import time

# Base URL options - try both if one fails
BASE_URLS = [
    "https://sdmx.ilo.org/rest",      # New URL (as of 2024)
    "https://www.ilo.org/sdmx/rest"    # Legacy URL (may still work)
]

def test_dns_and_connection():
    """Test DNS resolution and connection to ILOSTAT servers"""
    print("\n🔍 Testing connectivity to ILOSTAT servers...")
    
    for base_url in BASE_URLS:
        hostname = base_url.replace("https://", "").split("/")[0]
        print(f"\n   Testing: {hostname}")
        
        # Test DNS resolution
        try:
            ip = socket.gethostbyname(hostname)
            print(f"   ✅ DNS resolved: {hostname} -> {ip}")
            
            # Test HTTPS connection
            try:
                response = requests.get(f"{base_url}/dataflow/ILO", timeout=10)
                print(f"   ✅ Connection successful (Status: {response.status_code})")
                return base_url  # Return the working URL
            except Exception as e:
                print(f"   ⚠️  Connection failed: {e}")
                
        except socket.gaierror:
            print(f"   ❌ DNS resolution failed for {hostname}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None


def quick_query(countries="USA+GBR+DEU+FRA+JPN", start_year="2020", end_year="2023", 
                base_url=None, use_alternative_dataflow=False):
    """
    Quick query to get employment data by gender and occupation
    
    Args:
        countries: Plus-separated country codes (e.g., "USA+GBR+DEU")
        start_year: Start year (e.g., "2020")
        end_year: End year (e.g., "2023")
        base_url: Base URL to use (if None, will auto-detect)
        use_alternative_dataflow: If True, try alternative dataflows
    """
    
    # Auto-detect working base URL if not provided
    if base_url is None:
        base_url = test_dns_and_connection()
        if base_url is None:
            print("\n❌ Cannot connect to any ILOSTAT servers.")
            print("   This could be due to:")
            print("   1. Network/firewall blocking HTTPS connections")
            print("   2. DNS resolution issues")
            print("   3. ILOSTAT servers temporarily unavailable")
            print("\n💡 Troubleshooting steps:")
            print("   1. Check your internet connection")
            print("   2. Try disabling VPN if you're using one")
            print("   3. Check if your firewall/antivirus is blocking Python")
            print("   4. Try running: ping sdmx.ilo.org")
            print("   5. Try accessing https://sdmx.ilo.org/rest/dataflow/ILO in your browser")
            return None
    
    # Primary dataflow: Employment by Sex and Occupation
    dataflows = [
        "DF_EMP_TEMP_SEX_OCU_NB",      # Primary: Employment by Sex and Occupation
        "DF_EMP_TEMP_SEX_AGE_NB",      # Alternative: Employment by Sex and Age
    ]
    
    dataflow_id = dataflows[1] if use_alternative_dataflow else dataflows[0]
    
    # Build URL
    filters = f"{countries}.A..."  # Annual data, all other dimensions wildcarded
    
    url = f"{base_url}/data/ILO,{dataflow_id}/{filters}"
    params = {
        'format': 'jsondata',
        'startPeriod': start_year,
        'endPeriod': end_year
    }
    
    print(f"\n🌍 Querying ILOSTAT for gender-occupation data...")
    print(f"   Base URL: {base_url}")
    print(f"   Dataflow: {dataflow_id}")
    print(f"   Countries: {countries}")
    print(f"   Period: {start_year}-{end_year}")
    print(f"   Full URL: {url}")
    print(f"   Params: {params}")
    
    try:
        print(f"\n⏳ Making request (this may take 30-120 seconds)...")
        response = requests.get(url, params=params, timeout=120)
        
        # Check for common error codes
        if response.status_code == 404:
            print(f"\n⚠️  Dataflow not found (404). Trying alternative dataflow...")
            return quick_query(countries, start_year, end_year, base_url, True)
        
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Data retrieved successfully!")
        
        # Parse to DataFrame
        df = parse_sdmx_json(data)
        
        if not df.empty:
            print(f"\n📊 Data Summary:")
            print(f"   Total observations: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Unique countries: {df['REF_AREA'].nunique() if 'REF_AREA' in df.columns else 'N/A'}")
            print(f"   Date range: {df['TIME_PERIOD'].min() if 'TIME_PERIOD' in df.columns else 'N/A'} to {df['TIME_PERIOD'].max() if 'TIME_PERIOD' in df.columns else 'N/A'}")
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gender_occupation_data_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\n💾 Saved to: {filename}")
            
            # Display sample
            print(f"\n📋 Sample data (first 10 rows):")
            print(df.head(10).to_string())
            
            return df
        else:
            print("⚠️  No data found in response")
            print("   The query executed successfully but returned no data.")
            print("   This might mean:")
            print("   - No data available for these countries/years")
            print("   - The dataflow doesn't contain the requested dimensions")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 120 seconds.")
        print("   Try:")
        print("   - Fewer countries")
        print("   - Shorter time period")
        print("   - Running the query again (server might be slow)")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("   The connection to the server failed.")
        print("   This could be due to network issues or the server being down.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 500:
            print("   Server error. The ILOSTAT server encountered an error.")
            print("   Try again later or with different parameters.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print("   The server returned invalid JSON.")
        print(f"   Response preview: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
        return None


def parse_sdmx_json(json_data):
    """Parse SDMX JSON format to DataFrame with enriched labels"""
    
    try:
        # Check if data structure exists
        if 'data' not in json_data:
            print("⚠️  Response doesn't contain 'data' field")
            print(f"   Available keys: {list(json_data.keys())}")
            return pd.DataFrame()
        
        # ILOSTAT returns structures as an array, not a single object
        if 'structures' in json_data['data']:
            structure = json_data['data']['structures'][0]
        elif 'structure' in json_data['data']:
            structure = json_data['data']['structure']
        else:
            print("⚠️  Cannot find 'structure' or 'structures' in response")
            print(f"   Available data keys: {list(json_data['data'].keys())}")
            return pd.DataFrame()
            
        dataset = json_data['data']['dataSets'][0]
        
        # Get dimension info
        series_dims = structure['dimensions']['series']
        obs_dims = structure['dimensions']['observation']
        
        dim_names = [d['id'] for d in series_dims]
        dim_values = {d['id']: d['values'] for d in series_dims}
        
        # Create label mappings for better readability
        label_mappings = {}
        for dim in series_dims:
            dim_id = dim['id']
            label_mappings[dim_id] = {
                val['id']: val.get('name', val['id']) 
                for val in dim['values']
            }
        
        time_values = obs_dims[0]['values']
        time_name = obs_dims[0]['id']
        
        # Parse series
        rows = []
        for series_key, series_info in dataset['series'].items():
            indices = [int(i) for i in series_key.split(':')]
            
            # Build dimension values (both codes and labels)
            row_dims = {}
            for i, dim_name in enumerate(dim_names):
                idx = indices[i]
                code = dim_values[dim_name][idx]['id']
                row_dims[dim_name] = code
                # Add label column
                row_dims[f"{dim_name}_LABEL"] = label_mappings[dim_name].get(code, code)
            
            # Add observations
            for obs_idx, obs_value in series_info['observations'].items():
                row = row_dims.copy()
                row[time_name] = time_values[int(obs_idx)]['id']
                row['VALUE'] = obs_value[0]
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns for better readability
        # Put label columns after their corresponding code columns
        ordered_cols = []
        for dim_name in dim_names:
            ordered_cols.append(dim_name)
            if f"{dim_name}_LABEL" in df.columns:
                ordered_cols.append(f"{dim_name}_LABEL")
        ordered_cols.extend([time_name, 'VALUE'])
        
        df = df[ordered_cols]
        
        print(f"   Successfully parsed {len(rows)} observations")
        return df
        
    except KeyError as e:
        print(f"⚠️  Error parsing data - missing key: {e}")
        print("   The response structure might be different than expected")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Error parsing data: {type(e).__name__}: {e}")
        return pd.DataFrame()


def get_all_countries_data(start_year="2022", end_year="2023"):
    """
    Get data for ALL countries (warning: large dataset)
    """
    print("⚠️  WARNING: This will download a large dataset for all countries")
    print("   This may take several minutes and could fail due to size.")
    confirm = input("Continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return None
    
    return quick_query(countries="ALL", start_year=start_year, end_year=end_year)


def get_regional_data(region_code="X01_COU", start_year="2020", end_year="2023"):
    """
    Get data for a specific region
    
    Region codes:
    - X01_COU: World (all countries expanded)
    - X85_COU: BRICS countries
    - Use country group codes with _COU suffix
    """
    print(f"🌍 Querying region: {region_code}")
    return quick_query(countries=region_code, start_year=start_year, end_year=end_year)


def list_available_dataflows(base_url=None):
    """List all available dataflows from ILOSTAT"""
    if base_url is None:
        base_url = test_dns_and_connection()
        if base_url is None:
            return None
    
    print(f"\n📋 Fetching available dataflows from {base_url}...")
    
    try:
        url = f"{base_url}/dataflow/ILO"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print("✅ Successfully retrieved dataflow list")
        print(f"   Response size: {len(response.text)} bytes")
        
        # Save to file for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ilostat_dataflows_{timestamp}.xml"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"   Saved to: {filename}")
        
        return response.text
        
    except Exception as e:
        print(f"❌ Error fetching dataflows: {e}")
        return None


if __name__ == "__main__":
    print("=" * 80)
    print("ILOSTAT QUICK START - Gender by Occupation Data")
    print("=" * 80)
    
    print("\nOptions:")
    print("1. Sample countries (USA, GBR, DEU, FRA, JPN) - 2020-2023")
    print("2. Custom countries")
    print("3. All countries (large download)")
    print("4. BRICS countries")
    print("5. Test connectivity & list available dataflows")
    
    choice = input("\nSelect option (1-5) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        df = quick_query()
    elif choice == "2":
        countries = input("Enter country codes (e.g., USA+GBR+DEU): ").strip()
        start = input("Start year [2020]: ").strip() or "2020"
        end = input("End year [2023]: ").strip() or "2023"
        df = quick_query(countries=countries, start_year=start, end_year=end)
    elif choice == "3":
        df = get_all_countries_data()
    elif choice == "4":
        df = get_regional_data("X85_COU")
    elif choice == "5":
        list_available_dataflows()
        df = None
    else:
        print("Invalid choice")
        df = None
    
    if df is not None and not df.empty:
        print("\n✅ Complete! Your data is ready for analysis.")
        print(f"\n📁 Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
    elif df is not None and df.empty:
        print("\n⚠️  Query completed but no data was returned.")
    else:
        print("\n❌ Query failed or was cancelled.")