"""
Enhanced BLS Occupation Matching with Fuzzy Matching
======================================================

This script improves upon the basic matching by using multiple fuzzy matching strategies
to find the best BLS occupation for each profession.

Usage:
    python PracticalBLSDemographics.py <path_to_bls_excel_file>
    
Or if the file is named cpsaat11.xlsx or cpsaat11_2024.xlsx in the current directory:
    python PracticalBLSDemographics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    print("Warning: fuzzywuzzy not installed. Using basic matching only.")
    print("Install with: pip install fuzzywuzzy python-Levenshtein")
    FUZZY_AVAILABLE = False

# Your profession list
PROFESSION_LIST = ['Accountant', 'Actor', 'Actuary', 'Administrative Assistant',
 'Administrator', 'Air Traffic Controller', 'Animal Trainer', 'Anthropologist',
 'Appraiser', 'Archaeologist', 'Architect', 'Archivist', 'Art Director',
 'Artist', 'Astronaut', 'Astronomer', 'Athlete', 'Audio Technician', 'Auditor',
 'Automotive Designer', 'Baker', 'Banker', 'Bankruptcy Specialist', 'Barber',
 'Barista', 'Bartender', 'Basketball player', 'Biologist', 'Biomedical Engineer',
 'Blacksmith', 'Bodyguard', 'Bounty Hunter', 'Boxer', 'Brand Manager', 'Brewer',
 'Bricklayer', 'Broker', 'Builder', 'Butcher', 'CEO', 'Carer', 'Carpenter',
 'Cartographer', 'Cashier', 'Chef', 'Chemical Engineer', 'Chemist', 'Chiropractor',
 'Civil Engineer', 'Claims Adjuster', 'Cleaner', 'Clerk', 'Coach', 'Comedian',
 'Compliance Officer', 'Composer', 'Conservation Officer', 'Construction Worker',
 'Copywriter', 'Court Reporter', 'Crime Scene Investigator',
 'Customer Support Specialist', 'DJ', 'Dancer', 'Data Scientist',
 'Database Administrator', 'Debt Counselor', 'Dentist', 'Detective',
 'Development Officer', 'Dietitian', 'Director', 'Doctor', 'Dog Walker',
 'Draughtsperson', 'Driver', 'Economist', 'Editor', 'Electrician',
 'Emergency Management Specialist', 'Entrepreneur', 'Environmental Engineer',
 'Ergonomist', 'Estate Planner', 'Event Coordinator', 'Executive Assistant',
 'Exterminator', 'Facilities Manager', 'Farmer', 'Fashion Designer',
 'Firefighter', 'Fishmonger', 'Flight Attendant', 'Florist', 'Football player',
 'Forklift Operator', 'Gardener', 'Geologist', 'Graphic Designer', 'Grocer',
 'Hair dresser', 'Handyperson', 'Health Inspector', 'Historian',
 'Hotel Concierge', 'Hotel Manager', 'Human Resources Specialist',
 'IT Support Specialist', 'Illustrator', 'Industrial Designer',
 'Insurance Underwriter', 'Janitor', 'Jeweller', 'Journalist', 'Judge', 'Lawyer',
 'Librarian', 'Lifeguard', 'Loan Officer', 'Logger', 'Logistics Manager',
 'Magician', 'Makeup Artist', 'Marine Biologist', 'Marketing Manager', 'Masseur',
 'Mathematician', 'Mayor', 'Mechanic', 'Meteorologist', 'Midwife', 'Miner',
 'Model', 'Musician', 'News Reader', 'Nurse', 'Nutritionist', 'Oceanographer',
 'Office Assistant', 'Operations Manager', 'Optician', 'Painter', 'Paralegal',
 'Paramedic', 'Park Ranger', 'Payroll Specialist', 'Personal Trainer',
 'Pharmacist', 'Photographer', 'Physicist', 'Pilot', 'Plumber', 'Police Officer',
 'Politician', 'Postal Worker', 'Priest', 'Procurement Officer', 'Professor',
 'Property Manager', 'Psychologist', 'Quality Assurance Inspector',
 'Real Estate Agent', 'Receptionist', 'Researcher', 'Roofer', 'Safety Inspector',
 'Sailor', 'Salesperson', 'Scientist', 'Security Officer', 'Shopkeeper', 'Singer',
 'Skier', 'Social Worker', 'Software Engineer', 'Soldier', 'Sound Engineer',
 'Statistician', 'Street Vendor', 'Surfer', 'Surgeon', 'Swimmer', 'Tailor',
 'Tattoo Artist', 'Teacher', 'Technician', 'Tennis Player', 'Therapist',
 'Translator', 'Umpire', 'Urban Planner', 'Usher', 'Veterinarian', 'Videographer',
 'Waiter', 'Waste Collection Worker', 'Welder', 'Wholesaler', 'Writer',
 'Zoologist']

# Manual authoritative mappings (CPS-aligned)
MAPPINGS = {
    "Actuary": "Actuaries",
    "Anthropologist": "Miscellaneous social scientists and related workers",
    "Archaeologist": "Miscellaneous social scientists and related workers",
    "Art Director": "Artists and related workers",
    "Astronaut": "Aircraft pilots and flight engineers",
    "Audio Technician": "Broadcast, sound, and lighting technicians",
    "Automotive Designer": "Commercial and industrial designers",
    "Banker": "Financial managers",
    "Bankruptcy Specialist": "Lawyers",
    "Barista": "Fast food and counter workers",
    "Biologist": "Biological scientists",
    "Blacksmith": "Other metal workers and plastic workers",
    "Bodyguard": "Other protective service workers",
    "Bounty Hunter": "Private detectives and investigators",
    "Boxer": "Athletes and sports competitors",
    "Brand Manager": "Marketing managers",
    "Brewer": "Food processing workers, all other",
    "Bricklayer": "Brickmasons, blockmasons, and stonemasons",
    "Comedian": "Entertainers and performers, sports and related workers, all other",
    "Conservation Officer": "Conservation scientists and foresters",
    "Construction Worker": "Construction laborers",
    "Copywriter": "Writers and authors",
    "Crime Scene Investigator": "Detectives and criminal investigators",
    "Customer Support Specialist": "Customer service representatives",
    "Data Scientist": "Computer and information research scientists",
    "Debt Counselor": "Credit counselors and loan officers",
    "Development Officer": "Fundraisers",
    "Dog Walker": "Animal caretakers",
    "Draughtsperson": "Architectural and civil drafters",
    "Entrepreneur": "Chief executives",
    "Ergonomist": "Industrial engineers, including health and safety",
    "Estate Planner": "Personal financial advisors",
    "Event Coordinator": "Meeting, convention, and event planners",
    "Executive Assistant": "Executive secretaries and executive administrative assistants",
    "Exterminator": "Pest control workers",
    "Fishmonger": "Butchers and other meat, poultry, and fish processing workers",
    "Florist": "Floral designers",
    "Forklift Operator": "Industrial truck and tractor operators",
    "Gardener": "Landscaping and groundskeeping workers",
    "Geologist": "Geoscientists and hydrologists, except geographers",
    "Health Inspector": "Occupational health and safety specialists and technicians",
    "Historian": "Miscellaneous social scientists and related workers",
    "Hotel Concierge": "Baggage porters, bellhops, and concierges",
    "Hotel Manager": "Lodging managers",
    "Illustrator": "Artists and related workers",
    "Jeweller": "Jewelers and precious stone and metal workers",
    "Lifeguard": "Other protective service workers",
    "Logger": "Logging workers",
    "Logistics Manager": "Transportation, storage, and distribution managers",
    "Magician": "Entertainers and performers, sports and related workers, all other",
    "Makeup Artist": "Other personal appearance workers",
    "Marine Biologist": "Biological scientists",
    "Mayor": "Legislators",
    "Meteorologist": "Atmospheric and space scientists",
    "Midwife": "Nurse midwives",
    "Oceanographer": "Geoscientists and hydrologists, except geographers",
    "Office Assistant": "Office clerks, general",
    "Park Ranger": "Forest and conservation workers",
    "Payroll Specialist": "Payroll and timekeeping clerks",
    "Personal Trainer": "Exercise trainers and group fitness instructors",
    "Politician": "Legislators",
    "Postal Worker": "Postal service mail carriers",
    "Priest": "Clergy",
    "Procurement Officer": "Purchasing agents, except wholesale, retail, and farm products",
    "Professor": "Postsecondary teachers",
    "Property Manager": "Property, real estate, and community association managers",
    "Quality Assurance Inspector": "Inspectors, testers, sorters, samplers, and weighers",
    "Real Estate Agent": "Real estate brokers and sales agents",
    "Safety Inspector": "Occupational health and safety specialists and technicians",
    "Sailor": "Sailors and marine oilers",
    "Security Officer": "Security guards and gambling surveillance officers",
    "Shopkeeper": "Retail salespersons",
    "Skier": "Athletes and sports competitors",
    "Soldier": None,
    "Sound Engineer": "Broadcast, sound, and lighting technicians",
    "Surfer": "Athletes and sports competitors",
    "Swimmer": "Athletes and sports competitors",
    "Tattoo Artist": "Other personal appearance workers",
    "Urban Planner": "Urban and regional planners",
    "Videographer": "Television, video, and film camera operators and editors",
    "Waiter": "Waiters and waitresses",
    "Waste Collection Worker": "Refuse and recyclable material collectors",
    "Welder": "Welding, soldering, and brazing workers",
    "Wholesaler": "Sales representatives, wholesale and manufacturing",
    "Zoologist": "Biological scientists",
    "Actor": "Actors",
    "Artist": "Artists and related workers",
    "Builder": "Construction managers",
    "Clerk": "Office clerks, general",
    "DJ": "Disc jockeys, except radio",
    "Director": "Producers and directors",
    "Driver": "Driver/sales workers and truck drivers",
    "Doctor": "Other physicians",
    "Janitor": "Janitors and building cleaners",
    "Mechanic": "Automotive service technicians and mechanics",
    "Miner": "Underground mining machine operators",
    "Salesperson": "Retail salespersons",
    "Writer": "Writers and authors",
    "Athlete": "Athletes and sports competitors",
    "Basketball player": "Athletes and sports competitors",
    "Football player": "Athletes and sports competitors",
    "Tennis player": "Athletes and sports competitors",
    "Administrator": "Administrative services managers",
    "Architect": "Architects, except landscape and naval",
    "Detective": "Detectives and criminal investigators",
    "Human Resources Specialist": "Human resources workers",
    "Social Worker": "Social workers, all other",
    "Teacher": "Elementary and middle school teachers",
    "Researcher": None,   # "Survey researchers" (1k workers) is wrong; no clean BLS equivalent
    "Scientist": None,    # Too broad; BLS splits by domain
    "Therapist": "Therapists, all other",
    "Technician": "Other engineering technologists and technicians, except drafters",
    "Handyperson": "Maintenance and repair workers, general",
    "Broker": "Securities, commodities, and financial services sales agents",
    "Tennis Player": "Athletes and sports competitors",
}

def parse_bls_excel(file_path: str) -> pd.DataFrame:
    """
    Parse CPS Table 11 (ASEC) Excel file into a clean, flat dataframe.
    Robust to merged cells and blank header columns.
    """

    df = pd.read_excel(file_path, header=None)

    # CPS Table 11 layout (verified)
    HEADER_ROW = 5   # Excel row 6 (0-based)
    DATA_START = 6   # Excel row 7

    # Slice data
    df = df.iloc[DATA_START:].reset_index(drop=True)

    # Explicit column assignment by POSITION (not header text)
    df.columns = [
        "occupation",
        "total_employed_2024",
        "pct_women",
        "pct_white",
        "pct_black",
        "pct_asian",
        "pct_hispanic",
    ] + [f"extra_{i}" for i in range(len(df.columns) - 7)]

    # Drop rows without occupation labels
    df = df[df["occupation"].notna()]

    # Drop aggregate/category rows
    df = df[~df["occupation"].str.lower().str.contains(
        "total|occupations", na=False
    )]

    # Normalize CPS suppression symbol
    df = df.replace("–", pd.NA)

    # Convert numeric columns
    numeric_cols = [
        "total_employed_2024",
        "pct_women",
        "pct_white",
        "pct_black",
        "pct_asian",
        "pct_hispanic",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only meaningful columns
    df = df[["occupation"] + numeric_cols]

    return df

def find_exact_match(profession: str, occupation_list: List[str]) -> Optional[str]:
    """Try exact matching (case-insensitive)"""
    prof_lower = profession.lower().strip()
    
    for occ in occupation_list:
        occ_lower = occ.lower().strip()
        
        # Exact match
        if prof_lower == occ_lower:
            return occ
        
        # Profession is in occupation (e.g., "Nurse" in "Registered nurses")
        if prof_lower in occ_lower:
            return occ
        
        # Occupation is in profession (rare but possible)
        if occ_lower in prof_lower and len(occ_lower) > 5:  # Avoid short matches
            return occ
    
    return None


def find_keyword_match(profession: str, occupation_list: List[str]) -> Optional[Tuple[str, int]]:
    """Find matches based on keywords"""
    prof_words = set(profession.lower().split())
    
    best_match = None
    best_score = 0
    
    for occ in occupation_list:
        occ_words = set(occ.lower().split())
        
        # Count common words
        common_words = prof_words & occ_words
        
        if common_words:
            # Score based on proportion of words matched
            score = len(common_words) / max(len(prof_words), len(occ_words))
            
            if score > best_score:
                best_score = score
                best_match = occ
    
    if best_score > 0.5:  # At least 50% word overlap
        return (best_match, int(best_score * 100))
    
    return None


def find_fuzzy_match(profession: str, occupation_list: List[str], threshold: int = 70) -> Optional[Tuple[str, int]]:
    """Use fuzzy matching to find best match"""
    if not FUZZY_AVAILABLE:
        return None
    
    # Try different fuzzy matching strategies
    results = []
    
    # Strategy 1: Token sort ratio (handles word order differences)
    match1 = process.extractOne(profession, occupation_list, scorer=fuzz.token_sort_ratio)
    if match1:
        results.append(('token_sort', match1[0], match1[1]))
    
    # Strategy 2: Partial ratio (handles partial matches)
    match2 = process.extractOne(profession, occupation_list, scorer=fuzz.partial_ratio)
    if match2:
        results.append(('partial', match2[0], match2[1]))
    
    # Strategy 3: Token set ratio (handles subset/superset)
    match3 = process.extractOne(profession, occupation_list, scorer=fuzz.token_set_ratio)
    if match3:
        results.append(('token_set', match3[0], match3[1]))
    
    # Find best result
    if results:
        best = max(results, key=lambda x: x[2])
        if best[2] >= threshold:
            return (best[1], best[2])
    
    return None


def normalize_profession_name(profession: str) -> str:
    """Normalize profession names for better matching"""
    # Common replacements
    replacements = {
        'hair dresser': 'hairdresser',
        'basketball player': 'athlete',
        'football player': 'athlete',
        'tennis player': 'athlete',
        'ceo': 'chief executive',
        'it support specialist': 'computer support specialist',
        'data scientist': 'data analyst',
        'software engineer': 'software developer',
        'news reader': 'broadcast announcer',
        'masseur': 'massage therapist',
        'waiter': 'waiter and waitress',
        'sailor': 'water transportation worker',
        'doctor': 'physician',
        'carer': 'personal care aide',
        'handyperson': 'maintenance worker',
        'grocer': 'retail salesperson',
    }
    
    prof_lower = profession.lower()
    return replacements.get(prof_lower, profession)


def match_profession_enhanced(
    profession: str,
    occupation_list: List[str],
    fuzzy_threshold: int = 70
) -> Dict:
    """
    Manual-first matching with automatic fallback.
    """

    # --------------------------------------------------
    # Strategy 0: Manual mapping (authoritative)
    # --------------------------------------------------
    if profession in MAPPINGS:
        mapped = MAPPINGS[profession]

        if mapped is None:
            return {
                "matched": False,
                "bls_occupation": None,
                "match_method": "manual_none",
                "mapping_source": "manual",
                "match_score": 0,
            }

        if mapped in occupation_list:
            return {
                "matched": True,
                "bls_occupation": mapped,
                "match_method": "manual",
                "mapping_source": "manual",
                "match_score": 100,
            }

        return {
            "matched": False,
            "bls_occupation": mapped,
            "match_method": "manual_missing_in_bls",
            "mapping_source": "manual",
            "match_score": 0,
        }

    # --------------------------------------------------
    # Strategy 1+: Automatic matching (non-manual jobs)
    # --------------------------------------------------
    normalized = normalize_profession_name(profession)

    exact = find_exact_match(normalized, occupation_list)
    if exact:
        return {
            "matched": True,
            "bls_occupation": exact,
            "match_method": "exact",
            "mapping_source": "automatic",
            "match_score": 100,
        }

    keyword = find_keyword_match(normalized, occupation_list)
    if keyword:
        return {
            "matched": True,
            "bls_occupation": keyword[0],
            "match_method": "keyword",
            "mapping_source": "automatic",
            "match_score": keyword[1],
        }

    fuzzy = find_fuzzy_match(normalized, occupation_list, fuzzy_threshold)
    if fuzzy:
        return {
            "matched": True,
            "bls_occupation": fuzzy[0],
            "match_method": "fuzzy",
            "mapping_source": "automatic",
            "match_score": fuzzy[1],
        }

    return {
        "matched": False,
        "bls_occupation": None,
        "match_method": "none",
        "mapping_source": "automatic",
        "match_score": 0,
    }


def process_all_professions(
    profession_list: List[str],
    bls_df: pd.DataFrame,
    fuzzy_threshold: int = 70
) -> pd.DataFrame:
    """Process all professions with enhanced matching"""
    
    occupation_col = bls_df.columns[0]
    occupation_list = bls_df[occupation_col].tolist()
    
    results = []
    
    print(f"\nMatching {len(profession_list)} professions using enhanced fuzzy matching...")
    print(f"Fuzzy threshold: {fuzzy_threshold} (lower = more lenient)")
    
    for i, profession in enumerate(profession_list, 1):
        if i % 25 == 0:
            print(f"  Progress: {i}/{len(profession_list)}")
        
        # Get match
        match_result = match_profession_enhanced(
            profession,
            occupation_list,
            fuzzy_threshold
        )
        
        # Build result row
        row = {
            'profession': profession,
            'normalized_profession': normalize_profession_name(profession),
            **match_result
        }
        
        # Add demographic data if matched
        if match_result['matched']:
            occ_data = bls_df[bls_df[occupation_col] == match_result['bls_occupation']].iloc[0]
            # for col in bls_df.columns[1:]:
                # row[col] = occ_data[col]

            for col in bls_df.columns[1:]:
                val = occ_data[col]

                # If duplicate column produced a Series, collapse safely
                if isinstance(val, pd.Series):
                    val = val.iloc[0]

                row[col] = val

        
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    """Main execution"""
    
    print("="*70)
    print("ENHANCED BLS OCCUPATION MATCHER WITH FUZZY MATCHING")
    print("="*70)
    
    # Find BLS file
    possible_files = [
        'cpsaat11.xlsx',
        'cpsaat11_2024.xlsx',
        'table11.xlsx',
    ]
    
    bls_file = None
    if len(sys.argv) > 1:
        bls_file = sys.argv[1]
    else:
        for filename in possible_files:
            if Path(filename).exists():
                bls_file = filename
                break
    
    if not bls_file or not Path(bls_file).exists():
        print("\n✗ No BLS data file found!")
        print("\nPlease provide the file as an argument:")
        print("  python PracticalBLSDemographics.py <path_to_file.xlsx>")
        print("\nOr place one of these files in the current directory:")
        for f in possible_files:
            print(f"  - {f}")
        return
    
    # Parse BLS data
    print(f"\n{'='*70}")
    print("STEP 1: Loading BLS Data")
    print(f"{'='*70}")
    bls_df = parse_bls_excel(bls_file)
    
    # Process professions with different thresholds
    print(f"\n{'='*70}")
    print("STEP 2: Matching Professions")
    print(f"{'='*70}")
    
    # Try with threshold of 70 first
    results_df = process_all_professions(PROFESSION_LIST, bls_df, fuzzy_threshold=70)
    
    # Statistics
    print(f"\n{'='*70}")
    print("STEP 3: Results Summary")
    print(f"{'='*70}")
    
    total = len(results_df)
    matched = results_df['matched'].sum()
    exact = (results_df['match_method'] == 'exact').sum()
    keyword = (results_df['match_method'] == 'keyword').sum()
    fuzzy = (results_df['match_method'] == 'fuzzy').sum()
    unmatched = total - matched
    
    print(f"\nTotal professions:      {total}")
    print(f"Matched:                {matched} ({matched/total*100:.1f}%)")
    print(f"  - Exact matches:      {exact}")
    print(f"  - Keyword matches:    {keyword}")
    print(f"  - Fuzzy matches:      {fuzzy}")
    print(f"Unmatched:              {unmatched} ({unmatched/total*100:.1f}%)")
    
    # Show match quality distribution
    if matched > 0:
        matched_only = results_df[results_df['matched']]
        avg_score = matched_only['match_score'].mean()
        print(f"\nAverage match score:    {avg_score:.1f}/100")
        print(f"Score distribution:")
        print(f"  90-100 (Excellent):   {(matched_only['match_score'] >= 90).sum()}")
        print(f"  80-89  (Good):        {((matched_only['match_score'] >= 80) & (matched_only['match_score'] < 90)).sum()}")
        print(f"  70-79  (Fair):        {((matched_only['match_score'] >= 70) & (matched_only['match_score'] < 80)).sum()}")
        print(f"  <70    (Weak):        {(matched_only['match_score'] < 70).sum()}")
    
    # Save results
    print(f"\n{'='*70}")
    print("STEP 4: Saving Results")
    print(f"{'='*70}")
    
    output_file = r"US\USBureauOfLaborStatistics\profession_demographics_enhanced.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    # Save unmatched
    unmatched_df = results_df[~results_df['matched']][['profession', 'normalized_profession']]
    if len(unmatched_df) > 0:
        unmatched_file = r"US\USBureauOfLaborStatistics\unmatched_professions_enhanced.csv"
        unmatched_df.to_csv(unmatched_file, index=False)
        print(f"✓ Saved unmatched to: {unmatched_file}")
    
    # Display sample results
    print(f"\n{'='*70}")
    print("SAMPLE RESULTS (First 15 professions)")
    print(f"{'='*70}")
    
    display_cols = ['profession', 'bls_occupation', 'match_method', 'match_score']
    print(results_df[display_cols].head(15).to_string(index=False))
    
    # Show some fuzzy matches specifically
    fuzzy_matches = results_df[results_df['match_method'] == 'fuzzy'].head(10)
    if len(fuzzy_matches) > 0:
        print(f"\n{'='*70}")
        print("EXAMPLES OF FUZZY MATCHES")
        print(f"{'='*70}")
        for _, row in fuzzy_matches.iterrows():
            print(f"{row['profession']:25} → {row['bls_occupation']:40} (score: {row['match_score']})")
    
    # Show unmatched
    if len(unmatched_df) > 0:
        print(f"\n{'='*70}")
        print(f"UNMATCHED PROFESSIONS ({len(unmatched_df)})")
        print(f"{'='*70}")
        for prof in unmatched_df['profession'].head(20):
            print(f"  - {prof}")
        if len(unmatched_df) > 20:
            print(f"  ... and {len(unmatched_df) - 20} more")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


# https://www.bls.gov/cps/cpsaat11.htm


# python "US\PracticalBLSDemographics.py" "US\USBureauOfLaborStatistics\cpsaat11_2024.xlsx" 