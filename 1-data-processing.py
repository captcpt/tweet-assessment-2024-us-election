import os
import pandas as pd
import gzip
import ast
import re

data_folder = "./usc-x-24-us-election-main"
processed_data_folder = "./processed_data"
os.makedirs(processed_data_folder, exist_ok=True)

# Mapping states/abbreviations
state_mapping = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia", 
    "Washington, D.C.": "District of Columbia"
}

# Extracting .gz file to .csv
def extract_gz(input_file, output_file):
    with gzip.open(input_file, 'rb') as gz_file, open(output_file, 'wb') as out_file:
        out_file.write(gz_file.read())

# Extracting hashtags into standardized list format
def extract_hashtags_list_from_string(hashtags_string):
    try:
        hashtags = ast.literal_eval(hashtags_string)
        if isinstance(hashtags, list):
            return [hashtag['text'] for hashtag in hashtags if isinstance(hashtag, dict)]
        return []  # Return empty list if not a list of dictionaries
    except (ValueError, SyntaxError):
        return []  # Return empty list in case of error

# Extracting view count
def extract_view_count(view_count):
    try:
        view_count_dict = ast.literal_eval(view_count)

        if isinstance(view_count_dict, dict):
            return int(view_count_dict.get('count', 0))       
        elif isinstance(view_count_dict, (int, float)):
            return int(view_count_dict)
        else:
            return 0

    except (ValueError, SyntaxError, TypeError):
        return None 

# Extracting the location 
def extract_location(user_info):
    if isinstance(user_info, str):
        match = re.search(r"'location':\s*'([^']+)'", user_info)
        if match:
            return match.group(1)  # Return the captured location value
    return None

# Standardizing state labels
def get_standardized_state(location):
    if pd.isna(location) or location.strip() == '':
        return None
    
    location = location.strip().lower()
    
    # Checking for state patterns with abbreviations ("City, CA")
    for state_abbr in [state for state in state_mapping.keys() if len(state) == 2]:
        if re.search(rf',\s*{state_abbr.lower()}$', location):
            return state_mapping[state_abbr.upper()]
    
    # Checking exact state name matches ("California")
    for full_name in state_mapping.values():
        if full_name.lower() in location:
            return full_name
            
    # Checking state abbreviation matches ("CA")
    for abbr in state_mapping.keys():
        if abbr.lower() == location:
            return state_mapping[abbr]

    # Checking USA if no state matches
    if any(term in location for term in ['usa', 'united states', 'u.s.a', 'u.s.']):
        return "USA"

    # Otherwise not US
    return None

# Loop through each folder
for part in os.listdir(data_folder):
    part_path = os.path.join(data_folder, part)
    
    # Check directory (starts with part_)
    if os.path.isdir(part_path) and part.startswith("part_"):
        print(f"Processing {part}...")

        # Processing each .csv.gz file in the folder
        for filename in os.listdir(part_path):
            if filename.endswith(".csv.gz"):
                gz_file_path = os.path.join(part_path, filename)
                
                # Output .csv path (same filename but without .gz extension)
                csv_file_path = os.path.join(processed_data_folder, filename.replace('.gz', ''))
                
                # Extracting .gz file to .csv
                extract_gz(gz_file_path, csv_file_path)
                
                # Loading and processing the CSV
                df = pd.read_csv(csv_file_path, encoding='utf-8')
            
                df['hashtags_list'] = df['hashtags'].apply(extract_hashtags_list_from_string)
                df['viewCount'] = df['viewCount'].apply(extract_view_count)
                df['location'] = df['user'].apply(extract_location)
                
                # Apply the state standardization
                df['state'] = df['location'].apply(get_standardized_state)
                
                # Dropping original 'user' column since we only need 'location' from it
                columns_to_keep = [
                    'id', 'text', 'hashtags_list', 'replyCount', 'retweetCount', 
                    'likeCount', 'quoteCount', 'viewCount', 'location', 'state'
                ]
                df = df[columns_to_keep]
                df.dropna(subset=['location'], inplace=True)
                df.dropna(subset=['state'], inplace=True)

                # Saving the final processed data
                df.to_csv(csv_file_path, index=False)
                print(f"Processed and saved {csv_file_path}")

# After the processing loop, merge all files into one data file
all_files = [os.path.join(processed_data_folder, f) for f in os.listdir(processed_data_folder) if f.endswith('.csv')]
combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# Saving the merged dataset
combined_df.to_csv(os.path.join(processed_data_folder, 'combined_election_data.csv'), index=False)