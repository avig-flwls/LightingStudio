import argparse
from pathlib import Path
import ast
from collections import Counter
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Iterable

def find_max_not_bigger(values, threshold):
    """
    values: iterable of strings like "8k", "4k", etc.
    threshold: string like "8k". The result must be <= this.
    Returns the string in values with the highest numeric value ≤ threshold,
    or None if no such value.
    """
    # Parse threshold ("8k" → 8)
    thresh_num = int(threshold.rstrip('k'))

    # Filter out those bigger than the threshold
    valid = [v for v in values if int(v.rstrip('k')) <= thresh_num]

    if not valid:
        return None

    # Pick the one with the largest numeric part
    return max(valid, key=lambda v: int(v.rstrip('k')))

def fetch_polyhaven_hdris_information(max_resolution: str, max_items: int | None = None) -> pd.DataFrame:
    """Fetch HDRI metadata from Poly Haven API."""
    response = requests.get(POLYHAVEN_API_URL)
    assets = response.json()
    
    # Limit assets to process if specified
    assets_items = list(assets.items())
    if max_items is not None:
        assets_items = assets_items[:max_items]
    
    print(f"Total HDRI assets in Poly Haven Library: {len(assets)}")
    print(f"Processing: {len(assets_items)}")
    records = []
    for name, asset in tqdm(assets_items, total=len(assets_items), desc="Processing Poly Haven HDRI Library"):
        files_response = requests.get(POLYHAVEN_FILE_URL + name)
        files = files_response.json()
        resolution_list = list(files.get("hdri").keys())
        selected_resolution = find_max_not_bigger(resolution_list, max_resolution)
        format_list = list(files.get("hdri").get(selected_resolution).keys())
        if "exr" in format_list:
            record = {
                "name": asset.get("name"),
                "categories": asset.get("categories"),
                "tags": asset.get("tags"),
                "resolution": selected_resolution,
                "format": "exr",
                "file_url": files.get("hdri").get(selected_resolution).get("exr").get("url")
            }
        elif "hdr" in format_list:
            record = {
                "name": asset.get("name"),
                "categories": asset.get("categories"),
                "tags": asset.get("tags"),
                "resolution": selected_resolution,
                "format": "hdr",
                "file_url": files.get("hdri").get(selected_resolution).get("hdr").get("url")
            }
        else:
            record = {
                "name": asset.get("name"),
                "categories": asset.get("categories"),
                "tags": asset.get("tags"),
                "resolution": selected_resolution,
                "format": format_list[0],
                "file_url": files.get("hdri").get(selected_resolution).get(format_list[0]).get("url")
            }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    print(f"Total HDRI assets selected:{len(df)}")

    # Show the first few rows
    print(df.head())
    return df

def compare_database(df: pd.DataFrame, max_resolution: str) -> pd.DataFrame:
    update_database = fetch_polyhaven_hdris_information(max_resolution)
    # select rows in A where name is NOT in B['name']
    new_hdris_database = update_database[~update_database['name'].isin(df['name'])]
    print(new_hdris_database.head())
    return new_hdris_database

def download_hdris(df: pd.DataFrame, download_path: Path) -> None:
    """Download HDRIs from the dataframe to the specified path."""
    download_path.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(df.index, total=len(df), desc="Downloading HDRIs from Poly Haven"):
        file_url = df.at[idx, "file_url"]
        name = df.at[idx, "name"]
        file_format = df.at[idx, "format"]
        dest_path = download_path / f"{name}.{file_format}"
        try:
            r = requests.get(file_url, stream=True)
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            
def get_all_unique_tag(df:pd.DataFrame) -> list:
    
    df["categories"] = df["categories"].apply(ast.literal_eval)
    # Flatten the list of lists and get unique values  
    tag_counts = Counter(tag for tag_list in df['categories'] for tag in tag_list)
    
    # Remove keys that contain "collection: "
    filtered_tag_counts = Counter({k: v for k, v in tag_counts.items() if "collection: " not in k})
    
    print(filtered_tag_counts)
    return filtered_tag_counts

# -----------------------------
# Query helpers for multi-labels
# -----------------------------

def _ensure_parsed_multi_label_columns(df: pd.DataFrame) -> None:
    """Ensure df has `categories_parsed` and `tags_parsed` columns as Python lists.

    If `categories_parsed` / `tags_parsed` don't exist, they will be created by
    parsing the corresponding string columns. If they exist but contain strings,
    they will be parsed in-place.
    """
    if 'categories_parsed' not in df.columns:
        df['categories_parsed'] = df['categories']
    if 'tags_parsed' not in df.columns:
        df['tags_parsed'] = df['tags']

    def _parse_if_needed(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x

    df['categories_parsed'] = df['categories_parsed'].apply(_parse_if_needed)
    df['tags_parsed'] = df['tags_parsed'].apply(_parse_if_needed)


def list_all_categories(df: pd.DataFrame) -> List[str]:
    """Return sorted list of all unique categories across the dataset."""
    _ensure_parsed_multi_label_columns(df)
    all_categories = {c for row in df['categories_parsed'] for c in row}
    return sorted(all_categories)


def list_all_tags(df: pd.DataFrame) -> List[str]:
    """Return sorted list of all unique tags across the dataset."""
    _ensure_parsed_multi_label_columns(df)
    all_tags = {t for row in df['tags_parsed'] for t in row}
    return sorted(all_tags)


def get_names_by_categories(df: pd.DataFrame, required_categories: Iterable[str]) -> List[str]:
    """Return names whose categories include ALL of `required_categories` (AND logic)."""
    _ensure_parsed_multi_label_columns(df)
    required_set = set(required_categories)
    mask = df['categories_parsed'].apply(lambda row: required_set.issubset(set(row)))
    return df.loc[mask, 'name'].tolist()


def get_names_by_tags(df: pd.DataFrame, required_tags: Iterable[str]) -> List[str]:
    """Return names whose tags include ALL of `required_tags` (AND logic)."""
    _ensure_parsed_multi_label_columns(df)
    required_set = set(required_tags)
    mask = df['tags_parsed'].apply(lambda row: required_set.issubset(set(row)))
    return df.loc[mask, 'name'].tolist()


def print_label_pools(df: pd.DataFrame) -> None:
    """Print a simple, readable pool of all categories and tags (sorted)."""
    cats = list_all_categories(df)
    tags = list_all_tags(df)
    print(f"Total unique categories: {len(cats)}")
    print(", ".join(cats))
    print()
    print(f"Total unique tags: {len(tags)}")
    print(", ".join(tags))
            
"""
Run this script to get the HDRI database from Poly Haven
"""

# Set API endpoints
POLYHAVEN_API_URL = "https://api.polyhaven.com/assets?type=hdris"
POLYHAVEN_FILE_URL = "https://api.polyhaven.com/files/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and download Poly Haven HDRIs")
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of HDRIs to download (for quick testing). Defaults to all.",
    )
    parser.add_argument(
        "--max-resolution",
        type=str,
        default="1k",
        help="Maximum resolution to download (e.g., 1k, 2k, 4k, 8k, 16k).",
    )
    parser.add_argument(
        "--root-output-dir",
        type=Path,
        default=Path("tmp") / "source",
        help="Root directory for outputs (downloads and CSV).",
    )
    args = parser.parse_args()

    # Resolve paths based on CLI args
    root_output_dir = args.root_output_dir
    max_resolution = args.max_resolution
    download_dir = root_output_dir / max_resolution
    output_csv_path = root_output_dir / f"{max_resolution}_database.csv"

    # Fetch HDRI information (limited if max_downloads specified)
    database = fetch_polyhaven_hdris_information(max_resolution, max_items=args.max_downloads)
    
    # Save database to CSV
    root_output_dir.mkdir(parents=True, exist_ok=True)
    database.to_csv(output_csv_path, index=False)
    
    # Create download directory
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download HDRIs (database is already limited if max_downloads was specified)
    download_hdris(database, download_dir)

    # Optional: reload CSV to compute and print tag stats
    df = pd.read_csv(output_csv_path)
    get_all_unique_tag(df)