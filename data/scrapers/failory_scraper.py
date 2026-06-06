"""
Failory Cemetery Scraper
========================
Scrapes structured startup failure data from https://www.failory.com/cemetery
Produces:
  - failory_dataset.csv     (full dataset)
  - company_urls.txt         (one URL per line for deeper scraping)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.failory.com/cemetery"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

FIELDS = [
    ("title", "company"),
    ("description", "description"),
    ("category", "category"),
    ("country", "country"),
    ("failure", "failure_reason"),
    ("outcome", "outcome"),
    ("started", "started"),
    ("closed", "closed"),
    ("funding", "funding"),
    ("employees", "employees"),
    ("founders", "founders"),
]


def scrape_page(url: str) -> list[dict]:
    """Scrape a single page of the Failory Cemetery."""
    print(f"  Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Target the main CMS list (exclude the featured/recommended cards)
    main_list = soup.select_one('div[fs-list-element="list"]')
    if not main_list:
        # Fallback: try all w-dyn-items
        cards = soup.select("div.w-dyn-item")
    else:
        cards = main_list.select("div.w-dyn-item")

    rows = []
    for card in cards:
        link = card.find("a")
        if not link or not link.get("href", "").startswith("/cemetery/"):
            continue

        row = {}
        for fs_field, col_name in FIELDS:
            el = card.select_one(f'[fs-list-field="{fs_field}"]')
            row[col_name] = el.get_text(strip=True) if el else ""

        row["url"] = "https://www.failory.com" + link["href"]
        rows.append(row)

    return rows


def main():
    all_rows = []
    seen_urls = set()

    # Try pages 1 through 10 (dynamically stop when no new data)
    for page in range(1, 11):
        if page == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}?8bd93ea4_page={page}"

        rows = scrape_page(url)

        if not rows:
            print(f"  Page {page}: no cards found — stopping.")
            break

        # Deduplicate
        new_rows = []
        for r in rows:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                new_rows.append(r)

        if not new_rows:
            print(f"  Page {page}: all duplicates — stopping.")
            break

        print(f"  Page {page}: {len(new_rows)} new startups")
        all_rows.extend(new_rows)

        # Be polite
        time.sleep(1.5)

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = "failory_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(df)} startups to {csv_path}")

    # Save URL list
    urls_path = "company_urls.txt"
    with open(urls_path, "w", encoding="utf-8") as f:
        for url in df["url"]:
            f.write(url + "\n")
    print(f"✅ Saved {len(df)} URLs to {urls_path}")

    # Preview
    print(f"\n{'='*80}")
    print("DATASET PREVIEW")
    print(f"{'='*80}")
    print(df.to_string(index=False, max_rows=10))
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Quick stats
    print(f"\n{'='*80}")
    print("QUICK STATS")
    print(f"{'='*80}")
    if "category" in df.columns:
        print(f"\nTop Categories:\n{df['category'].value_counts().head(10).to_string()}")
    if "failure_reason" in df.columns:
        print(f"\nTop Failure Reasons:\n{df['failure_reason'].value_counts().head(10).to_string()}")
    if "country" in df.columns:
        print(f"\nTop Countries:\n{df['country'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
