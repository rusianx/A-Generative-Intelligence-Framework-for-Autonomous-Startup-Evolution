"""
YC Algolia Full Extraction Script - v2
Uses batch-based filtering to bypass Algolia's 1000-record pagination limit.
Fetches all 5954 YC companies and saves to JSON + CSV.
"""
import requests
import json
import csv
import time

URL = "https://45BWZJ1SGC-dsn.algolia.net/1/indexes/*/queries"

HEADERS = {
    "X-Algolia-Application-Id": "45BWZJ1SGC",
    "X-Algolia-API-Key": "NzllNTY5MzJiZGM2OTY2ZTQwMDEzOTNhYWZiZGRjODlhYzVkNjBmOGRjNzJiMWM4ZTU0ZDlhYTZjOTJiMjlhMWFuYWx5dGljc1RhZ3M9eWNkYyZyZXN0cmljdEluZGljZXM9WUNDb21wYW55X3Byb2R1Y3Rpb24lMkNZQ0NvbXBhbnlfQnlfTGF1bmNoX0RhdGVfcHJvZHVjdGlvbiZ0YWdGaWx0ZXJzPSU1QiUyMnljZGNfcHVibGljJTIyJTVE",
    "Content-Type": "application/json"
}

HITS_PER_PAGE = 100


def fetch_page(page_num, facet_filters=None):
    """Fetch a single page of results from Algolia with optional filters."""
    params = f"query=&hitsPerPage={HITS_PER_PAGE}&page={page_num}"
    if facet_filters:
        params += f"&facetFilters={json.dumps(facet_filters)}"

    payload = {
        "requests": [
            {
                "indexName": "YCCompany_production",
                "params": params
            }
        ]
    }
    r = requests.post(URL, json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()["results"][0]


def get_all_batches():
    """First, discover all unique batch values via faceting."""
    payload = {
        "requests": [
            {
                "indexName": "YCCompany_production",
                "params": "query=&hitsPerPage=0&facets=batch&maxValuesPerFacet=500"
            }
        ]
    }
    r = requests.post(URL, json=payload, headers=HEADERS)
    r.raise_for_status()
    result = r.json()["results"][0]
    facets = result.get("facets", {}).get("batch", {})
    return facets  # dict of batch_name -> count


def fetch_all_for_batch(batch_name):
    """Fetch all companies for a specific batch, handling pagination."""
    companies = []
    page = 0
    while True:
        result = fetch_page(page, facet_filters=[f"batch:{batch_name}"])
        hits = result["hits"]
        companies.extend(hits)
        if len(hits) < HITS_PER_PAGE:
            break
        page += 1
        if page >= 10:  # Safety: max 1000 per batch
            break
        time.sleep(0.1)
    return companies


def main():
    print("Step 1: Discovering all YC batches via faceting...")
    batches = get_all_batches()
    total_reported = sum(batches.values())
    print(f"Found {len(batches)} batches, reporting {total_reported} total companies\n")

    # Sort batches for nice output
    batch_names = sorted(batches.keys())

    all_companies = []
    seen_ids = set()

    print("Step 2: Fetching companies batch by batch...")
    for i, batch_name in enumerate(batch_names):
        expected = batches[batch_name]
        companies = fetch_all_for_batch(batch_name)

        # Deduplicate
        new_companies = []
        for c in companies:
            cid = c.get("objectID") or c.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                new_companies.append(c)

        all_companies.extend(new_companies)
        print(f"  [{i+1}/{len(batches)}] {batch_name}: expected={expected}, fetched={len(companies)}, new={len(new_companies)}, total={len(all_companies)}")
        time.sleep(0.15)

    # Also fetch companies with no batch (if any)
    print("\nStep 3: Checking for companies without a batch...")
    # Use a negative filter approach - fetch with empty batch
    try:
        result = fetch_page(0, facet_filters=[["batch:-"]])
        no_batch_hits = result["nbHits"]
        print(f"  Companies without batch: {no_batch_hits}")
        if no_batch_hits > 0:
            page = 0
            while True:
                result = fetch_page(page, facet_filters=[["batch:-"]])
                hits = result["hits"]
                for c in hits:
                    cid = c.get("objectID") or c.get("id")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_companies.append(c)
                if len(hits) < HITS_PER_PAGE:
                    break
                page += 1
                time.sleep(0.1)
    except Exception as e:
        print(f"  Could not fetch no-batch companies: {e}")

    print(f"\n{'='*60}")
    print(f"Total unique companies fetched: {len(all_companies)}")

    # Clean up: remove _highlightResult to reduce file size
    for company in all_companies:
        company.pop("_highlightResult", None)

    # Save JSON
    json_path = "d:/DATASETS/yc_companies_algolia.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_companies, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    # Save CSV with key fields
    csv_path = "d:/DATASETS/yc_companies_algolia.csv"
    csv_fields = [
        "id", "name", "slug", "website", "one_liner", "long_description",
        "batch", "status", "team_size", "all_locations",
        "industry", "subindustry", "tags", "industries", "regions",
        "stage", "isHiring", "nonprofit", "top_company",
        "small_logo_thumb_url", "launched_at"
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for company in all_companies:
            row = {}
            for field in csv_fields:
                val = company.get(field, "")
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                row[field] = val
            writer.writerow(row)
    print(f"Saved CSV: {csv_path}")
    print(f"\nDone! {len(all_companies)} YC companies extracted.")


if __name__ == "__main__":
    main()
