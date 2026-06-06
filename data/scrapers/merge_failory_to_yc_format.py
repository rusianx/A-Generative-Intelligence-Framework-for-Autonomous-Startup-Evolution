"""
Merge Failory datasets and reshape to the YC schema.
=====================================================
1. Combines `failory_dataset.csv` (base) and `failory_dataset_enriched.csv`
   (base + scraped article fields) into a single record set, keyed by company.
   The enriched file is a superset, so values from it take precedence; any row
   present in only one file is still kept (outer merge).
2. Rewrites every record into the exact same column schema as
   `yc_companies_algolia.csv`, so the failure data (failory) and the success
   data (YC) can be stacked / compared directly.

Output: data/failory_dataset_yc_format.csv

Run from the repo root:  python data/scrapers/merge_failory_to_yc_format.py
"""
import csv
import re
import calendar
import os

# Resolve paths relative to the data/ directory regardless of CWD.
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CSV = os.path.join(DATA_DIR, "failory_dataset.csv")
ENRICHED_CSV = os.path.join(DATA_DIR, "failory_dataset_enriched.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "failory_dataset_yc_format.csv")

# Target schema — identical to yc_companies_algolia.csv.
YC_COLUMNS = [
    "id", "name", "slug", "website", "one_liner", "long_description",
    "batch", "status", "team_size", "all_locations",
    "industry", "subindustry", "tags", "industries", "regions",
    "stage", "isHiring", "nonprofit", "top_company",
    "small_logo_thumb_url", "launched_at",
]

# Failory outcome -> YC status vocabulary.
OUTCOME_TO_STATUS = {
    "Shut Down": "Inactive",
    "Bankruptcy": "Inactive",
    "Acquired": "Acquired",
    "Still Active": "Active",
}


def merge_failory_rows():
    """Outer-merge base + enriched on company name, preferring enriched values."""
    merged = {}
    for path in (BASE_CSV, ENRICHED_CSV):
        if not os.path.exists(path):
            print(f"  ! skipping missing file: {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("company") or "").strip().lower()
                if not key:
                    continue
                record = merged.setdefault(key, {})
                for col, val in row.items():
                    val = (val or "").strip()
                    # Prefer non-empty values; enriched is read last so it wins.
                    if val:
                        record[col] = val
                    record.setdefault(col, "")
    return list(merged.values())


def slugify(company, url):
    """Reuse the Failory URL slug when available, else slugify the name."""
    if url:
        tail = url.rstrip("/").split("/")[-1]
        if tail:
            return tail
    return re.sub(r"[^a-z0-9]+", "-", (company or "").lower()).strip("-")


def employees_to_team_size(employees):
    """Convert a coarse range like '100-250' or '+10,000' into a midpoint int."""
    if not employees or employees.lower() == "no data":
        return ""
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", employees)]
    if not nums:
        return ""
    if len(nums) == 1:
        return str(nums[0])
    return str(round(sum(nums) / len(nums)))


def year_to_launched_at(started):
    """Convert a founding year to a Jan-1 unix timestamp (YC's launched_at format)."""
    m = re.search(r"\d{4}", started or "")
    if not m:
        return ""
    return str(calendar.timegm((int(m.group(0)), 1, 1, 0, 0, 0)))


def to_yc_format(rows, id_prefix="fl-"):
    out = []
    for i, r in enumerate(rows, start=1):
        company = r.get("company", "")
        category = r.get("category", "")
        country = r.get("country", "")
        long_desc = r.get("failure_story") or r.get("meta_description") or r.get("description", "")
        out.append({
            "id": f"{id_prefix}{i}",
            "name": company,
            "slug": slugify(company, r.get("url", "")),
            "website": "",
            "one_liner": r.get("description", ""),
            "long_description": long_desc,
            "batch": "",
            "status": OUTCOME_TO_STATUS.get(r.get("outcome", ""), "Inactive"),
            "team_size": employees_to_team_size(r.get("employees", "")),
            "all_locations": country,
            "industry": category,
            "subindustry": category,
            "tags": r.get("failure_reason", ""),
            "industries": category,
            "regions": country,
            "stage": "",
            "isHiring": "False",
            "nonprofit": "False",
            "top_company": "False",
            "small_logo_thumb_url": "",
            "launched_at": year_to_launched_at(r.get("started", "")),
        })
    return out


def main():
    print("Merging Failory base + enriched datasets...")
    merged = merge_failory_rows()
    print(f"  merged unique companies: {len(merged)}")

    yc_rows = to_yc_format(merged)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=YC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(yc_rows)

    print(f"Wrote {len(yc_rows)} rows to {OUTPUT_CSV}")
    from collections import Counter
    print("Status distribution:", dict(Counter(r["status"] for r in yc_rows)))


if __name__ == "__main__":
    main()
