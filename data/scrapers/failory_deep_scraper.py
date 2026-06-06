"""
Failory Deep Scraper (Phase 2)
==============================
Reads company_urls.txt and scrapes each individual article page to extract:
  - detailed_description: the article summary / intro paragraph
  - funding_detail: specific funding amounts mentioned
  - investors: investors mentioned in the article
  - failure_story: the full body of the failure analysis
  - lessons: key lessons / takeaways
  - article_title: the page title (often contains funding amount)

Merges with the base failory_dataset.csv and outputs:
  - failory_dataset_enriched.csv
"""

import requests
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import re
import time
import os
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

# Checkpoint file to resume after interruptions
CHECKPOINT_FILE = "deep_scrape_checkpoint.json"


def load_checkpoint() -> dict:
    """Load previously scraped articles from checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict):
    """Save progress to checkpoint file."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_funding_from_title(title: str) -> str:
    """Extract specific funding amount from article title (e.g., 'Raising $1.8B')."""
    patterns = [
        r'\$[\d,.]+[BMKbmk]',           # $1.8B, $50M, etc.
        r'\$[\d,.]+ (?:billion|million)', # $1.8 billion
    ]
    for p in patterns:
        match = re.search(p, title, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def extract_investors(text: str) -> str:
    """Extract investor names from article text using known patterns."""
    # Look for investor lineup patterns
    patterns = [
        r'investors?\s+included?\s+([^.]+)',
        r'backed\s+by\s+([^.]+)',
        r'funded\s+by\s+([^.]+)',
        r'raised\s+.*?from\s+([^.]+)',
        r'investment\s+from\s+([^.]+)',
    ]
    investors = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            investors.append(m.strip())
    return "; ".join(investors) if investors else ""


def scrape_article(url: str) -> dict:
    """Scrape a single Failory cemetery article page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ⚠ Error fetching {url}: {e}")
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Article title
    title_el = soup.find("title")
    article_title = title_el.get_text(strip=True) if title_el else ""

    # Meta description
    meta_desc = ""
    meta_el = soup.find("meta", attrs={"name": "description"})
    if meta_el:
        meta_desc = meta_el.get("content", "")

    # Main article body — look for rich text blocks
    body_parts = []
    headings = []

    # Failory uses .w-richtext for the main article content
    richtext = soup.select_one("div.w-richtext")
    if richtext:
        for el in richtext.children:
            if isinstance(el, NavigableString):
                text = el.strip()
                if text:
                    body_parts.append(text)
            elif el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                heading_text = el.get_text(strip=True)
                headings.append(heading_text)
                body_parts.append(f"\n## {heading_text}\n")
            elif el.name in ("p", "div", "blockquote", "li", "ul", "ol"):
                text = el.get_text(strip=True)
                if text and text != "\u200b":  # Skip zero-width spaces
                    body_parts.append(text)

    full_story = "\n".join(body_parts).strip()

    # Extract specific data
    funding_detail = extract_funding_from_title(article_title)
    investors = extract_investors(full_story)

    # Extract numbered failure reasons (sections like "### 1) Coronavirus")
    reason_pattern = re.compile(r'##\s*\d+\)\s*(.+)')
    failure_reasons_detailed = []
    for match in reason_pattern.finditer(full_story):
        failure_reasons_detailed.append(match.group(1).strip())

    # Extract lessons if present
    lessons = ""
    lessons_section = re.search(
        r'(?:lessons?\s+learned|key\s+takeaway|what\s+can\s+we\s+learn)(.*?)(?=##|\Z)',
        full_story, re.IGNORECASE | re.DOTALL
    )
    if lessons_section:
        lessons = lessons_section.group(1).strip()[:1000]

    return {
        "article_title": article_title,
        "meta_description": meta_desc,
        "funding_detail": funding_detail,
        "investors": investors[:500],  # Cap length
        "failure_reasons_detailed": "; ".join(failure_reasons_detailed) if failure_reasons_detailed else "",
        "failure_story": full_story[:3000],  # Cap at 3000 chars for CSV sanity
        "lessons": lessons,
        "headings": "; ".join(headings),
    }


def main():
    # Load base dataset
    if not os.path.exists("failory_dataset.csv"):
        print("❌ failory_dataset.csv not found. Run failory_scraper.py first.")
        return

    df = pd.read_csv("failory_dataset.csv")
    print(f"📊 Loaded {len(df)} companies from failory_dataset.csv")

    # Load URLs
    urls = df["url"].tolist()

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"📌 Checkpoint has {len(checkpoint)} previously scraped articles")

    # Scrape each article
    results = {}
    for i, url in enumerate(urls):
        slug = url.split("/")[-1]

        if slug in checkpoint:
            results[slug] = checkpoint[slug]
            continue

        print(f"  [{i+1}/{len(urls)}] Scraping {slug}...")
        article_data = scrape_article(url)
        results[slug] = article_data

        # Save checkpoint every 5 articles
        checkpoint[slug] = article_data
        if (i + 1) % 5 == 0:
            save_checkpoint(checkpoint)
            print(f"    💾 Checkpoint saved ({len(checkpoint)} articles)")

        # Be polite — 1.5s between requests
        time.sleep(1.5)

    # Final checkpoint save
    save_checkpoint(checkpoint)

    # Merge with base dataset
    enriched_rows = []
    for _, row in df.iterrows():
        slug = row["url"].split("/")[-1]
        article = results.get(slug, {})
        enriched = row.to_dict()
        enriched.update({
            "article_title": article.get("article_title", ""),
            "meta_description": article.get("meta_description", ""),
            "funding_detail": article.get("funding_detail", ""),
            "investors": article.get("investors", ""),
            "failure_reasons_detailed": article.get("failure_reasons_detailed", ""),
            "failure_story": article.get("failure_story", ""),
            "lessons": article.get("lessons", ""),
        })
        enriched_rows.append(enriched)

    enriched_df = pd.DataFrame(enriched_rows)

    # Save enriched dataset
    output_path = "failory_dataset_enriched.csv"
    enriched_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved enriched dataset to {output_path}")
    print(f"   Columns: {list(enriched_df.columns)}")
    print(f"   Rows: {len(enriched_df)}")

    # Stats
    has_story = enriched_df["failure_story"].str.len() > 10
    has_investors = enriched_df["investors"].str.len() > 2
    has_funding = enriched_df["funding_detail"].str.len() > 0
    has_detailed_reasons = enriched_df["failure_reasons_detailed"].str.len() > 2

    print(f"\n📈 Enrichment Stats:")
    print(f"   Articles with failure story:     {has_story.sum()}/{len(enriched_df)}")
    print(f"   Articles with investors:         {has_investors.sum()}/{len(enriched_df)}")
    print(f"   Articles with funding detail:    {has_funding.sum()}/{len(enriched_df)}")
    print(f"   Articles with detailed reasons:  {has_detailed_reasons.sum()}/{len(enriched_df)}")

    # Preview
    print(f"\n{'='*80}")
    print("SAMPLE (Quibi):")
    print(f"{'='*80}")
    quibi = enriched_df[enriched_df["company"] == "Quibi"]
    if not quibi.empty:
        for col in ["article_title", "funding_detail", "investors", "failure_reasons_detailed"]:
            print(f"  {col}: {quibi.iloc[0][col]}")


if __name__ == "__main__":
    main()
