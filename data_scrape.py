import os
import re
import requests
import zipfile
import csv
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

category_urls = {
    "Weapons": "http://darksouls.wikidot.com/weapons",
    "Shields": "http://darksouls.wikidot.com/shields",
    "Spell Tools": "http://darksouls.wikidot.com/spell-tools",
    "Weapon Upgrades": "http://darksouls.wikidot.com/upgrade",
    "Armor Sets": "http://darksouls.wikidot.com/armor",
    "Head Armor": "http://darksouls.wikidot.com/head",
    "Chest Armor": "http://darksouls.wikidot.com/chest",
    "Hands Armor": "http://darksouls.wikidot.com/hands",
    "Legs Armor": "http://darksouls.wikidot.com/legs",
    "Ammo": "http://darksouls.wikidot.com/ammo",
    "Bonfire Items": "http://darksouls.wikidot.com/bonfire-items",
    "Consumables": "http://darksouls.wikidot.com/consumables",
    "Multiplayer Items": "http://darksouls.wikidot.com/multiplayer-items",
    "Rings": "http://darksouls.wikidot.com/rings",
    "Keys": "http://darksouls.wikidot.com/keys",
    "Pyromancies": "http://darksouls.wikidot.com/pyromancies",
    "Sorceries": "http://darksouls.wikidot.com/sorceries",
    "Miracles": "http://darksouls.wikidot.com/miracles",
    "Story": "http://darksouls.wikidot.com/story",
    "Bosses": "http://darksouls.wikidot.com/bosses",
    "Enemies": "http://darksouls.wikidot.com/enemies",
    "Merchants": "http://darksouls.wikidot.com/merchants",
    "NPCs": "http://darksouls.wikidot.com/npcs",
    "Areas": "http://darksouls.wikidot.com/areas",
    "Stats": "http://darksouls.wikidot.com/stats",
    "Classes": "http://darksouls.wikidot.com/classes",
    "Gifts": "http://darksouls.wikidot.com/gifts",
    "Covenants": "http://darksouls.wikidot.com/covenants",
    "Trophies": "http://darksouls.wikidot.com/trophies",
    "Achievements": "http://darksouls.wikidot.com/achievements"
}


headers = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 20
BASE_FOLDER = 'darksouls_data'

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def extract_links_by_subcategory(url):
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        content = soup.find('div', id='page-content')

        sub_links = {}
        current_heading = "General"

        for tag in content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'a']):
            if tag.name in ['h1', 'h2', 'h3'] and tag.get_text(strip=True):
                current_heading = tag.get_text(strip=True)
                if current_heading not in sub_links:
                    sub_links[current_heading] = []

            if tag.name == 'a' and tag.get('href'):
                href = tag['href'].strip()
                if href.startswith(('#', 'javascript:', 'mailto:')):
                    continue
                full_url = urljoin(url, href)
                if "darksouls.wikidot.com" in full_url:
                    sub_links.setdefault(current_heading, []).append(full_url)

        for k in sub_links:
            sub_links[k] = sorted(set(sub_links[k]))
        return sub_links

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return {}

def scrape_page_content(url):
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        content_div = soup.find('div', id='page-content')
        sub_links = []

        if content_div:
            for tag in content_div.find_all(['div', 'script', 'style', 'table', 'img']):
                tag.decompose()

            for a_tag in content_div.find_all('a', href=True):
                href = a_tag['href']
                if not href.startswith(('mailto:', 'javascript:', '#')):
                    sub_links.append(urljoin(url, href))

            text = content_div.get_text(separator='\n', strip=True)
            return re.sub(r'\n{3,}', '\n\n', text), sorted(set(sub_links))
        return "", []
    except Exception as e:
        return f"❌ Failed to scrape {url}: {e}", []

def save_page_to_file(category, subcat, url, data_store):
    folder = os.path.join(BASE_FOLDER, sanitize_filename(category), sanitize_filename(subcat))
    os.makedirs(folder, exist_ok=True)
    filename = sanitize_filename(url.split('/')[-1] or "index") + ".txt"
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        content, sublinks = scrape_page_content(url)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        data_store.append({"Category": category, "Subcategory": subcat, "URL": url, "Content": content})
        return sublinks  # Return sub-links to process
    return []

def save_data_as_csv_xlsx(data_store):
    df = pd.DataFrame(data_store)
    df.to_csv(os.path.join(BASE_FOLDER, "all_data.csv"), index=False, encoding='utf-8')
    df.to_excel(os.path.join(BASE_FOLDER, "all_data.xlsx"), index=False, engine='openpyxl')

def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=folder_path)
                zipf.write(filepath, arcname=arcname)

def main():
    print("🔍 Starting enhanced Dark Souls wiki scraper...")
    all_links = {}
    all_data = []

    # Step 1: Extract main category sub-links
    for category, url in category_urls.items():
        print(f"📂 Extracting links for: {category}")
        subcats = extract_links_by_subcategory(url)
        all_links[category] = subcats

    # Step 2: Scrape pages + subpages
    print("🚀 Scraping all pages with multithreading...")
    seen_urls = set()
    tasks = []
    new_links = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for category, subcats in all_links.items():
            for subcat, links in subcats.items():
                for link in links:
                    if link not in seen_urls:
                        seen_urls.add(link)
                        tasks.append(executor.submit(save_page_to_file, category, subcat, link, all_data))

        for future in as_completed(tasks):
            sublinks = future.result()
            for sublink in sublinks:
                if sublink not in seen_urls and "darksouls.wikidot.com" in sublink:
                    new_links.append(sublink)
                    seen_urls.add(sublink)

        # Step 3: Scrape all discovered sub-links (1-level deep)
        print("🔁 Scraping discovered sub-links...")
        sublink_tasks = [executor.submit(save_page_to_file, "Discovered", "Sublinks", link, all_data)
                         for link in new_links]

        for future in as_completed(sublink_tasks):
            future.result()

    # Step 4: Save data in structured formats
    print("📄 Saving combined data to CSV and XLSX...")
    save_data_as_csv_xlsx(all_data)

    # Step 5: Zip the entire folder
    print("🗜️ Zipping all data...")
    zip_folder(BASE_FOLDER, "darksouls_wiki_dump.zip")

    print("✅ Done! Download your data from 'darksouls_wiki_dump.zip'")

if __name__ == "__main__":
    main()
