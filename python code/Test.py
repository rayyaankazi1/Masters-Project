# Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import requests
from bs4 import BeautifulSoup
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import time
import unicodedata
import re

# Creating the dictionary
KEYWORDS = ['superavit', 'deficit', 'ajuste', 'equilibrio', 'fiscal', 'motosierra', 'emision', 'gasto']

# Defining function to clean text to avoid missing words
def clean_text(text):
    if not text: return ""
    # Lowercase & remove accents (e.g., 'déficit' -> 'deficit')
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    # Remove everything except letters and spaces
    return re.sub(r'[^a-z\s]', '', text)

# Data Scraping function
def run_thesis_scraper(limit=200):
    base_url = "https://www.casarosada.gob.ar/informacion/discursos"
    # Improved headers to look more like a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    links = []
    # Use a Session for better performance
    session = requests.Session()
    
    print("--- PHASE 1: COLLECTING LINKS ---")
    for start in range(0, limit, 40):
        page_url = f"{base_url}?start={start}"
        print(f"Checking page: {page_url}...", end=" ")
        
        try:
            # Short timeout so it doesn't hang forever
            r = session.get(page_url, headers=headers, timeout=10)
            if r.status_code != 200:
                print(f"Failed! (Status: {r.status_code})")
                break
                
            soup = BeautifulSoup(r.text, 'html.parser')
            page_links = [
                "https://www.casarosada.gob.ar" + a['href'] 
                for a in soup.find_all('a', href=True) 
                if '/informacion/discursos/' in a['href']
            ]
            
            # Remove duplicates while keeping order
            for l in page_links:
                if l not in links: links.append(l)
            
            print(f"Found {len(page_links)} links. (Total so far: {len(links)})")
            
            if not page_links or len(links) >= limit:
                break
            
            time.sleep(2) # Be extra polite
            
        except Exception as e:
            print(f"\nCaught an error on page {start}: {e}")
            break

    links = links[:limit]
    
    print(f"\n--- PHASE 2: SCRAPING {len(links)} SPEECHES ---")
    results = []
    for i, link in enumerate(links):
        try:
            print(f"[{i+1}/{len(links)}] Scraping: {link.split('/')[-1][:30]}...", end="\r")
            res = session.get(link, headers=headers, timeout=10)
            s = BeautifulSoup(res.text, 'html.parser')
            
            # Text extraction logic
            content_div = s.find('div', {'class': 'articulo-contenido'})
            raw_text = content_div.get_text(separator=" ") if content_div else " ".join([p.get_text() for p in s.find_all('p')])
            
            normalized_body = clean_text(raw_text)
            words = normalized_body.split()
            total_words = len(words)
            count = sum(words.count(clean_text(k)) for k in KEYWORDS)
            
            results.append({
                'Title': s.find('h1').text.strip() if s.find('h1') else "No Title",
                'Fiscal_Count': count,
                'Fiscal_Density': (count / total_words) if total_words > 0 else 0,
                'Total_Words': total_words
            })
            time.sleep(1) # Delay between speech pages
            
        except Exception as e:
            print(f"\nError on {link}: {e}")

    print("\nScraping complete!")
    return pd.DataFrame(results)

# --- 3. EXECUTION ---

df = run_thesis_scraper(200)



# --- 4. THE DIAGNOSIS ---

if isinstance(df, pd.DataFrame):

print("\n--- RESULTS ---")

display(df)

# Validation

if df['Total_Words'].sum() == 0:

print("\n DIAGNOSIS: The scraper found the pages but extracted 0 words.")

print("Action: The website is blocking the content or using a new container tag.")

elif df['Fiscal_Count'].sum() == 0:

print("\n DIAGNOSIS: Text found, but zero keywords matched.")

print(f"Action: Check your KEYWORDS list. Current list: {KEYWORDS}")

else:

print("\n SUCCESS: Data captured.")

else:

print(df)

