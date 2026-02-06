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

# Data scraping function
def run_thesis_scraper(limit=200):
    base_url = "https://www.casarosada.gob.ar/informacion/discursos"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    links = []
    # 1. LOOP THROUGH PAGES (40 items per page usually)
    # This will check page 0, 40, 80, 120, etc.
    for start in range(0, limit, 40):
        page_url = f"{base_url}?start={start}"
        print(f"Fetching links from: {page_url}")
        
        try:
            r = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Find links on this specific page
            page_links = []
            for a in soup.find_all('a', href=True):
                if '/informacion/discursos/' in a['href']:
                    full_link = "https://www.casarosada.gob.ar" + a['href']
                    if full_link not in links: 
                        page_links.append(full_link)
            
            links.extend(page_links)
            
            # Stop if we found enough links or if the page has no links (end of results)
            if len(links) >= limit or not page_links:
                break
                
            time.sleep(1) # Be polite to the server
        except Exception as e:
            print(f"Error fetching page {start}: {e}")
            break

    # Truncate to the exact limit
    links = links[:limit]
    print(f"Total links gathered: {len(links)}")

    # 2. SCRAPE EACH LINK (Same logic as yours)
    results = []
    for link in links:
        try:
            print(f"Scraping content: {link.split('/')[-1]}...")
            res = requests.get(link, headers=headers, timeout=10)
            s = BeautifulSoup(res.text, 'html.parser')
            
            # Content extraction
            content_div = s.find('div', {'class': 'articulo-contenido'})
            raw_text = content_div.get_text(separator=" ") if content_div else " ".join([p.get_text() for p in s.find_all('p')])
            
            # Calculations
            normalized_body = clean_text(raw_text)
            words = normalized_body.split()
            total_words = len(words)
            count = sum(words.count(clean_text(k)) for k in KEYWORDS)
            density = (count / total_words) if total_words > 0 else 0
            
            results.append({
                'Title': s.find('h1').text.strip() if s.find('h1') else "No Title",
                'Fiscal_Count': count,
                'Fiscal_Density': density,
                'Total_Words': total_words,
                'Found_Text': "Yes" if total_words > 0 else "No",
                'URL': link
            })
            time.sleep(1)
        except Exception as e:
            print(f"Skipping link {link} due to error: {e}")

    return pd.DataFrame(results)


