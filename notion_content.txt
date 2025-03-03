from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from bs4 import BeautifulSoup

def remove_duplicates(text):
    """
    Remove duplicate paragraphs and sentences from text
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with duplicates removed
    """
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Remove duplicate paragraphs while preserving order
    unique_paragraphs = []
    seen_paragraphs = set()
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph:
            continue
            
        # Skip very short paragraphs that might be menu items or UI elements
        if len(paragraph) < 4:
            continue
            
        # Skip if already seen
        if paragraph in seen_paragraphs:
            continue
            
        unique_paragraphs.append(paragraph)
        seen_paragraphs.add(paragraph)
    
    # Join back into text
    cleaned_text = '\n\n'.join(unique_paragraphs)
    
    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text

def scrape_notion_with_selenium(url):
    """
    Advanced scraper for Notion pages with duplicate content removal
    
    Args:
        url: URL of the Notion page to scrape
        
    Returns:
        String containing the deduplicated text content of the page
    """
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    print("Starting Chrome WebDriver...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        # Navigate to the URL
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for page to load
        print("Waiting for page to load...")
        time.sleep(8)
        
        # Scroll to load all content
        print("Scrolling to load all content...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(5):  # Scroll up to 5 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # No more new content
            last_height = new_height
        
        # Wait for dynamic content
        print("Waiting for content to fully load...")
        time.sleep(5)
        
        # Get page source
        page_source = driver.page_source
        
        # Parse with BeautifulSoup for better text extraction
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get page title
        title = soup.title.string if soup.title else "DeepScaleR Document"
        
        # Find the main content area - typically a main div with lots of text
        main_content = soup.find('main') or soup.find('div', class_=lambda c: c and 'content' in c.lower()) or soup
        
        # Get all text blocks
        text_blocks = []
        
        # Process headings
        headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            text = heading.get_text().strip()
            if text:
                text_blocks.append(text)
        
        # Process paragraphs and other text elements
        paragraphs = main_content.find_all(['p', 'div'])
        for p in paragraphs:
            # Skip if it's just a container with no direct text
            if not p.get_text(strip=True):
                continue
                
            # Skip common navigation or UI elements
            if any(c in (p.get('class') or []) for c in ['menu', 'nav', 'header', 'footer']):
                continue
                
            text = p.get_text().strip()
            if text and len(text) > 10:  # Ensure it's substantial text
                text_blocks.append(text)
        
        # Get all list items
        list_items = main_content.find_all('li')
        for item in list_items:
            text = item.get_text().strip()
            if text:
                text_blocks.append(f"• {text}")
        
        # Combine all text
        full_text = f"{title}\n\n" + "\n\n".join(text_blocks)
        
        # Remove duplicates
        print("Removing duplicate content...")
        cleaned_text = remove_duplicates(full_text)
        
        return cleaned_text
    
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}"
    
    finally:
        # Close the browser
        driver.quit()
        print("Browser closed")

def save_to_file(content, filename="notion_content_deduped.txt"):
    """Save the scraped content to a text file"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Content saved to {filename}")
    return filename

# URL of the Notion page to scrape
url = "https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2"

# Scrape the content
print(f"Starting to scrape: {url}")
content = scrape_notion_with_selenium(url)

# Save the content to a file
filename = save_to_file(content)

# Print statistics
print(f"\nExtracted {len(content)} characters")
print("\nPreview of the first 1000 characters:")
print("=" * 50)
print(content[:1000] + "..." if len(content) > 1000 else content)
print("=" * 50)