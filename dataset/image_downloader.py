import requests
import os
import argparse

# 1. Argument parsing for the API key

# Define command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--api-key', required=True)  # API key is required
args = parser.parse_args()
API_KEY = args.api_key

# 2. Search parameters

# Define search categories
SEARCH_TERMS = ['portrait', 'landscape', 'object', 'street', 'animal']
PER_PAGE = 5  # Number of images per query
PAGES_PER_TERM = 2  # 2 pages x 5 images per page = 10 images per category
SAVE_DIR = 'dataset/real'  # Directory to save images

# 3. Pexels API headers

# Authorization header for the API
headers = {
    'Authorization': API_KEY
}

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# 4. Download loop

image_count = 0  # Counter for downloaded images

# Loop through search terms
for term in SEARCH_TERMS:
    for page in range(1, PAGES_PER_TERM + 1):
        params = {
            'query': term,
            'per_page': PER_PAGE,
            'page': page
        }
        print(f"Searching for '{term}' - page {page}")
        response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
        
        # Check for API errors
        if response.status_code != 200:
            print(f"Error for '{term}' page {page}: {response.status_code}")
            continue
        
        data = response.json()
        for photo in data.get('photos', []):
            image_url = photo['src']['large']
            filename = f"{term}_{image_count:04d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            try:
                # Download the image content and save it
                img_data = requests.get(image_url).content
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                print(f"Downloaded: {filename}")
                image_count += 1
            except Exception as e:
                print(f"Download error: {e}")

# Final message
print(f"\n✅ Total images downloaded: {image_count}")
