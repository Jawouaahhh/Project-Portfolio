"""
This script connects to a local MongoDB instance and stores a URL for scraping
in the 'book_scraping' database. It takes two command-line arguments:

1. url_to_scrap: The URL that needs to be scraped.
2. scope: The scope of the scraping operation (e.g., specific pages, entire site).

The script checks for the required arguments, prints the provided URL and scope,
and inserts the URL into the 'urls' collection with an initial status of 'to_be_scraped'.
"""

import sys
from pymongo import MongoClient

# MongoDB setup
# Connect to the local MongoDB instance on the default port 27017 and
# select the 'book_scraping' database. The 'urls' collection will be used
# to store URLs that need to be scraped.
client = MongoClient('localhost', 27017)
db = client['book_scraping']
collection_urls = db['urls']

# Check for command-line arguments
# The script expects two command-line arguments:
# 1. "url_to_scrap": The URL that the user wants to scrape.
# 2. "scope": The scope of the scraping, which may define how deep or
#    wide the scraping operation should go (e.g., specific pages, entire site, etc.).
if len(sys.argv) < 3:
    print("Please provide an URL to scrap as well as the scope of scraping")
    exit(1)

# Extract command-line arguments
url_to_scrap = sys.argv[1]
scope = sys.argv[2]

# Display the provided URL and scope
print("URL to be scraped:", url_to_scrap)
print("Scope:", scope)

# Insert the URL into the 'urls' collection in MongoDB
# The inserted document contains the URL, its status (initially 'to_be_scraped'),
# and the scope of the scraping operation.
collection_urls.insert_one({'url': url_to_scrap, 'status': 'to_be_scraped', 'scope': scope})
