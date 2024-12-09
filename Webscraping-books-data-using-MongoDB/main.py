"""
This script connects to a MongoDB instance and manages the scraping process of URLs.
It includes functions to retrieve URLs that need to be scraped, insert new URLs found
during scraping, and store relevant content from scraped pages. The workflow includes:

1. `get_url_to_scrap`: Retrieves a URL to scrape based on certain conditions to ensure
   it's eligible for scraping.
2. `insert_new_urls`: Searches for new URLs on the scraped page, checks their eligibility,
   and inserts them into the collection if they haven't been stored before.
3. `insert_one_document`: Saves the raw HTML of the scraped page and extracts important
   elements such as title, headings, and emphasis tags, and then stores them.
"""

import urllib
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime, timedelta
import time

# Connecting to MongoDB and initializing the database and the collections needed for scraping:
# 'logs' for recording scraping activities,
# 'urls' for managing URL statuses and subsequents scraping attempts,
# and 'data' for storing scraped content.
client = MongoClient('localhost', 27017)
db = client['book_scraping']
collection_logs = db['logs']
collection_urls = db['urls']
collection_data = db['data']


def scrape_urls():
    """
    Main function to manage the scraping process. It continuously retrieves URLs to scrape,
    performs the scraping, inserts new found URLs, and updates the status of the scraped URL.

    If an error occurs during scraping, it updates the URL status to 'error_scraping' and
    schedules the next attempt.
    """
    still_scraping = True
    print("The scraping has started !")
    while still_scraping:
        response = None
        try:
            # Get a URL to scrape
            url_doc = get_url_to_scrap()
            if url_doc:
                scope = url_doc['scope']
                print("URL being scraped:", url_doc['url'])

                # Scrape the URL content
                response = requests.get(url_doc['url'])

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Insert all new URLs found in the scraped HTML
                insert_new_urls(url_doc, scope, soup)

                # Insert the page content and extract important fields
                insert_one_document(url_doc, response, soup)

                # Update the status of the URL to 'scraped'
                collection_urls.update_one(
                    {'_id': url_doc['_id']},
                    {'$set': {'status': 'scraped', 'last_update': datetime.now()}}
                )
            else:
                still_scraping = False
        except Exception as e:
            print(f"An error occurred: {e}")
            collection_urls.update_one(
                {'_id': url_doc['_id']},
                {'$set': {'status': 'error_scraping', 'last_update': datetime.now(),
                          'next_scrap_date': datetime.now() + timedelta(minutes=10)}},
                {"$inc": {"nombre_de_trentative": 1}}
            )
        finally:
            if url_doc:
                collection_logs.insert_one({
                    'url': url_doc['url'],
                    'status': response.status_code if response else 'error',
                    'reason': response.reason if response else 'N/A',
                    "last_update": datetime.now()
                })


def get_url_to_scrap():
    """
    Retrieves the next URL to scrape based on the following conditions:
    - URLs that have never been scraped (status 'to_be_scraped').
    - URLs with scraping errors that can be retried (next_scrap_date reached).
    - URLs being scraped but exceeding a time limit, suggesting issues with another scraper.

    The function waits and retries if there are possible URLs but none are immediately available.

    Returns:
        dict: The document of the URL to scrape or None if no eligible URL is found.
    """
    wait = True
    while wait:
        # Check if there are possible URLs to scrape
        possible_urls = collection_urls.find_one({"$or": [
            {'status': 'to_be_scraped'},
            {'status': 'error_scraping', 'nombre_de_trentative': {"$lt": 10}},
            {'status': 'being_scraped'}
        ]})

        # Find a URL document eligible for scraping
        url_doc = collection_urls.find_one_and_update(
            {"$or": [
                {'status': 'to_be_scraped'},
                {'status': 'error_scraping', 'next_scrap_date': {"$lt": datetime.now()},
                 'nombre_de_trentative': {"$lt": 10}},
                {'status': 'being_scraped', 'last_update': {"$lt": datetime.now() - timedelta(minutes=10)}}
            ]},
            {'$set': {'status': 'being_scraped', "last_update": datetime.now()}}
        )

        # Stop waiting and start scraping if a URL is found
        if url_doc:
            wait = False

        # Stop the scraper if no possible URLs are available
        if url_doc is None and possible_urls is None:
            wait = False

        # Wait 10 seconds if there are possible URLs but none to scrape immediately
        if url_doc is None and possible_urls:
            print("Waiting 10 seconds for possible URLs")
            time.sleep(10)

    return url_doc


def insert_one_document(url_doc, response, soup):
    """
    Saves the full HTML document of the scraped page and extracts key elements.

    Extracts:
    - Title of the page
    - Heading tags (h1, h2)
    - Text inside bold (b), strong, and emphasis (em) tags

    The extracted data is saved in the 'data' collection with the URL and HTML content.

    Parameters:
        url_doc (dict): The URL document from the database.
        response (requests.Response): The response object containing the HTML content.
        soup (BeautifulSoup): The parsed HTML document.
    """
    title = soup.find('title').text
    h1 = soup.find('h1').text
    h2 = list(map(lambda x: x.text, soup.find_all('h2')))
    b = list(map(lambda x: x.text, soup.find_all('b')))
    em = list(map(lambda x: x.text, soup.find_all('em')))
    strong = list(map(lambda x: x.text, soup.find_all('strong')))
    collection_data.insert_one({
        'url': url_doc['url'],
        'html': response.text,
        "title": title,
        "h1": h1,
        "h2": h2,
        "b": b,
        "em": em,
        "strong": strong,
        "last_update": datetime.now()
    })


def insert_new_urls(url_doc, scope, soup):
    """
    Finds all new URLs on the scraped page, verifies their eligibility, and inserts them into the collection.

    - Searches all anchor tags (<a>) and parses their href links.
    - Converts relative URLs to absolute URLs based on the current page.
    - Checks if the URL is within the specified scope and hasn't been inserted already.
    - Adds new eligible URLs to the 'urls' collection with the status 'to_be_scraped'.

    Parameters:
        url_doc (dict): The URL document from the database.
        scope (str): The scope to verify if URLs are eligible.
        soup (BeautifulSoup): The parsed HTML document.
    """
    for link in soup.find_all('a'):
        absolute_url = urllib.parse.urljoin(url_doc['url'], link['href'])
        # Check if the URL is not already recorded in the database and is within the specified scope
        if absolute_url.startswith(scope):
            collection_urls.update_one(
                {'url': absolute_url},
                {"$setOnInsert": {'url': absolute_url, 'status': 'to_be_scraped', 'scope': scope}},
                upsert=True
            )


# Main execution
if __name__ == "__main__":
    scrape_urls()
