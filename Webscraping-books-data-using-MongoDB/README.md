# Web Scraper with MongoDB!

The scraper sets up and connects to a MongoDB database to retrieve a URL to scrape.
Once the URL is scraped, the HTML page is saved in a collection, with some important data extracted and stored separately in distinct fields. 
It also extracts all URLs linked on the page and retains only those within the defined scope.

## Programs

The scraper is divided into two parts:

1. **initialise_scraper** : Initializes the scraper with a starting URL and a defined SCOPE to adhere to.
2. **main** : Launches the scraper to begin the scraping process.

## Running the Scraper

To run the scraper, follow these steps:

- First, run **initialise_scraper** to set up **MongoDB** with the initial URL and scope.
- Then, run the **main** program as many times as needed (you can run it in parallel on different terminals) to continue the scraping process.


### Command to Run:
```console
	python .\initialise_scrapper.py 'first_url.com' 'scope'
	python .\main.py
```

## MongoDB Collections

3 collections are defined :
- The 'logs' collection for recording scraping activities
- The 'urls' collection for managing URL statuses and subsequents scraping attempts
- The 'data' collection for storing scraped content.