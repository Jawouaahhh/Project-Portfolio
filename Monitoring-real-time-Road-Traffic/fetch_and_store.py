"""
This script fetches traffic data from Rennes Metropole API, transforms the data,
and stores it in HBase while avoiding duplicates. The script maintains a set of
processed record IDs to skip already processed data.

Functions:
    fetch_data(): Fetches traffic data from the API.
    transform_data(fields): Transforms traffic data fields to add additional information.
    load_processed_ids(file_path): Loads processed record IDs from a file.
    save_processed_ids(file_path, processed_ids): Saves processed record IDs to a file.
    store_in_hbase(data, processed_ids): Stores transformed data in HBase and updates processed IDs.
    main(): Main function to fetch, transform, and store data, and handle processed IDs.

"""

import requests
import happybase
import time
from datetime import datetime, timedelta
from dateutil.parser import parse
import random
import os

PROCESSED_IDS_FETCH_FILE = "/data/processed_ids_fetch.txt"


def fetch_data():
    """
    Fetch traffic data from Rennes Metropole API.

    Returns:
        dict: The JSON response containing traffic data.
    """
    url = "https://data.rennesmetropole.fr/api/records/1.0/search/?dataset=etat-du-trafic-en-temps-reel&rows=1000"
    response = requests.get(url)
    return response.json()


def transform_data(fields):
    """
    Transform traffic data fields to add additional information.

    Args:
        fields (dict): The original traffic data fields.

    Returns:
        dict: The transformed traffic data fields.
    """
    # Concatenate denomination with speed limit to create a unique denomination value
    fields['unique_denomination'] = "{} - {}".format(fields.get('denomination', ''), fields.get('vitesse_maxi', ''))

    # Create additional fields based on datetime
    datetime_str = fields.get('datetime')
    if datetime_str:
        try:
            # Parsing the datetime string using dateutil.parser
            dt = parse(datetime_str)
            fields['day_of_week'] = dt.strftime('%A')  # Day of the week in text
            fields['hour_range'] = "{:02d}:00-{:02d}:59".format(dt.hour, dt.hour)  # Hour range

            # Adding random hour_range and day_of_week fields
            random_hour = random.randint(0, 23)
            random_day = random.choice(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
            fields['hour_range_random'] = "{:02d}:00-{:02d}:59".format(random_hour, random_hour)
            fields['day_of_week_random'] = random_day

            # Generate a random datetime named datetime_random
            random_date = datetime(2024, 7, random.randint(1, 10), random_hour, 0, 0)
            fields['datetime_random'] = random_date.isoformat()
        except Exception as e:
            print("Error parsing datetime for record: {}: {}".format(fields, e))
            fields['day_of_week'] = None
            fields['hour_range'] = None
            fields['hour_range_random'] = None
            fields['day_of_week_random'] = None
            fields['datetime_random'] = None
    else:
        fields['day_of_week'] = None
        fields['hour_range'] = None
        fields['hour_range_random'] = None
        fields['day_of_week_random'] = None
        fields['datetime_random'] = None

    # Calculate speed excess
    try:
        average_speed = int(fields.get('averagevehiclespeed', 0))
        speed_limit = int(fields.get('vitesse_maxi', 0))
        fields['speed_excess'] = average_speed - speed_limit
    except ValueError as e:
        print("Error calculating speed excess: {}".format(e))
        fields['speed_excess'] = None

    # Calculate speed compliance
    fields['speed_compliance'] = "Vitesse conforme" if average_speed <= speed_limit else "Excès de vitesse"

    # Categorize congestion level
    try:
        traveltime = int(fields.get('traveltime', 0))
        if traveltime <= 10:
            fields['congestion_level'] = 'Bas'
        elif traveltime <= 20:
            fields['congestion_level'] = 'Moyen'
        else:
            fields['congestion_level'] = 'Elevé'
    except ValueError as e:
        print("Error calculating congestion level: {}".format(e))
        fields['congestion_level'] = 'Unknown'

    # Categorize time of day
    if 6 <= dt.hour < 10:
        fields['time_of_day'] = 'Matin - rush'
    elif 10 <= dt.hour < 16:
        fields['time_of_day'] = 'Après-midi'
    elif 16 <= dt.hour < 19:
        fields['time_of_day'] = 'Après-midi - rush'
    else:
        fields['time_of_day'] = 'Nuit'

    # Placeholder for vehicle probe count and traffic density
    fields['vehicle_probe_count'] = int(fields.get('vehicleprobemeasurement', 0))
    fields['traffic_density'] = fields['vehicle_probe_count'] / traveltime if traveltime else 0

    return fields


def load_processed_ids(file_path):
    """
    Load processed record IDs from a file.

    Args:
        file_path (str): Path to the file containing processed record IDs.

    Returns:
        set: A set of processed record IDs.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def save_processed_ids(file_path, processed_ids):
    """
    Save processed record IDs to a file.

    Args:
        file_path (str): Path to the file to save processed record IDs.
        processed_ids (set): A set of processed record IDs.
    """
    with open(file_path, 'w') as f:
        for record_id in processed_ids:
            f.write("{}\n".format(record_id))


def store_in_hbase(data, processed_ids):
    """
    Store transformed data in HBase and update processed IDs.

    Args:
        data (dict): The JSON response containing traffic data.
        processed_ids (set): A set of already processed record IDs.
    """
    connection = happybase.Connection(
        'localhost',
        9090,
        transport='buffered',
        protocol='binary'
    )
    table = connection.table('traffic_data')
    for record in data['records']:
        key = record['recordid']
        if key not in processed_ids:
            fields = record['fields']
            transformed_fields = transform_data(fields)
            if not transformed_fields['datetime_random']:
                print("Missing datetime_random for record: {}".format(key))
            table.put(key, {"data:" + k: str(v) for k, v in transformed_fields.items()})
            processed_ids.add(key)
        else:
            print("Duplicate record found: {}, skipping...".format(key))


def main():
    """
    Main function to fetch, transform, and store data, and handle processed IDs.
    """
    processed_ids = load_processed_ids(PROCESSED_IDS_FETCH_FILE)
    data = fetch_data()
    store_in_hbase(data, processed_ids)
    save_processed_ids(PROCESSED_IDS_FETCH_FILE, processed_ids)


# Lines for continuous data fetching:
# def main():
#     processed_ids = load_processed_ids(PROCESSED_IDS_FETCH_FILE)
#     while True:
#         data = fetch_data()
#         store_in_hbase(data, processed_ids)
#         save_processed_ids(PROCESSED_IDS_FETCH_FILE, processed_ids)
#         time.sleep(60)

if __name__ == "__main__":
    main()
