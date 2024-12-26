"""
This script reads traffic data from HBase, transforms the data, and publishes it to a Kafka topic while avoiding duplicates.
The script maintains a set of processed record IDs to skip already processed data.

Functions:
    load_processed_ids(file_path): Loads processed record IDs from a file.
    save_processed_ids(file_path, processed_ids): Saves processed record IDs to a file.
    decode_data(data): Decodes byte data to string.
    transform_data(data): Transforms traffic data fields to add additional information.
    transform_geo_point(geo_point_str): Transforms geo_point data to a list of floats.
    transform_geo_shape(geo_shape_str): Transforms geo_shape data to a dictionary.
    read_from_hbase(): Reads data from HBase table.
    publish_to_kafka(producer, topic, data): Publishes data to a Kafka topic.
    main(): Main function to read, transform, and publish data, and handle processed IDs.

"""

import happybase
from kafka import KafkaProducer
import json
import os

PROCESSED_IDS_KAFKA_FILE = "/data/processed_ids_kafka.txt"


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


def decode_data(data):
    """
    Decode byte data to string.

    Args:
        data (dict): The original data with byte values.

    Returns:
        dict: The decoded data with string values.
    """
    decoded_data = {}
    for k, v in data.items():
        if isinstance(k, bytes):
            k = k.decode('utf-8')
        if isinstance(v, bytes):
            v = v.decode('utf-8')
        decoded_data[k] = v
    return decoded_data


def transform_data(data):
    """
    Transform traffic data fields to add additional information.

    Args:
        data (dict): The original traffic data fields.

    Returns:
        dict: The transformed traffic data fields.
    """
    transformed_data = {}
    for key, value in data.items():
        if key == 'data:geo_point_2d':
            transformed_data[key] = transform_geo_point(value)
        elif key == 'data:geo_shape':
            transformed_data[key] = transform_geo_shape(value)
        else:
            transformed_data[key] = value
    return transformed_data


def transform_geo_point(geo_point_str):
    """
    Transform geo_point data to a list of floats.

    Args:
        geo_point_str (str): The geo_point data as a string.

    Returns:
        list: The geo_point data as a list of floats, or None if transformation fails.
    """
    try:
        parts = json.loads(geo_point_str)
        if isinstance(parts, list) and len(parts) == 2:
            return parts  # Keep as list of floats
    except ValueError:
        return None
    return None


def transform_geo_shape(geo_shape_str):
    """
    Transform geo_shape data to a dictionary.

    Args:
        geo_shape_str (str): The geo_shape data as a string.

    Returns:
        dict: The geo_shape data as a dictionary, or None if transformation fails.
    """
    try:
        shape = json.loads(geo_shape_str.replace("'", '"'))
        return shape  # Keep as dictionary
    except json.JSONDecodeError:
        return None


def read_from_hbase():
    """
    Read data from HBase table.

    Yields:
        tuple: Key and data from HBase table.
    """
    connection = happybase.Connection(
        'hadoop-master',
        9090,
        transport='buffered',
        protocol='binary'
    )
    table = connection.table('traffic_data')
    for key, data in table.scan():
        yield key, data


def publish_to_kafka(producer, topic, data):
    """
    Publish data to a Kafka topic.

    Args:
        producer (KafkaProducer): The Kafka producer.
        topic (str): The Kafka topic.
        data (dict): The data to publish.
    """
    producer.send(topic, value=json.dumps(data).encode('utf-8'))
    producer.flush()  # Flush after sending


def main():
    """
    Main function to read, transform, and publish data, and handle processed IDs.
    """
    processed_ids = load_processed_ids(PROCESSED_IDS_KAFKA_FILE)
    producer = KafkaProducer(bootstrap_servers='kafka:29092')
    for key, data in read_from_hbase():
        if key not in processed_ids:
            decoded_data = decode_data(data)
            transformed_data = transform_data(decoded_data)
            publish_to_kafka(producer, 'rennesTrafficTopic', transformed_data)
            processed_ids.add(key)
        else:
            print("Duplicate record found: {}, skipping...".format(key))
    save_processed_ids(PROCESSED_IDS_KAFKA_FILE, processed_ids)


if __name__ == "__main__":
    main()
