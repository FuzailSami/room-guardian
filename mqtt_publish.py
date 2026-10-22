#!/usr/bin/env python3
"""
Publishes JSON messages to the local Mosquitto broker on the Pi.
Mosquitto bridges every topic under fuzail-pi/ to AWS IoT Core, where
IoT Rules write the messages to DynamoDB.
"""

import json

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_PREFIX = "fuzail-pi"


def publish(subtopic, payload, qos=1, timeout=10):
    """Publish payload (a dict) as JSON to fuzail-pi/<subtopic>.
    Raises RuntimeError if the local broker doesn't acknowledge it."""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt 2.x
    except AttributeError:
        client = mqtt.Client()  # paho-mqtt 1.x

    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    try:
        topic = f"{TOPIC_PREFIX}/{subtopic}"
        info = client.publish(topic, json.dumps(payload), qos=qos)
        info.wait_for_publish(timeout)
        if not info.is_published():
            raise RuntimeError(f"MQTT publish to {topic} not acknowledged")
    finally:
        client.loop_stop()
        client.disconnect()
