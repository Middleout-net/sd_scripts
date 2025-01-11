import sys
import requests
import time
import logging
from datetime import datetime


class LogStream:
    def __init__(self, flush_interval=5, api=True, client_id="-1"):
        self.console = sys.stdout
        self.flush_interval = flush_interval
        self.message_buffer = []
        self.last_flush_time = time.time()
        self.api = api
        self.client_id = client_id

    def write(self, message):
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if message.strip():  # Avoid sending empty messages
            timestamped_message = self.add_timestamp(message)
            self.console.write(message)
            self.console.flush()
            if self.api:
                self.buffer_message(timestamped_message)

    def buffer_message(self, message):
        self.message_buffer.append(message)
        current_time = time.time()
        if (
            len(self.message_buffer) >= self.flush_interval
            or (current_time - self.last_flush_time) >= self.flush_interval
        ):
            self.flush_buffer()

    def flush_buffer(self):
        if self.message_buffer:
            self.send_to_api(self.message_buffer)
            self.message_buffer = []
            self.last_flush_time = time.time()

    def send_to_api(self, messages):
        # Replace with your actual API endpoint and payload
        api_endpoint = "http://localhost:3002/stdout"
        payload = {"logs": messages}
        headers = {"Content-Type": "application/json", "Client-ID": self.client_id}
        try:
            response = requests.post(api_endpoint, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except Exception as e:
            print(f"Failed to send logs to API: {e}", file=self.console)

    def add_timestamp(self, message):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{current_time} - {message}"

    def flush(self):
        self.flush_buffer()  # Ensure any remaining messages are sent

class LogStreamHandler(logging.Handler):
    def __init__(self, log_stream):
        super().__init__()
        self.log_stream = log_stream

    def emit(self, record):
        try:
            # Get the log message from the record and add a timestamp
            log_message = self.format(record)
            # Write the message to the LogStream
            self.log_stream.write(log_message)
        except Exception as e:
            self.handleError(record)
