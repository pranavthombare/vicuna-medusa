import random
import pandas as pd
import logging
from datetime import datetime
import json
from locust import HttpUser, task
from locust.exception import RescheduleTask
import argparse

class SimpleUser(HttpUser):

    @task
    def endpoint_hit(self):
        payload = {
                "prompt": "Tell me something about hip hop.",
                "parameters": {
                        "temperature": 0.5,
                        "max_steps": 100,
                },
        }

        headers = {"Content-Type": "application/json"}
        response = self.client.post(self.host, json=payload, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            logging.info(f"{datetime.isoformat(datetime.now())} -  POST  success")
        else:
            logging.error(f"{datetime.isoformat(datetime.now())} -  POST  failed")
            raise RescheduleTask()