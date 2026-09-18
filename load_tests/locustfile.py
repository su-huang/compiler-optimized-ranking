"""Load test for the baseline /predict endpoint.

Run with: locust -f locustfile.py --host http://localhost:8000
"""

import random

from locust import HttpUser, task

SAMPLE_REVIEWS = [
    "Great place, food was amazing and service was fast. Will definitely come back!",
    "Not impressed. Waited 40 minutes and the order was still wrong when it arrived.",
    "Solid value for the money, friendly staff, nothing fancy but satisfying.",
    "Overpriced for the portion size. The ambiance was nice at least.",
    "Best meal I've had in months. Every dish was cooked perfectly.",
    "Mediocre at best. Wouldn't recommend unless you're really in a hurry.",
]


class RankingUser(HttpUser):
    @task
    def predict(self):
        text_a, text_b = random.sample(SAMPLE_REVIEWS, 2)
        self.client.post("/predict", json={"text_a": text_a, "text_b": text_b})
