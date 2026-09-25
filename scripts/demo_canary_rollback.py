"""Demo: a deliberately broken canary gets detected and auto-rolled-back (Phase 5).

Fires many /predict requests at a running server with CANARY_CHAOS_FAILURE_RATE
set high, and prints served_by + canary status after each request so the
transcript shows the canary's error rate climbing, the rollback triggering,
and all subsequent traffic reverting to stable.

Usage (in a separate terminal, start the server with a broken canary first):
    CANARY_TRAFFIC_FRACTION=0.5 CANARY_CHAOS_FAILURE_RATE=0.8 \\
        uvicorn src.serving.app:app --port 8000
    python -m scripts.demo_canary_rollback
"""

import argparse

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = httpx.Client(base_url=args.host)
    was_enabled = True

    for i in range(args.num_requests):
        response = client.post(
            "/predict",
            json={"text_a": "Great food and service!", "text_b": "Terrible, would not recommend."},
        )
        served_by = response.json()["served_by"]
        status = client.get("/canary/status").json()
        print(
            f"[{i:02d}] served_by={served_by:<7} canary_enabled={status['enabled']!s:<5} "
            f"error_rate={status['recent_error_rate']} samples={status['samples']}"
        )
        if was_enabled and not status["enabled"]:
            print("\ncanary disabled -- rollback complete, remaining requests all go to stable.\n")
        was_enabled = status["enabled"]


if __name__ == "__main__":
    main()
