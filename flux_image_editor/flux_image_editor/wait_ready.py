"""Wait until the FLUX HTTP inference service reports ready."""

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def wait_until_ready(url, timeout_sec, poll_interval_sec=1.0):
    deadline = time.monotonic() + timeout_sec
    last_error = "service has not responded"
    while time.monotonic() < deadline:
        try:
            request_timeout = min(5.0, max(0.1, deadline - time.monotonic()))
            with urlopen(url, timeout=request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("ready") is True:
                return payload
            last_error = f"HTTP {response.status}: {payload}"
        except (
            HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError
        ) as error:
            last_error = str(error)
        time.sleep(min(poll_interval_sec, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"FLUX service did not become ready within {timeout_sec:.1f}s: {last_error}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        payload = wait_until_ready(args.url, args.timeout, args.poll_interval)
    except (TimeoutError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "FLUX ready: "
        f"model={payload.get('model_id', 'unknown')} "
        f"load_sec={payload.get('model_load_sec', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
