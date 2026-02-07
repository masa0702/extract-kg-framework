#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple connectivity check for Wikidata API.

Usage:
  python src/check_wikidata_api.py
  python src/check_wikidata_api.py --user-agent "YourApp/1.0 (contact: you@example.com)"
  python src/check_wikidata_api.py --endpoint https://www.wikidata.org/w/api.php --lang ja --search 東京
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Any, Dict

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check connectivity to Wikidata API.")
    ap.add_argument("--endpoint", default="https://www.wikidata.org/w/api.php")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--search", default="東京")
    ap.add_argument(
        "--user-agent",
        default="Wikidata-Connectivity-Check/1.0 (contact: you@example.com)",
        help="User-Agent for Wikidata API",
    )
    ap.add_argument("--timeout-sec", type=int, default=10)
    return ap.parse_args()


def _print_env_proxy() -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        v = os.environ.get(k)
        if v:
            print(f"{k}={v}")


def _dns_check(host: str) -> None:
    try:
        socket.getaddrinfo(host, None)
        print(f"dns=OK host={host}")
    except Exception as e:  # pragma: no cover
        print(f"dns=NG host={host} error={type(e).__name__}: {e}")


def _request(endpoint: str, params: Dict[str, Any], headers: Dict[str, str], timeout_sec: int) -> None:
    if requests is None:
        print("requests=NG (requests library is not available)")
        sys.exit(2)
    try:
        r = requests.get(endpoint, params=params, headers=headers, timeout=timeout_sec)
        print(f"http_status={r.status_code}")
        r.raise_for_status()
        data = r.json()
        hits = data.get("search") or []
        top_id = hits[0].get("id") if hits else None
        print(f"hits={len(hits)} top_id={top_id}")
    except Exception as e:  # pragma: no cover
        print(f"request_failed {type(e).__name__}: {e}")
        sys.exit(1)


def main() -> int:
    args = parse_args()
    _print_env_proxy()
    _dns_check("www.wikidata.org")
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": args.lang,
        "type": "item",
        "search": args.search,
        "limit": 1,
        "origin": "*",
    }
    headers = {"User-Agent": args.user_agent}
    _request(args.endpoint, params, headers, args.timeout_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
