#!/usr/bin/env python3
"""Lightweight stdlib-only checks for compatibility golden fixtures."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "tests" / "golden"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    ts_req = load(G / "requests" / "typesafe-systemone.json")
    ai_req = load(G / "requests" / "ai-sdk-evaluation-v4.json")
    ts_resp = load(G / "responses" / "typesafe-systemone.json")
    ai_resp = load(G / "responses" / "ai-sdk-evaluation-v4.json")

    assert ts_req["questions"]["refund_review"]["type"] == "noul"
    assert ai_req["questions"]["refund_review"]["type"] == "boolean"
    assert ts_resp["answers"]["refund_review"]["noul"] == 0.9
    assert ai_resp["answers"]["refund_review"]["probability"] == 0.9
    assert set(ts_req["questions"]) == set(ai_req["questions"])
    assert set(ts_resp["answers"]) == set(ai_resp["answers"])
    assert ai_resp["providerMetadata"]["turbojev"]["contract"] == "turbojev.core.v4"
    print("OK: compatibility golden fixtures")

if __name__ == "__main__":
    main()
