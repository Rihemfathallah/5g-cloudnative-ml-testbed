#!/usr/bin/env python3
"""Same as provision.js, with pymongo, for the local (non-Kubernetes) config check."""
import json, sys
from pathlib import Path
from bson.int64 import Int64
from pymongo import MongoClient

uri = sys.argv[1] if len(sys.argv) > 1 else "mongodb://127.0.0.1/open5gs"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
base = json.loads((Path(__file__).parent / "subscriber.json").read_text())
db = MongoClient(uri, serverSelectionTimeoutMS=5000)["open5gs"]
for i in range(n):
    doc = json.loads(json.dumps(base))
    doc["imsi"] = str(int(base["imsi"]) + i).zfill(15)
    doc["security"]["sqn"] = Int64(doc["security"]["sqn"])
    db.subscribers.update_one({"imsi": doc["imsi"]}, {"$set": doc}, upsert=True)
print("subscribers:", db.subscribers.count_documents({}))
