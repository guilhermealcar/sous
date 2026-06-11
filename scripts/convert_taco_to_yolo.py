import json

ANNOTATIONS_FILE = "data/raw/TACO/data/annotations.json"

with open(ANNOTATIONS_FILE, "r") as f:
    data = json.load(f)

categories = {}

for category in data["categories"]:
    categories[category["id"]] = category["name"]

print("\n=== CATEGORY MAP ===\n")

for cid, name in categories.items():
    print(f"{cid:>2} -> {name}")
