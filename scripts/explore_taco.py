import json
from collections import Counter

ANNOTATIONS_PATH = "data/raw/TACO/data/annotations.json"

with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

print("\n=== DATASET INFO ===")
print(f"Images: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")

print("\n=== CATEGORIES ===")

for category in data["categories"]:
    print(
        f"{category['id']:>3} | "
        f"{category['name']}"
    )

counter = Counter()

for ann in data["annotations"]:
    counter[ann["category_id"]] += 1

print("\n=== TOP CLASSES ===")

for category in data["categories"]:
    cid = category["id"]
    name = category["name"]

    print(
        f"{name:<40} "
        f"{counter[cid]}"
    )
