import json
import random
import shutil
from pathlib import Path

random.seed(42)

# Caminhos absolutos/relativos baseados na execucao a partir da raiz do projeto
ANNOTATIONS_FILE = "data/raw/TACO/data/annotations.json"
IMAGES_ROOT = Path("data/raw/TACO/data")
OUTPUT_ROOT = Path("data/yolo")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

CLASS_MAP = {
    "plastic": 0,
    "metal": 1,
    "glass": 2,
    "paper": 3,
    "cigarette": 4,
    "other": 5,
}

def map_category(name):
    plastic = {
        "Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap",
        "Disposable plastic cup", "Other plastic cup", "Plastic lid",
        "Other plastic", "Plastic film", "Garbage bag", "Other plastic wrapper",
        "Single-use carrier bag", "Polypropylene bag", "Crisp packet",
        "Spread tub", "Tupperware", "Disposable food container",
        "Other plastic container", "Plastic glooves", "Plastic utensils",
        "Plastic straw", "Six pack rings", "Squeezable tube",
    }
    metal = {
        "Food Can", "Drink can", "Metal bottle cap", "Metal lid",
        "Pop tab", "Scrap metal", "Aerosol", "Aluminium foil", "Aluminium blister pack",
    }
    glass = {
        "Glass bottle", "Glass cup", "Glass jar", "Broken glass",
    }
    paper = {
        "Other carton", "Egg carton", "Drink carton", "Corrugated carton",
        "Meal carton", "Pizza box", "Paper cup", "Magazine paper",
        "Tissues", "Wrapping paper", "Normal paper", "Paper bag",
        "Plastified paper bag", "Paper straw", "Toilet tube",
    }

    if name in plastic: return CLASS_MAP["plastic"]
    if name in metal: return CLASS_MAP["metal"]
    if name in glass: return CLASS_MAP["glass"]
    if name in paper: return CLASS_MAP["paper"]
    if name == "Cigarette": return CLASS_MAP["cigarette"]
    return CLASS_MAP["other"]

def main():
    print("Iniciando processamento e mapeamento das 6 macro-classes...")

    # Limpa o diretorio yolo caso exista
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    with open(ANNOTATIONS_FILE) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    
    annotations_per_image = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_per_image:
            annotations_per_image[image_id] = []
        annotations_per_image[image_id].append(ann)

    image_ids = list(annotations_per_image.keys())
    random.shuffle(image_ids)

    n = len(image_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": image_ids[:train_end],
        "val": image_ids[train_end:val_end],
        "test": image_ids[val_end:]
    }

    for split in splits:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, ids in splits.items():
        for image_id in ids:
            img_info = images[image_id]
            image_path = IMAGES_ROOT / img_info["file_name"]

            if not image_path.exists():
                continue

            width = img_info["width"]
            height = img_info["height"]
            unique_name = img_info["file_name"].replace("/", "_")
            destination = OUTPUT_ROOT / "images" / split / unique_name
            
            shutil.copy(image_path, destination)

            label_name = unique_name.rsplit('.', 1)[0] + ".txt"
            label_file = OUTPUT_ROOT / "labels" / split / label_name

            lines = []
            for ann in annotations_per_image[image_id]:
                category_name = categories[ann["category_id"]]
                class_id = map_category(category_name)

                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(label_file, "w") as f:
                f.write("\n".join(lines))

    yaml_content = f"""path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['plastic', 'metal', 'glass', 'paper', 'cigarette', 'other']
"""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)

    print(f"Dataset convertido com sucesso no diretorio {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()