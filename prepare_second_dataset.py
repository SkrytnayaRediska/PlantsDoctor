import sys, re, json
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/olgamiskevich/PycharmProjects/PlantsDoctor/PlantDoc-Dataset")
OUT_CSV = ROOT.parent / "plantdoc_parsed.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# canonization of crops
CANON_CROP = {
    "apple": "apple",
    "tomato": "tomato",
    "potato": "potato",
    "cucumber": "cucumber",
    "wheat": "wheat",
    "grape": "grape",
    "corn": "corn", "maize": "corn",
    "pepper": "pepper", "bell pepper": "pepper", "pepper bell": "pepper",
    "peach": "peach",
    "strawberry": "strawberry",
    "blueberry": "blueberry",
    "cherry": "cherry",
    "orange": "orange",
    "raspberry": "raspberry",
    "soybean": "soybean",
    "squash": "squash",
}

# order of checking
CROP_ALIASES_ORDERED = sorted(CANON_CROP.keys(), key=lambda s: -len(s))


def norm_text(some_str):
    some_str = some_str.lower()
    some_str = some_str.replace("(", " ").replace(")", " ").replace("-", " ").replace("/", " ")
    some_str = re.sub(r"\s+", " ", some_str).strip()
    return some_str


def to_snake(some_str):
    some_str = some_str.lower()
    some_str = re.sub(r"[^a-z0-9]+", "_", some_str)
    some_str = re.sub(r"_+", "_", some_str).strip("_")
    return some_str or "healthy"


def parse_class_dir(name: str):
    """
    name examples:
      'Apple leaf' -> crop=apple, disease=healthy
      'Apple Scab leaf' -> crop=apple, disease=scab
      'Tomato Early blight leaf' -> crop=tomato, disease=early_blight
      'Bell Pepper leaf' -> crop=pepper, disease=healthy
    Steps:
      1) norm str (lowercase, spaces)
      2) find crop alias (substr) from CROP_ALIASES_ORDERED
      3) remain without 'leaf'/'leaves' is a disease name; if no remains -> healthy
    """
    raw = norm_text(name)
    crop_found = None
    rest = raw

    for alias in CROP_ALIASES_ORDERED:
        if alias in raw:
            crop_found = CANON_CROP[alias]
            rest = rest.replace(alias, " ")
            break

    rest = rest.replace("leaf", " ").replace("leaves", " ")
    rest = norm_text(rest)

    disease = "healthy" if (not rest or rest in {"", "leaf", "leaves"}) else rest
    return crop_found, disease


def guess_split(path):
    parts = [p.lower() for p in path.parts]
    for key in ("train", "test", "val", "valid", "validation"):
        if key in parts:
            return key
    return "unknown"


def is_image(path):
    return path.suffix.lower() in IMG_EXTS


def main():
    assert ROOT.exists(), f"No such a directory: {ROOT}"

    rows = []
    for p in ROOT.rglob("*"):
        if not (p.is_file() and is_image(p)):
            continue
        class_dir = p.parent.name
        crop, disease = parse_class_dir(class_dir)
        if crop is None:
            continue
        disease_snake = to_snake(disease)
        crop_snake = to_snake(crop)
        rows.append({
            "filepath": str(p.resolve()),
            "relpath": str(p.relative_to(ROOT)),
            "crop": crop_snake,
            "disease": disease_snake,
            "label_full": f"{crop_snake}___{disease_snake}",
            "split_guess": guess_split(p)
        })

    if not rows:
        raise SystemExit("Not found imgs or classes")

    df = pd.DataFrame(rows).drop_duplicates()
    print("Total imgs:", len(df))
    print("Crops:", sorted(df["crop"].unique().tolist()))
    print("Items example (crop,disease):")
    print(df[["crop", "disease"]].drop_duplicates().head(10))

    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
