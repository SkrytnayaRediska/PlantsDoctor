import os, re, json, random
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor/Plant_leave_diseases_dataset_with_augmentation")

OUT_DIR = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_labels_from_path(rel_path):
    """
    Try to find (crop, disease) from the relative path.
    If not mathing structure returns ('not_found','not_found').
    """
    parts: List[str] = rel_path.parts
    if len(parts) < 2:
        return None, None

    parent = parts[-2]
    if "___" in parent:
        crop, disease = parent.split("___", 1)
        return crop.strip(), disease.strip()

    # Not mathing structure f.e. for background-only imgs
    return "not_found", "not_found"


def is_image(path):
    return path.suffix.lower() in IMG_EXTS


def norm(some_str):
    return re.sub(r"\s+", "_", some_str.strip()).lower()


def save_csv(df: pd.DataFrame, path: Path):
    cols = ["filepath", "crop", "disease", "label_full", "relpath"]
    df[cols].to_csv(path, index=False)


def scan_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            rel = p.relative_to(root)
            crop, disease = parse_labels_from_path(rel)
            if crop is None:
                continue
            rows.append({
                "filepath": str(p.resolve()),
                "relpath": str(rel),
                "crop": crop,
                "disease": disease,
                "label_full": f"{crop}___{disease}"
            })
    if not rows:
        raise SystemExit(f"No imgs in {root}")
    return pd.DataFrame(rows)


df = scan_dataset(DATA_ROOT)

# normalization
df["crop"] = df["crop"].astype(str).map(norm)
df["disease"] = df["disease"].astype(str).map(norm)
df["label_full"] = df.apply(lambda r: f"{r['crop']}___{r['disease']}", axis=1)


# i guess we dont need all the 1000+ imgs of a background
cap = 400
nf_mask = (df["crop"] == "not_found") & (df["disease"] == "not_found")
nf_count = int(nf_mask.sum())
if nf_count > cap:
    keep_nf = df[nf_mask].sample(n=cap, random_state=RANDOM_STATE)
    df = pd.concat([df[~nf_mask], keep_nf], ignore_index=True).reset_index(drop=True)
    print(f"not_found/not_found: было {nf_count}, оставили {cap} (удалили {nf_count - cap})")
else:
    print(f"not_found/not_found: {nf_count} — ограничение {cap} не требуется")

# kinda stratification
strat_col = "label_full" if df["label_full"].nunique() > 1 else "crop"

df_trainval, df_test = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[strat_col]
)

val_size_rel = VAL_SIZE / (1.0 - TEST_SIZE)
df_train, df_val = train_test_split(
    df_trainval, test_size=val_size_rel, random_state=RANDOM_STATE, stratify=df_trainval[strat_col]
)

save_csv(df_train, OUT_DIR / "train.csv")
save_csv(df_val, OUT_DIR / "val.csv")
save_csv(df_test, OUT_DIR / "test.csv")

print(f"Train: {len(df_train)}   Val: {len(df_val)}   Test: {len(df_test)}")

# result of classes split
CROPS = sorted(df["crop"].unique().tolist())
diseases_by_crop = {c: sorted(df[df["crop"] == c]["disease"].unique().tolist()) for c in CROPS}
meta = {"crops": CROPS, "diseases": diseases_by_crop}
meta_path = OUT_DIR.parent / "reports/metrics/classes_from_manifest.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
