import re, json, hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PV_OUT_DIR = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor/out")
CSV_PV_TRAIN = PV_OUT_DIR / "train.csv"
CSV_PV_VAL = PV_OUT_DIR / "val.csv"
CSV_PV_TEST = PV_OUT_DIR / "test.csv"

CSV_PLANTDOC = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor/plantdoc_parsed.csv")

OUT_DIR = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor/out_merged")
REPORTS = OUT_DIR.parent / "reports/metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42


def norm_text(some_str):
    return re.sub(r"\s+", "_", str(some_str).strip().lower())


CROP_CANON = {
    "pepper": "pepper",
    "pepper_bell": "pepper",
    "bell_pepper": "pepper",
    "capsicum_annuum": "pepper",
    "corn": "corn",
    "maize": "corn",
    "zea_mays": "corn",
    "tomato": "tomato",
    "solanum_lycopersicum": "tomato",
    "potato": "potato",
    "solanum_tuberosum": "potato",
    "apple": "apple",
    "malus_domestica": "apple",
    "cucumber": "cucumber",
    "cucumis_sativus": "cucumber",
    "grape": "grape",
    "vitis_vinifera": "grape",
    "wheat": "wheat",
    "triticum_aestivum": "wheat",
    "soybean": "soybean",
    "glycine_max": "soybean",
    "orange": "orange",
    "citrus": "orange",
    "blueberry": "blueberry",
    "raspberry": "raspberry",
    "strawberry": "strawberry",
    "cherry": "cherry",
    "peach": "peach",
    "squash": "squash",
}


def canon_crop(some_str):
    s_norm = norm_text(some_str)
    return CROP_CANON.get(s_norm, s_norm)


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_csv_minimal(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["filepath", "crop", "disease", "label_full"]:
        assert col in df.columns, f"{path} не содержит столбец {col}"
    if "relpath" not in df.columns:
        df["relpath"] = ""
    return df[["filepath", "crop", "disease", "label_full", "relpath"]].copy()


def save_csv(df: pd.DataFrame, path: Path):
    df = df.copy()
    if "md5" in df.columns:
        df = df.drop(columns=["md5"])
    cols = ["filepath", "crop", "disease", "label_full", "relpath"]
    df[cols].to_csv(path, index=False)


assert CSV_PV_TRAIN.exists() and CSV_PV_VAL.exists() and CSV_PV_TEST.exists(), "Нет PV CSV (train/val/test)"
assert CSV_PLANTDOC.exists(), f"Нет PlantDoc CSV: {CSV_PLANTDOC}"

pv_train = load_csv_minimal(CSV_PV_TRAIN)
pv_val = load_csv_minimal(CSV_PV_VAL)
pv_test = load_csv_minimal(CSV_PV_TEST)
pd_all = load_csv_minimal(CSV_PLANTDOC)

pool = pd.concat([pv_train, pv_val, pv_test, pd_all], ignore_index=True)

pool["filepath"] = pool["filepath"].astype(str)
pool["crop"] = pool["crop"].astype(str).map(canon_crop)
pool["disease"] = pool["disease"].astype(str).map(norm_text)
pool["label_full"] = pool.apply(lambda r: f"{r['crop']}___{r['disease']}", axis=1)

before = len(pool)
pool = pool.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
print(f"Removed duplications by filepath: {before - len(pool)}")

pool["md5"] = pool["filepath"].apply(md5_file)
before = len(pool)
pool = pool.drop_duplicates(subset=["md5"]).reset_index(drop=True)
print(f"Removed duplications by md5: {before - len(pool)}")

strat_col = "label_full" if pool["label_full"].nunique() > 1 else "crop"

pool_trainval, pool_test = train_test_split(
    pool, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=pool[strat_col]
)

val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
pool_train, pool_val = train_test_split(
    pool_trainval, test_size=val_rel, random_state=RANDOM_STATE, stratify=pool_trainval[strat_col]
)

save_csv(pool_train, OUT_DIR / "train.csv")
save_csv(pool_val, OUT_DIR / "val.csv")
save_csv(pool_test, OUT_DIR / "test.csv")

print(f"Train: {len(pool_train)} | Val: {len(pool_val)} | Test: {len(pool_test)}")
print("CSV:", OUT_DIR)

CROPS = sorted(pool["crop"].unique().tolist())
diseases_by_crop = {c: sorted(pool[pool["crop"] == c]["disease"].unique().tolist()) for c in CROPS}

with open(REPORTS / "classes_from_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"crops": CROPS, "diseases": diseases_by_crop}, f, ensure_ascii=False, indent=2)

summary = {
    "total_images": int(len(pool)),
    "num_crops": int(len(CROPS)),
    "num_pairs": int(pool["label_full"].nunique()),
    "per_crop_counts": pool["crop"].value_counts().sort_index().to_dict(),
    "split_sizes": {"train": int(len(pool_train)), "val": int(len(pool_val)), "test": int(len(pool_test))}
}
with open(REPORTS / "build_merged_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
