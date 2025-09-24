import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre


def load_classes(path_json: Path):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    crops = data["crops"]
    return crops


def load_and_preprocess(img_path: Path, img_size: int):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    x = vgg_pre(x)  # VGG19 preprocessing
    return x  # (H,W,3)


def main():
    ap = argparse.ArgumentParser(description="Predict crop class (VGG19 Stage A) for a single image.")
    ap.add_argument("--image", type=Path, default="/Users/olgamiskevich/Downloads/indian_cherry.webp")
    ap.add_argument("--model", type=Path,
                    default="/Users/olgamiskevich/PycharmProjects/PlantsDoctor/checkpoints_crop_vgg19_merged/best_crop_vgg19_merged_noaug.keras")
    ap.add_argument("--classes", type=Path,
                    default="/Users/olgamiskevich/PycharmProjects/PlantsDoctor/reports/metrics/classes_from_manifest.json")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    crops = load_classes(args.classes)
    model = tf.keras.models.load_model(args.model, compile=False)

    x = load_and_preprocess(args.image, args.img_size)  # (H,W,3)
    probs = model.predict(x[None, ...], verbose=0)[0]  # (C,)

    k = max(1, min(args.topk, len(crops)))
    topk_idx = probs.argsort()[-k:][::-1]

    for rank, i in enumerate(topk_idx, start=1):
        print(f"#{rank}: {crops[i]:<20s} prob={probs[i]:.4f}")

    out = {
        "image": str(args.image),
        "topk": [
            {"crop": crops[i], "prob": float(probs[i])}
            for i in topk_idx
        ],
        "best": {"crop": crops[topk_idx[0]], "prob": float(probs[topk_idx[0]])}
    }
    print("\nJSON:")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
