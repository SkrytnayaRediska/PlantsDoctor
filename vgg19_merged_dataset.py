import json
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg_pre
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJ_DIR = Path("/Users/olgamiskevich/PycharmProjects/PlantsDoctor")
CSV_TRAIN = PROJ_DIR / "out_merged/train.csv"
CSV_VAL = PROJ_DIR / "out_merged/val.csv"
CKPT_DIR = PROJ_DIR / "checkpoints_crop_vgg19_merged"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES_JSON = PROJ_DIR / "reports/metrics/classes_from_manifest_merged.json"
CLASSES_JSON.parent.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH = 32
EPOCHS_WARM = 3
EPOCHS_FT = 5
LR_WARM = 3e-4
LR_FT = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH_WARM = 0.10
LABEL_SMOOTH_FT = 0.03
SEED = 42


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"filepath", "crop"}.issubset(df.columns), f"{path} must contain columns: filepath, crop"
    df = df[["filepath", "crop"]].copy()
    df["filepath"] = df["filepath"].astype(str)
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    return df


def make_opt(lr):
    if hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def decode(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    return vgg_pre(img)  # VGG19 preprocessing (RGB->BGR, centering)


def make_ds(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    p = tf.constant(df["filepath"].values)
    y = tf.constant(df["y"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if training:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    def _map(p_, y_):
        x = decode(p_)
        y1 = tf.one_hot(y_, NUM_CLASSES, dtype=tf.float32)
        return x, y1

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


tf.random.set_seed(SEED)

df_tr = read_split(CSV_TRAIN)
df_va = read_split(CSV_VAL)

CROPS = sorted(df_tr["crop"].unique().tolist())
CROP2ID = {c: i for i, c in enumerate(CROPS)}
NUM_CLASSES = len(CROPS)

with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump({"crops": CROPS}, f, ensure_ascii=False, indent=2)

df_tr["y"] = df_tr["crop"].map(CROP2ID).astype(int)
df_va["y"] = df_va["crop"].map(CROP2ID).astype(int)

counts = Counter(df_tr["y"].tolist())
n_samples = len(df_tr)
class_weights = {cls: float(n_samples) / (len(counts) * cnt) for cls, cnt in counts.items()}

print(f"Classes ({NUM_CLASSES}): {', '.join(CROPS)}")
print("Train size:", len(df_tr), "| Val size:", len(df_va))
print("Top-5 class counts (train):", Counter(df_tr['y']).most_common(5))

train_ds = make_ds(df_tr, training=True)
val_ds = make_ds(df_va, training=False)

base = VGG19(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False  # warmup

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
logits = tf.keras.layers.Dense(NUM_CLASSES, activation=None)(x)
probs = tf.keras.layers.Activation("softmax", name="probs")(logits)
model = tf.keras.Model(inp, probs)

model.compile(
    optimizer=make_opt(LR_WARM),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH_WARM),
    metrics=["accuracy"]
)
print("\nStep 1: head-only (VGG19 frozen)")
es_warm = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_WARM,
    callbacks=[es_warm], class_weight=class_weights, verbose=1
)

for layer in base.layers:
    layer.trainable = False
unfreeze = False
for layer in base.layers:
    if "block4" in layer.name or "block5" in layer.name:
        unfreeze = True
    if unfreeze and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization)):
        layer.trainable = True

model.compile(
    optimizer=make_opt(LR_FT),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH_FT),
    metrics=["accuracy"]
)

ckpt = tf.keras.callbacks.ModelCheckpoint(
    str(CKPT_DIR / "best_crop_vgg19_merged_noaug.keras"),
    monitor="val_accuracy", save_best_only=True
)
es_ft = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

print("\nStep 2: fine-tune VGG19 top blocks (no augmentation)")
hist = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
    callbacks=[ckpt, es_ft], class_weight=class_weights, verbose=1
)

print("Saved:", CKPT_DIR / "best_crop_vgg19_merged_noaug.keras")

probs = model.predict(val_ds, verbose=0)
y_true = df_va["y"].to_numpy()
y_pred = probs.argmax(1)

print("\nVAL metrics:")
print("Accuracy:", round(float(accuracy_score(y_true, y_pred)), 4))
print("Balanced accuracy:", round(float(balanced_accuracy_score(y_true, y_pred)), 4))
print("Macro-F1:", round(float(f1_score(y_true, y_pred, average='macro')), 4))
