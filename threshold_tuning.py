import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -----------------------
# 1️⃣ Load model
# -----------------------
model_path = "v4_crystals_final.keras"  # or "v4_crystals_final_savedmodel"
if model_path.endswith(".keras") and os.path.exists(model_path):
    print(f"Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path)
elif os.path.exists("v4_crystals_final_savedmodel"):
    print("Loading SavedModel...")
    model = tf.keras.models.load_model("v4_crystals_final_savedmodel")
else:
    raise FileNotFoundError("❌ No trained model found!")

# -----------------------
# 2️⃣ Load TFRecord test dataset
# -----------------------
feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_image(example["image/encoded"], channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image.set_shape([224, 224, 3])

    raw_label = tf.cast(example["image/class/label"], tf.int32)
    label = tf.where(raw_label == 1, 1, 0)

    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 16
test_files = tf.io.gfile.glob("test_balanced/test_*")

test_dataset_full = (
    tf.data.TFRecordDataset(test_files)
    .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# -----------------------
# 3️⃣ Collect predictions
# -----------------------
y_true, y_pred_probs = [], []

for images, labels in test_dataset_full:
    probs = model.predict(images, verbose=0).ravel()
    y_pred_probs.extend(probs)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

print(f"Collected {len(y_true)} predictions")

# -----------------------
# 4️⃣ Threshold tuning
# -----------------------
thresholds = np.arange(0.1, 0.91, 0.05)
results = []

for t in thresholds:
    y_pred = (y_pred_probs >= t).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    results.append((t, precision, recall, f1))

print("\n📊 Threshold results:")
for t, p, r, f1 in results:
    print(f"Threshold={t:.2f} | Precision={p:.3f} | Recall={r:.3f} | F1={f1:.3f}")

# -----------------------
# 5️⃣ Find best threshold by F1
# -----------------------
best_thr, best_p, best_r, best_f1 = max(results, key=lambda x: x[3])
print(f"\n✅ Best threshold = {best_thr:.2f} with F1={best_f1:.3f}, "
      f"Precision={best_p:.3f}, Recall={best_r:.3f}")

# -----------------------
# 6️⃣ Plot curves
# -----------------------
thr = [r[0] for r in results]
prec = [r[1] for r in results]
rec = [r[2] for r in results]
f1s = [r[3] for r in results]

plt.figure(figsize=(8,6))
plt.plot(thr, prec, marker='o', label="Precision")
plt.plot(thr, rec, marker='o', label="Recall")
plt.plot(thr, f1s, marker='o', label="F1", linewidth=2)
plt.axvline(best_thr, color="red", linestyle="--", label=f"Best Thr={best_thr:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning")
plt.legend()
plt.grid(True)
plt.show()
