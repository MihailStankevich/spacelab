import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------
# 0️⃣ Setup
# -----------------------
os.makedirs("checkpoints", exist_ok=True)

# Paths: combine old + new TFRecords
train_files = tf.io.gfile.glob("train_balanced/train_*")   # merged TFRecords (old + new)
test_files = tf.io.gfile.glob("test_balanced/test_*")
print(f"Found {len(train_files)} train files, {len(test_files)} test files")

# -----------------------
# 1️⃣ TFRecord Parsing
# -----------------------
feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example_proto, augment=False):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_image(example["image/encoded"], channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image.set_shape([224, 224, 3])

    raw_label = tf.cast(example["image/class/label"], tf.int32)
    label = tf.where(raw_label == 1, 1, 0)

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

    image = tf.cast(image, tf.float32) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

train_dataset_full = (
    tf.data.TFRecordDataset(train_files)
    .map(lambda x: _parse_function(x, augment=True), num_parallel_calls=AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset_full = (
    tf.data.TFRecordDataset(test_files)
    .map(lambda x: _parse_function(x, augment=False), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# -----------------------
# 2️⃣ Model Definition (load full model if available)
# -----------------------
model = None

# Prefer loading the exact graph that works in test_image.py to avoid kernel layout
# and axis transpose issues when re-constructing the model.
if os.path.exists("v3_crystals_balanced_final_savedmodel"):
    print("Loading SavedModel for fine-tuning...")
    model = tf.keras.models.load_model("v3_crystals_balanced_final_savedmodel", compile=False)
elif os.path.exists("v3_crystals_balanced_final_legacy.keras"):
    print("Loading legacy .keras model for fine-tuning...")
    model = tf.keras.models.load_model("v3_crystals_balanced_final_legacy.keras", compile=False)
else:
    print("No full model found. Rebuilding architecture and loading weights if available...")
    base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(224,224,3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    if os.path.exists("v3_crystals_balanced_final.weights.h5"):
        try:
            model.load_weights("v3_crystals_balanced_final.weights.h5")
            print("✅ Loaded previous weights for continued training")
        except Exception as e:
            print(f"⚠️  Could not load weights cleanly: {e}")
            try:
                model.load_weights("v3_crystals_balanced_final.weights.h5", skip_mismatch=True)
                print("✅ Loaded weights with skip_mismatch=True")
            except Exception as e2:
                print(f"❌ Failed to load weights even with skip_mismatch: {e2}")

print(model.summary())

# -----------------------
# 3️⃣ Fine-tuning setup
# -----------------------
# Unfreeze all layers for full fine-tuning (works for both loaded and rebuilt models)
for layer in model.layers:
    layer.trainable = True

# Recompile with very low LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="auc")]
)

# -----------------------
# 4️⃣ Callbacks
# -----------------------
class F1Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            p, r = logs.get('val_precision', 0), logs.get('val_recall', 0)
            if p + r > 0:
                f1 = 2 * (p*r) / (p+r)
                print(f"   Val F1 Score: {f1:.4f}")

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/v4_model_epoch_{epoch:02d}.keras",  # or .tf for SavedModel
    monitor="val_auc",
    save_best_only=False,      # save every epoch
    save_weights_only=False,   # save full model, not just weights
    verbose=1,
    mode='max'
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=3, mode="max", verbose=1
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=5, restore_best_weights=True, mode="max", verbose=1
)

# -----------------------
# 5️⃣ Training (Phase 3)
# -----------------------
print("\n🚀 Phase 3: Continue fine-tuning with full dataset")
history = model.fit(
    train_dataset_full,
    validation_data=test_dataset_full,
    epochs=6,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb, F1Callback()],
    verbose=1
)

# -----------------------
# 6️⃣ Save Final Model
# -----------------------
model.save_weights("v4_crystals_final.weights.h5")
print("✅ Final weights saved: v4_crystals_final.weights.h5")

# 2. Save full model as TensorFlow SavedModel (recommended for deployment)
try:
    model.save("v4_crystals_final_savedmodel", save_format="tf")
    print("✅ Full model saved as SavedModel: v4_crystals_final_savedmodel/")
except Exception as e:
    print(f"⚠️ Could not save as SavedModel: {e}")

# 3. Save as .keras (Keras v3 format, portable & compact)
try:
    model.save("v4_crystals_final.keras")
    print("✅ Full model saved as Keras v3: v4_crystals_final.keras")
except Exception as e:
    print(f"⚠️ Could not save as .keras: {e}")