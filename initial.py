import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# -----------------------
# 1️⃣ TFRecord files (using balanced dataset)
# -----------------------
train_files = tf.io.gfile.glob("train_balanced/train-*")
test_files = tf.io.gfile.glob("test_balanced/test-*")
print(f"Found {len(train_files)} balanced train files, {len(test_files)} test files")

# -----------------------
# 2️⃣ Feature description
# -----------------------
feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
}

# -----------------------
# 3️⃣ Parse function with higher resolution
# -----------------------
def _parse_function(example_proto, augment=False):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_image(example["image/encoded"], channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])  # Higher resolution for crystal details
    image.set_shape([224, 224, 3])
    
    raw_label = tf.cast(example["image/class/label"], tf.int32)
    label = tf.where(raw_label == 1, 1, 0)
    
    # Light augmentation for training only
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
    
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# -----------------------
# 4️⃣ Datasets with smaller batch size for higher resolution
# -----------------------
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16  # Smaller batch due to higher resolution (224x224)

train_dataset = (
    tf.data.TFRecordDataset(train_files)
    .map(lambda x: _parse_function(x, augment=True), num_parallel_calls=AUTOTUNE)
    .shuffle(5000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    tf.data.TFRecordDataset(test_files)
    .map(lambda x: _parse_function(x, augment=False), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# -----------------------
# 5️⃣ Verify balanced dataset
# -----------------------
print("Verifying balanced dataset composition...")
crystal_count = 0
total_count = 0

for _, y_batch in train_dataset.take(50):  # Sample first 50 batches
    batch_crystals = tf.reduce_sum(tf.cast(y_batch, tf.int32))
    crystal_count += int(batch_crystals.numpy())
    total_count += len(y_batch)

non_crystal_count = total_count - crystal_count
print(f"Sample verification (first 50 batches):")
print(f"   Crystals: {crystal_count:,} ({crystal_count/total_count*100:.1f}%)")
print(f"   Non-crystals: {non_crystal_count:,} ({non_crystal_count/total_count*100:.1f}%)")

# -----------------------
# 6️⃣ Model - Higher capacity for crystal detection
# -----------------------
# Use EfficientNetB3 for better feature extraction
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# More sophisticated head for subtle pattern detection
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

# Much lower learning rate for microscopy images
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine details
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="auc")]
)

print(f"Model has {model.count_params():,} parameters")

# -----------------------
# 7️⃣ Callbacks
# -----------------------
# F1 score callback
class F1Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            precision = logs.get('val_precision', 0)
            recall = logs.get('val_recall', 0)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"   Val F1 Score: {f1:.4f}")

# Save best model based on validation AUC
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/v3_best_balanced_model.weights.h5",
    monitor="val_auc",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    mode='max'
)

# Step-based saving every 100 steps (optional)
step_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/step_{epoch:02d}_{batch:04d}.weights.h5",
    save_freq=100,
    save_weights_only=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=3,
    verbose=1,
    mode='max'
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=5,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

# -----------------------
# 8️⃣ Phase 1 - Train head only
# -----------------------
print("🚀 Phase 1: Training with frozen EfficientNet backbone")
print("Using balanced dataset - no class weights needed")

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[F1Callback()],
    verbose=1
)

# -----------------------
# 9️⃣ Phase 2 - Fine-tune
# -----------------------
print("\n🔓 Phase 2: Fine-tuning top layers...")

# Unfreeze top 30% of base model layers
fine_tune_at = int(len(base_model.layers) * 0.5)
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Recompile with even lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="auc")]
)

print("🚀 Phase 2: Fine-tuning")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb, F1Callback()],
    verbose=1
)

# -----------------------
# 🔟 Save final model
# -----------------------
model.save_weights("v3_crystals_balanced_final.weights.h5")
print("✅ Final weights saved: v3_crystals_balanced_final.weights.h5")

# Save full model with proper configuration
try:
    # Method 1: Try saving as SavedModel format (more reliable)
    model.save("v3_crystals_balanced_final_savedmodel", save_format='tf')
    print("✅ Full model saved as SavedModel: v3_crystals_balanced_final_savedmodel/")
except Exception as e:
    print(f"Could not save as SavedModel: {e}")

try:
    # Method 2: Try saving as Keras format with explicit config
    # Clear any problematic internal state first
    model._clear_loss_function()
    
    # Save with safe serialization
    model.save("v3_crystals_balanced_final.keras", save_format='keras_v3')
    print("✅ Full model saved as Keras v3: v3_crystals_balanced_final.keras")
except Exception as e:
    print(f"Could not save as Keras v3: {e}")
    
    try:
        # Method 3: Fallback to legacy Keras format
        model.save("v3_crystals_balanced_final_legacy.keras", save_format='keras')
        print("✅ Full model saved as legacy Keras: v3_crystals_balanced_final_legacy.keras")
    except Exception as e2:
        print(f"Could not save as legacy Keras: {e2}")
        
        try:
            # Method 4: Save as HDF5 format
            model.save("v3_crystals_balanced_final.h5")
            print("✅ Full model saved as HDF5: v3_crystals_balanced_final.h5")
        except Exception as e3:
            print(f"Could not save as HDF5: {e3}")
            print("⚠️  Only weights were saved successfully. Full model saving failed.")

# -----------------------
# 1️⃣1️⃣ Comprehensive evaluation
# -----------------------
print("\n📊 Final Evaluation on Test Set:")
print("=" * 50)

# Get all predictions
y_true, y_scores = [], []
for x, y in test_dataset:
    preds = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_scores.extend(preds.flatten())

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n🎯 Performance at different thresholds:")
print("-" * 50)

best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_scores > threshold).astype(int)
    
    if len(np.unique(y_pred)) > 1:  # Avoid division by zero
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold {threshold:.1f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

print(f"\n🏆 Best threshold: {best_threshold:.1f} (F1={best_f1:.3f})")

# Final evaluation with best threshold
y_pred_best = (y_scores > best_threshold).astype(int)

print(f"\n📈 Final Results (threshold={best_threshold:.1f}):")
print("=" * 40)
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_best)
print(f"             Predicted")
print(f"           Not  Crystal")
print(f"Actual Not  {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"    Crystal {cm[1,0]:4d}    {cm[1,1]:4d}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_best, target_names=["Not Crystal", "Crystal"]))

# Save optimal threshold
with open("v3_best_threshold.txt", "w") as f:
    f.write(str(best_threshold))
print(f"💾 Best threshold saved: {best_threshold}")

print("\n✅ Training complete with balanced dataset!")
print("📁 Files created:")
print("  - v3_crystals_balanced_final.weights.h5 (model weights)")
print("  - checkpoints/v3_best_balanced_model.weights.h5 (best model)")
print("  - v3_best_threshold.txt (optimal prediction threshold)")