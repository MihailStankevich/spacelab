import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import json

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# -----------------------
# 1. TFRecord files (using balanced dataset)
# -----------------------
train_files = tf.io.gfile.glob("train_balanced/train-*")
test_files = tf.io.gfile.glob("test_balanced/test-*")
print(f"Found {len(train_files)} balanced train files, {len(test_files)} test files")

# -----------------------
# 2. Feature description
# -----------------------
feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64),
}

# -----------------------
# 3. Parse function with higher resolution
# -----------------------
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

# -----------------------
# 4. Datasets
# -----------------------
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

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
# 5. Model creation function
# -----------------------
def create_model():
    """Create model - use function to ensure clean model creation"""
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model

# -----------------------
# 6. Create and train model
# -----------------------
model = create_model()
print(f"Model has {model.count_params():,} parameters")

# F1 score callback
class F1Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            precision = logs.get('val_precision', 0)
            recall = logs.get('val_recall', 0)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"   Val F1 Score: {f1:.4f}")

# -----------------------
# 7. Phase 1 - Train head only (1 epoch)
# -----------------------
print("\nPhase 1: Training with frozen EfficientNet backbone (1 epoch)")

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1,
    callbacks=[F1Callback()],
    verbose=1
)

# Save Phase 1 model weights
model.save_weights("checkpoints/phase1_test.weights.h5")
print("Phase 1 weights saved")

# -----------------------
# 8. Phase 2 - Fine-tune (1 epoch)
# -----------------------
print("\nPhase 2: Fine-tuning top layers (1 epoch)...")

# Unfreeze top 50% of base model layers
fine_tune_at = int(len(model.layers[0].layers) * 0.5)
print(f"Unfreezing layers from {fine_tune_at} onwards (total base layers: {len(model.layers[0].layers)})")

for layer in model.layers[0].layers[fine_tune_at:]:
    layer.trainable = True

# Count trainable parameters
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"Trainable parameters after unfreezing: {trainable_params:,}")

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Much lower for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="auc")]
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1,
    callbacks=[F1Callback()],
    verbose=1
)

# -----------------------
# 9. Model saving - ROBUST APPROACH
# -----------------------
print("\nSaving model with multiple approaches...")

# Save final weights (this always works)
model.save_weights("checkpoints/final_test_model.weights.h5")
print("✅ Final weights saved: checkpoints/final_test_model.weights.h5")

# Save model architecture
print("Saving model architecture...")
try:
    model_config = model.get_config()
    with open("checkpoints/model_config_test.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print("✅ Model architecture saved: checkpoints/model_config_test.json")
except Exception as e:
    print(f"❌ Could not save model config: {e}")

# Save training configuration
training_config = {
    'fine_tune_at': fine_tune_at,
    'total_base_layers': len(model.layers[0].layers),
    'input_shape': [224, 224, 3],
    'base_model': 'EfficientNetB3',
    'phase1_epochs': 1,
    'phase2_epochs': 1,
    'phase1_lr': 1e-5,
    'phase2_lr': 1e-6,
    'classification_layers': [
        {'type': 'Dense', 'units': 512, 'activation': 'relu'},
        {'type': 'BatchNormalization'},
        {'type': 'Dropout', 'rate': 0.3},
        {'type': 'Dense', 'units': 256, 'activation': 'relu'},
        {'type': 'BatchNormalization'},
        {'type': 'Dropout', 'rate': 0.3},
        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
        {'type': 'Dropout', 'rate': 0.2},
        {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ]
}

with open("checkpoints/training_config_test.json", "w") as f:
    json.dump(training_config, f, indent=2)
print("✅ Training configuration saved: checkpoints/training_config_test.json")

# Create clean model for saving
def save_clean_model():
    """Create a fresh model and transfer weights for clean saving"""
    print("Creating clean model for saving...")
    
    # Create fresh model
    clean_model = create_model()
    
    # Set same fine-tuning state
    for layer in clean_model.layers[0].layers[fine_tune_at:]:
        layer.trainable = True
    
    # Recompile to match training state
    clean_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall", "auc"]
    )
    
    # Build the model with dummy input
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = clean_model(dummy_input)
    
    # Transfer weights from trained model
    clean_model.set_weights(model.get_weights())
    print(f"Weights transferred to clean model")
    
    return clean_model

# Try SavedModel format
try:
    clean_model = save_clean_model()
    clean_model.save("checkpoints/crystal_test_model_savedmodel", save_format='tf')
    print("✅ Full model saved as SavedModel: checkpoints/crystal_test_model_savedmodel/")
except Exception as e:
    print(f"❌ SavedModel failed: {e}")

# Try Keras format
try:
    clean_model = save_clean_model()
    clean_model.save("checkpoints/crystal_test_model.keras")
    print("✅ Full model saved as Keras: checkpoints/crystal_test_model.keras")
except Exception as e:
    print(f"❌ Keras format failed: {e}")

# -----------------------
# 10. Quick evaluation
# -----------------------
print("\nQuick evaluation on test set...")

# Sample evaluation on a subset
sample_size = 5  # Just test a few batches
y_true, y_scores = [], []

for i, (x, y) in enumerate(test_dataset.take(sample_size)):
    preds = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_scores.extend(preds.flatten())
    print(f"Batch {i+1}: {len(y)} samples processed")

y_true = np.array(y_true)
y_scores = np.array(y_scores)

print(f"\nEvaluated on {len(y_true)} samples:")
print(f"Score range: {y_scores.min():.3f} to {y_scores.max():.3f}")
print(f"Mean score: {y_scores.mean():.3f}")

# Test a threshold
threshold = 0.5
y_pred = (y_scores > threshold).astype(int)

if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"At threshold {threshold}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
else:
    print("Not enough variety in predictions/labels for full evaluation")

# Save threshold
with open("checkpoints/test_threshold.txt", "w") as f:
    f.write("0.5")

print("\n" + "="*60)
print("TEST TRAINING COMPLETE!")
print("="*60)
print("Files created:")
print("  📁 checkpoints/final_test_model.weights.h5 - Final model weights")
print("  📁 checkpoints/phase1_test.weights.h5 - Phase 1 weights")
print("  📁 checkpoints/model_config_test.json - Model architecture")
print("  📁 checkpoints/training_config_test.json - Training configuration")
print("  📁 checkpoints/test_threshold.txt - Test threshold")
if os.path.exists("checkpoints/crystal_test_model_savedmodel"):
    print("  📁 checkpoints/crystal_test_model_savedmodel/ - Full SavedModel")
if os.path.exists("checkpoints/crystal_test_model.keras"):
    print("  📁 checkpoints/crystal_test_model.keras - Full Keras model")

print(f"\nNext step: Test loading with your image using final_test_model.weights.h5")