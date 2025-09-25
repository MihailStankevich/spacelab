import tensorflow as tf
import numpy as np
import os

def test_single_image(image_path):
    """Test the trained crystal detection model on a single image"""
    
    print(f"Testing crystal detection model...")
    print(f"Image: {image_path}")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return None
    
    # Load the optimal threshold
    try:
        with open("v3_best_threshold.txt", "r") as f:
            optimal_threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {optimal_threshold}")
    except:
        optimal_threshold = 0.3
        print(f"Using default threshold: {optimal_threshold}")
    
    # Try to load the model (prioritize SavedModel)
    model = None
    model_paths = [
        ("SavedModel", "v3_crystals_balanced_final_savedmodel"),
        ("Legacy Keras", "v3_crystals_balanced_final_legacy.keras")
    ]
    
    for model_name, model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"\nLoading {model_name}...")
                
                # Custom objects for metrics
                custom_objects = {
                    'precision': tf.keras.metrics.Precision(),
                    'recall': tf.keras.metrics.Recall(), 
                    'auc': tf.keras.metrics.AUC()
                }
                
                if model_name == "SavedModel":
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                else:
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                
                print(f"✅ {model_name} loaded successfully!")
                break
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                try:
                    # Try loading without compilation
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"✅ {model_name} loaded without compilation")
                    break
                except Exception as e2:
                    print(f"❌ Complete failure for {model_name}: {e2}")
                    continue
    
    if model is None:
        print("❌ Could not load any model format")
        return None
    
    print(f"Model has {model.count_params():,} parameters")
    
    # Preprocess the image (same as training)
    try:
        print(f"\nPreprocessing image...")
        
        # Read and decode
        img_raw = tf.io.read_file(image_path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        
        # Resize to training size
        img = tf.image.resize(img, [224, 224])
        img.set_shape([224, 224, 3])
        
        # Normalize to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        
        # Add batch dimension
        img_batch = tf.expand_dims(img, axis=0)
        
        print(f"✅ Image preprocessed to shape: {img_batch.shape}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None
    
    # Make prediction
    try:
        print(f"\nMaking prediction...")
        
        prediction = model.predict(img_batch, verbose=0)
        score = float(prediction[0][0])
        
        # Apply optimal threshold
        is_crystal = score > optimal_threshold
        confidence = score if is_crystal else (1 - score)
        
        print(f"\n" + "=" * 50)
        print("CRYSTAL DETECTION RESULTS")
        print("=" * 50)
        print(f"Raw prediction score: {score:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.2f}")
        print(f"Classification: {'CRYSTAL ✨' if is_crystal else 'NOT CRYSTAL ❌'}")
        print(f"Confidence: {confidence:.1%}")
        
        # Confidence interpretation
        if confidence > 0.8:
            conf_level = "Very High"
        elif confidence > 0.65:
            conf_level = "High"
        elif confidence > 0.55:
            conf_level = "Moderate"
        else:
            conf_level = "Low"
        
        print(f"Confidence Level: {conf_level}")
        
        # Additional thresholds for comparison
        print(f"\n📊 Score interpretation:")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for thresh in thresholds:
            prediction_at_thresh = "Crystal" if score > thresh else "Not Crystal"
            marker = " ← OPTIMAL" if thresh == optimal_threshold else ""
            print(f"  At {thresh:.1f}: {prediction_at_thresh}{marker}")
        
        return {
            'score': score,
            'is_crystal': is_crystal,
            'confidence': confidence,
            'confidence_level': conf_level,
            'threshold_used': optimal_threshold
        }
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def test_multiple_images():
    """Test multiple images if available"""
    
    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Find images in current directory
    available_images = []
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in extensions):
            available_images.append(file)
    
    if not available_images:
        print("No images found in current directory")
        return
    
    print(f"Found {len(available_images)} images:")
    for i, img in enumerate(available_images[:10], 1):
        print(f"  {i}. {img}")
    
    # Test first few images
    for img in available_images[:3]:
        print(f"\n{'='*60}")
        result = test_single_image(img)
        if result:
            print(f"✅ Successfully tested {img}")
        else:
            print(f"❌ Failed to test {img}")

if __name__ == "__main__":
    print("🔬 Crystal Detection V3 - Single Image Testing")
    print("=" * 60)
    
    # Check if model files exist
    model_files = [
        "v3_crystals_balanced_final_savedmodel",
        "v3_crystals_balanced_final_legacy.keras",
        "v3_best_threshold.txt"
    ]
    
    print("Checking model files...")
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Test single image (change filename as needed)
    test_image = "11.jpg"
    
    if os.path.exists(test_image):
        result = test_single_image(test_image)
    else:
        print(f"\nTest image '{test_image}' not found.")
        print("Available images:")
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        available = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in extensions)]
        
        if available:
            for img in available[:5]:
                print(f"  - {img}")
            
            # Test the first available image
            first_image = available[0]
            print(f"\nTesting first available image: {first_image}")
            result = test_single_image(first_image)
        else:
            print("No images found in current directory.")
    
    print(f"\n💡 To test other images:")
    print(f"   result = test_single_image('your_image.jpg')")