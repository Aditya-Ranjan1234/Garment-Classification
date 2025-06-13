import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

def test_model():
    print("Testing model loading...")
    try:
        # Try loading the model with different approaches
        print("Attempt 1: Loading with default settings...")
        model = load_model('fashion_custom_cnn.h5')
        print("✓ Model loaded successfully with default settings")
        return True
    except Exception as e:
        print(f"✗ Error with default loading: {e}")
    
    try:
        print("\nAttempt 2: Loading with custom objects...")
        model = load_model('fashion_custom_cnn.h5', compile=False)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        print("✓ Model loaded successfully with custom compilation")
        return True
    except Exception as e:
        print(f"✗ Error with custom objects: {e}")
    
    try:
        print("\nAttempt 3: Loading model weights only...")
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
        
        # Rebuild the model architecture
        model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(32, 3, padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(32, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            
            Conv2D(64, 3, padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            
            Conv2D(128, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        # Try to load weights
        model.load_weights('fashion_custom_cnn.h5')
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        print("✓ Model loaded successfully by rebuilding architecture")
        
        # Test with sample data
        print("\nTesting with sample data...")
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test[..., None] / 255.0  # Add channel dimension and normalize
        
        # Make prediction
        sample = x_test[0:1]  # Take first test sample
        prediction = model.predict(sample)
        predicted_class = np.argmax(prediction[0])
        
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        print(f"\nPrediction results:")
        print(f"- Predicted class: {class_names[predicted_class]} (Confidence: {np.max(prediction[0]):.2f})")
        print("\nTop 3 predictions:")
        top3 = np.argsort(prediction[0])[-3:][::-1]
        for i in top3:
            print(f"- {class_names[i]}: {prediction[0][i]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with manual loading: {e}")
        return False

if __name__ == "__main__":
    test_model()
