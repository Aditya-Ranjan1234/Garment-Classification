import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Create cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache', 'fashion_mnist')
os.makedirs(CACHE_DIR, exist_ok=True)

def download_and_cache_data():
    """Download and cache the Fashion MNIST dataset"""
    print("Downloading Fashion MNIST dataset...")
    
    # This will download the dataset if not already in the default cache
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Save the data to our cache directory
    np.savez(os.path.join(CACHE_DIR, 'train.npz'), x=x_train, y=y_train)
    np.savez(os.path.join(CACHE_DIR, 'test.npz'), x=x_test, y=y_test)
    
    print(f"Dataset downloaded and cached to {CACHE_DIR}")
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

if __name__ == "__main__":
    download_and_cache_data()
