import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback
import io
import time

class TrainingPlot(Callback):
    def __init__(self, st_container):
        super(TrainingPlot, self).__init__()
        self.st_container = st_container
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.epoch_count = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.acc.append(logs['accuracy'])
        self.val_losses.append(logs['val_loss'])
        self.val_acc.append(logs['val_accuracy'])
        self.epoch_count.append(epoch + 1)
        
        # Update plots
        self.ax1.clear()
        self.ax1.plot(self.epoch_count, self.losses, 'b-', label='Training Loss')
        self.ax1.plot(self.epoch_count, self.val_losses, 'r-', label='Validation Loss')
        self.ax1.set_title('Training and Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(self.epoch_count, self.acc, 'b-', label='Training Accuracy')
        self.ax2.plot(self.epoch_count, self.val_acc, 'r-', label='Validation Accuracy')
        self.ax2.set_title('Training and Validation Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        
        self.fig.tight_layout()
        
        # Display in Streamlit
        self.st_container.pyplot(self.fig)
        time.sleep(0.1)  # Small delay to allow plot to update

def get_model():
    """Create and return the model architecture"""
    model = Sequential([
        InputLayer(input_shape=(28, 28, 1)),
        
        # Block 1
        Conv2D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),
        
        # Head
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def load_data():
    """Load and preprocess Fashion MNIST data"""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., None]  # Add channel dimension
    x_test = x_test[..., None]    # Add channel dimension
    
    return (x_train, y_train), (x_test, y_test)
