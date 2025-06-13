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
    def __init__(self, st_container, class_names):
        super(TrainingPlot, self).__init__()
        self.st_container = st_container
        self.class_names = class_names
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        
        # Loss and accuracy plots
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Loss
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Accuracy
        
        # Confusion matrix placeholder
        self.ax3 = self.fig.add_subplot(gs[:, 1])  # Confusion Matrix
        
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.epoch_count = []
        self.confusion_matrix = None
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.acc.append(logs['accuracy'])
        self.val_losses.append(logs['val_loss'])
        self.val_acc.append(logs['val_accuracy'])
        self.epoch_count.append(epoch + 1)
        
        # Clear the figure
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        
        # Plot training & validation loss values
        self.ax1.plot(self.epoch_count, self.losses, 'b-', label='Training Loss')
        self.ax1.plot(self.epoch_count, self.val_losses, 'r-', label='Validation Loss')
        self.ax1.set_title('Model Loss')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.legend(loc='upper right')
        
        # Plot training & validation accuracy values
        self.ax2.plot(self.epoch_count, self.acc, 'b-', label='Training Accuracy')
        self.ax2.plot(self.epoch_count, self.val_acc, 'r-', label='Validation Accuracy')
        self.ax2.set_title('Model Accuracy')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.legend(loc='lower right')
        
        # Plot confusion matrix every few epochs
        if epoch % 2 == 0 or epoch == self.params['epochs'] - 1:
            # Get predictions
            y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
            y_true = self.validation_data[1]
            
            # Calculate confusion matrix
            cm = tf.math.confusion_matrix(y_true, y_pred)
            cm = cm / tf.reduce_sum(cm, axis=1, keepdims=True)  # Normalize
            
            # Plot confusion matrix
            im = self.ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            self.fig.colorbar(im, ax=self.ax3)
            
            # Add labels
            tick_marks = np.arange(len(self.class_names))
            self.ax3.set_xticks(tick_marks)
            self.ax3.set_yticks(tick_marks)
            self.ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
            self.ax3.set_yticklabels(self.class_names)
            self.ax3.set_title('Confusion Matrix (Normalized)')
            
            # Add text annotations
            thresh = cm.numpy().max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.ax3.text(j, i, f"{cm[i, j]:.2f}",
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")
        
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

def load_data(train_split=0.8, show_sample_images=True, dataset_choice="Both"):
    """
    Load and preprocess Fashion MNIST data with options for visualization
    
    Args:
        train_split: Percentage of data to use for training (0-1)
        show_sample_images: Whether to display sample images
        dataset_choice: Which dataset to show samples from ("Training", "Test", "Both")
        
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    
    # Split training data into train/validation
    num_train = int(len(x_train_full) * train_split)
    x_train, x_val = x_train_full[:num_train], x_train_full[num_train:]
    y_train, y_val = y_train_full[:num_train], y_train_full[num_train:]
    
    # Normalize and reshape
    def preprocess(x):
        x = x.astype('float32') / 255.0
        return x[..., None]  # Add channel dimension
    
    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)
    
    # Show sample images if requested
    if show_sample_images and 'st' in globals():
        st.subheader("Sample Images")
        
        if dataset_choice in ["Training Samples", "Both"]:
            st.caption("Training Samples")
            show_samples(x_train, y_train, num_samples=5)
        
        if dataset_choice in ["Test Samples", "Both"]:
            st.caption("Test Samples")
            show_samples(x_test, y_test, num_samples=5)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def show_samples(images, labels, num_samples=5, class_names=None):
    """Display sample images with their labels"""
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(indices, axes):
        img = images[idx].squeeze()
        label = labels[idx]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{class_names[label]}\n(Class {label})")
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
