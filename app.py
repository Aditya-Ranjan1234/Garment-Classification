import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import time
import warnings
from io import StringIO, BytesIO
import tempfile

# Import training utilities
from train_utils import TrainingPlot, get_model, load_data
import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback  # Add this import

# Cache directory for dataset
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

def model_to_bytes(model, filename):
    """Save model to bytes for download"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
        model.save(tmp.name, save_format='h5')
        with open(tmp.name, 'rb') as f:
            model_bytes = f.read()
    return model_bytes

# Class labels for Fashion MNIST
CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Class descriptions for educational purposes
CLASS_DESCRIPTIONS = {
    "T-shirt/top": "A T-shirt or top, typically short-sleeved and casual",
    "Trouser": "Pants or trousers, including jeans, chinos, etc.",
    "Pullover": "A sweater or pullover, typically long-sleeved",
    "Dress": "A one-piece garment for women or girls",
    "Coat": "Outerwear like jackets or coats",
    "Sandal": "Open shoes with straps, typically worn in warm weather",
    "Shirt": "A button-up shirt, typically with a collar",
    "Sneaker": "Athletic shoes or trainers",
    "Bag": "Handbags, backpacks, or similar accessories",
    "Ankle boot": "Boots that cover the foot and ankle"
}

# Set page config with a wider layout - must be the first Streamlit command
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style
plt.style.use('seaborn')


# Class labels for Fashion MNIST
CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Model path
MODEL_PATH = os.path.join("D:", "My Projects", "somesh", "garmentcv-app", "fashion_custom_cnn.h5")

def load_custom_model():
    """Load the custom trained model from the specified path"""
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
            
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Compile the model
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        st.success("Successfully loaded pre-trained model!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
        return None

def preprocess_image(img, target_size=(28, 28)):
    """Preprocess the image for model prediction"""
    try:
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize and normalize
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Reshape for model input (28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_garment(model, img_array):
    """Make prediction using the model"""
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return 0, 0.0, np.zeros(len(CLASS_LABELS))

def create_model_architecture_image():
    """Create a visualization of the model architecture"""
    # Create a figure with better dimensions
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Model architecture description with layer details
    layers = [
        ("Input\n28×28×1", 0.1, 0.8),
        ("Conv2D (32)\n3×3, ReLU", 0.3, 0.8),
        ("BatchNorm", 0.3, 0.7),
        ("Conv2D (32)\n3×3, ReLU", 0.5, 0.7),
        ("BatchNorm", 0.5, 0.6),
        ("MaxPool2D 2×2", 0.7, 0.65),
        ("Dropout 0.25", 0.7, 0.75),
        ("Conv2D (64)\n3×3, ReLU", 0.3, 0.5),
        ("BatchNorm", 0.3, 0.4),
        ("Conv2D (64)\n3×3, ReLU", 0.5, 0.4),
        ("BatchNorm", 0.5, 0.3),
        ("MaxPool2D 2×2", 0.7, 0.35),
        ("Dropout 0.25", 0.7, 0.45),
        ("Conv2D (128)\n3×3, ReLU", 0.5, 0.2),
        ("BatchNorm", 0.5, 0.1),
        ("MaxPool2D 2×2", 0.7, 0.15),
        ("Dropout 0.25", 0.7, 0.25),
        ("Flatten", 0.3, 0.0),
        ("Dense (256)\nReLU", 0.5, 0.0),
        ("BatchNorm", 0.5, -0.1),
        ("Dropout 0.5", 0.5, -0.2),
        ("Dense (10)\nSoftmax", 0.7, -0.1)
    ]
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
        (19, 20), (20, 21)
    ]
    
    # Draw connections first (behind the nodes)
    for i, j in connections:
        ax.plot([layers[i][1], layers[j][1]], [layers[i][2], layers[j][2]], 
                'k-', alpha=0.3, linewidth=2)
    
    # Draw nodes on top
    for i, (name, x, y) in enumerate(layers):
        color = plt.cm.tab20(i / len(layers))
        circle = plt.Circle((x, y), 0.02, color=color, zorder=10)
        ax.add_artist(circle)
        ax.text(x, y + 0.03, name, ha='center', va='bottom', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img

def show_samples(images, labels, num_samples=5, class_names=None):
    """Display sample images with their labels"""
    if num_samples > len(images):
        num_samples = len(images)
        
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    samples = images[indices]
    sample_labels = labels[indices]
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i, (img, label_idx) in enumerate(zip(samples, sample_labels)):
        ax = axes[i]
        ax.imshow(img.squeeze(), cmap='gray')
        if class_names is not None:
            ax.set_title(f"{class_names[label_idx]} ({label_idx})")
        else:
            ax.set_title(f"Class {label_idx}")
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def load_data(train_split=0.8, show_sample_images=True, dataset_choice="Both"):
    """
    Load and preprocess Fashion MNIST data with caching
    
    Args:
        train_split: Percentage of data to use for training (0-1)
        show_sample_images: Whether to display sample images
        dataset_choice: Which dataset to show samples from ("Training", "Test", "Both")
        
    Returns:
        Tuple of (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Define cache paths
    cache_dir = os.path.join(CACHE_DIR, 'fashion_mnist')
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache = os.path.join(cache_dir, 'train.npz')
    test_cache = os.path.join(cache_dir, 'test.npz')
    
    # Try to load from cache first
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        try:
            train_data = np.load(train_cache, allow_pickle=True)
            x_train_full, y_train_full = train_data['x'], train_data['y']
            
            test_data = np.load(test_cache, allow_pickle=True)
            x_test, y_test = test_data['x'], test_data['y']
            
            st.sidebar.success("✅ Loaded dataset from cache")
        except Exception as e:
            st.sidebar.warning(f"Error loading cached data: {e}. Downloading fresh data...")
            (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
            
            # Save to cache
            np.savez(train_cache, x=x_train_full, y=y_train_full)
            np.savez(test_cache, x=x_test, y=y_test)
    else:
        # Download fresh data
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
        
        # Save to cache
        np.savez(train_cache, x=x_train_full, y=y_train_full)
        np.savez(test_cache, x=x_test, y=y_test)
    
    # Split training data into training and validation sets
    num_train = int(len(x_train_full) * train_split)
    x_train, x_val = x_train_full[:num_train], x_train_full[num_train:]
    y_train, y_val = y_train_full[:num_train], y_train_full[num_train:]
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Show sample images if requested
    if show_sample_images and 'st' in globals():
        st.subheader("Sample Images")
        if dataset_choice in ["Training Samples", "Both"]:
            st.write("### Training Samples")
            show_samples(x_train, y_train, num_samples=5, class_names=CLASS_LABELS)
        if dataset_choice in ["Test Samples", "Both"]:
            st.write("### Test Samples")
            show_samples(x_test, y_test, num_samples=5, class_names=CLASS_LABELS)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Header
    st.title("👕 Fashion Item Classifier")
    st.markdown("---")
    
    # Introduction
    st.header("About the Model")
    st.markdown("""
    This Fashion Item Classifier is a deep learning model that can identify different types of clothing items from images. 
    It's built using a Convolutional Neural Network (CNN) and trained on the Fashion MNIST dataset.
    """)
    
    # Model Architecture
    st.subheader("Model Architecture")
    st.markdown("""
    The model consists of multiple convolutional and pooling layers followed by fully connected layers:
    - **Input Layer**: 28x28 grayscale images
    - **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
    - **Batch Normalization**: For faster and more stable training
    - **Max Pooling**: For dimensionality reduction
    - **Dropout**: To prevent overfitting
    - **Dense Layers**: For final classification
    """)
    
    # Show model architecture visualization
    st.image(create_model_architecture_image(), caption="Model Architecture Visualization", use_column_width=True)
    
    # Training Section
    st.header("🎓 Train Your Own Model")
    st.markdown("""
    This interactive demo lets you explore how a neural network learns to classify fashion items. 
    You can customize the training process and visualize the learning progress in real-time.
    """)
    
    # Training parameters
    with st.expander("⚙️ Training Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("Number of Epochs", 1, 20, 5, 1,
                             help="Number of complete passes through the training dataset",
                             key="epochs_slider")
            batch_size = st.select_slider("Batch Size", 
                                       options=[32, 64, 128, 256], 
                                       value=64,
                                       help="Number of samples per gradient update",
                                       key="batch_size_slider")
        
        with col2:
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001,
                                   format="%.4f",
                                   help="Step size at each iteration while moving toward minimizing the loss",
                                   key="lr_slider")
            
            data_split = st.slider("Train/Validation Split", 70, 90, 80, 5,
                                       format="%d%%",
                                       help="Percentage of training data to use for training vs validation",
                                       key="split_slider")
        
        with col3:
            dataset_choice = st.selectbox("Dataset to Visualize",
                                       ["Training Samples", "Test Samples", "Both"],
                                       index=2,
                                       help="Which dataset to show samples from",
                                       key="dataset_choice_select")
            
            show_sample_images = st.checkbox("Show Sample Images", True,
                                          help="Display sample images from the selected dataset",
                                          key="show_samples_check")
    
    # Training controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        train_button = st.button("🚀 Start Training", key="train_button")
    with col2:
        stop_button = st.button("⏹️ Stop Training", key="stop_button", disabled=not train_button)
    with col3:
        show_samples_option = st.checkbox("Show Sample Images", value=True, key="show_samples_option")
    
    if stop_button:
        stop_training = True
        st.warning("Training will stop after the current epoch completes.")
    
    # Placeholder for training output
    training_output = st.empty()
    
    if train_button:
        stop_training = False
        with training_output.container():
            st.info("🚀 Starting training... This may take a few minutes.")
            
            # Initialize progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            plot_placeholder = st.empty()
            
            # Initialize callback with class names
            training_callback = TrainingPlot(plot_placeholder, CLASS_LABELS)
            
            # Load data
            with st.spinner("Loading Fashion MNIST dataset..."):
                (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(
                    train_split=data_split/100.0,
                    show_sample_images=show_samples_option,
                    dataset_choice=dataset_choice
                )
            
            # Create and compile model
            with st.spinner("Initializing model..."):
                model = get_model()
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
            
            # Display model summary
            with st.expander("📊 Model Architecture", expanded=False):
                buffer = StringIO()
                model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                st.text(buffer.getvalue())
            
            # Train model with our custom callback
            try:
                with st.spinner("Training in progress..."):
                    history = model.fit(
                        x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[training_callback],
                        verbose=0
                    )
                
                # Show final metrics on test set if training completed
                if not stop_training:
                    st.success("✅ Training complete!")
                    with st.spinner("Evaluating on test set..."):
                        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Loss", f"{test_loss:.4f}")
                    with col2:
                        st.metric("Test Accuracy", f"{test_acc*100:.2f}%")
                    
                    # Show some example predictions
                    st.subheader("Example Predictions on Test Set")
                    with st.spinner("Generating predictions..."):
                        show_predictions(model, x_test, y_test, CLASS_LABELS)
                    
                    # Download button for the trained model
                    st.download_button(
                        label="💾 Download Model",
                        data=model_to_bytes(model, 'fashion_mnist_model.h5'),
                        file_name="fashion_mnist_model.h5",
                        mime="application/octet-stream"
                    )
                else:
                    st.warning("⚠️ Training was stopped early.")
                    
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
            finally:
                # Reset the stop flag
                stop_training = False
    
    # Add a divider before the upload section
    st.markdown("---")
    
    # Image upload section
    st.header("🔍 Upload an Image for Classification")
    st.markdown("Upload an image of a fashion item to classify it using the pre-trained model.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True, width=200)
        
        # Add classify button
        if st.button("🔍 Classify Image", key="classify_button"):
            try:
                # Check file size (5MB limit)
                if uploaded_file.size > 5 * 1024 * 1024:  # 5MB in bytes
                    st.error("File size too large. Please upload an image smaller than 5MB.")
                else:
                    # Load the pre-trained model
                    with st.spinner("Loading model..."):
                        model = load_custom_model()
                    
                    if model is not None:
                        # Preprocess and predict
                        with st.spinner("Classifying image..."):
                            img_array = preprocess_image(image)
                            if img_array is not None:
                                predicted_class, confidence, predictions = predict_garment(model, img_array)
                                
                                # Display prediction
                                st.subheader("Prediction Results")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Predicted Class", CLASS_LABELS[predicted_class])
                                    st.metric("Confidence", f"{confidence*100:.2f}%")
                                
                                with col2:
                                    st.markdown("### Class Probabilities")
                                    for i, (label, prob) in enumerate(zip(CLASS_LABELS, predictions)):
                                        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
def show_predictions(model, x_test, y_test, class_names, num_examples=5):
    """Display model predictions on sample test images"""
    # Select random test samples
    indices = np.random.choice(len(x_test), num_examples, replace=False)
    x_samples = x_test[indices]
    y_true = y_test[indices]
    
    # Get model predictions
    y_pred = model.predict(x_samples)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    if num_examples == 1:
        axes = [axes]
    
    for i, (img, true_label, pred_label, ax) in enumerate(zip(x_samples, y_true, y_pred_classes, axes)):
        img = img.squeeze()
        ax.imshow(img, cmap='gray')
        
        # Set title with true and predicted labels
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Training information
    with st.expander("ℹ️ About the Training Process"):
        st.markdown("""
        The model is trained using the following configuration:
        - **Dataset**: Fashion MNIST (60,000 training, 10,000 test images)
        - **Optimizer**: Adam (learning_rate=0.001)
        - **Loss Function**: Sparse Categorical Crossentropy
        - **Metrics**: Accuracy
        - **Batch Size**: Configurable (32-256)
        - **Epochs**: Configurable (1-20)
        
        During training, you can observe:
        - Training and validation loss (left plot)
        - Training and validation accuracy (right plot)
        
        The model uses data augmentation (random rotations and translations) to improve generalization.
        """)
    
    # Process uploaded image if any
    uploaded_file = st.file_uploader("Choose an image... (Max 5MB)", 
                                   type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=False,
                                   key="image_uploader")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True, width=200)
        
        # Add classify button
        if st.button("🔍 Classify Image", key="classify_button"):
            try:
                # Check file size (5MB limit)
                if uploaded_file.size > 5 * 1024 * 1024:  # 5MB in bytes
                    st.error("File size too large. Please upload an image smaller than 5MB.")
                    return
                
                # Load the pre-trained model
                with st.spinner("Loading model..."):
                    model = load_custom_model()
                
                if model is None:
                    st.error("Failed to load the model. Please try again.")
                    return
                
                # Preprocess and predict
                with st.spinner("Classifying image..."):
                    img_array = preprocess_image(image)
                    if img_array is not None:
                        predicted_class, confidence, predictions = predict_garment(model, img_array)
                        
                        # Display prediction
                        st.subheader("Prediction Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Class", CLASS_LABELS[predicted_class])
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        with col2:
                            st.markdown("### Class Probabilities")
                            for i, (label, prob) in enumerate(zip(CLASS_LABELS, predictions)):
                                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Sidebar information
    st.sidebar.markdown("## Class Labels")
    st.sidebar.markdown("The model can identify the following items:")
    st.sidebar.markdown("""
    - T-shirt/top
    - Trouser
    - Pullover
    - Dress
    - Coat
    - Sandal
    - Shirt
    - Sneaker
    - Bag
    - Ankle boot
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "For best results, upload clear images of single clothing items on a plain background. "
        "The model works best with images similar to the Fashion MNIST dataset."
    )

if __name__ == "__main__":
    main()
