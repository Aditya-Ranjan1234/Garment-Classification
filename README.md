# Fashion Item Classifier

A Streamlit-based web application for classifying fashion items using deep learning. The app allows you to either classify uploaded images or train your own model on the Fashion MNIST dataset.

## Features

### 🔍 Image Classification
- Upload and classify fashion item images
- View prediction confidence scores
- See class probability distributions
- Supports JPG, JPEG, and PNG formats

### 🎓 Interactive Model Training
- Train a CNN model directly in the browser
- Real-time training visualization
- Adjustable training parameters (epochs, batch size, learning rate)
- Live loss and accuracy plots
- Early stopping capability
- Model download after training

### 📊 Data Visualization
- View sample images from the dataset
- Training/validation split visualization
- Confusion matrix
- Example predictions on test set

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/garment-classification.git
   cd garment-classification
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The app will open in your default web browser at `http://localhost:8501`

### Image Classification
1. Go to the "🔍 Image Classification" tab
2. Click "Browse files" to upload an image
3. View the prediction results and confidence scores

### Model Training
1. Go to the "🎓 Train Your Own Model" tab
2. Adjust the training parameters as needed
3. Click "Start Training" to begin
4. Monitor the training progress with live plots
5. Use "Stop Training" to halt the training process
6. After training, view test metrics and example predictions
7. Download the trained model if desired

## Requirements
- Python 3.8+
- TensorFlow 2.12.0
- Streamlit 1.31.0
- NumPy 1.24.3
- Matplotlib 3.7.1
- scikit-learn 1.2.2
- Pillow 10.0.0
- h5py 3.8.0
- seaborn 0.12.2

## Project Structure

```
garmentcv-app/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Integration

This application currently uses a mock model for demonstration purposes. To integrate with a real model:

1. Train or obtain a pre-trained model for garment classification
2. Save the model in the appropriate format (e.g., .h5 for Keras models)
3. Update the `app.py` to load your model instead of the mock model
4. Adjust the preprocessing and prediction functions as needed for your model

## Future Improvements

- Integrate with a pre-trained fashion model (e.g., ResNet, EfficientNet)
- Add support for multiple garments in a single image
- Implement virtual try-on features
- Add more detailed style recommendations
- Support for video input and real-time analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
