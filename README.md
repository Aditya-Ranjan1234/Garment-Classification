# GarmentCV - Fashion Analysis App

A Streamlit-based web application for analyzing garments and providing style recommendations using computer vision and deep learning.

## Features

- Upload and analyze garment images
- Automatic garment type classification
- Confidence scoring for predictions
- Style and pairing recommendations
- Interactive visualization of classification results
- User-friendly interface with adjustable settings

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd garmentcv-app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate  # On Windows
   source venv/bin/activate    # On macOS/Linux
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

2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

3. Upload an image of a garment using the file uploader

4. Adjust the confidence threshold in the sidebar if needed

5. View the analysis results, including:
   - Predicted garment type
   - Confidence scores for all categories
   - Style recommendations

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
