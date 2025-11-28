
# AsliNakli - Deepfake Detection Toolkit

**AsliNakli** is a unified deepfake detection system designed for **audio**, **image**, and **video** modalities. It uses state-of-the-art machine learning models to detect fake media and provides real-time predictions with interactive explanations like Grad-CAM for images and videos.

## Features
- **Audio Deepfake Detection**: Utilizes RawNetLite for audio classification.
- **Image Deepfake Detection**: Powered by EfficientNet for image analysis.
- **Video Deepfake Detection**: Combines EfficientNet for frame extraction and attention pooling for video classification.
- **Real-Time Predictions**: Run inference directly on uploaded media.
- **Explainability**: Visualize attention maps and Grad-CAM overlays for better understanding of model decisions.
- **CPU-Friendly**: Optimized for efficient performance even on CPU.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aslinakli.git
   cd aslinakli
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download or train the models (check `models/` directory for pre-trained weights).

## Usage

1. Start the Dash web app:
   ```bash
   python app.py
   ```

2. Access the web interface at [http://localhost:8050](http://localhost:8050).

3. Upload your audio, image, or video file and get the deepfake prediction.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
