# Sign Sense 🤟

**Sign Sense** is a real-time American Sign Language (ASL) detection application. It uses Google's **MediaPipe** for hand/pose tracking and a custom **Transformer-based TFLite model** to classify 250 signs instantly from your webcam feed.

The project is optimized for CPU inference on standard laptops using TensorFlow Lite.

## 🚀 Features

* **Real-time Detection:** Classifies 250 distinct ASL signs live.
* **Privacy-First:** All processing happens locally; no video is sent to the cloud.
* **Lightweight:** Uses `.tflite` format for fast CPU performance (no GPU required).
* **Robust Tracking:** Tracks hands, face, and pose simultaneously using MediaPipe Holistic.

## 🛠️ Installation

You can set up the project using **uv** (faster, recommended) or standard **pip**.

### Option 1: Using uv (Fastest)

If you have `uv` installed, this is the quickest way to get started with the correct Python version (3.10).

1.  **Create a virtual environment:**
    ```powershell
    # Create a fresh environment with Python 3.10
    uv python install 3.10
    uv venv .venv --python 3.10
    
    # Activate it (Windows)
    .\.venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```powershell
    uv pip install "numpy<2" opencv-python mediapipe pandas tensorflow
    ```

---

### Option 2: Using Standard pip

If you prefer standard Python tools, follow these steps. **Note:** You must ensure you are running **Python 3.10** or **3.11** (Python 3.13 is not yet supported).

1.  **Create a `requirements.txt` file** in your project folder with the following content:
    ```text
    numpy<2
    opencv-python
    mediapipe
    pandas
    tensorflow
    ```

2.  **Install from requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

```text
sign sense/
├── main.py              # Main application script
├── asl_model.tflite     # The converted TFLite model file
├── train.csv            # Label mappings (sign names)
├── requirements.txt     # Dependency list (optional, for pip users)
└── README.md            # This file

🏃 Usage
Ensure your webcam is connected.

Run the script:

PowerShell

python main.py
How to use:

Stand back so your upper body and hands are visible.

Perform a sign.

The prediction will appear at the top left of the window.

Press q to quit.

⚠️ Troubleshooting
"NumPy 2.x Error" / _ARRAY_API not found: This project requires NumPy 1.x because TensorFlow/MediaPipe are not yet updated for NumPy 2.0.

Fix: Run uv pip install "numpy<2" or pip install "numpy<2".

Laggy Video: Open main.py and find the mp_holistic.Holistic line. Change model_complexity=1 to model_complexity=0 for faster (but slightly less accurate) performance.
