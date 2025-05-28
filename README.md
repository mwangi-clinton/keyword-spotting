# Speech Command Recognition with LSTM

This project implements a speech command recognition model using an LSTM neural network. It trains a classifier to recognize audio commands such as "yes" or "no" using the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) dataset.

---

## 📁 Project Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download the Dataset

Download the dataset from Kaggle:

1. Go to: [https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
2. Download and extract the `train/audio/` folder.

Example structure:

```
/path/to/speech-data/
  └── train/
      └── audio/
          ├── yes/
          ├── no/
          └── ...
```

---

## Train the Model

Use the `speech_command_train.py` script to train an the classifier.

### Example Command:

```bash
python speech_command_train.py \
  -f /path/to/speech-data/train/audio/ \
  -win 800 \
  -hop 0.5 \
  -o ./saved_models/ \
  -m speech_model.pth
```

### Arguments

* `-f`: Path to the dataset folder (required)
* `-win`: Window length for feature extraction (e.g. 800)
* `-hop`: Hop percent (float between 0.0 and 1.0)
* `-o`: Output directory for saving the model
* `-m`: (Optional) Name of the saved model file (default: `ideal_model_repeat.pth`)

---

## 📄 Output

* A trained PyTorch model saved to the specified directory.
* Training logs with epoch-wise loss values.

---








