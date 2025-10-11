
# Sign-Language-Interpreter

> A project to recognize Indian Sign Language (ISL) gestures and convert them to text/speech.  
> Contains data-prep and model training notebooks, trained model files, and inference scripts.

---

## Table of contents

1. [Project overview](#project-overview)  
2. [Repository structure](#repository-structure)  
3. [Features](#features)  
4. [Quick start — run inference](#quick-start---run-inference)  
5. [Reproduce training (notebooks)](#reproduce-training-notebooks)  
6. [Dataset](#dataset)  
7. [Dependencies / Environment setup](#dependencies--environment-setup)  
8. [How the model works (summary)](#how-the-model-works-summary)  
9. [Files of interest](#files-of-interest)  


---

## Project overview

This repository implements an end-to-end pipeline for recognizing sign-language gestures (Indian Sign Language) from images/video frames and converting recognized gestures to textual output and speech. The project contains:

- Data processing & augmentation steps (notebook).
- Model training experiments (notebooks).
- Trained model files (`model_1.p`, `model_2.p`).
- Inference scripts that load a trained model and produce text/speech outputs.

---

## Repository structure

```
Sign-Language-Interpreter/
├─ .vscode/
├─ INDIAN SIGN LANGUAGE DATASET/     # dataset folder (present in repo)
├─ README.md                         # (this file)
├─ dataset_implementation.ipynb      # data prep + exploration
├─ image_classifier.ipynb            # model building / classifier notebook
├─ model.ipynb                       # training experiments / final model notebook
├─ model_1.p                         # trained model (pickle or saved object)
├─ model_2.p                         # alternative trained model
├─ output.py                         # inference / output handling script
├─ speech _language.py               # text-to-speech and language utilities
├─ test.ipynb                        # quick tests / evaluation
└─ working/
```

> Note: filenames include a space in `speech _language.py` on GitHub — be careful with exact filenames while running.

---

## Features

- Image-based sign recognition (ISL).
- Pre-built notebooks for dataset processing, training, and testing.
- Pretrained model files included for quick inference.
- Script to produce textual output and convert text to speech.

---

## Quick start — run inference

### 1. Clone the repo
```bash
git clone https://github.com/Nakul1009/Sign-Language-Interpreter.git
cd Sign-Language-Interpreter
```

### 2. Create & activate a Python virtual environment
```bash
python3 -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -U pip
pip install numpy pandas matplotlib scikit-learn opencv-python jupyter notebook pillow tensorflow keras joblib pyttsx3
```

### 4. Run inference
```bash
python output.py
```

If your filename includes spaces:
```bash
python "speech _language.py"
```

---

## Reproduce training (notebooks)

- `dataset_implementation.ipynb` — dataset loading, cleaning, augmentation, and preparing train/validation/test splits.  
- `image_classifier.ipynb` — model architecture, training, and visualizations.  
- `model.ipynb` — experiments and saving trained models (`model_1.p`, `model_2.p`).  
- `test.ipynb` — evaluation notebook.

To train:
```bash
jupyter notebook
```
Run cells in order, ensuring dataset paths are correct.

---

## Dataset

The included folder `INDIAN SIGN LANGUAGE DATASET` should have a structure like:

```
INDIAN SIGN LANGUAGE DATASET/
├─ train/
│  ├─ A/
│  ├─ B/
│  └─ ...
├─ val/
│  ├─ A/
│  └─ ...
└─ test/
   ├─ A/
   └─ ...
```

If missing, download or prepare the dataset in the same folder with similar structure.

---

## Dependencies / Environment setup

Create a `requirements.txt` file:

```
numpy
pandas
matplotlib
scikit-learn
opencv-python
pillow
tensorflow>=2.0
keras
joblib
jupyter
pyttsx3
```

Install using:
```bash
pip install -r requirements.txt
```

---

## How the model works (summary)

- **Input:** Image frames of ISL gestures.  
- **Preprocessing:** Resize, normalize, and augment.  
- **Model:** Pretrained Random Forest Model using hand landmarks
- **Output:** Predicted label → mapped to word → optionally converted to speech.  
- **Inference:** Done via `output.py` using trained models.

---

## Files of interest

| File | Purpose |
|------|----------|
| `dataset_implementation.ipynb` | Dataset processing and exploration |
| `image_classifier.ipynb` | Defines and trains CNN model |
| `model.ipynb` | Model experiments and saving |
| `model_1.p`, `model_2.p` | Trained model files |
| `output.py` | Inference script |
| `speech _language.py` | Text-to-speech functionality |

---

## Example usage

### Predict an image manually
```python
import joblib
import cv2
import numpy as np

model = joblib.load('model_1.p')

img = cv2.imread('sample.jpg')
img = cv2.resize(img, (128,128))
img = img.astype('float32')/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
label = np.argmax(pred, axis=1)
print("Predicted label:", label)
```

### Convert prediction to speech
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("This sign means A")
engine.runAndWait()
```

---


