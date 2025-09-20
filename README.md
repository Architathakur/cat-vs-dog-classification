# Cats vs Dogs Classification  

## Project Overview  
This project is a **binary image classification model** that predicts whether an image contains a **Cat** or a **Dog**.  
It is built using **TensorFlow/Keras** with a Convolutional Neural Network (CNN).  

---

## Tech Stack  
- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  

---

## Dataset  
- Source: [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogsvscats)  
- Classes:  
  - `Cat`  
  - `Dog`  
- Total Images: 25,000 (balanced dataset)  

*(Dataset is not uploaded here due to large size. Please download it from Kaggle.)*  

---

## Model Architecture  
- **Input Layer**: 256×256 RGB images  
- **Conv + Pool layers** for feature extraction  
- **Batch Normalization** for stable training  
- **Dense layers** for classification  
- **Sigmoid output** for binary classification  

---

## How to Run  

### Training  
1. Place dataset in:
dataset/train
dataset/test

2. Run training

3. The model will train and show accuracy/loss plots.

4. After training, the model is saved as:
cat_dog_model.h5

### Prediction

1. Ensure cat_dog_model.h5 is in the project root.

2. Run:

python predict.py sample.jpg


3. Example output:

Prediction: Dog (95.32% confidence)

---

### Results

- Training Accuracy: ~XX%

- Validation Accuracy: ~XX% 

---

### Future Improvements

1. Add data augmentation

2. Try transfer learning (VGG16, ResNet, EfficientNet)

3. Deploy as a Web App (Flask/Streamlit)

---

### Author

Archita Thakur
architath27@gmail.com
