# 🧠 Alzheimer Detection Web App

A deep learning-based web application for classifying Alzheimer’s disease from brain MRI images using a hybrid MobileNet-Transformer architecture. The solution is deployed using Streamlit, enabling real-time prediction, visualization, and report generation.

---

## 🌐 Live Demo

https://alzheimer-detection-app.streamlit.app/

---

## 🚀 Features

* Upload brain MRI images
* Predict Alzheimer’s disease stage
* Display confidence score for predictions
* Visualize affected regions using Grad-CAM
* Download detailed clinical-style PDF report

---

## 🧠 Model Architecture

This project uses a hybrid deep learning model:

* **MobileNetV2** → Efficient feature extraction from MRI images
* **Transformer Encoder** → Captures global contextual relationships
* **Fully Connected Layer** → Performs final classification

---

## 📊 Results

The hybrid MobileNet-Transformer model achieved strong performance on MRI image classification:

* **Training Accuracy:** 99.89%
* **Testing Accuracy:** 98.67%

### 📈 Performance Insights

* High training accuracy indicates effective feature learning
* Strong testing accuracy demonstrates good generalization capability
* Minimal overfitting observed between training and testing results

### 📊 Evaluation Metrics

* Confusion Matrix
* Accuracy & Loss Curves
* Grad-CAM Visualization for interpretability

---

## 📂 Project Structure

```
alzheimer-detection-streamlit-app/
│── app.py
│── requirements.txt
│── labels.json
│── accuracy_plot.png
│── confusion_matrix.png
│── loss_plot.png
│── heatmap.png
│── README.md
```

---

## 📓 Model Training Notebook

View the complete training process here:
https://drive.google.com/file/d/1hfyWsySDgQw-87CmlleYkVzikG2Cad3A/view?usp=sharing

---

## 🧠 Model Weights

Due to size limitations, the trained model is hosted externally:
https://drive.google.com/file/d/1uPC8rgv2pYCLC_12WfxvYx-RgoU7Tk7W/view?usp=drive_link

---

## 🛠️ Tech Stack

* Python
* Streamlit
* PyTorch
* OpenCV
* NumPy
* ReportLab

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Disclaimer

This application is intended for **screening and educational purposes only** and should not be used as a substitute for professional medical diagnosis.

---

## 👩‍💻 Author

Developed by Anusri and team as part of a final-year B.Tech project in Artificial Intelligence & Data Science, focusing on AI-driven medical image analysis.

