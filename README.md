# 🧠 Deep Learning Project

# Spam Detection & Sentiment Analysis using LSTM and BiLSTM

---

## 📌 Project Overview

This project applies **Deep Learning** techniques to solve two real-world Natural Language Processing (NLP) tasks:

* **Spam Detection** for SMS messages
* **Sentiment Analysis** for movie reviews

The system uses **LSTM** and **Bidirectional LSTM (BiLSTM)** neural networks built with **TensorFlow/Keras** and provides an interactive web interface using **Streamlit**.

---

## 🎯 Tasks and Outputs

| Problem            | Task                   | Output               |
| ------------------ | ---------------------- | -------------------- |
| Spam Detection     | Classify SMS messages  | SPAM or HAM          |
| Sentiment Analysis | Classify movie reviews | POSITIVE or NEGATIVE |

---

## 🧠 Neural Network Architectures

### 📩 Spam Detection Model (LSTM)

```text
Embedding(5000, 64)
        ↓
Bidirectional LSTM(64)
        ↓
Dropout(0.3)
        ↓
Dense(1, activation='sigmoid')
```

---

### 🎬 Sentiment Analysis Model (BiLSTM)

```text
Embedding(10000, 128)
        ↓
Bidirectional LSTM(64)
        ↓
Dense(64, activation='relu')
        ↓
Dropout(0.3)
        ↓
Dense(1, activation='sigmoid')
```

---

## 📊 Model Performance

| Model  | Task               | Accuracy |
| ------ | ------------------ | -------- |
| LSTM   | Spam Detection     | 98%+     |
| BiLSTM | Sentiment Analysis | 88%+     |

---

## 🛠 Technologies Used

| Category                | Technology         |
| ----------------------- | ------------------ |
| Deep Learning Framework | TensorFlow / Keras |
| Neural Networks         | LSTM, BiLSTM       |
| Web Interface           | Streamlit          |
| Programming Language    | Python 3.11        |
| Training Platform       | Google Colab       |

---

## 📂 Dataset Information

### 📩 SMS Spam Collection Dataset

* Source: UCI Machine Learning Repository
* Purpose: Spam vs Ham SMS classification

### 🎬 IMDB Movie Review Dataset

* Source: Kaggle
* Size: 50,000 movie reviews
* Purpose: Positive vs Negative sentiment classification

### 🔗 Dataset Download Links

* IMDB Dataset (50K Reviews):
  [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

* SMS Spam Collection Dataset:
  [https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## 📁 Project Structure

```text
DL_PROJECT/
│
├── app_dl.py                     # Streamlit web application
├── LabFinal.ipynb                # Google Colab training notebook
│
├── data/                         # Dataset folder
│   ├── spam.csv
│   └── IMDB_Dataset.csv
│
└── models_dl/                    # Saved trained models
    ├── spam_lstm_model.h5
    ├── spam_tokenizer.pkl
    ├── sentiment_bilstm_model.h5
    └── sentiment_tokenizer.pkl
```

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```bash
pip install tensorflow streamlit pandas numpy scikit-learn matplotlib
```

---

### Step 2: Run the Streamlit Web App

```bash
streamlit run app_dl.py
```

---

### Step 3: Open in Browser

```text
http://localhost:8501
```

---

## 📦 Requirements

```text
tensorflow>=2.13.0
streamlit>=1.28.0
pandas
numpy
scikit-learn
matplotlib
```

---

