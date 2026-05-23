# 🧠 Deep Learning Project

## Spam Detection + Sentiment Analysis using LSTM & BiLSTM

---

## 📌 Project Overview

This project uses **Deep Learning** to solve two real-world text classification problems:

| Problem | Task | Output |
|---------|------|--------|
| Spam Detection | Classify SMS messages | SPAM or HAM |
| Sentiment Analysis | Classify movie reviews | POSITIVE or NEGATIVE |

---

## 🧠 Neural Network Architecture

**Spam Detection (LSTM):**
Embedding(5000, 64) → Bidirectional LSTM(64) → Dropout(0.3) → Dense(1, sigmoid)


**Sentiment Analysis (BiLSTM):**
Embedding(10000, 128) → Bidirectional LSTM(64) → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)


---

## 📊 Results

| Model | Task | Accuracy |
|-------|------|----------|
| LSTM | Spam Detection | 98%+ |
| BiLSTM | Sentiment Analysis | 88%+ |

---

## 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| Deep Learning | TensorFlow / Keras |
| Neural Networks | LSTM, BiLSTM |
| Web Interface | Streamlit |
| Language | Python 3.11 |
| Training Platform | Google Colab |

---
## 📊 Dataset

## 📁 Project Structure
DL_PROJECT/
│
├── app_dl.py # Streamlit web application
├── LabFinal.ipynb # Google Colab training notebook
│
├── data/ # Datasets
│ ├── spam.csv # SMS Spam Collection
│ └── IMDB_Dataset.csv # IMDB Movie Reviews
│
└── models_dl/ # Trained models
├── spam_lstm_model.h5 # LSTM model for spam
├── spam_tokenizer.pkl # Tokenizer for spam
├── sentiment_bilstm_model.h5 # BiLSTM model for sentiment
└── sentiment_tokenizer.pkl # Tokenizer for sentiment

Due to GitHub file size limits, the dataset can be downloaded from:

- **IMDB Dataset (50,000 movie reviews):** [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **SMS Spam Collection:** [Download from UCI](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

After downloading, place files in `data/` folder:
data/
├── spam.csv
└── IMDB_Dataset.csv

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install tensorflow streamlit pandas numpy scikit-learn

Step 2: Run Web App
streamlit run app_dl.py

Step 3: Open Browser
http://localhost:8501

How to Test the Web App
Type This Text	Click Button	Expected Result
"You won $1000! Click here"	Check Spam/Ham	🚨 SPAM
"Hey, want to grab lunch?"	Check Spam/Ham	✅ HAM
"This movie was amazing!"	Check Sentiment	😊 POSITIVE
"Worst movie ever"	Check Sentiment	😞 NEGATIVE
-- 
## Requirements
tensorflow>=2.13.0
streamlit>=1.28.0
pandas
numpy
scikit-learn
matplotlib
