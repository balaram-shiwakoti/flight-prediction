
---

# ✈️ SkyPrice AI: Flight Ticket Price Predictor

An interactive machine learning application that predicts flight ticket prices in real-time. This project demonstrates end-to-end **Regression Modeling**, **Categorical Feature Engineering (One-Hot Encoding)**, and clean model serialization using Python, Scikit-Learn, and Streamlit.

---

## 🛠️ Project Architecture & Data Pipeline

```text
         User Input (Web GUI)
     [Airline, Route, Days Left]
                  │
                  ▼
      One-Hot Encoding Pipeline
  [Converts categories to binary 1s/0s]
                  │
                  ▼
      Linear Regression Weights
   [Calculates feature coefficients]
                  │
                  ▼
        Predicted Ticket Price
        [e.g., $1,450.50 AUD]

```

---

## 📂 Project Structure

```text
SkyPrice_AI/
│
├── flight_predictor_model.pkl   # Serialized model, features, and target metadata
├── app.py                       # Local Streamlit interactive travel dashboard
├── train_predictor.py           # Machine learning training and evaluation script
└── README.md                    # Project documentation

```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Installation

Clone this repository and install the required machine learning and web UI dependencies:

```bash
pip install -r requirements.txt
```

### 3. Training & Serializing the Model

To generate the synthetic dataset, train the regression model, and save the artifact, run:

```bash
python train_predictor.py

```

This script evaluates the model's performance metrics ($R^2$ score and Mean Absolute Error) and exports the pre-trained weights to `flight_predictor_model.pkl`.

### 4. Running the Web App

Launch the interactive prediction dashboard locally:

```bash
streamlit run app.py

```

Once the server starts, open `http://localhost:8501` in your web browser to test different itineraries live!

---

## 🧠 Key ML Techniques Implemented

> 💡 **Under the Hood:** Linear Regression calculates exact coefficients for each categorical feature, ensuring instantaneous predictions without deep learning overhead.

* **One-Hot Encoding (OHE):** Converts nominal string variables (like *Airline* and *Route*) into mathematical vectors without implying a numerical hierarchy.
* **Feature Scaling & Decay Curve:** Model variables weight the "Days Left" input to realistically simulate price surges as departure dates approach.
* **Model Serialization:** Uses Python's `pickle` library to package the trained estimator and feature mapping pipeline together for seamless deployment.
