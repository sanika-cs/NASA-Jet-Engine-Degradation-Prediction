# NASA-Jet-Engine-Degradation-Prediction


This project predicts the **Remaining Useful Life (RUL)** of NASA’s turbofan jet engines using **time-series sensor data** from the CMAPSS dataset.  
We apply **LSTM** and **Transformer** deep learning models to capture temporal patterns in sensor readings and estimate when an engine is likely to fail.  
The results are visualized through an **interactive Streamlit dashboard**.

---

##  Project Overview

- **Goal:** Predict how many operational cycles an engine has left before failure.
- **Dataset:** NASA C-MAPSS Jet Engine Degradation Dataset  
- **Problem Type:** Time Series Regression  
- **Models Used:**
  - LSTM (Long Short-Term Memory)
  - Transformer-based sequence model
- **Key Features:**
  - Walk-forward validation
  - Optuna hyperparameter tuning
  - MinMax scaling of sensor features
  - **Streamlit web app** for easy exploration & prediction

---

##  Dataset

**Source:** [NASA C-MAPSS Dataset](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data)  

**Columns:**
- `engine_id` — Engine identifier
- `cycle` — Time in cycles
- `Op_Setting_1`, `Op_Setting_2`, `Op_Setting_3` — Operational settings
- `Sensor_1` to `Sensor_21` — Sensor measurements
- `RUL` — Remaining Useful Life (target)

---

##  Tech Stack

- **Language:** Python   
- **Libraries:**  
  - Data: Pandas, NumPy, Scikit-learn  
  - Deep Learning: TensorFlow / Keras  
  - Optimization: Optuna  
  - Web App: Streamlit  

---

##  Model Performance (Example)

| Model       | Test RMSE | Test MAE | R² Score |
|-------------|----------:|---------:|---------:|
| LSTM        | 19.08    | 13.68    | 0.79     |
| Transformer | 20.60     | 16.28    | 0.75     |

---

