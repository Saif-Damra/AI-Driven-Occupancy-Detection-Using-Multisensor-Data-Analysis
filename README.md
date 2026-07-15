# 🏢 AI-Driven Occupancy Detection Using Multisensor Data Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project focuses on developing a machine learning-based model to estimate occupancy in indoor environments using data from non-intrusive sensors. These sensors monitor environmental parameters such as temperature, light, sound, CO2 levels, and motion to accurately detect the number of occupants in a room (0-3 people). The system can be used to optimize energy consumption in smart buildings by adjusting HVAC and lighting systems based on real-time occupancy data.

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Installation & Usage](#installation--usage)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

The goal of this project is to estimate the number of people present in a room based on environmental sensor readings. The pipeline includes:

1. **Data Preprocessing** – Extracting relevant time features and handling missing values
2. **Feature Engineering** – Utilizing CO2 slopes, Temperature, Light, and Sound variations
3. **Model Training** – Training 4 powerful ML models using GridSearchCV for hyperparameter tuning
4. **Model Comparison** – Comprehensive evaluation using Accuracy, Precision, Recall, and F1-Score

## 📊 Dataset

The dataset consists of sensor readings collected from a room over a period of 4 days.

| Sensor Type | Features |
|-------------|----------|
| **Temperature** | `S1_Temp`, `S2_Temp`, `S3_Temp`, `S4_Temp` |
| **Light** | `S1_Light`, `S2_Light`, `S3_Light`, `S4_Light` |
| **Sound** | `S1_Sound`, `S2_Sound`, `S3_Sound`, `S4_Sound` |
| **CO2** | `S5_CO2`, `S5_CO2_Slope` |
| **Motion (PIR)** | `S6_PIR`, `S7_PIR` |
| **Target** | `Room_Occupancy_Count` (0, 1, 2, or 3 occupants) |

> **Note:** The raw data is not included in this repository due to privacy considerations. See [data/README.md](data/README.md) for schema details.

## 📁 Project Structure

```
AI-Driven-Occupancy-Detection-Using-Multisensor-Data-Analysis/
├── 📂 data/
│   └── README.md           # Dataset schema & instructions
├── 📂 Notebook/
│   └── Occupancy_Estimation.ipynb  # Original Jupyter notebook (exploratory)
├── 📂 src/
│   ├── __init__.py
│   ├── main.py             # Main pipeline entry point
│   ├── data_processing.py  # Data loading, cleaning & feature engineering
│   ├── model_training.py   # Model training & hyperparameter tuning
│   └── visualization.py    # Professional charts & visualizations
├── 📂 outputs/             # Generated charts & results (after running)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🔬 Methodology

### Feature Engineering
- Extracted the `hour` from the time recordings, as EDA showed a strong correlation between the hour of the day and occupancy levels.
- Dropped irrelevant time features (`year`, `month`, `day`, `minutes`, `seconds`) to reduce noise.

### Feature Scaling
- Applied `StandardScaler` to normalize the sensor readings, which is crucial for models like SVM and improves convergence for others.

### Model Evaluation
- Used **5-Fold Cross Validation** during hyperparameter tuning (`GridSearchCV`) to ensure the models generalize well to unseen data.

## 🤖 Models & Results

Four classification models were evaluated:

| Model | Hyperparameter Tuning |
|-------|-----------------------|
| **Random Forest** | `n_estimators`, `max_depth`, `max_features`, `criterion` |
| **Gradient Boosting** | `n_estimators`, `learning_rate`, `max_depth` |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth` |
| **SVM** | `C`, `kernel`, `gamma` |

> 📝 **Note:** Run the pipeline to generate actual results. The best model parameters and evaluation metrics (Accuracy, Precision, Recall, F1-Score) will be saved to `outputs/model_results.csv`.

## 📈 Visualizations

The pipeline generates the following professional visualizations in the `outputs/` directory:

### Exploratory Data Analysis (EDA)
- **Room Occupancy Distribution** – Bar and pie charts of the target variable
- **Feature Correlation Heatmap** – Correlation matrix of all numerical features
- **Sensor Variation by Hour** – Line charts for Temp, Light, and Sound vs. Hour
- **Sensor Boxplots** – Distribution analysis for different sensor types
- **CO2 Levels by Occupancy** – CO2 trends throughout the day grouped by occupancy count

### Model Comparison
- **Model Comparison Dashboard** – Side-by-side metric comparison (Accuracy, F1, etc.)
- **Confusion Matrices** – Detailed true vs. predicted counts for all models
- **Performance Heatmap** – Color-coded overall metric comparison

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Saif-Damra/AI-Driven-Occupancy-Detection-Using-Multisensor-Data-Analysis.git
cd AI-Driven-Occupancy-Detection-Using-Multisensor-Data-Analysis

# Install dependencies
pip install -r requirements.txt

# Place your data file
# Copy Occupancy_Estimation.csv to the data/ directory
```

### Running the Pipeline

```bash
# Run with default settings (5-fold CV)
python src/main.py

# Specify custom data path
python src/main.py --data path/to/Occupancy_Estimation.csv

# Run with 10-fold CV
python src/main.py --cv 10
```

## 📖 Citation

This project is based on the research presented in the following paper:
*"Machine Learning-based Occupancy Estimation Using Multivariate Sensor Nodes"* by Adarsh Pal Singh, Vivek Jain, et al.
[IEEE Xplore Link](https://ieeexplore.ieee.org/document/8644432)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Developed by [Saif Damra](https://github.com/Saif-Damra)**
