# AI-Driven Occupancy Detection Using Multisensor Data Analysis

This project focuses on developing a machine learning-based model to estimate occupancy in indoor environments using data from non-intrusive sensors. These sensors monitor environmental parameters such as temperature, light, sound, CO2 levels, and motion to accurately detect the number of occupants in a room. The system can be used to optimize energy consumption in smart buildings by adjusting HVAC and lighting systems based on real-time occupancy data.

## Key Features:
- Utilizes multiple non-intrusive sensors to collect data.
- Machine learning models (Random Forest, SVM, Gradient Boosting, XGBoost) are implemented to predict occupancy levels.
- Data preprocessing, feature engineering (including CO2 slope calculations), and model training are key components.
- Achieves high accuracy and F1 scores, making it suitable for smart energy management applications.
- Extensive evaluation through k-fold cross-validation.

## Dataset:
The dataset consists of sensor readings from temperature, light, sound, CO2, and motion sensors collected from a room over a period of 4 days. The goal is to estimate the number of people present (0-3) based on the sensor data.

## Citation:
This project is based on the research presented in the following paper:  
*"Machine Learning-based Occupancy Estimation Using Multivariate Sensor Nodes"* by Adarsh Pal Singh, Vivek Jain, et al. You can find the paper [here](https://ieeexplore.ieee.org/document/8644432).
