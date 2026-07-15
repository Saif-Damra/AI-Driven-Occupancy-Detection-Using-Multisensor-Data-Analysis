# Dataset Information

## About the Data

The dataset (`Occupancy_Estimation.csv`) contains sensor readings from an indoor environment used to estimate room occupancy. Due to privacy considerations, the raw data is **not included** in this repository.

## Dataset Schema

| Column | Description | Type |
|--------|-------------|------|
| `Date` | Date of recording (YYYY/MM/DD) | String |
| `Time` | Time of recording (HH:MM:SS) | String |
| `S1_Temp` | Temperature sensor 1 (°C) | Float |
| `S2_Temp` | Temperature sensor 2 (°C) | Float |
| `S3_Temp` | Temperature sensor 3 (°C) | Float |
| `S4_Temp` | Temperature sensor 4 (°C) | Float |
| `S1_Light` | Light sensor 1 (lux) | Integer |
| `S2_Light` | Light sensor 2 (lux) | Integer |
| `S3_Light` | Light sensor 3 (lux) | Integer |
| `S4_Light` | Light sensor 4 (lux) | Integer |
| `S1_Sound` | Sound sensor 1 (voltage) | Float |
| `S2_Sound` | Sound sensor 2 (voltage) | Float |
| `S3_Sound` | Sound sensor 3 (voltage) | Float |
| `S4_Sound` | Sound sensor 4 (voltage) | Float |
| `S5_CO2` | CO2 sensor (ppm) | Integer |
| `S5_CO2_Slope` | CO2 slope (rate of change) | Float |
| `S6_PIR` | PIR motion sensor 1 (0/1) | Integer |
| `S7_PIR` | PIR motion sensor 2 (0/1) | Integer |
| `Room_Occupancy_Count` | **Target** - Number of occupants (0-3) | Integer |

## Source

This dataset is based on the research paper:
*"Machine Learning-based Occupancy Estimation Using Multivariate Sensor Nodes"* by Adarsh Pal Singh, Vivek Jain, et al.
[IEEE Paper Link](https://ieeexplore.ieee.org/document/8644432)

## How to Use

1. Place your `Occupancy_Estimation.csv` file in this `data/` directory
2. Run: `python src/main.py`
