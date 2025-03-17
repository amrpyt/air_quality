# Air Quality Prediction

## Overview
This Jupyter Notebook focuses on predicting air quality using deep learning models. It includes preprocessing steps, model building, and evaluation.

## Dataset
The dataset used for this project consists of time-series air quality data. It includes multiple features such as pollutant levels, temperature, humidity, and other environmental factors.

## Preprocessing
- Data is transformed into tensor format: `X_train_tensor (298463, 60, 10)`, `y_train_tensor (298463, 60)`.
- Time-series data preparation is performed to structure input for deep learning models.
- Normalization and feature engineering steps are applied.

## Models Implemented
- **DCNN Model (Deep Convolutional Neural Network)**
- **DNN Model (Deep Neural Network)**

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook step by step to preprocess the data and train the models.
3. Evaluate model performance using metrics like RMSE and MAE.

## Results
The models predict air quality levels based on historical data. Model performance can be improved with hyperparameter tuning and additional features.

## Author
Ahmed Gamal

## License
This project is open-source and free to use for educational and research purposes.

