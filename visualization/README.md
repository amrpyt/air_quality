# Visualization Tools for Air Quality Prediction

This directory contains scripts and tools for visualizing model performance and predictions from the air quality prediction models.

## Contents

- `model_comparison.py`: Script to compare the performance of different models (DCNN, DNN, LSTM, BiLSTM, RNN, CNN-BiLSTM) using metrics like RMSE and MAE.

## Usage

To generate model comparison visualizations:

```bash
# Navigate to the visualization directory
cd visualization

# Run the comparison script
python model_comparison.py
```

## Outputs

The scripts will generate visualizations in the following directories:

- Model comparison charts: `../visualizations/model_comparison.png`
- Prediction vs actual plots: `../visualizations/predictions_{model_name}.png`

## Extending

To add new visualization tools:

1. Create a new Python script in this directory
2. Use the common utility functions from existing scripts
3. Ensure outputs are saved to the visualizations directory
4. Update this README with information about the new script