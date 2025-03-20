import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os

# Set style for plots
plt.style.use('ggplot')
sns.set_palette('Set2')
sns.set_context('talk')

def load_model(model_path):
    """Load a Keras model from path"""
    return tf.keras.models.load_model(model_path)

def load_data(data_path):
    """Load test data for evaluation"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE and MAE metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def plot_model_comparison(model_names, rmse_values, mae_values, save_path=None):
    """Create bar chart comparing model performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # RMSE plot
    sns.barplot(x=model_names, y=rmse_values, ax=ax1)
    ax1.set_title('RMSE Comparison Across Models')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_xlabel('Model')
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # MAE plot
    sns.barplot(x=model_names, y=mae_values, ax=ax2)
    ax2.set_title('MAE Comparison Across Models')
    ax2.set_ylabel('MAE (lower is better)')
    ax2.set_xlabel('Model')
    for i, v in enumerate(mae_values):
        ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    # Plot a sample of predictions (first 100 points)
    sample_size = min(100, len(y_true))
    x = np.arange(sample_size)
    
    plt.plot(x, y_true[:sample_size], label='Actual', marker='o', markersize=4, linestyle='-')
    plt.plot(x, y_pred[:sample_size], label='Predicted', marker='x', markersize=4, linestyle='--')
    
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Define paths to models
    model_dir = '../models/'
    model_paths = {
        'DCNN': os.path.join(model_dir, 'DCNN_model_reg.keras'),
        'DNN': os.path.join(model_dir, 'DNN_model_reg.keras'),
        'LSTM': os.path.join(model_dir, 'lstm_model_reg.keras'),
        'BiLSTM': os.path.join(model_dir, 'bilstm_model_reg.keras'),
        'RNN': os.path.join(model_dir, 'RNN_model_reg.keras'),
        'CNN-BiLSTM': os.path.join(model_dir, 'CNN_BILSTM_reg.keras')
    }
    
    # Load test data (this is just a placeholder - implement your actual data loading)
    # X_test, y_test = load_data('path_to_test_data')
    
    # Placeholder for metrics
    rmse_values = [0.0456, 0.0512, 0.0489, 0.0423, 0.0534, 0.0401]  # Example values
    mae_values = [0.0341, 0.0398, 0.0372, 0.0318, 0.0408, 0.0295]   # Example values
    
    # Generate visualizations
    plot_model_comparison(
        list(model_paths.keys()), 
        rmse_values, 
        mae_values, 
        save_path='../visualizations/model_comparison.png'
    )
    
    # To implement prediction visualization, we would need:
    # 1. Load a model
    # 2. Make predictions on test data
    # 3. Call plot_predictions function
    
    print("Visualizations complete.")

if __name__ == "__main__":
    main()