# Stock Price Prediction using Neural Temporal Point Processes (NTPPs)

This repository contains the code and resources for implementing a **Neural Temporal Point Process (NTPP)** model to predict stock price movements. The project is based on a **Multivariate Temporal Point Process (MTPP)** framework, utilizing **Long Short-Term Memory (LSTM)** networks to capture temporal dependencies and patterns in financial data. By leveraging a five-year dataset of **S&P 500** stock prices, this model demonstrates how neural point processes can be applied to predict the timing and direction of stock price changes.

## Project Overview

The objective of this project is to predict stock price movements (**upward, downward, or stationary**) based on historical trading data. Temporal Point Processes are well-suited for this type of time-sensitive event prediction, and the use of neural networks in an MTPP framework allows the model to capture complex patterns and dependencies across multiple stocks.

### Features and Marks

The model was designed with various engineered features, including:
- **Lagged Prices**: Captures recent momentum in stock prices.
- **Moving Averages**: Provides insight into longer-term trends by smoothing price fluctuations.
- **Daily Price Changes and Volume**: Indicates daily volatility and liquidity, which can impact price movement.

Each event (daily price movement) is associated with one of three marks, indicating:
1. **Upward Movement**: Top 25% of price changes.
2. **Downward Movement**: Bottom 25% of price changes.
3. **Stationary or No Significant Movement**: Middle 50% of price changes.

### Model Architecture

The model uses an **LSTM-based architecture** to capture long-term dependencies in stock price sequences. Each stock is treated as a separate dimension in the multivariate framework, enabling the model to learn patterns across multiple stocks while accounting for individual trends. Key aspects of the model include:
- **Conditional Intensity Function**: Predicts the likelihood and timing of future events based on past sequences.
- **Recurrent Neural Network (LSTM)**: Processes sequential data with memory retention to understand the temporal dependencies of stock prices.

### Dataset

The dataset used in this project is the **[S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)** from Kaggle, which contains daily records of stock prices and trading volumes for companies in the S&P 500 index over five years. Each record includes:
- **Date**: Trading date for each record.
- **Open, High, Low, and Close Prices**: Daily price movements.
- **Volume**: Number of shares traded.

The dataset was preprocessed to include additional financial indicators, such as moving averages, volatility, and lagged prices, to provide the model with richer information.

## Repository Contents

- **data/**: Contains the preprocessed dataset files (not included here; please download from the Kaggle link above).
- **models/**: Model architecture and configurations for the NTPP model, including LSTM setup and MTPP framework.
- **scripts/**: Main scripts for data processing, training, and evaluation.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and visualization of model performance.
- **results/**: Contains training loss plots, confusion matrices, and other evaluation metrics.

## How to Run

1. **Download the S&P 500 dataset** from Kaggle and place it in the `data/` folder.
2. **Install the required dependencies** 
3. **Run the training script** to preprocess the data, train the NTPP model, and evaluate its performance:
# Results and Analysis

The model achieves high accuracy and balanced precision-recall scores across the training and test datasets. Training loss plots and confusion matrices are provided to showcase the model's learning progress and classification accuracy. This repository also includes code for visualizing these results to better understand the model's performance.
