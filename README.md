# House Price Prediction Project

## Overview
This project implements various machine learning models to predict house prices based on features such as square footage, number of bedrooms, lot size, and more. The implementation includes data preprocessing, feature engineering, model selection, and evaluation techniques.

## Repository Structure

### Data Files
- `house_price_regression_dataset.csv`: The main dataset containing house price information
- `df.pkl`: Pickled DataFrame for quick loading
- `pipeline.pkl`: Serialized model pipeline for production use
- `.DS_Store`: macOS system file

### Notebooks
- `Baselinemodel.ipynb`: Initial model implementation with baseline metrics
- `Feature_Engineering_&_Feature_Selection.ipynb`: Notebook dedicated to feature engineering and selection
- `HousePriceData_UnivariateEDA.ipynb`: Exploratory data analysis with univariate statistics
- `HousePriceData_Model_Selection.ipynb`: Comparison of different machine learning models
- `HousePriceData_MultiVariateAnalysis.ipynb`: In-depth multivariate analysis of features
- `House_Price_Prediction_After_Feature_Selection.ipynb`: Model implementation after feature selection
- `House_Price_Prediction_Basic_Questions.ipynb`: Analysis of basic questions related to house prices
- `Pandas_Profiling.ipynb`: Detailed profiling of the dataset

### Python Files
- `HousePrediction.py`: Main Python script for the prediction system

### Output
- `output_report.html`: Generated report with model performance metrics

## Installation
To set up this project, create a conda environment with the required packages:

```bash
# Create a new conda environment
conda create -n housepriceprediction python=3.9

# Activate the environment
conda activate housepriceprediction

# Install required packages
conda install -c conda-forge scikit-learn matplotlib pandas numpy seaborn jupyter
conda install -c conda-forge xgboost
pip install streamlit

# Clone this repository
git clone https://github.com/Jaswanth19-596/HousePricePrediction.git
cd HousePricePrediction
```

## Usage
To run the house price prediction model:

1. Start by exploring the EDA notebooks to understand the dataset.
2. Review the feature engineering process.
3. Examine model selection to understand which models were evaluated.
4. Load the finalized model pipeline for predictions:

```python
import pickle
import pandas as pd

# Load the model pipeline
with open('pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Make predictions on new data
sample_data = pd.DataFrame({
    'Square_Footage': [1500],
    'Num_Bedrooms': [3],
    'Lot_Size': [6000],
    # Add other required features
})

prediction = model_pipeline.predict(sample_data)
print(f"Predicted house price: ${prediction[0]:,.2f}")
```

## Models Implemented
- Linear Regression
- Lasso Regression
- Ridge Regression
- Random Forest
- XGBoost
- MLP (Neural Network)

## Feature Engineering
The project implements several feature engineering techniques:
- Feature scaling
- Categorical encoding
- Feature selection
- Multivariate analysis for feature interactions

## Evaluation Metrics
Model performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation scores

## Future Improvements
- Implement hyperparameter tuning for XGBoost and neural networks
- Add more features like neighborhood demographics
- Create a web interface for real-time predictions
- Deploy the model to a cloud service

## Author
[Jaswanth19-596](https://github.com/Jaswanth19-596)
