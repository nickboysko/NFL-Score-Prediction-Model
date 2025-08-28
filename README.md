# NFL Score Prediction Model

This project uses machine learning (XGBoost) to predict NFL game scores based on historical data and team statistics.

## Features

- **Data Processing**: Uses pandas, polars, and pyarrow for efficient data handling
- **Machine Learning**: XGBoost gradient boosting for score prediction
- **NFL Data**: Integration with nfl_data_py for comprehensive NFL statistics
- **Hyperparameter Tuning**: Optuna for automated model optimization
- **Visualization**: Matplotlib for data visualization and model analysis

## Installation

### Prerequisites

- Python 3.12 or higher
- Windows operating system
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd nfl-xgb-score
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Verification

To verify the installation, you can run:
```bash
python -c "import numpy, pandas, sklearn, xgboost, nfl_data_py, pyarrow, polars, matplotlib, optuna; print('All packages installed successfully!')"
```

## Usage

[Add usage instructions here once the model is implemented]

## Dependencies

The project uses the following key libraries:

- **numpy** (1.26.4): Numerical computing
- **pandas** (2.2.1): Data manipulation and analysis
- **scikit-learn** (1.6.1): Machine learning utilities
- **xgboost** (2.0.3): Gradient boosting framework
- **nfl_data_py** (0.3.2): NFL data access
- **pyarrow** (15.0.0): Fast data interchange format
- **polars** (0.20.10): Fast DataFrame library
- **matplotlib** (3.8.3): Data visualization
- **optuna** (3.6.0): Hyperparameter optimization

All versions are pinned for reproducibility and tested for compatibility with Python 3.12 on Windows.

## License

[Add license information here]
