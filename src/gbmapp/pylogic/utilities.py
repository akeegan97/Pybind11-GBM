"""Utility functions for GBM stock price predictions."""

import numpy as np
import pandas as pd


def ReadCsvData(file_path):
    """Read and process CSV data containing stock prices.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame with Date and Close columns, or None if invalid.
    """
    result = VerifyCsvFormat(file_path)
    if not result['valid']:
        print(f"CSV Validation Error: {result['error']}")
        return None
    
    try:
        data = pd.read_csv(file_path)
        
        # Map to standard column names
        date_col = result['date_column']
        close_col = result['close_column']
        
        # Extract and clean the data
        cleaned_data = pd.DataFrame({
            "Date": data[date_col],
            "Close": data[close_col]
        })
        
        # Clean Close column - remove $ and convert to float
        if cleaned_data['Close'].dtype == 'object':
            cleaned_data['Close'] = cleaned_data['Close'].astype(str).str.replace('$', '', regex=False)
            cleaned_data['Close'] = cleaned_data['Close'].str.replace(',', '', regex=False)
            cleaned_data['Close'] = pd.to_numeric(cleaned_data['Close'], errors='coerce')
        
        # Convert Date column
        cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], errors='coerce')
        
        # Remove any rows with NaN values
        cleaned_data = cleaned_data.dropna()
        
        # Sort by date
        cleaned_data.sort_values(by='Date', ascending=True, inplace=True)
        cleaned_data.reset_index(drop=True, inplace=True)
        
        return cleaned_data
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def VerifyCsvFormat(file_path):
    """Verify that CSV file has the required format.
    
    Args:
        file_path: Path to the CSV file to verify.
        
    Returns:
        Dictionary with 'valid' status, 'date_column', 'close_column', and 'error' message.
    """
    try:
        data = pd.read_csv(file_path, nrows=10)
    except Exception as e:
        return {'valid': False, 'error': f'Cannot read file: {e}'}
    
    # Look for Date column (case-insensitive, various formats)
    date_col = None
    for col in data.columns:
        if col.lower() in ['date', 'datetime', 'time']:
            date_col = col
            break
    
    if date_col is None:
        return {'valid': False, 'error': f'No Date column found. Available columns: {", ".join(data.columns)}'}
    
    # Look for Close/Price column (case-insensitive, various formats)
    close_col = None
    for col in data.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['close', 'price', 'last']):
            close_col = col
            break
    
    if close_col is None:
        return {'valid': False, 'error': f'No Close/Price column found. Available columns: {", ".join(data.columns)}'}
    
    return {
        'valid': True,
        'date_column': date_col,
        'close_column': close_col,
        'error': None
    }


def VerifyStartDate(data, start_date):
    """Check if start date exists in the dataset.
    
    Args:
        data: DataFrame containing the stock data.
        start_date: Date to verify.
        
    Returns:
        True if the date exists in the data.
    """
    start_date = pd.to_datetime(start_date)
    return start_date in data['Date'].values


def VerifyEndDate(data, end_date):
    """Check if end date exists in the dataset.
    
    Args:
        data: DataFrame containing the stock data.
        end_date: Date to verify.
        
    Returns:
        True if the date exists in the data.
    """
    end_date = pd.to_datetime(end_date)
    return end_date in data['Date'].values
    
def GetEndRange(data, end_date):
    """Calculate maximum number of prediction days available after end date.
    
    Args:
        data: DataFrame containing the stock data.
        end_date: The end date of the training period.
        
    Returns:
        Maximum number of days available for prediction, or None if date not found.
    """
    end_date = pd.to_datetime(end_date)
    try:
        end_index = data.index[data['Date'] == end_date][0]
        max_prediction_days = len(data) - end_index - 1
        return max_prediction_days
    except IndexError:
        return None

def CheckStartEndDate(start_date, end_date):
    """Verify that start date is before end date.
    
    Args:
        start_date: The start date.
        end_date: The end date.
        
    Returns:
        True if start_date is before end_date.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return start_date < end_date
    
class Statistics:
    """Container for statistical measures from training data."""
    
    def __init__(self, mu, deviation, variance, normalized_mu, normalized_variance, normalized_deviation):
        """Initialize statistics.
        
        Args:
            mu: Mean of log returns.
            deviation: Standard deviation of log returns.
            variance: Variance of log returns.
            normalized_mu: Drift term adjusted for prediction steps.
            normalized_variance: Variance adjusted for prediction steps.
            normalized_deviation: Volatility term for prediction.
        """
        self.trainingMu = mu
        self.trainingDeviation = deviation
        self.trainingVariance = variance
        self.normalizedMu = normalized_mu
        self.normalizedDeviation = normalized_deviation
        self.normalizedVariance = normalized_variance
    
def CalculateStatistics(data, start_date, end_date, steps):
    """Calculate statistical measures from training data.
    
    Args:
        data: DataFrame containing stock prices.
        start_date: Start date of training period.
        end_date: End date of training period.
        steps: Number of prediction steps.
        
    Returns:
        Statistics object containing training and normalized parameters.
    """
    start_index = data.index[data['Date'] == start_date][0]
    end_index = data.index[data['Date'] == end_date][0]
    training_data = data.iloc[start_index:end_index]
    
    # Calculate log returns
    log_returns = np.log(training_data['Close']) - np.log(training_data['Close'].shift(1))
    
    # Training statistics
    training_mu = log_returns.mean()
    training_deviation = log_returns.std()
    training_variance = training_deviation ** 2
    
    # Normalize for prediction period
    normalized_mu = training_mu * int(steps)
    normalized_variance = training_variance * np.sqrt(int(steps))
    normalized_deviation = np.sqrt(training_deviation)
    
    return Statistics(
        training_mu, training_deviation, training_variance,
        normalized_mu, normalized_variance, normalized_deviation
    )

def GBM(starting_price, normalized_mu, normalized_var, normalized_dev, steps, paths):
    """Simulate stock prices using Geometric Brownian Motion.
    
    Args:
        starting_price: Initial stock price.
        normalized_mu: Drift coefficient.
        normalized_var: Variance coefficient.
        normalized_dev: Volatility coefficient.
        steps: Number of time steps to simulate.
        paths: Number of simulation paths.
        
    Returns:
        Tuple of (display_paths, average_predicted_price) where display_paths
        contains the first 50 paths for visualization.
    """
    delta_t = 1 / steps
    simulated_paths = np.full((paths, steps), starting_price)
    display_paths = []
    
    for i in range(simulated_paths.shape[0]):
        path = [starting_price]
        
        for j in range(1, simulated_paths.shape[1]):
            previous_price = path[-1]
            random_shock = np.random.normal(0, np.sqrt(delta_t))
            predicted_price = previous_price * np.exp(
                (normalized_mu - normalized_var / 2) * delta_t + normalized_dev * random_shock
            )
            path.append(predicted_price)
        
        # Keep first 50 paths for visualization
        if i < 50:
            display_paths.append(path)
        
        simulated_paths[i] = path
    
    average_predicted_price = np.mean(simulated_paths[:, -1])
    return display_paths, average_predicted_price


