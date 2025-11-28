#!/usr/bin/env python
"""Statistical calculations and validation for GBM simulations."""

import numpy as np
import pandas as pd
from gbmapp.core.models import Statistics




class StatisticsCalculator:
    """Handles calculation of statistical measures for GBM."""
    
    @staticmethod
    def calculate_statistics(data: pd.DataFrame, start_date: str, 
                            end_date: str, steps: int) -> Statistics:
        """Calculate statistical measures from training data.
        
        Args:
            data: DataFrame containing stock prices.
            start_date: Start date of training period.
            end_date: End date of training period.
            steps: Number of prediction steps.
            
        Returns:
            Statistics object containing training and normalized parameters.
        """
        # Get training data slice
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        start_index = data.index[data['Date'] == start_date_obj][0]
        end_index = data.index[data['Date'] == end_date_obj][0]
        training_data = data.iloc[start_index:end_index + 1]  # Include end date
        
        # Calculate log returns
        log_returns = training_data['Close'].apply(np.log) - training_data['Close'].shift(1).apply(np.log)
        log_returns = log_returns.dropna()
        
        # Training statistics
        training_mu = log_returns.mean()
        training_deviation = log_returns.std()
        training_variance = training_deviation ** 2
        
        # Normalize for prediction period
        normalized_mu = training_mu * steps
        normalized_variance = training_variance * steps
        normalized_deviation = np.sqrt(normalized_variance)
        
        return Statistics(
            training_mu=training_mu,
            training_deviation=training_deviation,
            training_variance=training_variance,
            normalized_mu=normalized_mu,
            normalized_variance=normalized_variance,
            normalized_deviation=normalized_deviation
        )
