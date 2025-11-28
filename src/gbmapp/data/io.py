#!/usr/bin/env python
"""Data loading and validation for GBM application."""

import pandas as pd


class DataLoader:
    """Handles loading and validation of CSV data."""
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load and process CSV data containing stock prices.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            DataFrame with Date and Close columns.
            
        Raises:
            ValueError: If CSV format is invalid.
            FileNotFoundError: If file doesn't exist.
        """
        # Verify format first
        format_info = DataLoader.verify_csv_format(file_path)
        if not format_info['valid']:
            raise ValueError(f"CSV Validation Error: {format_info['error']}")
        
        try:
            data = pd.read_csv(file_path)
            
            # Map to standard column names
            date_col = format_info['date_column']
            close_col = format_info['close_column']
            
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
            raise ValueError(f"Error reading CSV: {e}")
    
    @staticmethod
    def verify_csv_format(file_path: str) -> dict:
        """Verify that CSV file has the required format.
        
        Args:
            file_path: Path to the CSV file to verify.
            
        Returns:
            Dictionary with 'valid' status, 'date_column', 'close_column', and 'error' message.
        """
        try:
            data = pd.read_csv(file_path, nrows=10)
        except Exception as e:
            return {'valid': False, 'error': f'Cannot read file: {e}', 
                    'date_column': None, 'close_column': None}
        
        # Look for Date column (case-insensitive, various formats)
        date_col = None
        for col in data.columns:
            if col.lower() in ['date', 'datetime', 'time']:
                date_col = col
                break
        
        if date_col is None:
            return {'valid': False, 
                    'error': f'No Date column found. Available columns: {", ".join(data.columns)}',
                    'date_column': None, 'close_column': None}
        
        # Look for Close/Price column (case-insensitive, various formats)
        close_col = None
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['close', 'price', 'last']):
                close_col = col
                break
        
        if close_col is None:
            return {'valid': False, 
                    'error': f'No Close/Price column found. Available columns: {", ".join(data.columns)}',
                    'date_column': None, 'close_column': None}
        
        return {
            'valid': True,
            'date_column': date_col,
            'close_column': close_col,
            'error': None
        }
    
    @staticmethod
    def verify_date_exists(data: pd.DataFrame, date: str) -> bool:
        """Check if date exists in the dataset.
        
        Args:
            data: DataFrame containing the stock data.
            date: Date to verify (string format - YYYY-MM-DD preferred).
            
        Returns:
            True if the date exists in the data.
            
        Raises:
            ValueError: If date format is invalid.
        """
        try:
            date_obj = pd.to_datetime(date, format='%Y-%m-%d')
        except:
            # Try with other common formats
            try:
                date_obj = pd.to_datetime(date)
            except Exception as e:
                raise ValueError(f"Invalid date format '{date}'. Please use YYYY-MM-DD (e.g., 2024-01-15)")
        return date_obj in data['Date'].values
    
    @staticmethod
    def get_max_prediction_days(data: pd.DataFrame, end_date: str) -> int | None:
        """Calculate maximum number of prediction days available after end date.
        
        Args:
            data: DataFrame containing the stock data.
            end_date: The end date of the training period.
            
        Returns:
            Maximum number of days available for prediction, or None if date not found.
        """
        end_date_obj = pd.to_datetime(end_date)
        try:
            end_index = data.index[data['Date'] == end_date_obj][0]
            max_prediction_days = len(data) - end_index - 1
            return max_prediction_days
        except IndexError:
            return None
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """Verify that start date is before end date.
        
        Args:
            start_date: The start date.
            end_date: The end date.
            
        Returns:
            True if start_date is before end_date.
        """
        start_obj = pd.to_datetime(start_date)
        end_obj = pd.to_datetime(end_date)
        return start_obj < end_obj
