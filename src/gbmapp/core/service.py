#!/usr/bin/env python
"""Business logic service for GBM simulations."""

import time
import numpy as np
import pandas as pd

from gbmapp.data.io import DataLoader
from gbmapp.core.validation import StatisticsCalculator
from gbmapp.native._dispatch import SimulationDispatcher, EngineType
from .models import SimConfig, SimResult


class GBMService:
    """Service layer for GBM simulation business logic."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and prepare CSV data.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with parsed dates
            
        Raises:
            ValueError: If file cannot be loaded
        """
        return DataLoader.load_csv(file_path)
    
    @staticmethod
    def validate_config(config: SimConfig, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate simulation configuration.
        
        Args:
            config: SimConfig to validate
            data: Data to validate against
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if data is None:
            return False, "No data loaded"
        
        if not DataLoader.verify_date_exists(data, config.start_date):
            return False, f"Invalid start date: {config.start_date}"
        
        if not DataLoader.verify_date_exists(data, config.end_date):
            return False, f"Invalid end date: {config.end_date}"
        
        if not DataLoader.validate_date_range(config.start_date, config.end_date):
            return False, "Start date must be before end date"
        
        if config.steps < 1:
            return False, "Steps must be positive"
        
        if config.paths < 1:
            return False, "Paths must be positive"
        
        return True, ""
    
    @staticmethod
    def run_simulation(config: SimConfig, data: pd.DataFrame) -> SimResult:
        """Run GBM simulation with given configuration.
        
        Args:
            config: Simulation configuration
            data: Historical price data
            
        Returns:
            SimResult with simulation outcomes
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        is_valid, error_msg = GBMService.validate_config(config, data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Calculate statistics from training data
        stats = StatisticsCalculator.calculate_statistics(
            data, 
            config.start_date, 
            config.end_date, 
            config.steps
        )
        
        # Get starting price and actual future price
        start_date_obj = pd.to_datetime(config.start_date)
        end_date_obj = pd.to_datetime(config.end_date)
        
        start_index = data.index[data['Date'] == start_date_obj][0]
        end_index = data.index[data['Date'] == end_date_obj][0]
        
        # Starting price is the price at the end of training period
        starting_price = float(data.iloc[end_index]["Close"])
        
        # Check if we have enough future data for comparison
        future_end_index = end_index + config.steps
        if future_end_index < len(data):
            compared_data = data.iloc[end_index:future_end_index + 1]
            real_price = compared_data['Close'].iloc[-1]
        else:
            real_price = None  # Predicting beyond available data
        
        # Convert engine string to EngineType enum
        try:
            engine_type = EngineType[config.engine.upper()]
        except KeyError:
            engine_type = EngineType.AUTO
        
        # Run simulation through dispatch layer
        start_time = time.perf_counter()
        
        walks, average_price = SimulationDispatcher.run_simulation(
            starting_price=starting_price,
            mu=stats.normalized_mu,
            variance=stats.normalized_variance,
            sigma=stats.normalized_deviation,
            steps=config.steps,
            paths=config.paths,
            engine=engine_type,
            threads=config.threads,
            seed=config.seed
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Prepare display paths
        np_paths = np.array(walks)
        display_paths = np_paths[:config.display_paths]
        
        # Calculate ratio if we have real price
        ratio = average_price / real_price if real_price else None
        
        return SimResult(
            avg_price=average_price,
            real_price=real_price,
            ratio=ratio,
            elapsed_time=elapsed,
            engine_used=engine_type.value,
            display_paths=display_paths
        )
    
    @staticmethod
    def get_available_engines() -> list[str]:
        """Get list of available simulation engines.
        
        Returns:
            List of engine names as strings.
        """
        engines = SimulationDispatcher.get_available_engines()
        return [engine.value for engine in engines]
