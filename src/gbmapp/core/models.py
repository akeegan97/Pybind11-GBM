from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class SimConfig:
    start_date: str  # Training start date
    end_date: str    # Training end date
    steps: int       # Prediction steps
    paths: int       # Number of simulation paths
    engine: str      # eg AUTO, SCALAR, MT, SIMD
    threads: int | None = None
    seed: int | None = None
    display_paths: int = 100  # Number of paths to display in plot


@dataclass
class SimResult:
    avg_price: float 
    real_price: float | None  # May be None if predicting beyond available data
    ratio: float | None       # May be None if real_price is None
    elapsed_time: float
    engine_used: str
    display_paths: np.ndarray

@dataclass
class Statistics:
    """Container for statistical measures from training data."""
    
    training_mu: float              # Mean of log returns
    training_deviation: float       # Standard deviation of log returns
    training_variance: float        # Variance of log returns
    normalized_mu: float           # Drift term adjusted for prediction steps
    normalized_variance: float     # Variance adjusted for prediction steps
    normalized_deviation: float    # Volatility term for prediction

class EngineType(Enum):
    """Available simulation engines."""
    AUTO = "AUTO"
    SCALAR = "SCALAR"
    MT = "MT"
    SIMD = "SIMD"