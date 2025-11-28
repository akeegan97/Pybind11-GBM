# GBM App: High-Performance Geometric Brownian Motion Simulator

A production-grade Python application for Monte Carlo simulation of stock prices using Geometric Brownian Motion (GBM) with three optimized C++ computation engines.

## Features

### Core Capabilities
- **Historical Data Analysis**: Load and analyze CSV stock price data with flexible date format support
- **GBM Parameter Estimation**: Automatically compute drift (μ) and volatility (σ) from historical returns
- **Monte Carlo Simulation**: Generate millions of price path predictions with configurable parameters
- **Risk Analysis**: Compare predicted prices against real future prices to validate model accuracy
- **Interactive Visualization**: Plot simulated paths and historical data with customizable display

### Performance Engines
Three computation backends optimized for different workload characteristics:

| Engine | Technology | Best For | Performance |
|--------|-----------|----------|-------------|
| Scalar | Single-threaded baseline | Small simulations, benchmarking | Baseline reference |
| MT | Multi-threaded | General-purpose, <100 steps | 15s for 1B paths × 30 steps |
| SIMD | AVX2 vectorized | Large simulations, >100 steps | 95s for 1B paths × 252 steps |
| AUTO | Intelligent selection | Default choice | Automatically selects optimal engine |

### Numerical Stability
- **Log-space computation**: Accumulate log-returns during simulation, exponentiate only at end
- **Kahan summation**: Compensated summation prevents floating-point drift with large path counts
- **Cache-line padding**: False sharing elimination in multi-threaded execution
- **Proper averaging**: Correct aggregate computation from distributed thread results

### User Interface
- Clean two-pane layout: Configuration panel (left) with tabbed results (right)
- Calendar widgets for date selection with YYYY-MM-DD format validation
- Real-time visualization of historical data, predicted paths, and statistics
- Dark theme with professional styling
- Engine selection between Scalar, MT, SIMD, or AUTO modes

## Installation

### Requirements
- Python 3.10 or later
- C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.15 or later
- AVX2 CPU support (optional, for SIMD engine)

### Build from Source

```bash
# Clone repository
git clone https://github.com/akeegan97/Pybind11-GBM.git
cd Pybind11-GBM

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and build C++ extensions
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run application
gbmapp
```

The build process will:
1. Detect system capabilities (AVX2, thread count)
2. Compile three C++ simulation engines with appropriate optimization flags
3. Generate Python bindings via pybind11
4. Install the `gbmapp` command-line entry point

## Usage

### Quick Start

```python
from gbmapp.core.models import SimConfig
from gbmapp.core.service import GBMService

# Configure simulation
config = SimConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    steps=252,              # One year of trading days
    paths=1_000_000,        # One million paths
    engine="AUTO",          # Automatic engine selection
    display_paths=100       # Show 100 paths in visualization
)

# Load data and run simulation
service = GBMService()
result = service.run_simulation(
    data_file="datasets/AAPL_historicals.csv",
    config=config
)

print(f"Average predicted price: ${result.avg_price:.2f}")
print(f"Actual future price: ${result.real_price:.2f}")
print(f"Accuracy ratio: {result.ratio:.2%}")
print(f"Computation time: {result.elapsed_time:.2f}s using {result.engine_used}")
```

### GUI Application

Launch the graphical interface:
```bash
gbmapp
```

Then:
1. Select start and end dates for training data using calendar widgets
2. Load CSV file with historical prices
3. Configure simulation parameters (steps, paths, display paths)
4. Choose computation engine (Scalar/MT/SIMD/AUTO)
5. Click Run Simulation to generate predictions
6. View results in three tabs:
   - Historical: Original price data
   - Predictions: Simulated GBM paths with statistics
   - Results: Detailed metrics and accuracy analysis

## Architecture

### Layer Structure

The application follows a clean layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         GUI Layer (Tkinter)             │
│  app.py, widgets.py, plots.py, theme.py │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Service Layer (Business Logic)      │
│           service.py                    │
│  - Parameter validation                 │
│  - Statistics computation               │
│  - Result formatting                    │
└──────────────┬──────────────────────────┘
               │
      ┌────────┼────────┐
      │        │        │
┌─────▼──┐ ┌──▼────┐ ┌─▼──────────┐
│  Data  │ │Dispatch│ │ Validation │
│  io.py │ │_dispatch│ │ validation│
└────────┘ └────────┘ └────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Native Module (C++ via Pybind11)      │
│  simulation.native                      │
│                                         │
│  Scalar | MT | SIMD engines             │
└─────────────────────────────────────────┘
```

### Key Components

**Models** (core/models.py):
- `SimConfig`: User-provided simulation parameters
- `SimResult`: Computed results with timing and metrics
- `Statistics`: Training data statistics and normalized parameters
- `EngineType`: Enumeration of available computation engines

**Services** (core/service.py):
- `GBMService`: Main orchestration layer coordinating data loading, statistical computation, and simulation execution

**Data** (data/io.py):
- `DataLoader`: CSV parsing with automatic date format detection

**Validation** (core/validation.py):
- `StatisticsCalculator`: Log-return analysis and parameter normalization for prediction periods

**Dispatch** (native/_dispatch.py):
- `SimulationDispatcher`: Routes computations to optimal C++ engine with automatic capability detection

## Algorithm Details

### Geometric Brownian Motion Model

The GBM model simulates stock prices under the stochastic differential equation:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Discretized using the Euler method:

$$S_{t+\Delta t} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \, Z_t\right]$$

Where:
- $S_t$ = Stock price at time $t$
- $\mu$ = Drift coefficient (expected return)
- $\sigma$ = Volatility (standard deviation of returns)
- $Z_t$ = Standard normal random variable

### Parameter Estimation

Historical parameters are estimated from training data log-returns $r_t = \ln(S_t / S_{t-1})$:

1. **Training statistics**:
   - $\mu_{train} = \mathbb{E}[r_t]$ (mean of log-returns)
   - $\sigma_{train} = \sqrt{\text{Var}(r_t)}$ (standard deviation)

2. **Normalization for prediction period** (e.g., 252 trading days):
   - $\mu_{pred} = \mu_{train} \times T$ 
   - $\sigma_{pred} = \sigma_{train} \times \sqrt{T}$

### Log-Space Computation

To maintain numerical stability with large path counts, the implementation:

1. Accumulates log-returns in log-space rather than compounding prices
2. Applies Kahan summation to maintain precision
3. Exponentiates only once at the end of each path

This approach prevents overflow and underflow issues while maintaining precision across millions of additions.

## Performance Analysis

### Benchmark Results

Performance comparison for 1 billion paths:

| Steps | Scalar | MT | SIMD | Selected |
|-------|--------|-----|------|----------|
| 30 | 45s | 15s | 32s | MT |
| 252 | 380s | 130s | 95s | SIMD |
| 504 | 760s | 260s | 160s | SIMD |

Performance observations:
- MT dominates for small step counts due to lower RNG overhead
- SIMD achieves breakeven at approximately 100-120 steps
- SIMD scales better with computational complexity


### Memory Usage

Typical memory consumption for standard workloads:
- Display arrays: O(min(N, displayPaths) × T)
- Per-thread state: O(1) with cache-line padding
- Total: 100-500 MB for typical configurations

## Building and Configuration

### Build Process

The build system automatically applies:
- `-O3` optimization level
- `-march=native` for CPU-specific instruction sets
- `-mavx2 -mfma` when available
- OpenMP support for multi-threading

To rebuild with custom optimization flags:

```bash
export CXXFLAGS="-O3 -march=native"
pip install --force-reinstall -e .
```

### System Capability Detection

The application automatically detects available CPU features:
- AVX2 support for SIMD engine selection
- Thread count for automatic work distribution
- Cache line size for optimal padding alignment

## Troubleshooting

### Build Issues

**Missing dependencies**:
```bash
pip install pybind11 numpy
```

**AVX2 not available**:
- Check CPU support: `lscpu | grep avx2`
- The SIMD engine gracefully falls back to scalar implementation if AVX2 is unavailable

**Compilation errors**:
- Ensure C++ compiler is C++17 compatible
- On systems with restricted environments: `pip install -e . --no-cache-dir`

### Runtime Issues

**ImportError for native module**:
```bash
pip install --force-reinstall -e .
python -c "import gbmapp.native.simulation; print('Success')"
```

**Slow display path generation**:
- Large display_paths values (>1000) require significant memory
- Default value of 100 paths is recommended for optimal responsiveness

## Project Structure

```
Pybind11-GBM/
├── src/gbmapp/
│   ├── __init__.py
│   ├── core/
│   │   ├── models.py          # Data classes and enums
│   │   ├── service.py         # Business logic orchestration
│   │   └── validation.py      # Statistical calculations
│   ├── cpp/
│   │   ├── bindings.cpp       # Pybind11 Python interface
│   │   ├── simulation.cpp     # System capability detection
│   │   ├── simulation_common.h # Shared utilities (Kahan, types)
│   │   ├── simulation_scalar.cpp
│   │   ├── simulation_mt.cpp
│   │   └── simulation_simd.cpp
│   ├── data/
│   │   └── io.py              # CSV loading and validation
│   ├── gui/
│   │   ├── app.py             # Main window and orchestration
│   │   ├── widgets.py         # Custom UI components
│   │   ├── plots.py           # Matplotlib visualization
│   │   └── theme.py           # Styling constants
│   └── native/
│       └── _dispatch.py       # C++ engine routing
├── datasets/                  # Sample CSV files
├── pyproject.toml             # Project metadata and entry points
└── setup.py                   # Build configuration
```

### Extending the System

To implement a new simulation engine:

1. Create `src/gbmapp/cpp/simulation_custom.cpp`
2. Implement the simulation function with matching signature:
   ```cpp
   SimulationResult SimulateGBMCustom(
       double startingPrice, double normalizedMu, double normalizedVar,
       double normalizedStd, int steps, int paths, int displayPathsRequested
   )
   ```
3. Add Python binding in `bindings.cpp`
4. Add routing logic in `_dispatch.py`
5. Update `setup.py` to include the new source file


## License

This project is provided as-is for educational and research purposes.

## Author

Andrew Keegan

---

**Last Updated**: November 2025  
**Version**: 0.1.0  
**Status**: Active Development
