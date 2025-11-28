#!/usr/bin/env python
"""Dispatch layer for routing simulation requests to appropriate engines."""

from typing import Tuple
import numpy as np
from gbmapp.core.models import EngineType

try:
    from . import simulation  # type: ignore # C++ extension module
    _SIMULATION_AVAILABLE = True
except ImportError:
    simulation = None
    _SIMULATION_AVAILABLE = False


class SimulationDispatcher:
    """Dispatches simulation requests to appropriate C++ engines."""

    _system_caps = None  # Cache system capabilities

    @staticmethod
    def is_available() -> bool:
        """Check if C++ simulation engines are available.

        Returns:
            True if C++ module is loaded successfully.
        """
        return _SIMULATION_AVAILABLE

    @staticmethod
    def get_system_capabilities() -> dict:
        """Get system hardware capabilities.

        Returns:
            Dictionary with system capabilities (AVX2, AVX512, threads, etc.)
        """
        if not SimulationDispatcher.is_available():
            return {
                'has_avx2': False,
                'has_avx512': False,
                'num_threads': 1,
                'cache_line_size': 64
            }

        # Cache capabilities
        if SimulationDispatcher._system_caps is None:
            if simulation is not None:
                caps = simulation.GetSystemCapabilities()
                SimulationDispatcher._system_caps = {
                    'has_avx2': caps.has_avx2,
                    'has_avx512': caps.has_avx512,
                    'num_threads': caps.num_threads,
                    'cache_line_size': caps.cache_line_size
                }
            else:
                SimulationDispatcher._system_caps = {
                    'has_avx2': False,
                    'has_avx512': False,
                    'num_threads': 1,
                    'cache_line_size': 64
                }

        return SimulationDispatcher._system_caps

    @staticmethod
    def run_simulation(
        starting_price: float,
        mu: float,
        variance: float,
        sigma: float,
        steps: int,
        paths: int,
        engine: EngineType = EngineType.AUTO,
        threads: int | None = None,
        seed: int | None = None
    ) -> Tuple[np.ndarray, float]:
        """Execute GBM simulation using specified engine.

        Args:
            starting_price: Initial stock price
            mu: Drift coefficient (normalized)
            variance: Variance coefficient (normalized)
            sigma: Volatility coefficient (normalized)
            steps: Number of time steps
            paths: Number of simulation paths
            engine: Engine type to use
            threads: Number of threads (for MT engines)
            seed: Random seed (optional)

        Returns:
            Tuple of (walks array, average final price)

        Raises:
            RuntimeError: If C++ module is not available
            ValueError: If engine type is not supported or parameters are invalid
        """
        if not SimulationDispatcher.is_available():
            raise RuntimeError("C++ simulation module not available")

        # Validate parameters using C++ validation
        if simulation is not None and hasattr(simulation, 'ValidateParameters'):
            try:
                simulation.ValidateParameters(
                    starting_price, mu, variance, sigma, steps, paths
                )
            except Exception as e:
                raise ValueError(f"Parameter validation failed: {e}")

        # Route to appropriate engine
        if engine == EngineType.SCALAR:
            return SimulationDispatcher._run_scalar(
                starting_price, mu, variance, sigma, steps, paths
            )
        elif engine == EngineType.MT:
            return SimulationDispatcher._run_multithreaded(
                starting_price, mu, variance, sigma, steps, paths, threads
            )
        elif engine == EngineType.SIMD:
            return SimulationDispatcher._run_simd(
                starting_price, mu, variance, sigma, steps, paths, threads
            )
        elif engine == EngineType.AUTO:
            return SimulationDispatcher._run_auto(
                starting_price, mu, variance, sigma, steps, paths
            )
        else:
            raise ValueError(f"Unsupported engine type: {engine}")

    @staticmethod
    def _run_scalar(
        starting_price: float,
        mu: float,
        variance: float,
        sigma: float,
        steps: int,
        paths: int
    ) -> Tuple[np.ndarray, float]:
        """Run scalar (single-threaded) simulation."""
        if simulation is not None and hasattr(simulation, 'SimulateGBMScalar'):
            return simulation.SimulateGBMScalar(
                starting_price, mu, variance, sigma, steps, paths
            )
        elif simulation is not None:
            # Fallback to basic implementation
            return simulation.SimulateGBM(
                starting_price, mu, variance, sigma, steps, paths
            )
        else:
            raise RuntimeError("C++ simulation module not available")

    @staticmethod
    def _run_multithreaded(
        starting_price: float,
        mu: float,
        variance: float,
        sigma: float,
        steps: int,
        paths: int,
        threads: int | None
    ) -> Tuple[np.ndarray, float]:
        """Run multi-threaded simulation."""
        if simulation is not None and hasattr(simulation, 'SimulateGBMMultiThreaded'):
            return simulation.SimulateGBMMultiThreaded(
                starting_price, mu, variance, sigma, steps, paths
            )
        elif simulation is not None:
            # Fallback to basic implementation
            return simulation.SimulateGBM(
                starting_price, mu, variance, sigma, steps, paths
            )
        else:
            raise RuntimeError("C++ simulation module not available")

    @staticmethod
    def _run_simd(
        starting_price: float,
        mu: float,
        variance: float,
        sigma: float,
        steps: int,
        paths: int,
        threads: int | None
    ) -> Tuple[np.ndarray, float]:
        """Run SIMD-optimized multi-threaded simulation."""
        if simulation is not None and hasattr(simulation, 'SimulateGBMIntrinsicMT'):
            return simulation.SimulateGBMIntrinsicMT(
                starting_price, mu, variance, sigma, steps, paths
            )
        else:
            # Fallback to multi-threaded
            return SimulationDispatcher._run_multithreaded(
                starting_price, mu, variance, sigma, steps, paths, threads
            )

    @staticmethod
    def _run_auto(
        starting_price: float,
        mu: float,
        variance: float,
        sigma: float,
        steps: int,
        paths: int
    ) -> Tuple[np.ndarray, float]:
        """Run simulation with automatic engine selection."""
        # Auto-select best available engine based on hardware capabilities
        # Priority: SIMD (if AVX2 available) > MT > Scalar
        caps = SimulationDispatcher.get_system_capabilities()

        if (simulation is not None and
                hasattr(simulation, 'SimulateGBMIntrinsicMT') and
                caps.get('has_avx2', False)):
            return SimulationDispatcher._run_simd(
                starting_price, mu, variance, sigma, steps, paths, None
            )
        elif simulation is not None and hasattr(simulation, 'SimulateGBMMultiThreaded'):
            return SimulationDispatcher._run_multithreaded(
                starting_price, mu, variance, sigma, steps, paths, None
            )
        else:
            return SimulationDispatcher._run_scalar(
                starting_price, mu, variance, sigma, steps, paths
            )

    @staticmethod
    def get_available_engines() -> list[EngineType]:
        """Get list of available engines.

        Returns:
            List of available engine types.
        """
        if not SimulationDispatcher.is_available():
            return []

        available = [EngineType.AUTO]  # AUTO is always available

        if simulation is not None and hasattr(simulation, 'SimulateGBMScalar'):
            available.append(EngineType.SCALAR)
        if simulation is not None and hasattr(simulation, 'SimulateGBMMultiThreaded'):
            available.append(EngineType.MT)
        if simulation is not None and hasattr(simulation, 'SimulateGBMIntrinsicMT'):
            available.append(EngineType.SIMD)

        return available
