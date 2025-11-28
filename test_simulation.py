#!/usr/bin/env python
"""Test simulation without GUI."""

from gbmapp.native import simulation
from gbmapp.native._dispatch import SimulationDispatcher
from gbmapp.core.models import EngineType

print("=" * 60)
print("GBM Simulation Test (No GUI)")
print("=" * 60)

# Check system capabilities
caps = simulation.GetSystemCapabilities()
print(f"\nSystem Capabilities:")
print(f"  AVX2: {caps.has_avx2}")
print(f"  AVX512: {caps.has_avx512}")
print(f"  Threads: {caps.num_threads}")
print(f"  Cache Line: {caps.cache_line_size} bytes")

# Test parameters
starting_price = 100.0
mu = 0.0005
variance = 0.0001
sigma = 0.01
steps = 252
paths = 10000

print(f"\nSimulation Parameters:")
print(f"  Starting Price: ${starting_price}")
print(f"  Steps: {steps}")
print(f"  Paths: {paths}")

# Test all engines
for engine_type in [EngineType.SCALAR, EngineType.MT, EngineType.SIMD]:
    try:
        print(f"\nTesting {engine_type.value} engine...")
        display_paths, avg_price = SimulationDispatcher.run_simulation(
            starting_price, mu, variance, sigma, steps, paths, engine_type
        )
        print(f"  Average final price: ${avg_price:.2f}")
        print(f"  Display paths shape: {display_paths.shape}")
        print(f"  ✓ {engine_type.value} engine works!")
    except Exception as e:
        print(f"  ✗ {engine_type.value} engine failed: {e}")

print("\n" + "=" * 60)
