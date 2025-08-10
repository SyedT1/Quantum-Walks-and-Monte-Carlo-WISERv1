"""
Test the improved Gaussian verification with proper statistical tests
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np

print("="*70)
print("TESTING IMPROVED GAUSSIAN VERIFICATION")
print("="*70)

# Test with different layer counts
test_layers = [2, 4, 6]

for n_layers in test_layers:
    print(f"\nTesting {n_layers}-layer Galton Box:")
    print("-" * 50)
    
    # Create and simulate
    qgb = QuantumGaltonBox(num_layers=n_layers)
    circuit = qgb.generate_optimized_circuit()
    
    # Run with more shots for better statistics
    results = qgb.simulate(shots=10000)
    
    # New verification method
    verification = qgb.verify_gaussian(plot=False)
    
    print(f"Circuit: {circuit.num_qubits} qubits, {circuit.depth()} depth")
    print(f"Mean: {results['mean']:.4f} (Theory: {verification['theoretical_mean']:.4f})")
    print(f"Std:  {results['std']:.4f} (Theory: {verification['theoretical_std']:.4f})")
    print(f"Mean Error: {verification['mean_error_pct']:.1%}")
    print(f"Std Error:  {verification['std_error_pct']:.1%}")
    
    if verification['chi2_valid']:
        print(f"Chi² p-value: {verification['chi2_pvalue']:.4f}")
        print(f"Chi² test: {'PASS' if verification['is_gaussian_chi2'] else 'FAIL'}")
    else:
        print("Chi² test: N/A (insufficient expected frequencies)")
    
    print(f"Mean/Std test: {'PASS' if verification['is_gaussian_means'] else 'FAIL'}")
    
    if verification['ad_valid']:
        print(f"Anderson-Darling: {'PASS' if verification['is_gaussian_ad'] else 'FAIL'}")
    else:
        print("Anderson-Darling: N/A")
    
    print(f"Overall Assessment: {'GAUSSIAN' if verification['is_gaussian'] else 'NOT GAUSSIAN'}")
    print(f"Tests Passed: {verification['tests_passed']}")
    
    # Show frequency comparison for small cases
    if n_layers <= 4:
        print("\nFrequency Analysis:")
        obs_freq = verification['observed_frequencies']
        exp_freq = verification['expected_frequencies']
        
        print("Pos | Observed | Expected | Ratio")
        print("----|----------|----------|------")
        for pos in sorted(obs_freq.keys()):
            obs = obs_freq[pos]
            exp = exp_freq.get(pos, 0)
            ratio = obs/exp if exp > 0 else float('inf')
            print(f"{pos:3d} | {obs:8d} | {exp:8.1f} | {ratio:5.2f}")

print("\n" + "="*70)
print("VERIFICATION TEST COMPLETE")
print("The new method provides proper statistical assessment for discrete distributions")
print("="*70)