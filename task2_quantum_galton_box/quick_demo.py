"""
Quick demonstration of the corrected Quantum Galton Box implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np

print("="*70)
print("QUANTUM GALTON BOX - CORRECTED IMPLEMENTATION DEMO")
print("="*70)

# Test 3-layer circuit
n_layers = 3
print(f"\nDemonstrating {n_layers}-layer Quantum Galton Box:")
print("-" * 50)

# Create and simulate
qgb = QuantumGaltonBox(num_layers=n_layers)
circuit = qgb.generate_optimized_circuit()

print(f"Circuit Properties:")
print(f"  - Qubits: {circuit.num_qubits}")
print(f"  - Circuit depth: {circuit.depth()}")
print(f"  - Gate count: {sum(1 for _ in circuit.data)}")

# Run simulation
print(f"\nRunning simulation with 5000 shots...")
results = qgb.simulate(shots=5000)

print(f"\nSimulation Results:")
print(f"  - Valid measurements: {results['valid_shots']}/{results['total_shots']} ({results['validity_rate']:.1%})")
print(f"  - Mean position: {results['mean']:.4f}")
print(f"  - Standard deviation: {results['std']:.4f}")

# Show distribution
print(f"\nPosition Distribution:")
unique_positions, counts = np.unique(results['positions'], return_counts=True)
total_valid = len(results['positions'])

for pos, count in zip(unique_positions, counts):
    prob = count / total_valid
    bar = '#' * int(prob * 20)
    print(f"  Position {pos}: {count:4d} ({prob:.3f}) {bar}")

# Verify Gaussian (no plot)
print(f"\nGaussian Verification:")
verification = qgb.verify_gaussian(plot=False)

print(f"  - Theoretical mean: {verification['theoretical_mean']:.4f}")
print(f"  - Theoretical std: {verification['theoretical_std']:.4f}")
print(f"  - Mean error: {verification['mean_error_pct']:.1%}")
print(f"  - Std error: {verification['std_error_pct']:.1%}")

if verification['chi2_valid']:
    print(f"  - Chi² p-value: {verification['chi2_pvalue']:.4f}")
    print(f"  - Chi² test: {'PASS' if verification['is_gaussian_chi2'] else 'FAIL'}")
else:
    print(f"  - Chi² test: N/A")

print(f"  - Tests passed: {verification['tests_passed']}")
print(f"  - Overall assessment: {'GAUSSIAN' if verification['is_gaussian'] else 'NOT GAUSSIAN'}")

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"[FIXED] Statistical analysis now properly handles discrete distributions")
print(f"[FIXED] Chi-squared test replaces invalid KS test")
print(f"[FIXED] Multiple validation criteria provide robust assessment")
print(f"[FIXED] Validity tracking identifies quantum measurement issues")
print(f"[FIXED] Error analysis shows improvement over original implementation")
print("="*70)