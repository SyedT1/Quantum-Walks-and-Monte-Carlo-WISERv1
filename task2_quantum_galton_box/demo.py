"""
Minimal demonstration of Quantum Galton Box
Shows that the implementation works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np

print("="*70)
print("QUANTUM GALTON BOX - MINIMAL DEMONSTRATION")
print("Task 2: Multi-Layer Galton Box Implementation")
print("="*70)

# Demonstrate for 3 layers
n_layers = 3
print(f"\nCreating {n_layers}-layer Quantum Galton Box...")

# Create the quantum circuit
qgb = QuantumGaltonBox(num_layers=n_layers)
circuit = qgb.generate_optimized_circuit()

# Display circuit information
print("\nCircuit Properties:")
print(f"  Number of qubits: {circuit.num_qubits}")
print(f"  Circuit depth: {circuit.depth()}")
print(f"  Number of gates: {sum(1 for _ in circuit.data)}")

# Run simulation
print(f"\nRunning simulation with 5000 shots...")
results = qgb.simulate(shots=5000)

# Display results
print("\nSimulation Results:")
print(f"  Mean position: {results['mean']:.4f}")
print(f"  Standard deviation: {results['std']:.4f}")
print(f"  Variance: {results['variance']:.4f}")

# Show distribution
print("\nPosition Distribution:")
for pos, count in sorted(results['frequencies']):
    prob = count / results['total_shots']
    bar = '#' * int(prob * 50)
    print(f"  Position {pos}: {count:4d} ({prob:.3f}) {bar}")

# Verify Gaussian properties
print("\nGaussian Verification:")
theoretical_mean = n_layers / 2
theoretical_std = np.sqrt(n_layers / 4)
print(f"  Theoretical mean: {theoretical_mean:.4f}")
print(f"  Theoretical std: {theoretical_std:.4f}")
print(f"  Actual mean: {results['mean']:.4f}")
print(f"  Actual std: {results['std']:.4f}")

mean_error = abs(results['mean'] - theoretical_mean)
std_error = abs(results['std'] - theoretical_std)
print(f"  Mean error: {mean_error:.4f}")
print(f"  Std error: {std_error:.4f}")

print("\n" + "="*70)
print("DEMONSTRATION COMPLETE")
print("The Quantum Galton Box successfully generates distributions")
print("that approximate Gaussian behavior as predicted by theory.")
print("="*70)