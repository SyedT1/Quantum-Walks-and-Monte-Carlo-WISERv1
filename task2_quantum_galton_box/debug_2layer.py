"""
Debug the 2-layer Quantum Galton Box circuit to understand why positions are missing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np
from collections import Counter

print("="*80)
print("DEBUG: 2-LAYER QUANTUM GALTON BOX")
print("="*80)

# Test both simple and complex implementations
for circuit_name, use_complex in [("Simple", False), ("Complex", True)]:
    print(f"\n{circuit_name} Implementation:")
    print("-" * 50)
    
    qgb = QuantumGaltonBox(num_layers=2)
    
    if use_complex:
        circuit = qgb.generate_complex_circuit()
    else:
        circuit = qgb.generate_optimized_circuit()
    
    print(f"Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    print(f"Circuit structure: (skipped due to encoding)")
    # print(circuit)  # Skip circuit printing due to Unicode issues
    
    # Run with high shot count
    results = qgb.simulate(shots=10000, use_complex=use_complex)
    
    print(f"\nRaw measurement counts: {results['counts']}")
    print(f"Positions array (first 20): {results['positions'][:20]}")
    print(f"Position range: {results['positions'].min()} to {results['positions'].max()}")
    
    # Count positions manually
    position_counts = Counter(results['positions'])
    sorted_positions = sorted(position_counts.items())
    print(f"Position frequencies: {sorted_positions}")
    
    # Calculate statistics
    positions_array = np.array(results['positions'])
    print(f"Mean: {positions_array.mean():.4f}")
    print(f"Std: {positions_array.std():.4f}")
    print(f"Unique positions: {np.unique(positions_array)}")
    
    # Expected for 2-layer binomial (n=2, p=0.5)
    expected_mean = 2 * 0.5  # n * p = 1.0
    expected_std = np.sqrt(2 * 0.5 * 0.5)  # sqrt(n * p * (1-p)) = sqrt(0.5) = 0.707
    print(f"Expected mean: {expected_mean:.4f}")
    print(f"Expected std: {expected_std:.4f}")
    
    # Expected probabilities for binomial(2, 0.5)
    # P(X=0) = C(2,0) * 0.5^0 * 0.5^2 = 1 * 1 * 0.25 = 0.25
    # P(X=1) = C(2,1) * 0.5^1 * 0.5^1 = 2 * 0.5 * 0.5 = 0.50  
    # P(X=2) = C(2,2) * 0.5^2 * 0.5^0 = 1 * 0.25 * 1 = 0.25
    expected_probs = [0.25, 0.50, 0.25]  # For positions 0, 1, 2
    print(f"Expected probabilities: {expected_probs}")
    
    # Calculate actual probabilities
    total_shots = len(results['positions'])
    actual_probs = []
    for pos in [0, 1, 2]:
        count = position_counts.get(pos, 0)
        prob = count / total_shots
        actual_probs.append(prob)
        print(f"Position {pos}: {count:5d} shots ({prob:.3f}) - Expected: {expected_probs[pos]:.3f}")
    
    print(f"Actual probabilities: {actual_probs}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\nFor a 2-layer Galton Box, we should see:")
print("- Position 0: 25% (both qubits measure 0)")
print("- Position 1: 50% (one qubit measures 1, one measures 0)")  
print("- Position 2: 25% (both qubits measure 1)")
print("- Total positions: 3 (not 2!)")

print("\nThe issue might be:")
print("1. Circuit implementation not generating all possible outcomes")
print("2. Position calculation from bitstring incorrect")
print("3. Complex circuit altering the distribution")
print("4. Measurement or processing bug")

print("\nLet's examine the raw bitstrings more carefully...")

# Additional debugging - examine raw bitstrings
qgb_simple = QuantumGaltonBox(num_layers=2)
circuit_simple = qgb_simple.generate_optimized_circuit()
results_simple = qgb_simple.simulate(shots=1000, use_complex=False)

print(f"\nRaw bitstring analysis (Simple circuit):")
bitstring_counts = Counter()
for bitstring, count in results_simple['counts'].items():
    bitstring_counts[bitstring] = count
    # Calculate position from bitstring
    pos = bitstring.count('1')
    print(f"Bitstring '{bitstring}' -> Position {pos}: {count} times")

print(f"\nBitstring distribution: {dict(bitstring_counts)}")
print("Expected bitstrings: '00' (pos 0), '01' (pos 1), '10' (pos 1), '11' (pos 2)")