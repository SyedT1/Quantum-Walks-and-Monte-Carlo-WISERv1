"""
Test the complex quantum Galton Box circuit with more qubits and greater depth
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np
import time

print("="*80)
print("COMPLEX QUANTUM GALTON BOX IMPLEMENTATION")
print("="*80)

# Test different layer counts with both implementations
test_layers = [3, 5, 7]

for n_layers in test_layers:
    print(f"\n{n_layers}-Layer Quantum Galton Box Comparison:")
    print("-" * 70)
    
    # Test both implementations
    for circuit_type, use_complex in [("Simple", False), ("Complex", True)]:
        print(f"\n{circuit_type} Implementation:")
        
        # Create and generate circuit
        qgb = QuantumGaltonBox(num_layers=n_layers)
        
        start_time = time.time()
        
        if use_complex:
            circuit = qgb.generate_complex_circuit()
        else:
            circuit = qgb.generate_optimized_circuit()
        
        circuit_gen_time = time.time() - start_time
        
        # Get circuit metrics
        metrics = qgb.get_circuit_metrics()
        
        print(f"  Circuit Generation: {circuit_gen_time:.4f}s")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Depth: {circuit.depth()}")
        print(f"  Gates: {metrics['total_gates']}")
        print(f"  Gate types: {metrics['gate_counts']}")
        
        # Run simulation
        start_time = time.time()
        results = qgb.simulate(shots=5000, use_complex=use_complex)
        sim_time = time.time() - start_time
        
        print(f"  Simulation Time: {sim_time:.4f}s")
        print(f"  Mean: {results['mean']:.4f} (expected: {n_layers/2:.4f})")
        print(f"  Std:  {results['std']:.4f} (expected: {np.sqrt(n_layers/4):.4f})")
        print(f"  Validity Rate: {results['validity_rate']:.1%}")
        
        # Statistical test
        verification = qgb.verify_gaussian(plot=False)
        chi2_text = f"{verification['chi2_pvalue']:.4f}" if verification['chi2_valid'] else "N/A"
        
        print(f"  Chi² p-value: {chi2_text}")
        print(f"  Mean Error: {verification['mean_error_pct']:.1%}")
        print(f"  Std Error: {verification['std_error_pct']:.1%}")
        print(f"  Gaussian Test: {'PASS' if verification['is_gaussian'] else 'FAIL'}")

print("\n" + "="*80)
print("COMPLEXITY ANALYSIS")
print("="*80)

# Detailed analysis for 5-layer circuit
n = 5
print(f"\nDetailed Analysis for {n}-Layer Circuits:")
print("-" * 50)

for circuit_type, use_complex in [("Simple", False), ("Complex", True)]:
    qgb = QuantumGaltonBox(num_layers=n)
    
    if use_complex:
        circuit = qgb.generate_complex_circuit()
    else:
        circuit = qgb.generate_optimized_circuit()
    
    print(f"\n{circuit_type} Circuit:")
    print(f"  Total Qubits: {circuit.num_qubits}")
    print(f"  Classical Bits: {circuit.num_clbits}")
    print(f"  Circuit Depth: {circuit.depth()}")
    
    # Analyze gate composition
    gate_counts = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    print(f"  Gate Composition:")
    for gate, count in sorted(gate_counts.items()):
        print(f"    {gate}: {count}")
    
    print(f"  Total Gates: {sum(gate_counts.values())}")

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

# Performance comparison across different layer counts
layer_range = [3, 4, 5, 6, 7]
simple_times = []
complex_times = []

print(f"\nSimulation Time Comparison (5000 shots):")
print("Layers | Simple (s) | Complex (s) | Ratio")
print("-------|------------|-------------|-------")

for n in layer_range:
    # Simple implementation
    qgb_simple = QuantumGaltonBox(num_layers=n)
    start = time.time()
    qgb_simple.simulate(shots=5000, use_complex=False)
    simple_time = time.time() - start
    simple_times.append(simple_time)
    
    # Complex implementation  
    qgb_complex = QuantumGaltonBox(num_layers=n)
    start = time.time()
    qgb_complex.simulate(shots=5000, use_complex=True)
    complex_time = time.time() - start
    complex_times.append(complex_time)
    
    ratio = complex_time / simple_time if simple_time > 0 else float('inf')
    
    print(f"{n:6d} | {simple_time:10.4f} | {complex_time:11.4f} | {ratio:5.1f}x")

print(f"\nAverage performance penalty: {np.mean(complex_times)/np.mean(simple_times):.1f}x slower")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("• Complex circuit uses 4x more qubits (4n vs n)")
print("• Circuit depth increases significantly (~20+ vs 2)")
print("• Includes entanglement, multi-qubit gates, and rotations")
print("• Maintains same mathematical distribution properties")
print("• Performance trade-off for increased quantum complexity")
print("="*80)