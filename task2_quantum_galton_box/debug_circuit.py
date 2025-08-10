"""
Debug the quantum circuit to understand why distributions aren't Gaussian
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def test_simple_peg():
    """Test a simple quantum peg following the paper exactly."""
    print("Testing simple quantum peg (Figure 3 from paper):")
    
    # Create exact circuit from Figure 3
    qreg = QuantumRegister(4, 'q')
    creg = ClassicalRegister(2, 'c')  # Only measure output qubits
    qc = QuantumCircuit(qreg, creg)
    
    # Initialize as in paper: q0 in superposition, q2 has the "ball"
    qc.h(qreg[0])  # Control qubit in superposition
    qc.x(qreg[2])  # Place "ball" on middle qubit
    
    # Quantum peg operations from paper
    qc.cswap(qreg[0], qreg[1], qreg[2])  # Controlled-SWAP q1,q2 
    qc.cx(qreg[2], qreg[0])              # CNOT q2 -> q0
    qc.cswap(qreg[0], qreg[2], qreg[3])  # Controlled-SWAP q2,q3
    
    # Measure output qubits (q1 and q3)
    qc.measure(qreg[1], creg[0])
    qc.measure(qreg[3], creg[1])
    
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    
    # Simulate
    simulator = AerSimulator(method='statevector')
    job = simulator.run(qc, shots=10000)
    counts = job.result().get_counts()
    
    print("Results:")
    total = sum(counts.values())
    for bitstring, count in sorted(counts.items()):
        prob = count / total
        print(f"  {bitstring}: {count:4d} ({prob:.3f})")
    
    return counts

def test_layers_separately():
    """Test each layer count to understand the pattern."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT LAYER COUNTS")
    print("="*60)
    
    for n in range(1, 5):
        print(f"\nTesting {n} layers:")
        print("-" * 30)
        
        qgb = QuantumGaltonBox(n)
        circuit = qgb.generate_optimized_circuit()
        
        print(f"Circuit: {circuit.num_qubits} qubits, {circuit.depth()} depth")
        
        # Quick simulation
        results = qgb.simulate(shots=5000)
        
        # Show frequency distribution
        unique, counts = np.unique(results['positions'], return_counts=True)
        probs = counts / counts.sum()
        
        print("Position distribution:")
        for pos, prob in zip(unique, probs):
            bar = '#' * int(prob * 30)
            print(f"  Pos {pos}: {prob:.3f} {bar}")
        
        print(f"  Mean: {results['mean']:.3f}")
        print(f"  Std:  {results['std']:.3f}")
        
        # Expected for true Galton board
        expected_mean = n / 2
        expected_std = np.sqrt(n / 4)
        print(f"  Expected mean: {expected_mean:.3f}")
        print(f"  Expected std:  {expected_std:.3f}")

def analyze_measurement_outputs():
    """Analyze what qubits are actually being measured."""
    print("\n" + "="*60)
    print("ANALYZING MEASUREMENT STRUCTURE")
    print("="*60)
    
    for n in [2, 3]:
        print(f"\n{n}-layer circuit measurement analysis:")
        qgb = QuantumGaltonBox(n)
        circuit = qgb.generate_optimized_circuit()
        
        # Find measurement operations
        measurements = []
        for i, (instruction, qargs, cargs) in enumerate(circuit.data):
            if instruction.name == 'measure':
                qubit_idx = qargs[0].index
                classical_idx = cargs[0].index
                measurements.append((qubit_idx, classical_idx))
        
        print(f"  Measurements: qubit -> classical")
        for q_idx, c_idx in measurements:
            print(f"    q{q_idx} -> c{c_idx}")
        
        # Analyze how results are processed
        results = qgb.simulate(shots=1000)
        print(f"  Raw measurement positions: {np.unique(results['positions'])}")

if __name__ == "__main__":
    print("QUANTUM GALTON BOX - CIRCUIT DEBUGGING")
    print("=" * 70)
    
    # Test simple peg first
    test_simple_peg()
    
    # Test layer scaling
    test_layers_separately() 
    
    # Analyze measurement structure
    analyze_measurement_outputs()
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)