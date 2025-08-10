"""
Analyze why the quantum circuit is not producing Gaussian distributions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def analyze_circuit_structure():
    """Analyze the structure of generated circuits."""
    print("ANALYZING CIRCUIT STRUCTURE")
    print("="*50)
    
    for n_layers in [2, 3]:
        print(f"\n{n_layers}-layer circuit analysis:")
        qgb = QuantumGaltonBox(n_layers)
        circuit = qgb.generate_optimized_circuit()
        
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Classical bits: {circuit.num_clbits}")
        
        # Analyze gate operations
        gate_sequence = []
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            qubits = [q.index for q in instruction.qubits]
            gate_sequence.append((gate_name, qubits))
        
        print("  Gate sequence:")
        for i, (gate, qubits) in enumerate(gate_sequence[:15]):  # First 15 gates
            print(f"    {i+1:2d}. {gate:8s} on qubits {qubits}")
        if len(gate_sequence) > 15:
            print(f"    ... and {len(gate_sequence)-15} more gates")

def create_correct_galton_circuit(n_layers):
    """Create a corrected Galton box circuit following the paper exactly."""
    print(f"\nCreating corrected {n_layers}-layer circuit:")
    
    # Following the paper: we need 2n qubits + 1 control qubit
    total_qubits = 2 * n_layers + 1
    qreg = QuantumRegister(total_qubits, 'q')
    creg = ClassicalRegister(n_layers + 1, 'c')
    qc = QuantumCircuit(qreg, creg)
    
    # Control qubit is q[0]
    control = 0
    
    # Initialize ball at center
    ball_start = n_layers  # Start at middle position
    qc.x(qreg[ball_start])
    
    # Build layer by layer
    current_ball_positions = [ball_start]
    
    for layer in range(n_layers):
        print(f"  Layer {layer + 1}: Processing positions {current_ball_positions}")
        
        # Reset control qubit (except first layer)
        if layer > 0:
            qc.reset(qreg[control])
        
        # Put control in superposition
        qc.h(qreg[control])
        
        # Process each ball position
        next_positions = []
        for pos in current_ball_positions:
            left = pos - 1
            right = pos + 1
            
            if left >= 1 and right < total_qubits:
                # Apply quantum peg: controlled-SWAP operations
                qc.cswap(qreg[control], qreg[left], qreg[pos])
                qc.cx(qreg[pos], qreg[control])  # Balancing CNOT
                qc.cswap(qreg[control], qreg[pos], qreg[right])
                
                # Ball can now be at left or right position
                if left not in next_positions:
                    next_positions.append(left)
                if right not in next_positions:
                    next_positions.append(right)
        
        current_ball_positions = next_positions
        print(f"    Next layer positions: {current_ball_positions}")
    
    # Measure all output qubits (except control)
    for i in range(1, min(total_qubits, n_layers + 2)):
        if i < len(creg):
            qc.measure(qreg[i], creg[i-1])
    
    return qc

def test_corrected_circuit():
    """Test the corrected circuit implementation."""
    print("\nTESTING CORRECTED CIRCUIT")
    print("="*50)
    
    for n_layers in [2, 3, 4]:
        print(f"\n{n_layers}-layer corrected circuit:")
        
        # Create corrected circuit
        circuit = create_correct_galton_circuit(n_layers)
        
        # Simulate
        simulator = AerSimulator(method='statevector')
        job = simulator.run(circuit, shots=5000)
        counts = job.result().get_counts()
        
        # Process results (same as before)
        positions = []
        valid_shots = 0
        
        for bitstring, count in counts.items():
            pos = bitstring[::-1].find('1')
            if pos != -1:
                positions.extend([pos] * count)
                valid_shots += count
        
        if positions:
            positions = np.array(positions)
            mean = np.mean(positions)
            std = np.std(positions)
            
            # Expected values
            expected_mean = n_layers / 2.0
            expected_std = np.sqrt(n_layers / 4.0)
            
            print(f"  Valid shots: {valid_shots}/{sum(counts.values())} ({valid_shots/sum(counts.values()):.1%})")
            print(f"  Mean: {mean:.3f} (Expected: {expected_mean:.3f})")
            print(f"  Std:  {std:.3f} (Expected: {expected_std:.3f})")
            
            # Show distribution
            unique, counts_arr = np.unique(positions, return_counts=True)
            print("  Distribution:")
            for pos, count in zip(unique, counts_arr):
                prob = count / len(positions)
                bar = '#' * int(prob * 20)
                print(f"    Pos {pos}: {prob:.3f} {bar}")

def compare_implementations():
    """Compare original vs corrected implementation."""
    print("\nCOMPARING IMPLEMENTATIONS")
    print("="*50)
    
    n_layers = 3
    
    # Original implementation
    print("Original implementation:")
    qgb = QuantumGaltonBox(n_layers)
    orig_circuit = qgb.generate_optimized_circuit()
    orig_results = qgb.simulate(shots=5000)
    
    print(f"  Validity rate: {orig_results['validity_rate']:.1%}")
    print(f"  Mean: {orig_results['mean']:.3f}")
    print(f"  Std: {orig_results['std']:.3f}")
    
    # Corrected implementation
    print("\nCorrected implementation:")
    corr_circuit = create_correct_galton_circuit(n_layers)
    simulator = AerSimulator(method='statevector')
    job = simulator.run(corr_circuit, shots=5000)
    counts = job.result().get_counts()
    
    positions = []
    valid_shots = 0
    
    for bitstring, count in counts.items():
        pos = bitstring[::-1].find('1')
        if pos != -1:
            positions.extend([pos] * count)
            valid_shots += count
    
    if positions:
        positions = np.array(positions)
        validity_rate = valid_shots / sum(counts.values())
        mean = np.mean(positions)
        std = np.std(positions)
        
        print(f"  Validity rate: {validity_rate:.1%}")
        print(f"  Mean: {mean:.3f}")
        print(f"  Std: {std:.3f}")

if __name__ == "__main__":
    print("QUANTUM GALTON BOX - CIRCUIT ANALYSIS")
    print("="*70)
    
    analyze_circuit_structure()
    test_corrected_circuit()
    compare_implementations()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)