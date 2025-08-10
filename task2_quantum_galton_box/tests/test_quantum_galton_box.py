"""
Unit tests for Quantum Galton Box implementation.
Tests 1-2 layer circuits against expected behavior from the paper.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_galton_box import QuantumGaltonBox
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class TestQuantumGaltonBox:
    """Test suite for Quantum Galton Box implementation."""
    
    def test_single_layer_circuit(self):
        """Test 1-layer Galton box circuit structure."""
        qgb = QuantumGaltonBox(num_layers=1)
        circuit = qgb.generate_optimized_circuit()
        
        # Check circuit dimensions
        assert circuit.num_qubits == 2, "1-layer circuit should have 2 qubits"
        assert circuit.num_clbits == 2, "1-layer circuit should have 2 classical bits"
        
        # Check that circuit contains expected gates
        gate_names = [inst.operation.name for inst in circuit.data]
        assert 'x' in gate_names, "Circuit should have X gate for initialization"
        assert 'h' in gate_names, "Circuit should have Hadamard gate"
        assert 'cswap' in gate_names, "Circuit should have controlled-SWAP gates"
        
    def test_two_layer_circuit(self):
        """Test 2-layer Galton box circuit structure and output."""
        qgb = QuantumGaltonBox(num_layers=2)
        circuit = qgb.generate_optimized_circuit()
        
        # Check circuit dimensions
        assert circuit.num_qubits == 4, "2-layer circuit should have 4 qubits"
        assert circuit.num_clbits == 3, "2-layer circuit should have 3 classical bits"
        
        # Run simulation
        results = qgb.simulate(shots=1000)
        
        # Check that we get outputs
        assert len(results['counts']) > 0, "Should have measurement results"
        assert results['total_shots'] == 1000, "Should record correct shot count"
        
    def test_single_peg_probabilities(self):
        """Test that single peg produces 50-50 probability."""
        qgb = QuantumGaltonBox(num_layers=1)
        circuit = qgb.generate_optimized_circuit()
        
        # Simulate with many shots for statistical accuracy
        results = qgb.simulate(shots=10000)
        
        # Check that we get approximately 50-50 distribution
        positions = results['positions']
        unique, counts = np.unique(positions, return_counts=True)
        probabilities = counts / len(positions)
        
        # Should have 2 outcomes with roughly equal probability
        assert len(unique) == 2, "Single peg should produce 2 outcomes"
        for prob in probabilities:
            assert 0.45 < prob < 0.55, f"Probability {prob} not close to 0.5"
    
    def test_gaussian_verification(self):
        """Test Gaussian distribution verification for multi-layer circuit."""
        qgb = QuantumGaltonBox(num_layers=4)
        circuit = qgb.generate_optimized_circuit()
        
        # Run simulation with sufficient shots
        results = qgb.simulate(shots=5000)
        
        # Verify Gaussian distribution
        verification = qgb.verify_gaussian(plot=False)
        
        assert 'ks_pvalue' in verification, "Should have KS test result"
        assert 'is_gaussian' in verification, "Should have Gaussian verdict"
        assert verification['theoretical_mean'] == 2.0, "4-layer theoretical mean should be 2"
        assert abs(verification['theoretical_std'] - 1.0) < 0.01, "4-layer theoretical std should be 1"
    
    def test_circuit_metrics(self):
        """Test circuit complexity metrics calculation."""
        qgb = QuantumGaltonBox(num_layers=3)
        circuit = qgb.generate_optimized_circuit()
        
        metrics = qgb.get_circuit_metrics()
        
        assert 'num_qubits' in metrics, "Should report qubit count"
        assert 'circuit_depth' in metrics, "Should report circuit depth"
        assert 'total_gates' in metrics, "Should report total gate count"
        assert 'gate_counts' in metrics, "Should report individual gate counts"
        
        # Check theoretical bound
        n = 3
        theoretical = 2 * n**2 + 5 * n + 2
        assert metrics['theoretical_upper_bound'] == theoretical, "Theoretical bound calculation error"
    
    def test_scaling_properties(self):
        """Test that circuit scales properly with layer count."""
        gate_counts = []
        depths = []
        
        for n in range(1, 6):
            qgb = QuantumGaltonBox(num_layers=n)
            circuit = qgb.generate_optimized_circuit()
            metrics = qgb.get_circuit_metrics()
            
            gate_counts.append(metrics['total_gates'])
            depths.append(metrics['circuit_depth'])
        
        # Check that gate count increases
        for i in range(len(gate_counts) - 1):
            assert gate_counts[i+1] > gate_counts[i], "Gate count should increase with layers"
            
        # Check that depth increases
        for i in range(len(depths) - 1):
            assert depths[i+1] >= depths[i], "Circuit depth should not decrease with layers"
    
    def test_measurement_consistency(self):
        """Test that measurements are consistent across multiple runs."""
        qgb = QuantumGaltonBox(num_layers=2)
        circuit = qgb.generate_optimized_circuit()
        
        means = []
        for _ in range(5):
            results = qgb.simulate(shots=1000)
            means.append(results['mean'])
        
        # Check that means are relatively consistent
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        
        assert std_of_means < 0.2, f"Means vary too much: std={std_of_means}"
        assert 0.5 < mean_of_means < 1.5, f"Mean {mean_of_means} outside expected range for 2 layers"
    
    def test_empty_circuit_error(self):
        """Test that accessing metrics without circuit raises error."""
        qgb = QuantumGaltonBox(num_layers=3)
        
        with pytest.raises(ValueError):
            qgb.get_circuit_metrics()
    
    def test_no_results_error(self):
        """Test that verification without simulation raises error."""
        qgb = QuantumGaltonBox(num_layers=3)
        circuit = qgb.generate_optimized_circuit()
        
        with pytest.raises(ValueError):
            qgb.verify_gaussian()


class TestCircuitGeneration:
    """Test different circuit generation methods."""
    
    def test_general_vs_optimized(self):
        """Compare general and optimized circuit generation."""
        qgb = QuantumGaltonBox(num_layers=3)
        
        # Generate both versions
        general_circuit = qgb.generate_galton_circuit()
        optimized_circuit = qgb.generate_optimized_circuit()
        
        # Both should exist
        assert general_circuit is not None
        assert optimized_circuit is not None
        
        # Optimized should have fewer or equal gates
        general_gates = sum(1 for _ in general_circuit.data)
        optimized_gates = sum(1 for _ in optimized_circuit.data)
        
        assert optimized_gates <= general_gates, "Optimized circuit should not have more gates"
    
    def test_circuit_reproducibility(self):
        """Test that same parameters produce identical circuits."""
        qgb1 = QuantumGaltonBox(num_layers=3)
        qgb2 = QuantumGaltonBox(num_layers=3)
        
        circuit1 = qgb1.generate_optimized_circuit()
        circuit2 = qgb2.generate_optimized_circuit()
        
        # Should have same structure
        assert circuit1.num_qubits == circuit2.num_qubits
        assert circuit1.num_clbits == circuit2.num_clbits
        assert circuit1.depth() == circuit2.depth()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])