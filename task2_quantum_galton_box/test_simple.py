"""
Simple test script for Quantum Galton Box
Tests basic functionality without full analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import matplotlib.pyplot as plt

def test_basic_functionality():
    """Test basic Quantum Galton Box functionality."""
    
    print("="*60)
    print("QUANTUM GALTON BOX - SIMPLE TEST")
    print("="*60)
    
    # Test different layer counts
    for n_layers in [1, 2, 3, 4]:
        print(f"\nTesting {n_layers}-layer Galton Box:")
        print("-" * 40)
        
        try:
            # Create circuit
            qgb = QuantumGaltonBox(num_layers=n_layers)
            circuit = qgb.generate_optimized_circuit()
            
            # Get metrics
            metrics = qgb.get_circuit_metrics()
            print(f"  Circuit created successfully")
            print(f"  - Qubits: {metrics['num_qubits']}")
            print(f"  - Gates: {metrics['total_gates']}")
            print(f"  - Depth: {metrics['circuit_depth']}")
            
            # Run small simulation
            results = qgb.simulate(shots=1000)
            print(f"  Simulation completed")
            print(f"  - Mean: {results['mean']:.3f}")
            print(f"  - Std: {results['std']:.3f}")
            
            # Quick Gaussian check
            verification = qgb.verify_gaussian(plot=False)
            print(f"  - Gaussian test p-value: {verification['ks_pvalue']:.4f}")
            print(f"  - Is Gaussian: {verification['is_gaussian']}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run test
    test_basic_functionality()
    
    # Create a simple plot for one example
    print("\nCreating example plot for 4-layer circuit...")
    qgb = QuantumGaltonBox(num_layers=4)
    circuit = qgb.generate_optimized_circuit()
    results = qgb.simulate(shots=5000)
    verification = qgb.verify_gaussian(plot=True)
    
    print("\nDone! Check the results folder for the plot.")