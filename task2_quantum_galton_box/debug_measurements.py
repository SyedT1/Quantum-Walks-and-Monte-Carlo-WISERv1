"""
Debug the measurement results to understand the encoding
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_galton_box import QuantumGaltonBox
import numpy as np

def analyze_raw_measurements():
    """Analyze raw measurement bitstrings."""
    print("ANALYZING RAW MEASUREMENT RESULTS")
    print("="*50)
    
    for n_layers in [2, 3]:
        print(f"\n{n_layers}-layer Galton Box:")
        print("-" * 30)
        
        qgb = QuantumGaltonBox(n_layers)
        circuit = qgb.generate_optimized_circuit()
        
        # Run simulation and get raw counts
        results = qgb.simulate(shots=1000)
        raw_counts = results['counts']
        
        print("Raw measurement results:")
        total_shots = sum(raw_counts.values())
        
        for bitstring, count in sorted(raw_counts.items()):
            prob = count / total_shots
            ones_count = bitstring.count('1')
            ones_positions = [i for i, bit in enumerate(bitstring[::-1]) if bit == '1']
            
            print(f"  {bitstring}: {count:3d} ({prob:.3f}) - {ones_count} ones at positions {ones_positions}")
        
        # Check how many results have exactly one '1'
        valid_results = 0
        invalid_results = 0
        
        for bitstring, count in raw_counts.items():
            ones_count = bitstring.count('1')
            if ones_count == 1:
                valid_results += count
            else:
                invalid_results += count
        
        print(f"  Valid (exactly 1 bit set): {valid_results} ({valid_results/total_shots:.1%})")
        print(f"  Invalid (!=1 bits set): {invalid_results} ({invalid_results/total_shots:.1%})")
        
        # Show position extraction
        print("  Position extraction:")
        for bitstring, count in sorted(raw_counts.items()):
            if count > 50:  # Only show frequent results
                pos = bitstring[::-1].find('1')
                print(f"    {bitstring} -> position {pos}")

def test_measurement_interpretation():
    """Test different ways of interpreting measurements."""
    print("\n" + "="*50)
    print("TESTING MEASUREMENT INTERPRETATION")
    print("="*50)
    
    # Test with 3 layers
    qgb = QuantumGaltonBox(3)
    circuit = qgb.generate_optimized_circuit()
    results = qgb.simulate(shots=5000)
    raw_counts = results['counts']
    
    print("Different interpretation methods:")
    
    # Method 1: Find first '1' (current method)
    positions_method1 = []
    for bitstring, count in raw_counts.items():
        pos = bitstring[::-1].find('1')
        if pos != -1:
            positions_method1.extend([pos] * count)
    
    # Method 2: Binary to decimal conversion
    positions_method2 = []
    for bitstring, count in raw_counts.items():
        decimal_val = int(bitstring, 2)
        positions_method2.extend([decimal_val] * count)
    
    # Method 3: Position of rightmost '1'
    positions_method3 = []
    for bitstring, count in raw_counts.items():
        for i in range(len(bitstring)):
            if bitstring[-(i+1)] == '1':
                positions_method3.extend([i] * count)
                break
    
    print(f"Method 1 (find first '1'):")
    unique1, counts1 = np.unique(positions_method1, return_counts=True)
    for pos, cnt in zip(unique1, counts1):
        print(f"  Position {pos}: {cnt} ({cnt/len(positions_method1):.3f})")
    print(f"  Mean: {np.mean(positions_method1):.3f}")
    
    print(f"\nMethod 2 (binary to decimal):")
    unique2, counts2 = np.unique(positions_method2, return_counts=True)
    for pos, cnt in zip(unique2, counts2):
        print(f"  Position {pos}: {cnt} ({cnt/len(positions_method2):.3f})")
    print(f"  Mean: {np.mean(positions_method2):.3f}")
    
    print(f"\nMethod 3 (rightmost '1'):")
    unique3, counts3 = np.unique(positions_method3, return_counts=True)
    for pos, cnt in zip(unique3, counts3):
        print(f"  Position {pos}: {cnt} ({cnt/len(positions_method3):.3f})")
    print(f"  Mean: {np.mean(positions_method3):.3f}")

if __name__ == "__main__":
    analyze_raw_measurements()
    test_measurement_interpretation()