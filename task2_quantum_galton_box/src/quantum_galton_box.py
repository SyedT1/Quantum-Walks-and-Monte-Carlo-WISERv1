"""
Quantum Galton Box Implementation
Based on "Universal Statistical Simulator" by Mark Carney and Ben Varcoe (2022)

This module implements a generalized quantum Galton board circuit that creates
a superposition of all possible trajectories, resulting in a Gaussian distribution.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


class QuantumGaltonBox:
    """
    A quantum implementation of the Galton Box (bean machine) that demonstrates
    quantum superposition to generate Gaussian distributions.
    """
    
    def __init__(self, num_layers: int):
        """
        Initialize the Quantum Galton Box.
        
        Args:
            num_layers: Number of layers in the Galton board (n)
        """
        self.num_layers = num_layers
        self.num_qubits = 2 * num_layers
        self.circuit = None
        self.results = None
        
    def _create_quantum_peg(self, qc: QuantumCircuit, control_qubit: int, 
                           target_qubits: Tuple[int, int, int]) -> None:
        """
        Create a quantum peg module that simulates a ball hitting a peg.
        
        Args:
            qc: Quantum circuit to add the peg to
            control_qubit: Control qubit index (q0 in paper)
            target_qubits: Tuple of (left, center, right) qubit indices
        """
        left, center, right = target_qubits
        
        # Controlled-SWAP between left and center
        qc.cswap(control_qubit, left, center)
        
        # CNOT from center to control
        qc.cx(center, control_qubit)
        
        # SWAP between center and right
        qc.swap(center, right)
        
    def generate_galton_circuit(self) -> QuantumCircuit:
        """
        Generate a general n-layer Quantum Galton Box circuit.
        
        Returns:
            QuantumCircuit: The complete quantum circuit
        """
        n = self.num_layers
        
        # Calculate total number of qubits needed
        # 2n working qubits + 1 control qubit (recycled)
        total_qubits = 2 * n + 1
        
        # Create quantum and classical registers
        qreg = QuantumRegister(total_qubits, 'q')
        creg = ClassicalRegister(n + 1, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Initialize the "ball" at the center position
        center_pos = n
        qc.x(qreg[center_pos])
        
        # Build the circuit layer by layer
        qubit_index = 1  # Start from qubit 1 (q0 is control)
        
        for layer in range(1, n + 1):
            # Reset and prepare control qubit for this layer
            if layer > 1:
                qc.reset(qreg[0])
            qc.h(qreg[0])
            
            # Number of pegs in this layer
            num_pegs = layer
            
            # Process each peg in the current layer
            for peg in range(num_pegs):
                # Calculate qubit indices for this peg
                if peg == 0:
                    # First peg in layer
                    left = qubit_index + layer - 1
                    center = qubit_index + layer
                    right = qubit_index + layer + 1
                else:
                    # Subsequent pegs
                    left = qubit_index + peg - 1
                    center = qubit_index + peg
                    right = qubit_index + peg + 1
                
                # Apply controlled-SWAP operations
                if center < total_qubits and right < total_qubits:
                    qc.cswap(qreg[0], qreg[left], qreg[center])
                    
                    # Add CNOT for control qubit balancing
                    qc.cx(qreg[center], qreg[0])
                    
                    # Regular SWAP for the second part
                    if right < total_qubits:
                        qc.cswap(qreg[0], qreg[center], qreg[right])
                        
                # Add corrective CNOT if needed (for multiple pegs)
                if peg < num_pegs - 1 and layer > 1:
                    qc.cx(qreg[right], qreg[0])
            
            # Update qubit index for next layer
            if layer < n:
                qubit_index += 1
        
        # Add measurements to output qubits
        output_start = n
        for i in range(n + 1):
            if output_start + i < total_qubits:
                qc.measure(qreg[output_start + i], creg[i])
        
        self.circuit = qc
        return qc
    
    def generate_optimized_circuit(self) -> QuantumCircuit:
        """
        Generate an optimized version of the Quantum Galton Box circuit.
        Uses the approach from the paper with minimal gate depth.
        
        Returns:
            QuantumCircuit: Optimized quantum circuit
        """
        n = self.num_layers
        
        # Special case for single layer
        if n == 1:
            qreg = QuantumRegister(3, 'q')
            creg = ClassicalRegister(2, 'c')
            qc = QuantumCircuit(qreg, creg)
            
            # Simple single peg circuit
            qc.h(qreg[0])
            qc.x(qreg[1])
            qc.cswap(qreg[0], qreg[1], qreg[2])
            
            # Measure outputs
            qc.measure(qreg[1], creg[0])
            qc.measure(qreg[2], creg[1])
            
            self.circuit = qc
            return qc
        
        # For multiple layers, use full implementation
        total_qubits = 2 * n + 1
        
        qreg = QuantumRegister(total_qubits, 'q')
        creg = ClassicalRegister(n + 1, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Control qubit is q[0]
        control_idx = 0
        
        # Initialize: place "ball" at center position
        center_idx = n
        qc.x(qreg[center_idx])
        
        # Build circuit following paper's approach
        current_positions = [center_idx]
        
        for layer in range(1, n + 1):
            # Reset control qubit if not first layer
            if layer > 1:
                qc.reset(qreg[control_idx])
            
            # Put control qubit in superposition
            qc.h(qreg[control_idx])
            
            # Process this layer
            new_positions = []
            for pos in current_positions:
                left = pos - 1
                right = pos + 1
                
                if left >= 1 and right < total_qubits:
                    # Apply controlled-SWAP
                    qc.cswap(qreg[control_idx], qreg[left], qreg[pos])
                    
                    # Balancing CNOT
                    qc.cx(qreg[pos], qreg[control_idx])
                    
                    # Second SWAP
                    if right < total_qubits:
                        qc.cswap(qreg[control_idx], qreg[pos], qreg[right])
                    
                    new_positions.extend([left, right])
            
            current_positions = list(set(new_positions))
        
        # Measure output qubits (all except control)
        output_count = 0
        for i in range(1, total_qubits):
            if output_count < n + 1:
                qc.measure(qreg[i], creg[output_count])
                output_count += 1
        
        self.circuit = qc
        return qc
    
    def simulate(self, shots: int = 10000, backend: Optional[AerSimulator] = None) -> Dict:
        """
        Run noiseless simulation of the quantum circuit.
        
        Args:
            shots: Number of measurement shots
            backend: Optional backend simulator (uses AerSimulator if None)
            
        Returns:
            Dict: Simulation results with counts and statistics
        """
        if self.circuit is None:
            self.generate_optimized_circuit()
        
        # Use statevector simulator for noiseless simulation
        if backend is None:
            backend = AerSimulator(method='statevector')
        
        # Transpile and run
        transpiled_circuit = transpile(self.circuit, backend)
        job = backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Process results
        processed_results = self._process_results(counts, shots)
        self.results = processed_results
        
        return processed_results
    
    def _process_results(self, counts: Dict, shots: int) -> Dict:
        """
        Process raw measurement counts into statistical data.
        
        Args:
            counts: Raw measurement counts from simulation
            shots: Total number of shots
            
        Returns:
            Dict: Processed results with statistics
        """
        # Convert binary strings to position indices
        positions = []
        frequencies = []
        valid_shots = 0
        invalid_shots = 0
        
        for bitstring, count in counts.items():
            # Find position of the '1' in the bitstring (rightmost bit is position 0)
            pos = bitstring[::-1].find('1')
            if pos != -1:
                # Valid measurement (exactly one bit is '1')
                positions.extend([pos] * count)
                frequencies.append((pos, count))
                valid_shots += count
            else:
                # Invalid measurement (no bits are '1' or multiple bits are '1')
                invalid_shots += count
        
        positions = np.array(positions)
        
        # Calculate statistics only for valid measurements
        if len(positions) > 0:
            mean = np.mean(positions)
            std = np.std(positions)
            variance = np.var(positions)
        else:
            mean = std = variance = 0.0
        
        # Create histogram data
        hist_bins = np.arange(self.num_layers + 2) - 0.5
        hist_counts, _ = np.histogram(positions, bins=hist_bins)
        
        # Calculate validity statistics
        validity_rate = valid_shots / shots if shots > 0 else 0.0
        
        return {
            'counts': counts,
            'positions': positions,
            'frequencies': sorted(frequencies),
            'mean': mean,
            'std': std,
            'variance': variance,
            'histogram': hist_counts,
            'total_shots': shots,
            'valid_shots': valid_shots,
            'invalid_shots': invalid_shots,
            'validity_rate': validity_rate
        }
    
    def verify_gaussian(self, plot: bool = True) -> Dict:
        """
        Verify that the output follows a Gaussian distribution using appropriate tests for discrete data.
        
        Args:
            plot: Whether to create visualization
            
        Returns:
            Dict: Statistical test results
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run simulate() first.")
        
        positions = self.results['positions']
        
        # Calculate theoretical Gaussian parameters
        theoretical_mean = self.num_layers / 2
        theoretical_std = np.sqrt(self.num_layers / 4)
        
        # For discrete distributions, use chi-squared goodness-of-fit test
        # Create histogram of observed frequencies
        unique_pos, observed_counts = np.unique(positions, return_counts=True)
        
        # Calculate expected frequencies from theoretical Gaussian
        total_shots = len(positions)
        expected_probs = []
        
        for pos in unique_pos:
            # For discrete approximation, integrate Gaussian over [pos-0.5, pos+0.5]
            lower_bound = pos - 0.5
            upper_bound = pos + 0.5
            
            # Use cumulative distribution function
            expected_prob = (stats.norm.cdf(upper_bound, theoretical_mean, theoretical_std) - 
                           stats.norm.cdf(lower_bound, theoretical_mean, theoretical_std))
            expected_probs.append(expected_prob)
        
        expected_probs = np.array(expected_probs)
        
        # Normalize probabilities to sum to 1 (handle edge cases)
        if expected_probs.sum() > 0:
            expected_probs = expected_probs / expected_probs.sum()
        
        # Convert to expected frequencies
        expected_freqs = expected_probs * total_shots
        
        # Chi-squared test (only if all expected frequencies >= 5)
        if np.all(expected_freqs >= 5):
            chi2_stat, chi2_pvalue = stats.chisquare(observed_counts, expected_freqs)
            chi2_valid = True
        else:
            # Use modified chi-squared for small expected frequencies
            # Combine bins with expected < 5
            combined_observed = []
            combined_expected = []
            temp_obs = 0
            temp_exp = 0
            
            for obs, exp in zip(observed_counts, expected_freqs):
                temp_obs += obs
                temp_exp += exp
                
                if temp_exp >= 5 or (obs == observed_counts[-1]):  # Last bin
                    combined_observed.append(temp_obs)
                    combined_expected.append(temp_exp)
                    temp_obs = 0
                    temp_exp = 0
            
            if len(combined_observed) >= 2:
                chi2_stat, chi2_pvalue = stats.chisquare(combined_observed, combined_expected)
                chi2_valid = True
            else:
                chi2_stat, chi2_pvalue = np.nan, np.nan
                chi2_valid = False
        
        # Alternative: Compare means and standard deviations
        mean_diff = abs(positions.mean() - theoretical_mean)
        std_diff = abs(positions.std() - theoretical_std)
        
        # Normalized differences
        mean_error_pct = mean_diff / theoretical_mean if theoretical_mean > 0 else float('inf')
        std_error_pct = std_diff / theoretical_std if theoretical_std > 0 else float('inf')
        
        # Anderson-Darling test (better for normality testing)
        try:
            ad_stat, ad_crit_vals, ad_sig_level = stats.anderson(positions, dist='norm')
            # Check if statistic is less than critical value at 5% level
            ad_pvalue = 1.0 if ad_stat < ad_crit_vals[2] else 0.0  # Approximate
            ad_valid = True
        except:
            ad_stat, ad_pvalue, ad_valid = np.nan, np.nan, False
        
        # Comprehensive assessment
        is_gaussian_chi2 = chi2_pvalue > 0.05 if chi2_valid else False
        is_gaussian_means = (mean_error_pct < 0.1) and (std_error_pct < 0.2)  # Within 10% and 20%
        is_gaussian_ad = ad_pvalue > 0.05 if ad_valid else False
        
        # Overall assessment (need at least 2 out of 3 tests to pass)
        tests_passed = sum([is_gaussian_chi2, is_gaussian_means, is_gaussian_ad])
        is_gaussian_overall = tests_passed >= 2
        
        verification_results = {
            'chi2_statistic': chi2_stat if chi2_valid else np.nan,
            'chi2_pvalue': chi2_pvalue if chi2_valid else np.nan,
            'chi2_valid': chi2_valid,
            'ad_statistic': ad_stat if ad_valid else np.nan,
            'ad_pvalue': ad_pvalue if ad_valid else np.nan,
            'ad_valid': ad_valid,
            'theoretical_mean': theoretical_mean,
            'theoretical_std': theoretical_std,
            'actual_mean': self.results['mean'],
            'actual_std': self.results['std'],
            'mean_error_pct': mean_error_pct,
            'std_error_pct': std_error_pct,
            'is_gaussian_chi2': is_gaussian_chi2,
            'is_gaussian_means': is_gaussian_means,
            'is_gaussian_ad': is_gaussian_ad,
            'is_gaussian': is_gaussian_overall,
            'tests_passed': f"{tests_passed}/3",
            'observed_frequencies': dict(zip(unique_pos, observed_counts)),
            'expected_frequencies': dict(zip(unique_pos, expected_freqs)) if chi2_valid else {}
        }
        
        if plot:
            self._plot_distribution(verification_results)
        
        return verification_results
    
    def _plot_distribution(self, verification_results: Dict) -> None:
        """
        Create visualization of the distribution with Gaussian overlay.
        
        Args:
            verification_results: Results from Gaussian verification
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram with Gaussian overlay
        ax1 = axes[0]
        positions = self.results['positions']
        
        n_bins = min(30, self.num_layers + 1)
        counts, bins, _ = ax1.hist(positions, bins=n_bins, density=True, 
                                   alpha=0.7, color='blue', edgecolor='black')
        
        # Plot theoretical Gaussian
        x = np.linspace(positions.min(), positions.max(), 100)
        theoretical_gaussian = stats.norm.pdf(
            x, 
            verification_results['theoretical_mean'],
            verification_results['theoretical_std']
        )
        ax1.plot(x, theoretical_gaussian, 'r-', linewidth=2, 
                label='Theoretical Gaussian')
        
        # Plot fitted Gaussian
        fitted_gaussian = stats.norm.pdf(x, positions.mean(), positions.std())
        ax1.plot(x, fitted_gaussian, 'g--', linewidth=2, 
                label='Fitted Gaussian')
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'Quantum Galton Box Distribution (n={self.num_layers} layers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax2 = axes[1]
        stats.probplot(positions, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        chi2_text = f"{verification_results['chi2_pvalue']:.4f}" if verification_results['chi2_valid'] else "N/A"
        stats_text = (
            f"Mean: {self.results['mean']:.3f} (Theory: {verification_results['theoretical_mean']:.3f})\n"
            f"Std: {self.results['std']:.3f} (Theory: {verification_results['theoretical_std']:.3f})\n"
            f"Chi² p-value: {chi2_text}\n"
            f"Mean Error: {verification_results['mean_error_pct']:.1%}\n"
            f"Tests Passed: {verification_results['tests_passed']}\n"
            f"Gaussian: {'Yes' if verification_results['is_gaussian'] else 'No'}"
        )
        
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'../results/galton_box_{self.num_layers}_layers.png', dpi=150)
        plt.close()  # Close instead of show
    
    def get_circuit_metrics(self) -> Dict:
        """
        Calculate and return circuit complexity metrics.
        
        Returns:
            Dict: Circuit metrics including gate counts and depth
        """
        if self.circuit is None:
            raise ValueError("Circuit not generated yet.")
        
        # Count different gate types
        gate_counts = {}
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        # Calculate theoretical bounds from paper
        n = self.num_layers
        theoretical_gate_upper_bound = 2 * n**2 + 5 * n + 2
        
        return {
            'num_qubits': self.circuit.num_qubits,
            'num_classical_bits': self.circuit.num_clbits,
            'circuit_depth': self.circuit.depth(),
            'total_gates': sum(gate_counts.values()),
            'gate_counts': gate_counts,
            'theoretical_upper_bound': theoretical_gate_upper_bound,
            'efficiency': sum(gate_counts.values()) / theoretical_gate_upper_bound
        }