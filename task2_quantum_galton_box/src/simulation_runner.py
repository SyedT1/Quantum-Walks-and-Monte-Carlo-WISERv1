"""
Simulation Runner for Multi-Layer Quantum Galton Box
Handles batch simulations and comparative analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from quantum_galton_box import QuantumGaltonBox
from qiskit_aer import AerSimulator
import time
from scipy import stats

try:
    import seaborn as sns
    sns.set_style('darkgrid')
except ImportError:
    print("Warning: seaborn not installed, using default matplotlib styles")
    pass


class GaltonBoxSimulator:
    """
    Manages simulations of Quantum Galton Box across multiple configurations.
    """
    
    def __init__(self, layer_range: List[int] = None, shots: int = 10000):
        """
        Initialize the simulator.
        
        Args:
            layer_range: List of layer counts to simulate
            shots: Number of measurement shots per simulation
        """
        self.layer_range = layer_range or list(range(1, 11))
        self.shots = shots
        self.results = {}
        self.backend = AerSimulator(method='statevector')
        
    def run_all_simulations(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run simulations for all specified layer counts.
        
        Args:
            verbose: Print progress information
            
        Returns:
            DataFrame with simulation results
        """
        simulation_data = []
        
        for n_layers in self.layer_range:
            if verbose:
                print(f"Simulating {n_layers}-layer Quantum Galton Box...")
            
            start_time = time.time()
            
            # Create and simulate the circuit
            qgb = QuantumGaltonBox(n_layers)
            circuit = qgb.generate_optimized_circuit()
            
            # Run simulation
            sim_results = qgb.simulate(shots=self.shots, backend=self.backend)
            
            # Verify Gaussian distribution
            verification = qgb.verify_gaussian(plot=False)
            
            # Get circuit metrics
            metrics = qgb.get_circuit_metrics()
            
            elapsed_time = time.time() - start_time
            
            # Store results
            self.results[n_layers] = {
                'qgb': qgb,
                'simulation': sim_results,
                'verification': verification,
                'metrics': metrics
            }
            
            # Collect data for DataFrame
            simulation_data.append({
                'layers': n_layers,
                'mean': sim_results['mean'],
                'std': sim_results['std'],
                'variance': sim_results['variance'],
                'theoretical_mean': verification['theoretical_mean'],
                'theoretical_std': verification['theoretical_std'],
                'ks_pvalue': verification['ks_pvalue'],
                'is_gaussian': verification['is_gaussian'],
                'num_qubits': metrics['num_qubits'],
                'circuit_depth': metrics['circuit_depth'],
                'total_gates': metrics['total_gates'],
                'simulation_time': elapsed_time
            })
            
            if verbose:
                print(f"  Mean: {sim_results['mean']:.3f}, Std: {sim_results['std']:.3f}")
                print(f"  KS p-value: {verification['ks_pvalue']:.4f}")
                print(f"  Circuit depth: {metrics['circuit_depth']}, Gates: {metrics['total_gates']}")
                print(f"  Time: {elapsed_time:.2f}s\n")
        
        return pd.DataFrame(simulation_data)
    
    def plot_scaling_analysis(self, df: pd.DataFrame) -> None:
        """
        Create plots showing how various metrics scale with layer count.
        
        Args:
            df: DataFrame with simulation results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Mean convergence
        ax = axes[0, 0]
        ax.plot(df['layers'], df['mean'], 'bo-', label='Simulated')
        ax.plot(df['layers'], df['theoretical_mean'], 'r--', label='Theoretical')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Mean Position')
        ax.set_title('Mean Position Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Standard deviation scaling
        ax = axes[0, 1]
        ax.plot(df['layers'], df['std'], 'bo-', label='Simulated')
        ax.plot(df['layers'], df['theoretical_std'], 'r--', label='Theoretical')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Standard Deviation Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Gaussian test p-values
        ax = axes[0, 2]
        ax.plot(df['layers'], df['ks_pvalue'], 'go-')
        ax.axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('KS Test p-value')
        ax.set_title('Gaussian Distribution Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Circuit depth scaling
        ax = axes[1, 0]
        ax.plot(df['layers'], df['circuit_depth'], 'mo-')
        # Fit polynomial
        z = np.polyfit(df['layers'], df['circuit_depth'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(df['layers'].min(), df['layers'].max(), 100)
        ax.plot(x_smooth, p(x_smooth), 'r--', 
                label=f'Fit: {z[0]:.2f}n² + {z[1]:.2f}n + {z[2]:.2f}')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Circuit Depth')
        ax.set_title('Circuit Depth Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Gate count scaling
        ax = axes[1, 1]
        ax.plot(df['layers'], df['total_gates'], 'co-')
        # Theoretical O(n²) scaling
        theoretical_gates = 2 * df['layers']**2 + 5 * df['layers'] + 2
        ax.plot(df['layers'], theoretical_gates, 'r--', label='Theoretical O(n²)')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Total Gate Count')
        ax.set_title('Gate Count Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Simulation time
        ax = axes[1, 2]
        ax.plot(df['layers'], df['simulation_time'], 'yo-')
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Simulation Time (s)')
        ax.set_title('Computational Time Scaling')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Quantum Galton Box Scaling Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('../results/scaling_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close instead of show
    
    def plot_distribution_comparison(self, layers_to_plot: List[int] = None) -> None:
        """
        Plot distributions for multiple layer counts on same figure.
        
        Args:
            layers_to_plot: Specific layer counts to plot
        """
        if layers_to_plot is None:
            layers_to_plot = [3, 5, 7, 10] if 10 in self.layer_range else self.layer_range[:4]
        
        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(4*len(layers_to_plot), 4))
        
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for idx, n_layers in enumerate(layers_to_plot):
            if n_layers not in self.results:
                continue
                
            ax = axes[idx]
            positions = self.results[n_layers]['simulation']['positions']
            verification = self.results[n_layers]['verification']
            
            # Histogram
            n_bins = min(20, n_layers + 1)
            counts, bins, _ = ax.hist(positions, bins=n_bins, density=True,
                                     alpha=0.7, color='blue', edgecolor='black')
            
            # Theoretical Gaussian
            x = np.linspace(positions.min(), positions.max(), 100)
            theoretical = stats.norm.pdf(x, verification['theoretical_mean'],
                                        verification['theoretical_std'])
            ax.plot(x, theoretical, 'r-', linewidth=2)
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Probability Density' if idx == 0 else '')
            ax.set_title(f'{n_layers} Layers')
            ax.grid(True, alpha=0.3)
            
            # Add p-value annotation
            ax.text(0.95, 0.95, f"p={verification['ks_pvalue']:.3f}",
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Distribution Comparison Across Different Layer Counts', fontsize=14)
        plt.tight_layout()
        plt.savefig('../results/distribution_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close instead of show
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text report of the simulation results.
        
        Args:
            df: DataFrame with simulation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("QUANTUM GALTON BOX SIMULATION REPORT")
        report.append("=" * 70)
        report.append(f"\nSimulation Parameters:")
        report.append(f"  - Layer range: {self.layer_range[0]} to {self.layer_range[-1]}")
        report.append(f"  - Shots per simulation: {self.shots}")
        report.append(f"  - Total simulations: {len(df)}")
        
        report.append(f"\n{'='*70}")
        report.append("GAUSSIAN DISTRIBUTION VERIFICATION")
        report.append("=" * 70)
        
        # Check which simulations passed Gaussian test
        passed = df[df['is_gaussian']]
        failed = df[~df['is_gaussian']]
        
        report.append(f"\nPassed Gaussian Test (p > 0.05): {len(passed)}/{len(df)}")
        if len(failed) > 0:
            report.append(f"Failed for layers: {list(failed['layers'].values)}")
        
        # Statistical accuracy
        report.append(f"\n{'='*70}")
        report.append("STATISTICAL ACCURACY")
        report.append("=" * 70)
        
        mean_error = np.abs(df['mean'] - df['theoretical_mean']).mean()
        std_error = np.abs(df['std'] - df['theoretical_std']).mean()
        
        report.append(f"\nAverage Mean Error: {mean_error:.4f}")
        report.append(f"Average Std Error: {std_error:.4f}")
        
        # Circuit complexity
        report.append(f"\n{'='*70}")
        report.append("CIRCUIT COMPLEXITY SCALING")
        report.append("=" * 70)
        
        # Fit polynomial to gate count
        z = np.polyfit(df['layers'], df['total_gates'], 2)
        report.append(f"\nGate Count Scaling: {z[0]:.2f}n² + {z[1]:.2f}n + {z[2]:.2f}")
        report.append(f"Theoretical Upper Bound: 2n² + 5n + 2")
        
        # Fit polynomial to circuit depth
        z_depth = np.polyfit(df['layers'], df['circuit_depth'], 2)
        report.append(f"\nCircuit Depth Scaling: {z_depth[0]:.2f}n² + {z_depth[1]:.2f}n + {z_depth[2]:.2f}")
        
        # Performance
        report.append(f"\n{'='*70}")
        report.append("PERFORMANCE METRICS")
        report.append("=" * 70)
        
        report.append(f"\nTotal Simulation Time: {df['simulation_time'].sum():.2f} seconds")
        report.append(f"Average Time per Layer: {df['simulation_time'].mean():.2f} seconds")
        
        # Detailed results table
        report.append(f"\n{'='*70}")
        report.append("DETAILED RESULTS BY LAYER")
        report.append("=" * 70)
        report.append("\n" + df.to_string(index=False))
        
        report_text = "\n".join(report)
        
        # Save to file
        with open('../results/simulation_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def export_results(self, filename: str = 'simulation_results.csv') -> None:
        """
        Export simulation results to CSV file.
        
        Args:
            filename: Output filename
        """
        df = self.run_all_simulations(verbose=False)
        df.to_csv(f'../results/{filename}', index=False)
        print(f"Results exported to ../results/{filename}")


if __name__ == "__main__":
    """Run simulation when executed directly."""
    import os
    
    # Create results directory if needed
    os.makedirs('../results', exist_ok=True)
    
    print("="*70)
    print("QUANTUM GALTON BOX SIMULATION RUNNER")
    print("="*70)
    print("\nRunning simulations for layers 1-7...")
    
    # Create simulator for layers 1-7
    simulator = GaltonBoxSimulator(layer_range=list(range(1, 8)), shots=5000)
    
    # Run simulations
    df = simulator.run_all_simulations(verbose=True)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    simulator.plot_scaling_analysis(df)
    simulator.plot_distribution_comparison([2, 4, 6])
    
    # Generate report
    print("\nGenerating report...")
    report = simulator.generate_report(df)
    print("\nSimulation complete! Check the 'results' folder for outputs.")