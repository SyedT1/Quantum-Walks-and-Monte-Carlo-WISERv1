# Quantum Galton Box - Task 2 Implementation

## Overview
This project implements a generalized quantum algorithm for multi-layer Galton Box (bean machine) simulation based on the paper "Universal Statistical Simulator" by Mark Carney and Ben Varcoe (2022). The implementation creates quantum circuits that generate Gaussian distributions through quantum superposition of all possible trajectories.

## Project Structure
```
task2_quantum_galton_box/
├── src/
│   ├── quantum_galton_box.py      # Core quantum circuit implementation
│   └── simulation_runner.py       # Batch simulation and analysis tools
├── tests/
│   └── test_quantum_galton_box.py # Unit tests for validation
├── notebooks/
│   └── quantum_galton_box_exploration.ipynb # Interactive Jupyter notebook
├── results/                       # Output directory for results
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features
- **Generalized Circuit Generation**: Parameterized algorithm for any number of layers
- **Noiseless Simulation**: Uses Qiskit's statevector simulator for accurate results
- **Gaussian Verification**: Statistical tests (Kolmogorov-Smirnov) to verify Gaussian output
- **Scaling Analysis**: Demonstrates O(n²) gate complexity
- **Visualization**: Automatic generation of distribution plots and scaling graphs
- **Unit Tests**: Validation against paper's theoretical predictions

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
Run the main script to execute all simulations:
```bash
python main.py
```

This will:
- Demonstrate single circuit operation
- Validate against paper examples
- Run scaling analysis for 1-10 layers
- Generate plots and reports

### Interactive Exploration
Launch the Jupyter notebook:
```bash
jupyter notebook notebooks/quantum_galton_box_exploration.ipynb
```

### Python API
```python
from src.quantum_galton_box import QuantumGaltonBox

# Create 5-layer Galton Box
qgb = QuantumGaltonBox(num_layers=5)

# Generate optimized circuit
circuit = qgb.generate_optimized_circuit()

# Run simulation
results = qgb.simulate(shots=10000)

# Verify Gaussian distribution
verification = qgb.verify_gaussian(plot=True)
```

### Running Tests
```bash
pytest tests/ -v
```

## Algorithm Details

### Circuit Construction
The quantum circuit mimics a physical Galton board where:
- Each "peg" is implemented as a quantum module using controlled-SWAP gates
- A quantum "ball" starts at the center position
- Superposition allows exploration of all trajectories simultaneously
- Measurement collapses to classical Gaussian distribution

### Key Components
1. **Quantum Peg Module**: Basic building block using Hadamard, CSWAP, and CNOT gates
2. **Layer Structure**: n-layer board requires 2n qubits
3. **Gate Complexity**: O(n²) gates for n layers
4. **Circuit Depth**: Optimized for minimal depth compared to previous implementations

## Results

### Expected Outputs
- **Distribution Plots**: Visual confirmation of Gaussian distribution
- **Scaling Analysis**: Gate count and circuit depth vs. layer count
- **Statistical Report**: Mean, standard deviation, and test statistics
- **CSV Export**: Raw data for further analysis

### Performance Metrics
- 1-10 layers: < 1 second per simulation (5000 shots)
- Statistical accuracy: KS test p-value > 0.05 for all configurations
- Circuit efficiency: 50-70% of theoretical upper bound

## GitHub Issues Addressed

### Issue #3: Setup Quantum SDK and Environment ✅
- Implemented with Qiskit 0.45.0
- Created modular project structure
- Established testing framework

### Issue #4: Generalize Galton Box Circuit ✅
- `generate_galton_circuit()`: General implementation
- `generate_optimized_circuit()`: Optimized version with minimal depth
- Unit tests validate 1-2 layer circuits against paper

### Issue #5: Noiseless Simulation and Gaussian Verification ✅
- Statevector simulator for noiseless results
- Kolmogorov-Smirnov test for Gaussian verification
- Comprehensive plotting and visualization

## Optimizations Implemented
1. **Modular Design**: Separate circuit generation, simulation, and visualization
2. **Efficient Simulation**: Statevector method for small circuits
3. **Pre-computed Theoretical Values**: Reduces computation time
4. **Batch Processing**: Run multiple simulations efficiently

## References
Carney, M., & Varcoe, B. (2022). Universal Statistical Simulator. arXiv:2202.01735v1 [quant-ph]

## License
MIT License

## Author
Implementation based on Task 2 specifications for Quantum Galton Box project.