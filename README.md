# Rockfall Risk Assessment for Road Networks

This project provides a comprehensive framework for assessing rockfall risk for road networks. It implements a semi-quantitative risk assessment methodology based on the fundamental risk equation: **Risk = Hazard × Vulnerability × Exposure**.

## Project Structure

```
rockfall-risk-roads/
│
├── config.ini                # Configuration settings
├── requirements.txt          # Required Python packages
│
├── data/                     # Data directory
│   ├── input/                # Input data files
│   └── output/               # Output data files
│
├── examples/                 # Example scripts
│   └── hazard_assessment_example.py
│
└── src/                      # Source code
    ├── __init__.py
    │
    ├── hazard/               # Hazard assessment module
    │   ├── __init__.py
    │   ├── hazard_assessment.py
    │   ├── hazard_classification.py
    │   ├── parameter_extraction.py
    │   ├── runout_analysis.py
    │   └── README.md
    │
    ├── vulnerability/        # Vulnerability assessment module
    │   └── vulnerability_assessment.py
    │
    ├── exposure/             # Exposure assessment module
    │   ├── intrinsic_value.py
    │   └── network_relevance.py
    │
    ├── risk/                 # Risk calculation module
    │   ├── risk_calculation.py
    │   └── risk_classification.py
    │
    ├── road/                 # Road network handling module
    │   ├── classification.py
    │   ├── network_analysis.py
    │   └── segmentation.py
    │
    └── utils/                # Utility functions
        ├── geo_utils.py
        ├── io_utils.py
        └── visualization.py
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rockfall-risk-roads.git
   cd rockfall-risk-roads
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Before running the assessment, you need to configure the input data paths and parameters in the `config.ini` file:

```ini
[INPUT_DATA]
# Data paths
road_network_file = /path/to/road_network.shp
runout_extent_raster = /path/to/runout_extent.tif
susceptibility_raster = /path/to/max_susceptibility.tif
energy_raster = /path/to/max_energy.tif
velocity_raster = /path/to/max_velocity.tif
```

### Running the Assessment

The project can be used either from the command line or as a Python module:

#### Command Line

Run the example hazard assessment script:

```
python examples/hazard_assessment_example.py
```

#### Python Module

```python
from src.hazard import HazardAssessment
from src.utils import io_utils

# Create hazard assessment from config file
hazard_assessment = HazardAssessment.from_config("config.ini", io_utils_module=io_utils)

# Load road network and perform assessment
road_segments = io_utils.read_vector("path/to/road_network.shp")
segments_with_hazard, segments_not_in_runout = hazard_assessment.assess_hazard(
    road_segments,
    simulate_parameters=False
)
```

## Input Data Requirements

### Road Network Data

- Vector data (line geometry) representing the road network
- Essential attributes: ID, type, purpose, state, width, roadbed, level
- Format: Shapefile or any GeoPandas-compatible vector format

### Rockfall Data

- **Runout Extent Raster**: Binary raster (1 = runout zone) defining the extent of potential rockfall runout
- **Susceptibility Raster**: Continuous values representing rockfall susceptibility
- **Velocity Raster**: Continuous values (m/s) representing rockfall velocity
- **Energy Raster**: Continuous values (kJ) representing rockfall energy

### Additional Data (Optional)

- **Slope Units**: Vector data (polygon geometry) representing geomorphological units
- **Source Areas**: Vector data (point/polygon geometry) representing rockfall initiation locations
- **Rockfall Clumps**: Vector data (polygon geometry) representing rockfall potential source areas

## Output

The assessment produces the following outputs:

1. **Hazard Assessment**: Road segments with hazard parameters and classifications
2. **Vulnerability Assessment**: Road segments with vulnerability scores and classifications
3. **Exposure Assessment**: Road segments with exposure scores and classifications
4. **Risk Assessment**: Comprehensive risk assessment combining all components
5. **Summary Statistics**: Statistics and visualizations of the risk assessment results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was developed based on research in rockfall risk assessment for road networks in the Province of Rome, Italy.
