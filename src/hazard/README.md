# Hazard Assessment Module

This module provides a comprehensive framework for assessing rockfall hazard by integrating runout analysis, parameter extraction, and hazard classification.

## Components

The hazard module consists of the following components:

### 1. Runout Analysis

The `RunoutAnalysis` class handles the conversion of runout rasters to vector polygons and the spatial analysis of runout zone intersection with road segments.

```python
from hazard.runout_analysis import RunoutAnalysis

# Initialize with a runout raster
runout_analysis = RunoutAnalysis(runout_raster, runout_value=1.0)

# Identify road segments intersecting the runout zone
runout_segments = runout_analysis.identify_intersecting_segments(road_segments)
```

### 2. Parameter Extraction

The `HazardParameterExtraction` class handles the extraction of various hazard parameters such as susceptibility, velocity, and energy from raster data for road segments.

```python
from hazard.parameter_extraction import HazardParameterExtraction

# Initialize with buffer distance
parameter_extraction = HazardParameterExtraction(buffer_distance=15.0)

# Add parameter rasters
parameter_extraction.add_parameter('susceptibility', susceptibility_raster)
parameter_extraction.add_parameter('velocity', velocity_raster)
parameter_extraction.add_parameter('energy', energy_raster)

# Extract parameters for road segments
segments_with_parameters = parameter_extraction.extract_parameters(road_segments)

# For testing purposes, simulate parameter values
simulated_segments = parameter_extraction.simulate_parameters(road_segments)
```

### 3. Hazard Classification

The `HazardClassification` class handles the classification of continuous hazard parameters into discrete hazard classes and calculates weighted hazard scores.

```python
from hazard.hazard_classification import HazardClassification

# Initialize with parameter weights and class thresholds
hazard_classification = HazardClassification(
    parameter_weights={'susceptibility': 0.4, 'velocity': 0.3, 'energy': 0.3},
    class_thresholds=None  # Use default thresholds
)

# Classify segments with parameters
classified_segments = hazard_classification.classify_dataframe(segments_with_parameters)

# Calculate hazard score for a specific set of parameters
hazard_score, hazard_class = hazard_classification.calculate_hazard_score({
    'susceptibility': 0.75,
    'velocity': 15.0,
    'energy': 250.0
})
```

### 4. Hazard Assessment

The `HazardAssessment` class integrates the above components to provide a comprehensive hazard assessment for road segments.

```python
from hazard.hazard_assessment import HazardAssessment

# Initialize with all components
hazard_assessment = HazardAssessment(
    runout_raster=runout_raster,
    runout_value=1.0,
    buffer_distance=15.0,
    parameter_weights={'susceptibility': 0.4, 'velocity': 0.3, 'energy': 0.3},
    class_thresholds=None  # Use default thresholds
)

# Add parameter rasters
hazard_assessment.add_parameter_raster('susceptibility', susceptibility_raster)
hazard_assessment.add_parameter_raster('velocity', velocity_raster)
hazard_assessment.add_parameter_raster('energy', energy_raster)

# Perform complete hazard assessment
segments_with_hazard, segments_not_in_runout = hazard_assessment.assess_hazard(
    road_segments,
    simulate_parameters=False
)
```

## Configuration

The hazard module can be configured using the `config.ini` file. Here's an example configuration:

```ini
[INPUT_DATA]
# Path to the road network data (vector, line geometry)
road_network_file = /path/to/road_network.shp

# Path to the rockfall runout model results (raster)
# Total runout extent (binary: 1 = within runout zone)
runout_extent_raster = /path/to/runout_extent.tif

# Maximum susceptibility (continuous values)
susceptibility_raster = /path/to/max_susceptibility.tif

# Energy (continuous values, kJ)
energy_raster = /path/to/max_energy.tif

# Velocity (continuous values, m/s)
velocity_raster = /path/to/max_velocity.tif

[HAZARD]
# Weights for hazard parameters
susceptibility_weight = 0.4
velocity_weight = 0.3
energy_weight = 0.3

# Thresholds for susceptibility classification
susceptibility_very_low_max = 0.2
susceptibility_low_max = 0.4
susceptibility_moderate_max = 0.6
susceptibility_high_max = 0.8

# Thresholds for velocity classification (m/s)
velocity_very_low_max = 4.0
velocity_low_max = 8.0
velocity_moderate_max = 12.0
velocity_high_max = 16.0

# Thresholds for energy classification (kJ)
energy_very_low_max = 100.0
energy_low_max = 200.0
energy_moderate_max = 300.0
energy_high_max = 400.0
```

## Usage

To use the hazard module in the rockfall risk assessment workflow:

```python
from src.hazard import HazardAssessment
from src.utils import io_utils

# Create hazard assessment from config file
hazard_assessment = HazardAssessment.from_config("config.ini", io_utils_module=io_utils)

# Or load data manually
from src.utils.io_utils import read_vector, read_raster

# Load data
road_segments = read_vector(config['INPUT_DATA']['road_network_file'])

# Perform hazard assessment
segments_with_hazard, segments_not_in_runout = hazard_assessment.assess_hazard(
    road_segments,
    simulate_parameters=False  # Use actual raster data
)
```

## Raster Data Requirements

The hazard assessment requires the following raster datasets:

1. **Runout Extent Raster**: A binary raster (1 = runout zone) defining the total extent of potential rockfall runout.

2. **Susceptibility Raster**: Continuous values representing the maximum rockfall susceptibility.

3. **Velocity Raster**: Continuous values (m/s) representing the maximum rockfall velocity.

4. **Energy Raster**: Continuous values (kJ) representing the maximum rockfall energy.

All rasters should:
- Have the same spatial resolution and extent
- Be properly georeferenced (preferably in UTM Zone 33N, EPSG:32633)
- Have NoData values defined for areas outside the study area

## Outputs

The hazard assessment produces the following outputs:

1. **Runout Segments**: Road segments intersecting the rockfall runout zone.
2. **Hazard Parameters**: Extracted or simulated hazard parameters for each segment.
3. **Hazard Classification**: Classified hazard parameters and overall hazard scores and classes.

These outputs are used in the subsequent stages of the risk assessment workflow, particularly in the risk calculation module where they are combined with vulnerability and exposure assessments.
