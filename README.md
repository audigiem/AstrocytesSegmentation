# AstroCa

**3D+time fluorescence image astrocyte segmentation with feature extraction**

A comprehensive Python pipeline for analyzing calcium signaling in astrocytes from 3D+time fluorescence microscopy data. This tool provides automated segmentation, event detection, and feature extraction capabilities for studying astrocyte dynamics.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

- **3D+Time Image Processing**: Advanced processing of fluorescence microscopy time series
- **Automated Segmentation**: Intelligent astrocyte boundary detection and segmentation
- **Event Detection**: Calcium event identification and characterization
- **Feature Extraction**: Comprehensive spatial and temporal feature analysis
- **Hot Spot Analysis**: Detection and analysis of recurring activity patterns
- **Coactive Event Detection**: Identification of synchronized calcium events
- **Visualization Tools**: Built-in visualization for results validation
- **Performance Profiling**: Memory and time profiling capabilities

## Installation

### Prerequisites

- Python 3.10+
- Poetry (Python dependency manager)
- Doxygen (for documentation generation, optional)

### Clone Repository

```bash
git clone git@github.com:audigiem/AstrocytesSegmentation.git
cd AstrocytesSegmentation
```

### Install Poetry
Linux/WSL:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

macOS:
```bash
brew install poetry
```
### Install Dependencies
```bash
make install
# or manually:
poetry install
```

## Quick Start

1. Configure parameters in `config.ini` (see [Configuration](#configuration))
2. Run the pipeline:
   ```bash
   make run
   ```
3. View results in the specified output directory

## Usage
### Using Make (Recommended)
The project includes a comprehensive Makefile for easy operation:
```bash
# Show all available commands
make help

# Basic operations
make run                # Run main processing pipeline
make run-stats          # Run with performance statistics
make visualize          # Launch visualization tool
make test               # Run component tests

# Documentation
make doc                # Generate and open documentation
make doc-latex          # Generate LaTeX documentation

# Development
make lint               # Check code style
make format             # Format code automatically
make profile            # Run with line-by-line profiling

# Maintenance
make clean              # Remove all generated files
make clean-docs         # Remove documentation only
make clean-runs         # Remove run outputs only
```


### Direct Poetry Commands
```bash
# Activate environment
poetry shell

# Run main pipeline
poetry run python tests/main.py

# With options
poetry run python tests/main.py --stats
poetry run python tests/main.py --memstats

# Visualization
poetry run python tests/visualizationSegmentation.py

# Profiling (add @profile decorator to target functions)
poetry run kernprof -l -v tests/main.py

```

### Available Options
- `--stats`: Enable performance statistics collection
- `--memstats`: Enable memory profiling ⚠️ Significantly increases computation time

## Configuration
The main configuration is stored in `config.ini`. Key sections include:
### Input/Output Paths
```ini
[paths]
input_folder = path/to/your/tif/files
output_dir = path/to/output/directory
```
### Processing Parameters
```ini
[background_estimation]
acquisition_frequency = 4
amplification_factor = 2.0
method = percentile
percentile = 10.0

[active_voxels]
threshold_zscore = 2.8
radius_closing_morphology = 1
border_condition = # classic scipy border condition (reflect, wrap, etc.) and 'ignore' which reshape 
# the 4D array to 3D by joining frames (t, zmax, y, x) and (t+1, zmin, y, x) frames and then ignore border
median_size = 1.5

[events_extraction]
threshold_size_3d = 400
threshold_corr = 0.6

[features_extraction]
voxel_size_x = 0.1025
voxel_size_y = 0.1025
voxel_size_z = 0.1344
threshold_hot_spots = 0.5
```
### Save Options
```ini
[save]
save_boundaries = 1
save_background_estimation = 1
save_events = 1
save_features = 1
# ... other save options
```

For complete configuration details, see the example `config.ini` file.

## Documentation
### Generate Documentatio
```bash
make doc
# or manually:
doxygen Doxyfile
```


### View Documentation
- HTML: Open `docs/html/index.html` in your browser
- LaTeX: Generated in `docs/latex/` directory 

### Install Doxygen (if needed)
Linux/WSL:
```bash
sudo apt install doxygen
```
macOS:
```bash
brew install doxygen
```


## Development
### Code Quality
```bash
# Check code style
make lint

# Auto-format code
make format

# Run tests
make test
```
### Profiling

For performance analysis:
```bash
# Time profiling
make profile

# Memory profiling (warning: very slow)
make run-memstats
```

Add `@profile` decorator above functions you want to analyze (except `@njit/@jit` decorated functions).


### Environment Management
```bash
# Show environment info
make env-info

# Update dependencies
make deps-update
```

## Project Structure

```plaintext
AstrocytesSegmentation/
├── astroca/                    # Main processing modules
│   ├── activeVoxels/          # Active voxel detection
│   ├── croppingBoundaries/    # Boundary computation and cropping
│   ├── dynamicImage/          # Background estimation
│   ├── events/                # Event detection algorithms
│   ├── features/              # Feature extraction and analysis
│   ├── parametersNoise/       # Noise parameter estimation
│   ├── tools/                 # Utility functions
│   └── varianceStabilization/ # Variance stabilization
├── tests/                     # Test scripts and validation
│   ├── componentTest/         # Unit tests for modules
│   ├── comparingTools/        # Result comparison utilities
│   ├── main.py               # Main execution script
│   └── visualizationSegmentation.py # Visualization tool
├── docs/                      # Generated documentation
├── config.ini                # Main configuration file
├── Makefile                  # Build and execution commands
├── pyproject.toml            # Poetry project configuration
└── README.md                 # This file
```

### Key Modules
- **activeVoxels**: Z-score based detection of active voxels with morphological operations
- **events**: Connected component analysis for calcium event detection
- **features**: Spatial and temporal feature extraction, hot spot analysis
- **dynamicImage**: Background estimation using percentile methods
- **tools**: Data loading, export utilities, and logging


## Contributing
1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request


### Development Workflow
```bash
# Set up development environment
make install

# Run tests before committing
make test
make lint

# Format code
make format

# Update documentation
make doc
```

### License
This project is part of an ongoing internship at INRIA. Please contact the authors for usage permissions.


### Contact
For questions or collaboration opportunities, please open an issue or contact the development team.

---

**Quick Command Reference:**
- `make help` - Show all available commands
- `make run` - Run the main pipeline
- `make doc` - Generate documentation
- `make test` - Run tests
- `make clean` - Clean up generated files
