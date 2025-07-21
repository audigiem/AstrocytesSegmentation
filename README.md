# AstroCa

3D+time fluorescence image astrocyte segmentation.

## Architecture

The project structure is organized as follows:
- `astroca/`: Contains the main processing modules for analyzing astrocyte signals.
- `config.ini`: Configuration file for setting parameters for the main script.
- `docs/`: Documentation files.
- `Doxyfile`: Configuration file for Doxygen to generate documentation.
- `poetry.lock`: Lock file for Poetry dependencies.
- `pyproject.toml`: Poetry project file defining dependencies and project metadata.
- `README.md`: This file, providing an overview of the project.
- `tests/`: Contains test scripts and component tests for verifying the functionality of the modules.
```
.
├── ./astroca
│   ├── ./astroca/activeVoxels
│   │   ├── ./astroca/activeVoxels/activeVoxelsFinder.py
│   │   ├── ./astroca/activeVoxels/medianFilter.py
│   │   ├── ./astroca/activeVoxels/spaceMorphology.py
│   │   └── ./astroca/activeVoxels/zScore.py
│   ├── ./astroca/croppingBoundaries
│   │   ├── ./astroca/croppingBoundaries/computeBoundaries.py
│   │   ├── ./astroca/croppingBoundaries/cropper.py
│   ├── ./astroca/dynamicImage
│   │   ├── ./astroca/dynamicImage/backgroundEstimator.py
│   │   ├── ./astroca/dynamicImage/dynamicImage.py
│   ├── ./astroca/events
│   │   ├── ./astroca/events/eventDetectorCorrected.py
│   │   └── ./astroca/events/tools.py
│   ├── ./astroca/features
│   │   ├── ./astroca/features/featuresComputation.py
│   ├── ./astroca/parametersNoise
│   │   ├── ./astroca/parametersNoise/parametersNoise.py
│   ├── ./astroca/tools
│   │   ├── ./astroca/tools/converter.py
│   │   ├── ./astroca/tools/exportData.py
│   │   ├── ./astroca/tools/loadData.py
│   └── ./astroca/varianceStabilization
│       └── ./astroca/varianceStabilization/varianceStabilization.py
├── ./config.ini
├── ./docs
├── ./Doxyfile
├── ./poetry.lock
├── ./pyproject.toml
├── ./README.md
└── ./tests
    ├── ./tests/comparingTools
    │   ├── ./tests/comparingTools/compareCSVFiles.py
    │   ├── ./tests/comparingTools/compareFileHist.py
    │   ├── ./tests/comparingTools/compareFiles.py
    │   ├── ./tests/comparingTools/comparetextFile.py
    ├── ./tests/componentTest
    │   ├── ./tests/componentTest/eventDetectionTest.py
    │   ├── ./tests/componentTest/featuresTest.py
    │   ├── ./tests/componentTest/medianFilterTest.py
    │   ├── ./tests/componentTest/morphologyClosing.py
    └── ./tests/main.py
```

## Installation

Clone the repository:
```bash
git clone https://gitlab.inria.fr/anbadoua/analyzeastrocasignals.git
cd analyzeastrocasignals
```

Install dependencies and create a Poetry virtual environment:
First check if Poetry is installed:
```bash
poetry --version
```

If not installed, install it with (Linux)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
(Mac)
```bash
pip3 install poetry
```

Then, install the dependencies using Poetry:
```bash
poetry install
```

## Usage

Activate the Poetry environment:
```bash
poetry env list
poetry env activate
```

### Global parameters
(optional) Configure the main script parameters in `config.ini` 
```text
[paths]
input_folder = <path to folder containing .tif series or single .tif file>
output_dir = <directory where intermediate results will be saved (ignored if save_results=0)>

[save]
save_cropp_boundaries = 0
save_boundaries = 1
save_variance_stabilization = 0
save_background_estimation = 1
save_df = 0
save_av = 0
save_events = 1
save_anscombe_inverse = 0
save_amplitude = 1
save_features = 1

[preprocessing]
pixel_cropped = 10
x_min = 0
x_max = 319

[background_estimation]
moving_window = 7
method = percentile
method2 = Med
percentile = 10.0

[active_voxels]
radius_closing_morphology = 1
median_size = 1.5

border_condition = <classic scipy border condition, e.g. reflect', 'constant', ... or 'ignore' to mimic the behavior of the original code>
// If 'ignore', all the 3D volumes are stacked together and processed as a single 3D volume. On the border, the values considered to apply
// the morphological closing or the median filter are only the ones from the single 3D volume (no padding).

threshold_zscore = 2.8

[events_extraction]
threshold_size_3d = 400
threshold_size_3d_removed = 20
threshold_corr = 0.6

[features_extraction]
voxel_size_x = 0.1025
voxel_size_y = 0.1025
voxel_size_z = 0.1344
threshold_median_localized = 0.5
volume_localized = 0.0434
```


Run the main processing pipeline:
```bash
chmod +x tests/main.py
poetry run ./tests/main.py
```

Available options for time profiling:
```bash
poetry run ./tests/main.py --stats
```

Silent execution option:
```bash
poetry run ./tests/main.py --quiet
```

Memory and time profiling options:
```bash
poetry run ./tests/main.py --memstats
```
NOTE: Memory profiling significantly increases computation time.

## Documentation

### Check if Doxygen is installed

```bash
doxygen --version
```
If not installed (Linux)
```bash
sudo apt install doxygen
```
(Mac)
```bash
brew install doxygen
```

### Generate documentation

```bash
doxygen Doxyfile
```

### View documentation

Open `docs/html/index.html` in a web browser to view the generated documentation.
LaTeX documentation is available in `docs/latex/`.
To compile LaTeX documentation:
```bash
cd docs/latex
make all
```

