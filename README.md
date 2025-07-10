# AstroCa

3D+time fluorescence image astrocyte segmentation.

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

If not installed, install it with:
```bash
curl -sSL https://install.python-poetry.org | python3 -
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

[files]
save_results = <1 to save intermediate results, 0 otherwise>

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
threshold_median_localized = 4.0
threshold_distance_localized = 6.0
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
If not installed:
```bash
sudo apt install doxygen
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

