# Remote Sensing - Solar and Telluric Spectra Analysis

This project is part of the **MVA P2 Remote Sensing** course. It focuses on the analysis and visualization of solar spectra and telluric (atmospheric) absorption using high-resolution data from the Institute for Astrophysics Göttingen (IAG).

## Project Overview

The core of the project is the study of the Sun's atmosphere and the impact of the Earth's atmosphere on astronomical observations. It provides tools to:
- Load and process FITS data from the IAG Solar Atlas.
- Analyze telluric spectra grouped by atmospheric conditions (Airmass, H2O column density).
- Visualize absorption lines and convert wavenumbers to wavelengths.
- Render visible spectra with realistic color mapping.

## Features

- **IAG Data Interface**: A specialized `open_iag` class to handle FITS files, headers, and spectral data.
- **Spectral Scaling**: Options to scale telluric spectra by airmass and precipitable water vapor (tau) to align absorption lines.
- **Solar Atlas Visualization**: Plotting the IAG Solar Atlas with error bars and data flags.
- **Visible Spectrum Rendering**: Functionality to convert infrared/visible wavenumbers to wavelengths (nm) and display them as absorption spectra with a rainbow gradient.

## Project Structure

```text
├── data/
│   └── IAG/                # FITS data files (Solar Atlas & Telluric groups)
├── scripts/
│   └── IAG_open_data.py    # Core data loading and plotting utilities
├── sun_atmosphere.ipynb    # Main research notebook with analysis and visualizations
├── main.py                 # Entry point (template)
└── pyproject.toml          # Project dependencies and metadata
```

## Setup and Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package and environment management.

### Prerequisites
- Python >= 3.8
- [uv](https://docs.astral-sh.com/uv/getting-started/installation/) installed on your system.

### Environment Setup
To set up the project environment and install all dependencies:

```bash
# Create a virtual environment and install dependencies from pyproject.toml
uv venv
uv sync
```

### Dependency Management
To add a new dependency:
```bash
uv add <package_name>
```

## Usage

The primary way to interact with the project is through the `sun_atmosphere.ipynb` notebook. If you are using `uv`, you can run the notebook in the managed environment:

```bash
# Run Jupyter within the uv environment
uv run jupyter notebook
```

Or use the `open_iag` class in your own scripts:

```python
from scripts.IAG_open_data import open_iag

# Load data for group 0
ot = open_iag(0)

# Plot telluric spectra scaled by H2O
ot.plot_tel(scale='H2O')

# Plot solar atlas
ot.plot_stel()
```

You can run your scripts directly using `uv run`:
```bash
uv run python your_script.py
```

## Data Source
The data used in this project is based on the [IAG Solar Atlas](https://zenodo.org/records/3598136).
