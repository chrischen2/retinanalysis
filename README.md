# RetinAnalysis
MEA and Single Cell Ephys Analysis Package

## End-to-End Setup Guide

This guide walks through the complete setup: prerequisites, installation, database, and launching the DJ-GUI web interface.

### Prerequisites

| Requirement | Purpose | Install link |
|---|---|---|
| **Conda** (Miniconda or Anaconda) | Python environment management | [miniconda](https://docs.anaconda.com/miniconda/) |
| **Docker Desktop** | Runs the MySQL database | [docker.com/desktop](https://docs.docker.com/desktop/) |
| **Node.js 18+** | DJ-GUI web frontend | [nodejs.org](https://nodejs.org/) |
| **Git** | Clone repositories | [git-scm.com](https://git-scm.com/) |

### Step 1: Clone repositories

```bash
cd ~/repositories  # or wherever you keep your code
git clone https://github.com/chrischen2/retinanalysis.git
git clone https://github.com/chrischen2/datajoint-1.git
```

### Step 2: Create conda environment and install packages

```bash
# Create and activate a Python 3.11 environment
conda create --name retinanalysis python=3.11.13
conda activate retinanalysis

# Install dj-server (the DataJoint web backend) first
cd ~/repositories/datajoint-1
pip install -e .

# Install retinanalysis (includes dj-server as a dependency)
cd ~/repositories/retinanalysis
pip install -e .
```

If you need the Chichilnisky lab utilities (optional, for MEA spike sorting):
```bash
cd ~/repositories/artificial-retina-software-pipeline/utilities/
pip install --no-build-isolation .
```

### Step 3: Configure retinanalysis

Create a `config.ini` file and place it in `retinanalysis/src/retinanalysis/config/`.
See the [sample config.ini](#sample-configini-file) section below for a template.

### Step 4: Start the MySQL database

```bash
# Copy docker-compose.yaml to a directory where you want to store DB data
mkdir -p ~/retinanalysis-db
cp ~/repositories/retinanalysis/docker-compose.yaml ~/retinanalysis-db/
cd ~/retinanalysis-db

# Start the MySQL container (runs on port 3306)
docker compose up -d
```

Make sure the container is running before proceeding. You can check in Docker Desktop or with:
```bash
docker ps  # should show datajoint/mysql:8.0 running
```

### Step 5: Populate the database

```python
import retinanalysis as ra
ra.populate_database()
```

This reads your `config.ini` paths and fills the database with experiment metadata. It can take a while for large datasets.

### Step 6: Launch DJ-GUI

```python
import retinanalysis as ra
ra.DJ_GUI.launch()
```

This starts:
- **Flask backend** on http://127.0.0.1:5000 (the DataJoint API server)
- **Next.js frontend** on http://localhost:3000 (the web GUI)

Your browser will open automatically. On first launch, npm dependencies are installed automatically (~1 min).

#### Launch options

```python
# API-only mode (no frontend, useful for notebooks)
ra.DJ_GUI.launch(frontend=False)

# Custom ports
ra.DJ_GUI.launch(flask_port=5001, frontend_port=3001)

# Don't open browser
ra.DJ_GUI.launch(open_browser=False)
```

#### Standalone Flask server (without retinanalysis)

You can also run the dj-server backend directly:
```bash
dj-server                          # console script
flask --app dj_server.app run     # Flask CLI
```

### Quick Reference

```bash
# Daily workflow — every time you start working:
conda activate retinanalysis       # 1. Activate environment
docker compose up -d               # 2. Start DB (from your DB directory)
```
```python
import retinanalysis as ra         # 3. Import
ra.DJ_GUI.launch()                 # 4. Launch GUI
```

---

## Note for Windows Users

The above requirements have been tested to work on both Mac and Linux (Ubuntu 24.04 LTS).

For windows, you may receive a DLL error when the package attempts to import matplotlib
for the first time. To fix this, run:
```
pip uninstall Pillow
pip install -U Pillow
```

## Sample config.ini file
```
[DEFAULT]
analysis = /Volumes/Vyom MEA/analysis
data = /Volumes/Vyom MEA/data/sorted
raw = /Volumes/Vyom MEA/data/raw
h5 = /Volumes/Vyom MEA/data/datajoint_testbed/data_dirs/data
meta = /Volumes/Vyom MEA/data/datajoint_testbed/data_dirs/meta
tags = /Volumes/Vyom MEA/data/datajoint_testbed/data_dirs/tags
query = /Volumes/data-1/analysis
user = vyomr

[SECONDARY]
analysis = /Volumes/data-1/analysis
data = /Volumes/data-1/data/sorted
raw = /Volumes/data-1/data/raw
h5 = /Volumes/data-1/data/h5
meta = /Volumes/data-1/datajoint_testbed/mea/meta
tags = /Volumes/data-1/datajoint_testbed/mea/tags
query = /Volumes/data-1/analysis
user = vyomr

[LINUX_DEFAULT]
...

[LINUX_SECONDARY]
...

[WINDOWS_DEFAULT]
...

[WINDOWS_SECONDARY]
...
```
Note: The `query` dir is used by `datajoint_utils.plot_mosaics_for_all_datasets` and it's useful to have it set to the NAS analysis dir even when all other paths are SSD. This allows loading and plotting mosaics and cell typing from all the data on the NAS instead of just the data on your SSD's `analysis` dir.

## Docker Details

Retinanalysis uses a DataJoint MySQL database to store all experiment metadata. This uses the datajoint/mysql docker image found at <a href='https://hub.docker.com/r/datajoint/mysql'>https://hub.docker.com/r/datajoint/mysql</a>.

The `docker-compose.yaml` in the repo root starts a MySQL 8.0 container on port 3306. The DJ-GUI web app connects to this same database — no separate database setup is needed.

For single-cell data, there is also a `docker-compose.sc.yaml` that runs on port 3307.

NOTE: Before importing retinanalysis, you will need to make sure the Docker container is running in Docker
Desktop (or through the terminal). If it is running, you will see a stop icon; otherwise, click the play button.

<img width="1382" height="832" alt="Screenshot 2025-10-24 at 3 00 20 PM" src="https://github.com/user-attachments/assets/45ee0d03-6dd7-48c4-ad38-c75e558259ed" />
