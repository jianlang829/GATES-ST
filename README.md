# GATES Project: Spatial Transcriptomics Analysis

This project implements the **G**raph **A**ttention **T**ranscriptomics **E**ncoder (GATES) for analyzing spatial transcriptomics data using modern PyTorch Geometric.

## 📁 Project Structure

```
gates_project/
C:.
│  .gitignore
│  default_yaml.txt
│  gates_model.txt
│  LICENSE
│  README.md
│  requirements.txt
│  run_analysis.txt
│  trainer.txt
│  utils.txt
│
├─configs
│      default.yaml
│
├─data
│  ├─151673
│  │  │  filtered_feature_bc_matrix.h5
│  │  │  metadata.tsv
│  │  │  position.tsv
│  │  │  RNA_counts.tsv
│  │  │  truth.txt
│  │  │  used_barcodes.txt
│  │  │
│  │  └─spatial
│  │          scalefactors_json.json
│  │          tissue_hires_image.png
│  │          tissue_lowres_image.png
│  │          tissue_positions_list.csv
│  │
│  ├─151674
│  │  │  filtered_feature_bc_matrix.h5
│  │  │  metadata.tsv
│  │  │  truth.txt
│  │  │  V1_Breast_Cancer_Block_A_Section_1_raw_feature_bc_matrix.h5
│  │  │
│  │  └─spatial
│  │          aligned_fiducials.jpg
│  │          detected_tissue_image.jpg
│  │          scalefactors_json.json
│  │          tissue_hires_image.png
│  │          tissue_lowres_image.png
│  │          tissue_positions_list.csv
│  │
│  └─151675
│      │  _filtered_feature_bc_matrix.h5
│      │  _truth.txt
│      │
│      └─spatial
│              scalefactors_json.json
│              tissue_hires_image.png
│              tissue_lowres_image.png
│              tissue_positions_list.csv
│
├─scripts
│  │  run_analysis.py
│  │  __init__.py
│  │
│  └─__pycache__
│          run_analysis.cpython-310.pyc
│          __init__.cpython-310.pyc
│
└─src
    │  Check_gpu_available.py
    │  convert_visium_to_stereo.py
    │  gates_model.py
    │  pyg.py
    │  trainer.py
    │  utils.py
    │  __init__.py
    │
    └─__pycache__
            gates_model.cpython-310.pyc
            trainer.cpython-310.pyc
            utils.cpython-310.pyc
            __init__.cpython-310.pyc
```

## ⚙️ Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the main analysis script:

```bash
cd scripts
python run_analysis.py
```

## 🛠 Configuration

All parameters (paths, hyperparameters) are managed in `configs/default.yaml`. Modify this file to customize your analysis.

## 📈 Output

The script will:
1. Preprocess the Stereo-seq Mouse Brain dataset.
2. Construct spatial and gene-similarity graphs.
3. Train the GATES model.
4. Perform clustering and generate evaluation metrics (Silhouette, Davies-Bouldin).
5. Display spatial clustering plots.
