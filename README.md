# GATES Project: Spatial Transcriptomics Analysis

This project implements the **G**raph **A**ttention **T**ranscriptomics **E**ncoder (GATES) for analyzing spatial transcriptomics data using modern PyTorch Geometric.

## 📁 Project Structure

```
gates_project/
├── src/
│   ├── __init__.py
│   ├── gates_model.py          # Model definition
│   ├── trainer.py              # Training logic
│   └── utils.py                # Data processing utilities
├── configs/
│   └── default.yaml            # Configuration file
├── scripts/
│   └── run_analysis.py         # Main script
├── README.md
└── requirements.txt
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
