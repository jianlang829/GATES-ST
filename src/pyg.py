import torch

torch_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda
cuda_str = f"cu{cuda_version.replace('.', '')}" if cuda_version else "cpu"

print(f"pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_str}.html")
