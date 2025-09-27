import sys
sys.modules['torch'] = None  # 阻止 anndata 导入 torch

import scanpy as sc
import pandas as pd
import os
# ... 后续代码不变
import scanpy as sc
import pandas as pd
import os

print("开始转换...")
# ===== 配置路径 =====
input_h5 = r"C:\Users\admini\Documents\GATES-ST\data\151673\filtered_feature_bc_matrix.h5"
input_pos = r"C:\Users\admini\Documents\GATES-ST\data\151673\spatial\tissue_positions_list.csv"
output_dir = r"C:\Users\admini\Documents\GATES-ST\data\151673"

# ===== 1. 读取表达矩阵 =====
adata = sc.read_10x_h5(input_h5)
# 转置：行=基因，列=barcode（TSV 通常这样）
df_counts = adata.to_df().T  # shape: (n_genes, n_barcodes)
df_counts.to_csv(os.path.join(output_dir, "RNA_counts.tsv"), sep="\t")

print("RNA_counts.tsv 已保存")
# ===== 2. 读取坐标文件 =====
# Visium 的 tissue_positions_list.csv 格式（无 header）：
# barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
pos_df = pd.read_csv(input_pos, header=None)
pos_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]

# 只保留 in_tissue == 1 的 spots（即组织内的有效点）
used_barcodes = pos_df[pos_df["in_tissue"] == 1]["barcode"].tolist()

# 保存 used_barcodes.txt
with open(os.path.join(output_dir, "used_barcodes.txt"), "w") as f:
    f.write("\n".join(used_barcodes))

print("✅ 已保存 used_barcodes.txt")
# 保存 position.tsv（格式：barcode, x, y）
# 注意：Visium 坐标常用 pxl_col (x), pxl_row (y)，但有些工具用 array_col/array_row
# 这里用高分辨率图像坐标（pxl_col, pxl_row），你也可以根据需求换
# ✅ 正确：使用高分辨率像素坐标 (pxl_col, pxl_row)
pos_used = pos_df[pos_df["in_tissue"] == 1][["barcode", "pxl_col", "pxl_row"]]
pos_used.to_csv(os.path.join(output_dir, "position.tsv"), sep="\t", index=False, header=False)

print("✅ 三个文件已生成！")
print("- RNA_counts.tsv")
print("- position.tsv")
print("- used_barcodes.txt")
