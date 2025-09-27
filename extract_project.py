import os

# ================== 配置区 ==================
PROJECT_ROOT = r"."  # 当前目录，可改为绝对路径如 r"C:\your\project"
OUTPUT_FILE = "project_summary.txt"

# 定义要提取的文件路径（相对于项目根目录）
TARGET_PATHS = [
    # 根目录
    ".gitignore",
    "README.md",
    "requirements.txt",

    # 配置文件
    "configs/default.yaml",

    # 脚本
    "scripts/run_analysis.py",

    # 源代码（只取 .py，自动跳过 __pycache__）
    "src/Check_gpu_available.py",
    "src/convert_visium_to_stereo.py",
    "src/gates_model.py",
    "src/pyg.py",
    "src/trainer.py",
    "src/utils.py",
    "src/__init__.py",
]

# ============================================

def read_file_safely(file_path):
    """安全读取文件，自动处理编码问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return f"[无法读取二进制文件或编码错误: {file_path}]"
    except Exception as e:
        return f"[读取错误: {e}]"

# 写入汇总文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    out_f.write("=== 项目核心文件内容汇总 ===\n\n")

    for rel_path in TARGET_PATHS:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(full_path):
            out_f.write(f"\n{'='*60}\n")
            out_f.write(f"文件: {rel_path}\n")
            out_f.write(f"{'='*60}\n\n")

            content = read_file_safely(full_path)
            out_f.write(content)
            out_f.write("\n\n")
        else:
            out_f.write(f"\n[警告] 文件未找到: {rel_path}\n\n")
            print(f"⚠️ 警告: {rel_path} 不存在")

print(f"✅ 项目必要文件内容已提取到: {OUTPUT_FILE}")
