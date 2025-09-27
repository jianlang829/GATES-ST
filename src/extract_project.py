import os
from pathlib import Path

# 定义要收集的文本文件扩展名（源码 + 配置）
TEXT_EXTENSIONS = {
    '.py',      # Python 源码
    '.yaml',    # 配置
    '.yml',
    '.txt',     # 文本文件（如 trainer.txt, gates_model.txt 等）
    '.md',      # Markdown
    # 注意：不包括 .h5, .h5ad, .png, .jpg 等二进制文件
}

# 要排除的目录（避免读取缓存或二进制目录）
EXCLUDE_DIRS = {'__pycache__', 'cache', 'figures', 'results'}

def is_text_file(file_path):
    """简单判断是否为文本文件（基于扩展名）"""
    return file_path.suffix.lower() in TEXT_EXTENSIONS

def should_skip_dir(dir_path: Path, root: Path):
    """判断是否应跳过该目录"""
    rel_parts = dir_path.relative_to(root).parts
    return any(part in EXCLUDE_DIRS for part in rel_parts)

def collect_files_to_txt(root_dir: str, output_file: str):
    root = Path(root_dir).resolve()
    output_path = Path(output_file)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for file_path in root.rglob('*'):
            if file_path.is_file():
                # 跳过排除目录中的文件
                if should_skip_dir(file_path.parent, root):
                    continue
                # 只处理文本类源码/配置文件
                if is_text_file(file_path):
                    try:
                        # 写入分隔符和文件路径
                        out_f.write(f"\n{'='*80}\n")
                        out_f.write(f"File: {file_path.relative_to(root)}\n")
                        out_f.write(f"{'='*80}\n\n")
                        # 读取并写入内容
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        out_f.write(content)
                        out_f.write('\n\n')
                        print(f"Collected: {file_path.relative_to(root)}")
                    except (UnicodeDecodeError, PermissionError) as e:
                        print(f"Skipped (not text or unreadable): {file_path.relative_to(root)} - {e}")

    print(f"\n✅ All source and config files collected into: {output_path}")

if __name__ == "__main__":
    # 假设脚本在项目根目录运行
    project_root = "."  # 或指定绝对路径，如 r"C:\your\project"
    output_txt = "collected_sources_and_configs.txt"
    collect_files_to_txt(project_root, output_txt)
