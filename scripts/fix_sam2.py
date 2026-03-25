import os

# 定位到你粘贴进来的 configs 文件夹
config_dir = r"F:\cor\sam2\configs"

# 1. 在 configs 根目录创建一个空的 __init__.py
with open(os.path.join(config_dir, "__init__.py"), "w") as f:
    pass

# 2. 如果里面还有 sam2 子目录，也给它塞一个 __init__.py
sam2_sub = os.path.join(config_dir, "sam2")
if os.path.exists(sam2_sub):
    with open(os.path.join(sam2_sub, "__init__.py"), "w") as f:
        pass

print("✅ Python 包结构修复完成！Hydra 已经恢复视力！")