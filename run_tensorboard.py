import os
import sys
from pathlib import Path
from tensorboard import program

# 导入我们的全局配置
from configs import config

# ================= 配置 =================
# 动态获取当前项目的 logs 总目录
LOG_DIR = os.path.join(config.PROJECT_ROOT, "logs")
PORT = 6006
# ========================================

def run_tensorboard():
    if not os.path.exists(LOG_DIR):
        print(f"❌ 错误：找不到日志目录: {LOG_DIR}")
        print("💡 提示：可能是你还没有开始训练，logs 文件夹尚未生成。请先运行一遍 train.py。")
        return

    print(f"📂 正在加载日志目录: {LOG_DIR}")
    print("🚀 正在启动 TensorBoard (全览模式)...")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', LOG_DIR, '--port', str(PORT)])

    try:
        url = tb.launch()
        print(f"\n✅ TensorBoard 已启动！")
        print(f"👉 请按住 Ctrl 键并点击链接在浏览器中访问: {url}")
        print(f"\n(按 Ctrl+C 可以停止服务)")

        # 保持后台运行
        input("按 [Enter] 键退出服务...\n")

    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    run_tensorboard()