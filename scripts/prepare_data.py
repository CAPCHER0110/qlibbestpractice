import os
import sys
import shutil
import tarfile
import time
import datetime
import urllib.request
from pathlib import Path

# === 配置区域 ===
# 假设脚本位于 scripts/ 目录，我们需要保存到 ../data/cn_data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "cn_data"
TEMP_FILE = PROJECT_ROOT / "data" / "qlib_bin.tar.gz"
VERSION_FILE = TARGET_DIR / "version.txt" # 用于记录上次更新时间

DATA_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"

def get_last_update_time():
    """读取本地记录的上次更新时间"""
    if not VERSION_FILE.exists():
        return None
    try:
        with open(VERSION_FILE, 'r') as f:
            timestamp = float(f.read().strip())
        return datetime.datetime.fromtimestamp(timestamp)
    except Exception:
        return None

def set_last_update_time():
    """写入当前时间作为更新标记"""
    with open(VERSION_FILE, 'w') as f:
        f.write(str(time.time()))

def download_and_extract():
    """执行核心下载和解压逻辑"""
    # 确保父目录存在
    if not TARGET_DIR.parent.exists():
        TARGET_DIR.parent.mkdir(parents=True)
        
    try:
        print(f"🔗 [Download] 正在从 GitHub 镜像源拉取最新数据...")
        print(f"   URL: {DATA_URL}")
        
        # 下载 (带简单进度条)
        urllib.request.urlretrieve(DATA_URL, str(TEMP_FILE), report_progress)
        print("\n✅ 下载完成，开始解压...")

        # 如果之前有旧数据，建议先清空，防止文件冲突（可选，视具体需求定）
        # if TARGET_DIR.exists():
        #     shutil.rmtree(TARGET_DIR)
        # TARGET_DIR.mkdir(parents=True, exist_ok=True)

        # 解压
        with tarfile.open(TEMP_FILE, "r:gz") as tar:
            members = tar.getmembers()
            # 兼容性处理：找到压缩包内的根目录名
            root_dir = members[0].name.split('/')[0]
            
            for member in members:
                if member.name.startswith(root_dir + '/'):
                    # 去掉顶层目录前缀 (strip components)
                    member.name = member.name[len(root_dir) + 1:]
                    # 避免解压出空文件名
                    if member.name: 
                        tar.extract(member, path=TARGET_DIR)
        
        # 写入版本文件
        set_last_update_time()
        print(f"✅ 数据已更新并解压至: {TARGET_DIR}")

    except Exception as e:
        print(f"\n❌ [Error] 数据下载/解压失败: {e}")
        # 失败清理
        if TEMP_FILE.exists():
            os.remove(TEMP_FILE)
        sys.exit(1)
    finally:
        # 清理压缩包
        if TEMP_FILE.exists():
            os.remove(TEMP_FILE)

def ensure_qlib_data(max_age_days=1):
    """
    主入口：确保数据存在且新鲜
    :param max_age_days: 数据被认为“新鲜”的天数，默认 1 天
    """
    
    # 1. 检查数据是否存在
    if not TARGET_DIR.exists() or not any(TARGET_DIR.iterdir()):
        print(f"⚠️ 本地未检测到数据，准备初始化...")
        download_and_extract()
        return

    # 2. 检查数据是否过期
    last_update = get_last_update_time()
    
    if last_update is None:
        print(f"⚠️ 数据存在但版本未知，准备更新...")
        download_and_extract()
        return

    # 计算时间差
    now = datetime.datetime.now()
    delta = now - last_update
    
    if delta.days >= max_age_days:
        print(f"⚠️ 本地数据已过期 (上次更新: {last_update.strftime('%Y-%m-%d %H:%M')})。")
        print(f"   过期阈值: {max_age_days} 天。正在自动拉取最新数据...")
        download_and_extract()
    else:
        print(f"✅ 本地数据有效 (上次更新: {last_update.strftime('%Y-%m-%d %H:%M')})。无需重新下载。")

def report_progress(block_num, block_size, total_size):
    """下载进度条"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = 100 * downloaded / total_size
        sys.stdout.write(f"\r📥 Downloading... {downloaded / (1024*1024):.1f} MB ({percent:.1f}%)")
    else:
        sys.stdout.write(f"\r📥 Downloading... {downloaded / (1024*1024):.1f} MB")
    sys.stdout.flush()

if __name__ == "__main__":
    ensure_qlib_data()