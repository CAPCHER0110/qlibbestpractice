# models/stock_utils.py

import os
import pandas as pd

class StockNameMap:
    def __init__(self, cache_path="data/stock_names.csv"):
        self.cache_path = cache_path
        self.name_map = {}
        self._load_or_fetch()

    def _load_or_fetch(self):
        """核心逻辑：有缓存读缓存，没缓存下数据"""
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path, dtype=str)
                self.name_map = dict(zip(df['code'], df['name']))
                return
            except Exception as e:
                print(f"⚠️ 缓存文件读取失败: {e}")

        self._fetch_from_akshare()

    def _fetch_from_akshare(self):
        # 为了不影响主流程速度，只有在真正需要下载时才 import akshare
        print("🌐 正在通过 Akshare 拉取全市场股票名单 (首次运行较慢)...")
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            
            codes = []
            names = []
            
            for _, row in df.iterrows():
                raw_code = str(row['代码'])
                name = str(row['名称'])
                
                if raw_code.startswith('6'):
                    qlib_code = f"SH{raw_code}"
                elif raw_code.startswith('8') or raw_code.startswith('4'):
                    qlib_code = f"BJ{raw_code}"
                else:
                    qlib_code = f"SZ{raw_code}"
                
                codes.append(qlib_code)
                names.append(name)
            
            self.name_map = dict(zip(codes, names))
            
            # 保存缓存
            save_df = pd.DataFrame({'code': codes, 'name': names})
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            save_df.to_csv(self.cache_path, index=False, encoding="utf_8_sig")
            print(f"✅ 股票名单已保存至: {self.cache_path} (共 {len(codes)} 只)")
            
        except ImportError:
            print("❌ 未安装 akshare，无法下载名称。请运行: pip install akshare")
        except Exception as e:
            print(f"❌ Akshare 下载失败: {e}")

    def get_name(self, instrument):
        """获取名称，如果找不到返回代码本身"""
        return self.name_map.get(instrument, instrument)

# 全局单例
stock_mapper = StockNameMap()