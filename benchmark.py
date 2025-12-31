import subprocess
import os
import sys
import pandas as pd
import numpy as np
import argparse
import re
from tabulate import tabulate
from collections import Counter

# ==============================================================================
# ⚙️ 全局配置
# ==============================================================================
MODELS = ['lgbm', 'lstm', 'alstm', 'transformer', 'tra']
POOLS = ['csi300', 'all']

def run_command(cmd, capture_output=True):
    """执行 Shell 命令并返回输出"""
    print(f"⚡ Running: {cmd}")
    try:
        # 使用 unbuffered 模式 (-u) 确保日志实时输出
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT,
            text=True
        )
        return result.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stdout: print(e.stdout)
        return None

def calculate_metrics(returns):
    """计算年化收益、夏普、回撤"""
    if returns.empty: return None
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    cum_ret = (1 + returns).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    return {
        "Ann Ret": ann_ret,
        "Sharpe": sharpe,
        "Max DD": max_dd,
        "Vol": ann_vol
    }

def get_backtest_metrics(csv_path):
    """读取 CSV 并分别计算 Strategy 和 Benchmark 的指标"""
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        strat_metrics = calculate_metrics(df['strategy'])
        bench_metrics = calculate_metrics(df['benchmark'])
        return strat_metrics, bench_metrics
    except Exception as e:
        print(f"⚠️ Error reading {csv_path}: {e}")
        return None, None

def parse_ic_metrics(log_output):
    """从日志中正则提取 Mean IC 和 Mean Rank IC"""
    if not log_output: return {}
    
    # 匹配日志: [Metrics] Mean IC: 0.0483, Mean Rank IC: 0.0493
    ic_pattern = re.search(r"Mean IC:\s*([-\d\.]+)", log_output)
    rank_ic_pattern = re.search(r"Mean Rank IC:\s*([-\d\.]+)", log_output)
    
    metrics = {}
    if ic_pattern:
        metrics['Mean IC'] = float(ic_pattern.group(1))
    if rank_ic_pattern:
        metrics['Mean Rank IC'] = float(rank_ic_pattern.group(1))
        
    return metrics

def parse_top_stocks_detailed(stdout_log, top_n=5):
    """从日志解析 Top N 股票的详细信息"""
    stocks = []
    if not stdout_log: return []
    
    capture = False
    for line in stdout_log.split('\n'):
        if "Top 10" in line and "Pool:" in line:
            capture = True
            continue
        if capture:
            if "Redis" in line or "-------" in line and len(stocks) > 0:
                break
            if "|" in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                # 排除表头
                if len(parts) >= 3 and parts[0] != "代码" and not parts[0].startswith("-"):
                    stocks.append({
                        'code': parts[0],
                        'name': parts[1],
                        'score': parts[2]
                    })
    
    return stocks[:top_n]

def format_percentage(val):
    return f"{val:.2%}" if isinstance(val, (float, int)) else val

def format_float(val, digits=4):
    return f"{val:.{digits}f}" if isinstance(val, (float, int)) else val

def main():
    parser = argparse.ArgumentParser(description="Qlib 自动化 Benchmark (Markdown修复版)")
    parser.add_argument('--models', type=str, default="all", help='模型列表')
    parser.add_argument('--train', action='store_true', help='是否重新训练')
    parser.add_argument('--output', type=str, default="benchmark_report.md", help='报告保存文件名')
    args = parser.parse_args()

    target_models = MODELS if args.models == "all" else args.models.split(',')
    
    bt_data = [] 
    pred_data = []

    print(f"🚀 Starting Benchmark for: {target_models}")

    # ==========================================================================
    # 1. 执行任务循环
    # ==========================================================================
    for model in target_models:
        m_lower = model.lower()
        
        # --- A. 训练 (Training) ---
        model_file = f"artifacts/{m_lower}_model_latest.pkl"
        if args.train or not os.path.exists(model_file):
            print(f"\n[Training] {model}...")
            run_command(f"python -u main.py --model {m_lower} --mode train", capture_output=False)
        
        for pool in POOLS:
            print(f"\n--- {model.upper()} @ {pool.upper()} ---")
            
            # --- B. 回测 (Backtest) ---
            bt_log = run_command(f"python main.py --model {m_lower} --mode backtest --filter {pool}")
            ic_metrics = parse_ic_metrics(bt_log)
            
            csv_path = f"predictions/{m_lower}_{pool}_daily_returns.csv"
            s_met, b_met = get_backtest_metrics(csv_path)
            
            if s_met:
                s_met.update(ic_metrics)
                bt_data.append({
                    "Pool": pool.upper(),
                    "Model": model.upper(),
                    "Type": "Strategy",
                    **s_met
                })
                bt_data.append({
                    "Pool": pool.upper(),
                    "Model": "Benchmark", 
                    "Type": "Benchmark",
                    **b_met
                })

            # --- C. 预测 (Predict) ---
            pred_log = run_command(f"python main.py --model {m_lower} --mode predict --filter {pool}")
            tops = parse_top_stocks_detailed(pred_log, top_n=5)
            if tops:
                pred_data.append({
                    "Pool": pool.upper(),
                    "Model": model.upper(),
                    "TopList": tops
                })

    # ==========================================================================
    # 2. 生成报告
    # ==========================================================================
    report_lines = []
    def log(text=""):
        print(text)
        report_lines.append(text)

    log("\n" + "="*80)
    log("📊 Qlib 策略综合评测报告")
    log("="*80)
    log(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Part 1: 回测绩效 ---
    for pool_name in [p.upper() for p in POOLS]:
        log(f"\n### 1.{POOLS.index(pool_name.lower()) + 1} {pool_name} 绩效对比 (Performance)")
        
        pool_records = [d for d in bt_data if d['Pool'] == pool_name]
        if not pool_records:
            log("暂无数据 (No data)")
            continue
            
        bench_row = next((r for r in pool_records if r['Type'] == 'Benchmark'), None)
        strat_rows = [r for r in pool_records if r['Type'] == 'Strategy']
        
        final_rows = strat_rows
        if bench_row:
            bench_row['Model'] = "🛑 基准 (Benchmark)"
            bench_row['Mean IC'] = np.nan
            bench_row['Mean Rank IC'] = np.nan
            final_rows.append(bench_row)
            
        table_list = []
        for r in final_rows:
            table_list.append({
                "模型名称 (Model)": r['Model'],
                "年化收益 (Ann Ret)": format_percentage(r['Ann Ret']),
                "夏普比率 (Sharpe)": format_float(r['Sharpe']),
                "最大回撤 (Max DD)": format_percentage(r['Max DD']),
                "信息系数 (Mean IC)": format_float(r.get('Mean IC', 0)),
                "秩信息系数 (Rank IC)": format_float(r.get('Mean Rank IC', 0))
            })
        
        # 【修复点 1】强制使用 pipe 格式，这是最标准的 Markdown 表格格式
        log(tabulate(table_list, headers="keys", tablefmt="pipe"))

    # --- Part 2: 预测信号 ---
    log("\n\n### 2. Top 5 预测信号 (Prediction Signals)")
    
    for pool_name in [p.upper() for p in POOLS]:
        log(f"\n**[{pool_name}] 信号详情**")
        
        pool_preds = [d for d in pred_data if d['Pool'] == pool_name]
        if not pool_preds:
            log("暂无数据 (No data)")
            continue

        all_stocks_names = []
        for p in pool_preds:
            for item in p['TopList']:
                all_stocks_names.append(item['name'])
        
        counts = Counter(all_stocks_names)
        consensus_stock_name = counts.most_common(1)[0][0] if counts else None
        
        table_data = []
        for p in pool_preds:
            row = { "模型名称 (Model)": p['Model'] }
            for i in range(5):
                if i < len(p['TopList']):
                    item = p['TopList'][i]
                    # 格式: 代码 名称 (得分)
                    display = f"{item['code']} {item['name']} ({float(item['score']):.3f})"
                    
                    if item['name'] == consensus_stock_name:
                        display += " 🔥"
                else:
                    display = "-"
                
                # 【修复点 2】将表头中的 `|` 替换为 `/`，防止 Markdown 表格破损
                row[f"第{i+1}名 (代码/名称/得分)"] = display
            table_data.append(row)
            
        # 【修复点 1】强制使用 pipe 格式
        log(tabulate(table_data, headers="keys", tablefmt="pipe"))
        
        if consensus_stock_name:
            count = counts[consensus_stock_name]
            log(f"\n> 🏆 **最强推荐 (Consensus):** **{consensus_stock_name}** (获得 {count} 个模型同时推荐)")

    # 保存
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n✅ 报告已生成: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"\n❌ 保存失败: {e}")

if __name__ == "__main__":
    main()