import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qlib.data import D

# =============================================================================
# 辅助函数
# =============================================================================
def calculate_metrics(returns):
    """计算夏普、最大回撤等核心指标 (中文版)"""
    if returns.empty or returns.sum() == 0:
        return {
            "年化收益率": "0.00%",
            "年化波动率": "0.00%",
            "夏普比率": "0.0000",
            "最大回撤": "0.00%"
        }
        
    # 年化收益
    ann_return = returns.mean() * 252
    # 年化波动率
    ann_vol = returns.std() * np.sqrt(252)
    # 夏普比率 (假设无风险利率为0)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # 最大回撤
    cum_ret = (1 + returns).cumprod()
    drawdown = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
    max_dd = drawdown.min()
    
    return {
        "年化收益率": f"{ann_return:.2%}",
        "年化波动率": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.4f}",
        "最大回撤": f"{max_dd:.2%}"
    }

def run_backtest_analysis(pred_df, output_prefix, topk=50, benchmark='SH000300'):
    """
    通用回测入口 (Vectorized Version)
    """
    print(">>> [Backtest] 开始执行回测分析 (Vectorized)...")
    
    # -------------------------------------------------------------------------
    # 1. 数据清洗与对齐
    # -------------------------------------------------------------------------
    if isinstance(pred_df, pd.Series):
        pred_df = pred_df.to_frame(name='score')
        
    pred_df = pred_df.sort_index()
    
    # 确保索引是 datetime 类型
    if not isinstance(pred_df.index.get_level_values(0)[0], pd.Timestamp):
        pred_df.index = pred_df.index.set_levels(pd.to_datetime(pred_df.index.levels[0]), level=0)

    start_date = pred_df.index.get_level_values("datetime").min()
    end_date = pred_df.index.get_level_values("datetime").max()
    print(f"    回测区间: {start_date} ~ {end_date}")

    # -------------------------------------------------------------------------
    # 2. 准备真实收益数据 (Label)
    # -------------------------------------------------------------------------
    print("    正在加载真实行情数据...")
    stocks = pred_df.index.get_level_values(1).unique().tolist()
    
    try:
        # 获取 T+1 收益率
        label_df = D.features(stocks, ['Ref($close, -1)/$close - 1'], start_time=start_date, end_time=end_date)
        label_df.columns = ['next_ret']
        
        merged = pred_df.join(label_df, how='inner')
        merged['next_ret'] = merged['next_ret'].fillna(0)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # -------------------------------------------------------------------------
    # 3. 计算 IC (选股能力)
    # -------------------------------------------------------------------------
    try:
        ic = merged.groupby(level='datetime').apply(lambda x: x['score'].corr(x['next_ret']))
        rank_ic = merged.groupby(level='datetime').apply(lambda x: x['score'].corr(x['next_ret'], method='spearman'))
        print(f"    [Metrics] Mean IC: {ic.mean():.4f}, Mean Rank IC: {rank_ic.mean():.4f}")
    except Exception as e:
        print(f"    ⚠️ IC 计算跳过: {e}")

    # -------------------------------------------------------------------------
    # 4. 向量化回测 (Top-K)
    # -------------------------------------------------------------------------
    print(f"    模拟 Top-{topk} 等权重交易策略...")
    
    merged['rank'] = merged.groupby(level='datetime')['score'].rank(method='first', ascending=False)
    portfolio_df = merged[merged['rank'] <= topk]
    strategy_ret = portfolio_df.groupby(level='datetime')['next_ret'].mean()

    # -------------------------------------------------------------------------
    # 5. 基准收益提取
    # -------------------------------------------------------------------------
    print(f"    提取基准收益 ({benchmark})...")
    try:
        bench_data = D.features([benchmark], ['$close'], start_time=start_date, end_time=end_date)
        
        bench_flat = bench_data.reset_index()
        date_col = next((c for c in bench_flat.columns if pd.api.types.is_datetime64_any_dtype(bench_flat[c])), 'datetime')
        
        bench_flat = bench_flat.set_index(date_col)
        bench_series = bench_flat['$close'].sort_index()
        
        bench_series = bench_series.groupby(level=0).first()
        benchmark_ret = bench_series.pct_change().fillna(0)
        benchmark_ret = benchmark_ret.reindex(strategy_ret.index).fillna(0)
        
    except Exception as e:
        print(f"    ⚠️ 基准加载失败 ({e})，使用 0 填充。")
        benchmark_ret = pd.Series(0, index=strategy_ret.index)

    alpha_ret = strategy_ret - benchmark_ret

    # -------------------------------------------------------------------------
    # 6. 结果输出与绘图 (含基准对比)
    # -------------------------------------------------------------------------
    strat_metrics = calculate_metrics(strategy_ret)
    bench_metrics = calculate_metrics(benchmark_ret)

    print("\n" + " " * 4 + "="*55)
    print(f"    {'[Performance] 策略 vs 基准':^45}")
    print(" " * 4 + "="*55)
    print(f"    {'指标':<12} | {'AI 策略':<15} | {'基准指数':<15}")
    print(" " * 4 + "-"*55)
    
    for k in strat_metrics.keys():
        s_val = strat_metrics[k]
        b_val = bench_metrics.get(k, "N/A")
        print(f"    {k:<12} | {s_val:<15} | {b_val:<15}")
    print(" " * 4 + "="*55 + "\n")

    print("    正在绘制收益曲线图...")
    plt.figure(figsize=(12, 6))
    
    cum_strategy = (1 + strategy_ret).cumprod()
    cum_bench = (1 + benchmark_ret).cumprod()
    cum_alpha = (1 + alpha_ret).cumprod()
    
    plt.plot(cum_strategy.index, cum_strategy.values, label=f'Strategy (Top{topk})', color='red', linewidth=1.5)
    plt.plot(cum_bench.index, cum_bench.values, label=f'Benchmark ({benchmark})', color='gray', linestyle='--', linewidth=1.5)
    plt.fill_between(cum_alpha.index, cum_alpha.values, 1, color='blue', alpha=0.1, label='Excess Return (Alpha)')
    
    plt.title(f"Backtest Performance: {output_prefix.split('/')[-1]}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f"{output_prefix}_backtest_plot.png"
    plt.savefig(plot_path)
    print(f"✅ 回测曲线图已保存: {plot_path}")
    
    res_df = pd.DataFrame({
        'strategy': strategy_ret,
        'benchmark': benchmark_ret,
        'alpha': alpha_ret
    })
    res_df.to_csv(f"{output_prefix}_daily_returns.csv")