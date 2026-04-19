import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 风险平价模型函数 ====================
def risk_parity_allocation(cov_matrix):
    n = cov_matrix.shape[0]
    
    def objective(weights):
        weights = np.array(weights)
        port_var = weights.dot(cov_matrix).dot(weights)
        risk_contrib = weights * (cov_matrix.dot(weights)) / port_var
        return np.sum((risk_contrib - risk_contrib.mean())**2)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: w}
    ]
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method='SLSQP', constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-8})
    return res.x

# ==================== 数据加载 ====================
def load_fund_data(fund_files):
    dfs = []
    for file in fund_files:
        fund_code = os.path.basename(file).split('_')[1].split('.xls')[0]
        df = pd.read_excel(file, sheet_name='每日净值')
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df = df[['复权单位净值(元)']].rename(columns={'复权单位净值(元)': fund_code})
        dfs.append(df)
    combined = pd.concat(dfs, axis=1).sort_index()
    combined.ffill(inplace=True)
    combined.dropna(inplace=True)
    return combined

def load_benchmark_data(benchmark_file):
    df = pd.read_excel(benchmark_file)
    df.rename(columns={'日期Date': '日期', '收盘Close': 'close'}, inplace=True)
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
    df.set_index('日期', inplace=True)
    df = df[['close']].rename(columns={'close': 'benchmark'})
    df.sort_index(inplace=True)
    return df

# ==================== 回测函数（修正调仓日期） ====================
def risk_parity_backtest(fund_data, benchmark_data, start_date='2020-01-01', rebalance_freq='6M'):
    # 过滤起始日期
    fund_data = fund_data[fund_data.index >= start_date]
    benchmark_data = benchmark_data[benchmark_data.index >= start_date]
    
    # 对齐数据
    combined = pd.concat([fund_data, benchmark_data], axis=1).dropna()
    fund_data = combined[fund_data.columns]
    benchmark_data = combined[['benchmark']]
    
    # 日收益率
    returns = fund_data.pct_change().dropna()
    all_dates = fund_data.index
    n_assets = len(fund_data.columns)
    
    # 修正：生成实际存在的交易日作为调仓日
    if rebalance_freq == 'M':
        rebalance_dates = fund_data.groupby(fund_data.index.to_period('M')).apply(lambda x: x.index[-1])
    elif rebalance_freq == 'Q':
        rebalance_dates = fund_data.groupby(fund_data.index.to_period('Q')).apply(lambda x: x.index[-1])
    elif rebalance_freq == '6M':
        # 半年周期：上半年（1-6月）和下半年（7-12月）
        half_year = (fund_data.index.year * 2 + (fund_data.index.month > 6))
        rebalance_dates = fund_data.groupby(half_year).apply(lambda x: x.index[-1])
    else:
        # 其他频率（'Y', 'W'等）回退到resample
        rebalance_dates = fund_data.resample(rebalance_freq).last().dropna().index
    
    # 确保是DatetimeIndex并过滤起始日期
    if isinstance(rebalance_dates, pd.Series):
        rebalance_dates = pd.DatetimeIndex(rebalance_dates.values)
    rebalance_dates = rebalance_dates[rebalance_dates >= start_date]
    
    # 转换为set加速判断
    rebalance_set = set(rebalance_dates)
    
    # 初始化
    weights = pd.DataFrame(index=all_dates, columns=fund_data.columns)
    portfolio_value = pd.Series(index=all_dates, dtype=float)
    portfolio_value.iloc[0] = 1.0
    current_weights = np.ones(n_assets) / n_assets
    weights.loc[all_dates[0]] = current_weights
    rebalance_flags = pd.Series(False, index=all_dates)
    
    # 回测循环
    for i in range(1, len(all_dates)):
        current_date = all_dates[i]
        prev_date = all_dates[i-1]
        
        # 检查调仓日
        if current_date in rebalance_set:
            lookback = 252
            start_idx = max(0, i - lookback)
            cov_matrix = returns.iloc[start_idx:i].cov().values
            current_weights = risk_parity_allocation(cov_matrix)
            rebalance_flags.loc[current_date] = True
        
        weights.loc[current_date] = current_weights
        asset_returns = fund_data.loc[current_date] / fund_data.loc[prev_date] - 1
        portfolio_return = np.dot(current_weights, asset_returns)
        portfolio_value.loc[current_date] = portfolio_value.loc[prev_date] * (1 + portfolio_return)
    
    # 构建结果DataFrame
    results = pd.DataFrame({
        'strategy': portfolio_value,
        'benchmark': benchmark_data['benchmark'] / benchmark_data['benchmark'].iloc[0]
    }, index=all_dates)
    results['excess_return'] = results['strategy'] - results['benchmark']
    results['rebalance'] = rebalance_flags
    
    return results, weights

# ==================== 绩效指标计算 ====================
def calculate_performance_metrics(results):
    daily_returns = results[['strategy', 'benchmark']].pct_change().dropna()
    annual_returns = (1 + daily_returns.mean()) ** 252 - 1
    annual_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_returns / annual_volatility
    max_drawdown = (results[['strategy', 'benchmark']] /
                   results[['strategy', 'benchmark']].cummax() - 1).min()
    excess_return_annual = annual_returns['strategy'] - annual_returns['benchmark']
    active_return = daily_returns['strategy'] - daily_returns['benchmark']
    information_ratio = active_return.mean() / active_return.std() * np.sqrt(252)
    
    metrics = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    })
    metrics.loc['excess_return'] = [excess_return_annual, np.nan, information_ratio, np.nan]
    return metrics

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 文件路径（请根据实际路径修改）
    fund_files = [
        r'MA_data\每日基金净值与行情_000071.OF.xls',
        r'MA_data\每日基金净值与行情_000216.OF.xls',
        r'MA_data\每日基金净值与行情_000614.OF.xls',
        r'MA_data\每日基金净值与行情_050025.OF.xls',
        r'MA_data\每日基金净值与行情_270042.OF.xls'
    ]
    benchmark_file = r'Benchmark\930929perf.xlsx'
    
    # 创建输出目录（如果不存在）
    os.makedirs('Output', exist_ok=True)
    
    # 加载数据
    print("加载基金数据...")
    fund_data = load_fund_data(fund_files)
    print("加载基准数据...")
    benchmark_data = load_benchmark_data(benchmark_file)
    
    # 运行回测
    print("运行风险平价回测...")
    results, weights = risk_parity_backtest(fund_data, benchmark_data, start_date='2020-01-01', rebalance_freq='6M')
    
    # 绩效指标
    metrics = calculate_performance_metrics(results)
    print("\n策略绩效指标:")
    print(metrics)
    
    # 重命名权重列（基金代码 -> 名称）
    dict_fund = {
        '000071.OF': '华夏恒生ETF联接A',
        '000216.OF': '华安黄金易ETF联接A',
        '000614.OF': '华安德国 (DAX)联接 (QDII)A',
        '050025.OF': '博时标普500ETF联接A',
        '270042.OF': '广发纳斯达克100ETF联接人民币'
    }
    weights.rename(columns=dict_fund, inplace=True)
    
    # 保存结果
    results.to_csv(r'Output\MARP_results.csv', encoding='utf-8-sig')
    weights.to_csv(r'Output\MARP_weights.csv', encoding='utf-8-sig')
    
    # ==================== 绘图1：净值曲线与超额净值 ====================
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.plot(results.index, results['strategy'], 'b-', label='策略净值')
    ax1.plot(results.index, results['benchmark'], 'g-', label='基准净值')
    
    # 超额净值（策略净值 / 基准净值）
    excess_value = results['strategy'] / results['benchmark']
    ax1.plot(results.index, excess_value, 'm-', linewidth=1, alpha=0.7, label='超额净值')
    
    # 填充正超额区域
    ax1.fill_between(results.index, 1, excess_value,
                     where=(excess_value > 1), facecolor='pink', alpha=0.5, interpolate=True,
                     label='超额正收益区域')
    
    # 标记调仓点（红色三角形）
    rebalance_dates = results[results['rebalance']].index
    ax1.scatter(rebalance_dates, results.loc[rebalance_dates, 'strategy'],
                marker='^', color='red', s=80, label='调仓日', zorder=10)
    
    ax1.set_title('风险平价策略净值 vs 基准净值', fontsize=15)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('净值', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r'Output\MARP_performance.png', dpi=300)
    plt.show()
    
    # ==================== 绘图2：资产权重变化 ====================
    plt.figure(figsize=(12, 8))
    for asset in weights.columns:
        plt.plot(weights.index, weights[asset], label=asset)
    plt.title('资产配置权重变化', fontsize=15)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('权重', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r'Output\MARP_weights.png', dpi=300)
    plt.show()
    
    # 输出调仓日期列表（供检查）
    print(f"\n实际调仓日期（共{len(rebalance_dates)}个）：")
    for d in rebalance_dates:
        print(d.strftime('%Y-%m-%d'), end='  ')
    print()
