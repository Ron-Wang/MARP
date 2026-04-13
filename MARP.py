import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 风险平价模型函数
def risk_parity_allocation(cov_matrix):
    n = cov_matrix.shape[0]
    
    # 目标函数：最小化风险贡献差异
    def objective(weights):
        weights = np.array(weights)
        port_var = weights.dot(cov_matrix).dot(weights)
        risk_contrib = weights * (cov_matrix.dot(weights)) / port_var
        return np.sum((risk_contrib - risk_contrib.mean())**2)
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: w}             # 权重大于0
    ]
    
    # 初始值（等权重）
    x0 = np.ones(n) / n
    # 优化求解
    res = minimize(objective, x0, method='SLSQP',
                   constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-8})
    
    return res.x

# 读取基金净值数据
def load_fund_data(fund_files):
    """
    加载多个基金净值数据
    fund_files: 基金文件路径列表
    """
    dfs = []
    for file in fund_files:
        # 提取基金代码作为列名
        fund_code = os.path.basename(file).split('_')[1].split('.xls')[0]
        # 读取Excel文件
        df = pd.read_excel(file, sheet_name='每日净值')
        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'])
        # 设置日期索引
        df.set_index('日期', inplace=True)
        # 使用复权单位净值
        df = df[['复权单位净值(元)']].rename(columns={'复权单位净值(元)': fund_code})
        dfs.append(df)
    
    # 合并所有基金数据
    combined = pd.concat(dfs, axis=1).sort_index()
    # 前向填充缺失值
    combined.ffill(inplace=True)
    # 删除仍有缺失的行
    combined.dropna(inplace=True)
    
    return combined

# 读取基准数据
def load_benchmark_data(benchmark_file):
    """
    加载基准指数数据
    """
    df = pd.read_excel(benchmark_file)
    # 重命名列
    df.rename(columns={'日期Date': '日期', '收盘Close': 'close'}, inplace=True)
    # 转换日期格式
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
    # 设置日期索引
    df.set_index('日期', inplace=True)
    # 只保留收盘价
    df = df[['close']].rename(columns={'close': 'benchmark'})
    # 按日期排序
    df.sort_index(inplace=True)
    
    return df

# 风险平价回测函数
def risk_parity_backtest(fund_data, benchmark_data, start_date='2020-01-01', rebalance_freq='6M'):
    """
    风险平价策略回测
    fund_data: 基金净值数据
    benchmark_data: 基准数据
    start_date: 回测开始日期
    rebalance_freq: 调仓频率
    """
    # 过滤回测开始日期之后的数据
    fund_data = fund_data[fund_data.index >= start_date]
    benchmark_data = benchmark_data[benchmark_data.index >= start_date]
    
    # 对齐基金和基准数据
    combined = pd.concat([fund_data, benchmark_data], axis=1).dropna()
    fund_data = combined[fund_data.columns]
    benchmark_data = combined[['benchmark']]
    
    # 计算日收益率
    returns = fund_data.pct_change().dropna()
    
    # 确定调仓日期
    all_dates = fund_data.index
    rebalance_dates = pd.date_range(start=start_date, end=all_dates[-1], freq=rebalance_freq)
    
    # 初始化权重和净值
    n_assets = len(fund_data.columns)
    weights = pd.DataFrame(index=all_dates, columns=fund_data.columns)
    portfolio_value = pd.Series(index=all_dates, dtype=float)
    portfolio_value.iloc[0] = 1.0  # 初始净值设为1
    
    # 初始权重 - 等权重
    current_weights = np.ones(n_assets) / n_assets
    weights.loc[all_dates[0]] = current_weights
    
    # 记录再平衡日期
    rebalance_flags = pd.Series(False, index=all_dates)
    
    # 回测循环
    for i in range(1, len(all_dates)):
        current_date = all_dates[i]
        prev_date = all_dates[i-1]
        
        # 检查是否是调仓日
        if current_date in rebalance_dates:
            # 使用过去252个交易日（约1年）的数据计算协方差
            lookback = 252
            start_idx = max(0, i - lookback)
            cov_matrix = returns.iloc[start_idx:i].cov().values
            
            # 计算风险平价权重
            current_weights = risk_parity_allocation(cov_matrix)
            rebalance_flags.loc[current_date] = True
        
        # 记录当前权重
        weights.loc[current_date] = current_weights
        
        # 计算组合收益率
        asset_returns = fund_data.loc[current_date] / fund_data.loc[prev_date] - 1
        portfolio_return = np.dot(current_weights, asset_returns)
        
        # 计算组合净值
        portfolio_value.loc[current_date] = portfolio_value.loc[prev_date] * (1 + portfolio_return)
    
    # 创建结果数据框
    results = pd.DataFrame({
        'strategy': portfolio_value,
        'benchmark': benchmark_data['benchmark'] / benchmark_data['benchmark'].iloc[0]
    }, index=all_dates)
    
    # 计算超额收益
    results['excess_return'] = results['strategy'] - results['benchmark']
    
    # 标记再平衡日期
    results['rebalance'] = rebalance_flags
    
    return results, weights

# 计算绩效指标
def calculate_performance_metrics(results):
    """
    计算策略绩效指标
    """
    # 计算日收益率
    daily_returns = results[['strategy', 'benchmark']].pct_change().dropna()
    # 年化收益率
    annual_returns = (1 + daily_returns.mean()) ** 252 - 1
    # 年化波动率
    annual_volatility = daily_returns.std() * np.sqrt(252)
    # 夏普比率 (假设无风险利率为0)
    sharpe_ratio = annual_returns / annual_volatility
    # 最大回撤
    max_drawdown = (results[['strategy', 'benchmark']] / 
                   results[['strategy', 'benchmark']].cummax() - 1).min()
    # 超额收益年化
    excess_return_annual = annual_returns['strategy'] - annual_returns['benchmark']
    # 信息比率
    active_return = daily_returns['strategy'] - daily_returns['benchmark']
    information_ratio = active_return.mean() / active_return.std() * np.sqrt(252)
    # 汇总指标
    metrics = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    })
    
    # 添加额外指标
    metrics.loc['excess_return'] = [excess_return_annual, np.nan, information_ratio, np.nan]
    
    return metrics

# 主程序
if __name__ == "__main__":
    
    # 基金文件列表
    fund_files = [
        r'MA_data\每日基金净值与行情_000071.OF.xls',  # 华夏恒生ETF联接A
        r'MA_data\每日基金净值与行情_000216.OF.xls',  # 华安黄金易ETF联接A
        r'MA_data\每日基金净值与行情_000614.OF.xls',  # 华安德国 (DAX)联接 (QDII)A
        r'MA_data\每日基金净值与行情_050025.OF.xls',  # 博时标普500ETF联接A
        r'MA_data\每日基金净值与行情_270042.OF.xls'   # 广发纳斯达克100ETF联接人民币 (QDII)A
    ]    
    # 基准文件
    benchmark_file = r'Benchmark\930929perf.xlsx'
    
    # 加载数据
    print("加载基金数据...")
    fund_data = load_fund_data(fund_files)
    print("加载基准数据...")
    benchmark_data = load_benchmark_data(benchmark_file)
    
    # 运行回测
    print("运行风险平价回测...")
    results, weights = risk_parity_backtest(fund_data, benchmark_data, start_date='2020-01-01')
    
    # 计算绩效指标
    metrics = calculate_performance_metrics(results)
    
    # 输出结果
    print("\n策略绩效指标:")
    print(metrics)
    
    # 基金代码替换为名称
    dict_fund = {'000071.OF':'华夏恒生ETF联接A', 
                 '000216.OF':'华安黄金易ETF联接A', 
                 '000614.OF':'华安德国 (DAX)联接 (QDII)A', 
                 '050025.OF':'博时标普500ETF联接A', 
                 '270042.OF':'广发纳斯达克100ETF联接人民币'}
    weights.rename(columns=dict_fund, inplace=True)
    
    # 保存结果
    results.to_csv(r'Output\MARP_results.csv', encoding='utf-8-sig')
    weights.to_csv(r'Output\MARP_weights.csv', encoding='utf-8-sig')
    
    ###### 绘制净值曲线和超额净值 ###### 
    plt.figure(figsize=(12, 8))
    
    # 创建双y轴
    ax1 = plt.gca()
    
    # 绘制净值曲线
    ax1.plot(results.index, results['strategy'], 'b-', label='策略净值')
    ax1.plot(results.index, results['benchmark'], 'g-', label='基准净值')
    
    # 计算超额净值
    excess_value = results['strategy'] / results['benchmark']
    
    # 绘制超额净值曲线
    ax1.plot(results.index, excess_value, 'm-', linewidth=0, alpha=0.7, label='超额净值')
    
    # 填充正超额净值区域（粉色）
    ax1.fill_between(
        results.index, 
        0,  # 基准线为0
        excess_value, 
        where=(excess_value > 0),
        facecolor='pink', 
        alpha=0.8, 
        interpolate=True
    )
    
    # 标记再平衡点
    rebalance_dates = results[results['rebalance']].index
    ax1.scatter(
        rebalance_dates, 
        results.loc[rebalance_dates, 'strategy'], 
        marker='^', 
        color='red', 
        s=80, 
        label='调仓日',
        zorder=10
    )
    
    # 设置标签和图例
    ax1.set_title('风险平价策略净值 vs 基准净值', fontsize=15)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('净值', fontsize=12)
    
    # 图例
    ax1.legend(loc='upper left', fontsize=10)
    # 网格线
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(r'Output\MARP_performance.png', dpi=300)
    plt.show()
    
    ###### 绘制权重变化 ###### 
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
    