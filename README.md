# MARP
# 风险平价基金组合策略（Risk Parity）
基于**风险平价模型**的多资产基金组合回测框架，实现自动调仓、绩效分析与可视化输出，适用于全球大类资产配置（股票/商品/海外指数等）。

## 核心功能
- ✅ **风险平价权重计算**：让组合中每类资产对组合总风险的贡献相等
- ✅ **滚动窗口回测**：支持自定义调仓周期（月/季/半年）
- ✅ **基准对比**：支持接入指数基准，计算超额收益与信息比率
- ✅ **完整绩效指标**：年化收益、波动率、夏普比率、最大回撤、信息比率
- ✅ **可视化输出**：净值曲线+调仓标记、资产权重变化趋势图
- ✅ **结果自动保存**：净值、权重、图片一键导出

## 项目目录结构
```
项目根目录/
├── MA_data/            # 基金每日净值数据（Excel）
├── Benchmark/          # 基准指数数据（Excel）
├── Output/             # 输出结果（自动生成）
│   ├── MARP_results.csv    # 策略+基准+超额净值
│   ├── MARP_weights.csv    # 每日资产权重
│   ├── MARP_performance.png  # 净值曲线图
│   └── MARP_weights.png     # 权重变化图
└── MARP.py      # 主程序代码
```

## 数据格式要求
### 基金数据
- 格式：Excel（.xls）
- Sheet：`每日净值`
- 字段：`日期`、`复权单位净值(元)`
- 命名示例：`每日基金净值与行情_000071.OF.xls`

### 基准数据
- 格式：Excel（.xlsx）
- 字段：`日期Date`（YYYYMMDD）、`收盘Close`

## 环境依赖
```bash
pip install pandas numpy matplotlib scipy openpyxl
```

## 快速使用
### 1. 修改文件路径
在 `if __name__ == "__main__":` 中配置你的文件路径：
```python
fund_files = [
    r"MA_data\每日基金净值与行情_000071.OF.xls",
    r"MA_data\每日基金净值与行情_000216.OF.xls",
    ...
]
benchmark_file = r"Benchmark\930929perf.xlsx"
```

### 2. 运行策略
```bash
python risk_parity.py
```

### 3. 核心参数（可自定义）
```python
# 回测开始日期
start_date='2020-01-01'

# 调仓频率：6M=半年，3M=季度，1M=月度
rebalance_freq='6M'

# 回看协方差窗口（默认252交易日=1年）
lookback = 252
```

## 策略原理
### 风险平价（Risk Parity）
目标：让组合中**每个资产的风险贡献相等**
- 优化目标：最小化风险贡献离差
- 约束：权重和为1、权重非负
- 优势：不依赖收益预测，仅用协方差矩阵，稳健性强

数学目标：
```
min Σ(rc_i - rc_mean)²
s.t. Σw=1, w≥0
```

## 输出结果说明
### 1. 策略净值曲线
- 蓝色：策略净值
- 绿色：基准净值
- 粉色填充：超额收益
- 红色三角：调仓日期

### 2. 资产权重变化
展示每只基金在组合中的**动态权重**，清晰反映风险平价调仓逻辑。

### 3. 绩效指标表
| 指标 | 说明 |
|------|------|
| Annual Return | 年化收益率 |
| Annual Volatility | 年化波动率 |
| Sharpe Ratio | 夏普比率 |
| Max Drawdown | 最大回撤 |
| excess_return | 年化超额收益 |
| information_ratio | 信息比率 |

## 核心函数
- `risk_parity_allocation()`：风险平价优化求解
- `load_fund_data()`：批量读取基金净值
- `load_benchmark_data()`：读取基准指数
- `risk_parity_backtest()`：滚动回测主函数
- `calculate_performance_metrics()`：绩效指标计算

## 适用场景
- 全球多资产基金配置
- 低波动、风险分散组合构建
- 基金投顾策略研发与回测
- 量化资产配置研究

## 默认配置（示例）
- 资产：恒生ETF + 黄金ETF + 德国DAX + 标普500 + 纳斯达克100
- 回测区间：2020-01-01 至今
- 调仓频率：每半年一次
- 协方差窗口：1年（252交易日）

---

## 一键运行示例代码
```python
if __name__ == "__main__":
    fund_files = [
        r'MA_data\每日基金净值与行情_000071.OF.xls',
        r'MA_data\每日基金净值与行情_000216.OF.xls',
        r'MA_data\每日基金净值与行情_000614.OF.xls',
        r'MA_data\每日基金净值与行情_050025.OF.xls',
        r'MA_data\每日基金净值与行情_270042.OF.xls'
    ]
    benchmark_file = r'Benchmark\930929perf.xlsx'

    fund_data = load_fund_data(fund_files)
    benchmark_data = load_benchmark_data(benchmark_file)

    results, weights = risk_parity_backtest(
        fund_data, benchmark_data,
        start_date='2020-01-01',
        rebalance_freq='6M'
    )

    metrics = calculate_performance_metrics(results)
    print(metrics)

    # 自动保存 + 绘图
