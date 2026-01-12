
# 📈 Tadawul Stock Analysis Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Finance](https://img.shields.io/badge/Finance-Quant%20Analysis-yellow)

A comprehensive Python tool for analyzing Saudi stock market (Tadawul) data with advanced financial metrics and visualizations. Built as a portfolio project for Financial Engineering roles.

## ✨ Features

- **📊 Data Retrieval**: Historical stock data from Yahoo Finance API
- **📈 Technical Analysis**: RSI, Moving Averages, Bollinger Bands, MACD
- **📉 Risk Metrics**: Volatility, Sharpe Ratio, Value at Risk (VaR)
- **📱 Interactive Visualizations**: Plotly charts with real-time updates
- **🔗 Multi-Stock Comparison**: Correlation matrix and performance comparison
- **💼 Portfolio Simulation**: Monte Carlo simulation and optimization
- **📤 Export Capabilities**: CSV, Excel, PDF reports

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tadawul-stock-analyzer.git
cd tadawul-stock-analyzer

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Basic Usage

```python
# Analyze a single stock
python main.py --stock 2222.SR --period 6mo

# Compare multiple stocks
python main.py --compare 2222.SR 2010.SR 7010.SR

# Generate portfolio report
python portfolio_analysis.py

# Run with specific parameters
python main.py --stock 1120.SR --period 1y --output report.html
```

## 📁 Project Structure
```
tadawul-stock-analyzer/
├── src/                    # Source modules
│   ├── data_fetcher.py    # Data collection from Yahoo Finance
│   ├── technical.py       # Technical indicators calculation
│   ├── risk_metrics.py    # Risk analysis and metrics
│   ├── portfolio.py       # Portfolio simulation and optimization
│   └── visualizer.py      # Interactive chart generation
├── notebooks/             # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   ├── technical_analysis.ipynb
│   └── portfolio_simulation.ipynb
├── tests/                # Unit tests
│   ├── test_data_fetcher.py
│   └── test_technical.py
├── data/                 # Sample data and cache
│   ├── sample_portfolio.csv
│   └── cache/
├── results/              # Generated outputs
│   ├── charts/
│   ├── reports/
│   └── exports/
├── docs/                 # Documentation
│   ├── api.md
│   └── user_guide.md
├── main.py              # Main entry point
├── portfolio_analysis.py # Portfolio analysis script
├── requirements.txt     # Python dependencies
├── LICENSE             # MIT License
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## 📊 Supported Analysis

### Technical Indicators
- Simple Moving Average (SMA) - 20, 50, 200 days
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI) - Overbought/Oversold signals
- Bollinger Bands with volatility indicators
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Fibonacci Retracement levels
- Volume Weighted Average Price (VWAP)

### 📉 Risk Metrics
- **Daily/Annual Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall (CVaR)**: Average loss beyond VaR
- **Beta Coefficient**: Sensitivity to market movements
- **Alpha**: Excess return relative to benchmark

### 💼 Portfolio Analysis
- **Efficient Frontier**: Optimal risk-return portfolios
- **Monte Carlo Simulation**: 10,000+ random portfolio scenarios
- **Portfolio Optimization**: Markowitz mean-variance optimization
- **Correlation Analysis**: Heatmap of stock correlations
- **Diversification Metrics**: Portfolio concentration and diversity
- **Rebalancing Strategies**: Periodic portfolio rebalancing
- **Performance Attribution**: Return decomposition by asset
- **Stress Testing**: Performance under market shocks

## 🏦 Supported Saudi Stocks

| Stock | Symbol | Sector | Market Cap |
|-------|--------|--------|------------|
| Saudi Aramco | 2222.SR | Energy | ~$2.1T |
| SABIC | 2010.SR | Materials | ~$80B |
| Al Rajhi Bank | 1120.SR | Financials | ~$90B |
| Saudi National Bank | 1180.SR | Financials | ~$70B |
| STC | 7010.SR | Telecommunications | ~$50B |
| Ma'aden | 1211.SR | Materials | ~$40B |
| Alinma Bank | 1150.SR | Financials | ~$20B |
| Riyad Bank | 1010.SR | Financials | ~$30B |
| Saudi Electricity | 5110.SR | Utilities | ~$25B |
| Jarir Marketing | 4190.SR | Consumer Discretionary | ~$10B |
| Savola Group | 2050.SR | Consumer Staples | ~$8B |
| Mobily | 7020.SR | Telecommunications | ~$12B |

> **Note**: Currently supports 50+ Tadawul listed companies with real-time data.

## 🛠️ Technologies Used

### Core Stack
- **Python 3.8+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Static data visualizations
- **Plotly & Dash**: Interactive web-based charts and dashboards
- **yFinance**: Yahoo Finance API wrapper for market data
- **SciPy & Statsmodels**: Statistical analysis and modeling

### Advanced Features
- **Scikit-learn**: Machine learning for price prediction
- **Streamlit**: Rapid web application deployment
- **Jupyter**: Interactive notebooks for analysis
- **SQLAlchemy**: Database integration (optional)
- **FastAPI**: REST API for data access (planned)

### Development Tools
- **Git & GitHub**: Version control and collaboration
- **PyCharm**: IDE for Python development
- **Docker**: Containerization (optional)
- **pytest**: Testing framework
- **Black & Flake8**: Code formatting and linting

## 📈 Sample Outputs

### 1. Stock Price Analysis
![Stock Analysis](https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=Stock+Price+with+Moving+Averages+and+Bollinger+Bands)

### 2. Technical Indicators
![Technical Indicators](https://via.placeholder.com/800x400/50E3C2/FFFFFF?text=RSI+MACD+and+Volume+Analysis)

### 3. Portfolio Simulation
![Portfolio Simulation](https://via.placeholder.com/800x400/9013FE/FFFFFF?text=Monte+Carlo+Portfolio+Simulation+with+Efficient+Frontier)

### 4. Correlation Matrix
![Correlation Matrix](https://via.placeholder.com/800x400/F5A623/FFFFFF?text=Stock+Correlation+Heatmap+Analysis)

### 5. Risk Metrics Dashboard
![Risk Dashboard](https://via.placeholder.com/800x400/B8E986/FFFFFF?text=Value+at+Risk+and+Maximum+Drawdown+Analysis)

## 📚 API Reference

### Data Fetching
```python
from src.data_fetcher import get_stock_data

# Get historical data
data = get_stock_data('2222.SR', period='1y', interval='1d')

# Get multiple stocks
portfolio_data = get_stock_data(['2222.SR', '2010.SR'], period='6mo')
```

### Technical Analysis
```python
from src.technical import calculate_indicators

# Calculate all indicators
indicators = calculate_indicators(data, 
                                 ma_periods=[20, 50, 200],
                                 rsi_period=14,
                                 bb_period=20)
```

### Risk Analysis
```python
from src.risk_metrics import calculate_var

# Calculate Value at Risk
var_95 = calculate_var(returns, confidence_level=0.95)
var_99 = calculate_var(returns, confidence_level=0.99)
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_data_fetcher.py

# Run with coverage report
pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Reporting Issues

Found a bug? Please open an issue on GitHub with:
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable

## 👨‍💻 Author

**REDHWAN MOHAMMED** - Financial Engineering
- 📧 Email: malekdxn9@gmail.com  
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)  
- 🐙 GitHub: [@REDHWAN MOHAMMED]([https://github.com/YourUsername](https://github.com/REDHWAN0))  

**Education**:  
- Master's in Financial Engineering  
- Master's in Business Administration 
- Certifications: Data Science, Data Analytics, Machine Learining, Deep Learning, Sql, and Python  

**Skills**:  
- Quantitative Analysis & Financial Modeling  
- Python Programming & Data Science  
- Risk Management & Portfolio Optimization  
- Machine Learning in Finance  

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 REDHWAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

Special thanks to:

- **Yahoo Finance** for providing free market data API
- **Python financial community** for open-source libraries and tools
- **Saudi Stock Exchange (Tadawul)** for market transparency
- **Contributors** who have helped improve this project
- **Financial Engineering professors** for their guidance and knowledge
- **Open-source community** for inspiration and best practices

### Libraries Used
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance market data downloader
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [plotly](https://plotly.com/python/) - Interactive graphing library
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python

### Inspiration
- QuantConnect and Quantopian for quantitative finance platforms
- Bloomberg Terminal for financial data visualization
- Academic research in financial engineering and risk management

### Educational Resources
- WorldQuant University. Financial Engineering Master: Courses Materials. 
- Coursera courses in Data Science, Data Analytics, and financial engineering
- Financial mathematics textbooks and papers

---

## 📊 Project Status

| Component | Status | Progress |
|-----------|--------|----------|
| Data Fetching | ✅ Complete | 100% |
| Technical Indicators | ✅ Complete | 100% |
| Risk Metrics | ✅ Complete | 100% |
| Portfolio Analysis | 🟡 In Progress | 80% |
| Web Dashboard | 🟡 Planned | 40% |
| API Development | 🔴 Future | 10% |
| Documentation | 🟡 In Progress | 70% |

**Last Updated**: January 2024  
**Version**: 1.0.0  
**Python Version**: 3.8+

---

<div align="center">
  
### ⭐ If you liked the project, don't forget to give it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=YourUsername/tadawul-stock-analyzer&type=Date)](https://star-history.com/#YourUsername/tadawul-stock-analyzer&Date)

</div>
```
