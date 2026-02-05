# Algorithmic Finance: Risk Parity & Black-Scholes Analysis 📊⚖️

This repository implements a professional quantitative workflow in Python, bridging the gap between modern portfolio management and derivative pricing. 

The project was developed for the **Programming Techniques** course in the **MSc in Mathematical Finance at ISEG**.

## 🔄 Project Workflow

The system is built using a modular Object-Oriented Programming (OOP) approach, divided into two main stages:

### Phase 1: Risk Parity Portfolio Optimization
Using the `RiskParityPortfolio` class, the system constructs a portfolio where risk is distributed equally among all assets, regardless of their individual volatility.
* **Risk Contribution Analysis:** Calculates the marginal risk contribution of each asset.
* **Numerical Optimization:** Uses `scipy.optimize` to find weights that minimize the difference between risk contributions.
* **Dynamic Selection:** Automatically identifies the asset with the **highest allocation** (the "safest" or least volatile asset in the risk parity context) to proceed to the options analysis.

### Phase 2: Advanced Options Analysis (Black-Scholes)
Using the `BlackScholesPricer` class, the system performs a deep dive into the derivatives of the top-ranked asset:
* **Analytical Pricing:** Calculates European Call and Put prices.
* **Greeks Analysis:** Full computation of **Delta, Gamma, Theta, Vega, and Rho**, essential for risk hedging.
* **Market Comparison:** Fetches real-time option chains via `yfinance` to compare theoretical Black-Scholes prices with actual market bids and asks.



---

## 🛠️ Tech Stack
* **Python:** Core implementation using OOP (Classes).
* **NumPy & Pandas:** High-performance data structures and matrix operations.
* **SciPy:** Numerical optimization and statistical functions (norm CDF).
* **yfinance:** Real-time financial data acquisition.
* **Matplotlib:** Visualization of risk distributions, cumulative returns, and Greeks.

---

## 📂 Project Structure
* `risk_parity_and_black_scholes.ipynb`: The main entry point. A Jupyter Notebook that executes the full pipeline with visualizations.
* `risk_parity_functions.py`: Module containing the `RiskParityPortfolio` class logic.
* `black_scholes_functions.py`: Module containing the `BlackScholesPricer` class and Greeks calculations.

---

## 🚀 How to Run
1. Ensure you have the dependencies installed: `pip install numpy pandas scipy yfinance matplotlib`.
2. Open the `risk_parity_and_black_scholes.ipynb` notebook.
3. Run the cells to see the portfolio optimization and the subsequent options market analysis.

---
**Author:** [Petr Terletskiy](https://www.linkedin.com/in/petr-terletskiy/)  
**Context:** MSc in Mathematical Finance 
