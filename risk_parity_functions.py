"""
Risk Parity Portfolio Optimization Module
Contains all functions for risk parity portfolio construction and analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta


class RiskParityPortfolio:
    """Main class for risk parity portfolio optimization and analysis"""
    
    def __init__(self, tickers, lookback_years=3, end_date=None):
        """
        Initialize portfolio with list of tickers
        
        Parameters:
        -----------
        tickers : list
            List of asset ticker symbols
        lookback_years : int
            Number of years of historical data to use
        end_date : datetime, optional
            End date for historical data. If None, uses current date
        """
        self.tickers = tickers
        self.end_date = end_date if end_date else datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_years*365)
        self.weights = None
        self.data = None
        self.cov_matrix = None
        self.returns = None
        self.risk_contributions = None
        
    def fetch_data(self):
        """Fetch historical price data for all tickers"""
        print(f"Fetching data from {self.start_date.date()} to {self.end_date.date()}...")
        
        try:
            # Download all data at once
            all_data = []
            
            for ticker in self.tickers:
                try:
                    # Download data for individual ticker
                    df = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if df.empty:
                        print(f"  Warning: No data found for {ticker}")
                        continue
                    
                    # Use 'Close' price for all assets (cryptos don't have 'Adj Close')
                    if 'Close' in df.columns:
                        price_series = df['Close']
                    elif 'Adj Close' in df.columns:
                        price_series = df['Adj Close']
                    else:
                        print(f"  Warning: No price data for {ticker}")
                        continue
                    
                    price_series.name = ticker
                    all_data.append(price_series)
                    print(f"  ✓ {ticker}: {len(price_series)} days of data")
                    
                except Exception as e:
                    print(f"  Error fetching {ticker}: {e}")
            
            if not all_data:
                raise ValueError("No assets could be fetched")
            
            # Combine all data into a single DataFrame
            self.data = pd.concat(all_data, axis=1)
            
            # Drop rows with NaN values
            self.data = self.data.dropna()
            
            if len(self.data) < 50:
                raise ValueError(f"Insufficient data. Only {len(self.data)} days of data available.")
            
            print(f"\n✅ Successfully fetched {len(self.data)} days of data for {len(self.data.columns)} assets")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def calculate_returns_and_covariance(self):
        """Calculate returns and covariance matrix"""
        if self.data is None:
            self.fetch_data()
        
        # Calculate daily returns
        self.returns = self.data.pct_change(fill_method=None).dropna()
        
        # Calculate covariance matrix
        self.cov_matrix = self.returns.cov().values
        
        return self.returns, self.cov_matrix
    
    def optimize_weights(self):
        """Calculate risk parity weights using optimization"""
        if self.cov_matrix is None:
            self.calculate_returns_and_covariance()
        
        n_assets = len(self.data.columns)  # Use actual number of assets with data
        
        # Helper functions
        def portfolio_variance(w, cov):
            return w.T @ cov @ w
        
        def risk_contribution(w, cov):
            port_var = portfolio_variance(w, cov)
            marginal_risk = cov @ w
            return w * marginal_risk / port_var
        
        def risk_parity_objective(w, cov):
            risk_contrib = risk_contribution(w, cov)
            target_risk = np.ones_like(w) / len(w)  # Equal risk contribution target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Initial weights (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        bounds = [(0.01, 0.99) for _ in range(n_assets)]  # Min 1%, Max 99% per asset
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            args=(self.cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            self.weights = result.x / np.sum(result.x)  # Ensure weights sum to 1
            self.risk_contributions = risk_contribution(self.weights, self.cov_matrix)
        else:
            print("Optimization failed, using equal weights")
            self.weights = initial_weights
            self.risk_contributions = risk_contribution(self.weights, self.cov_matrix)
        
        return self.weights
    
    def calculate_cumulative_returns(self, lookback_days=252):
        """
        Calculate cumulative returns for portfolio and individual assets
        """
        if self.weights is None:
            self.optimize_weights()
        
        # Use last lookback_days of data
        recent_data = self.data.iloc[-lookback_days:] if len(self.data) > lookback_days else self.data
        recent_returns = recent_data.pct_change(fill_method=None).dropna()
        
        # Calculate portfolio returns
        portfolio_daily_returns = recent_returns.dot(self.weights)
        portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
        
        # Calculate individual asset cumulative returns
        assets_cumulative = (1 + recent_returns).cumprod()
        
        return portfolio_cumulative, assets_cumulative
    
    def get_top_asset(self):
        """Get the asset with the highest weight allocation"""
        if self.weights is None:
            self.optimize_weights()
        
        max_idx = np.argmax(self.weights)
        top_asset = self.data.columns[max_idx]  # Get actual ticker from data
        top_weight = self.weights[max_idx]
        
        return top_asset, top_weight, max_idx
    
    def plot_allocations(self):
        """Plot portfolio weights and risk contributions"""
        if self.weights is None:
            self.optimize_weights()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Portfolio weights pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.weights)))
        axes[0].pie(self.weights, labels=self.data.columns, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[0].set_title('Portfolio Weights', fontsize=14, fontweight='bold')
        
        # Risk contributions bar chart
        x_pos = np.arange(len(self.weights))
        axes[1].bar(x_pos, self.risk_contributions, color=colors)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(self.data.columns, rotation=45, ha='right')
        axes[1].set_title('Risk Contributions', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Risk Contribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_cumulative_returns(self, lookback_days=252):
        """Plot cumulative returns of portfolio vs individual assets"""
        portfolio_cumulative, assets_cumulative = self.calculate_cumulative_returns(lookback_days)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio
        ax.plot(portfolio_cumulative.index, portfolio_cumulative.values,
                label='Risk Parity Portfolio', linewidth=3, color='black')
        
        # Plot individual assets
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.data.columns)))
        for i, ticker in enumerate(self.data.columns):
            ax.plot(assets_cumulative.index, assets_cumulative[ticker].values,
                    label=ticker, alpha=0.7, linewidth=1.5, color=colors[i])
        
        ax.set_title(f'Cumulative Returns (Last {lookback_days} Trading Days)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (Normalized to 1)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def calculate_performance_metrics(self, lookback_days=252):
        """Calculate performance metrics for the portfolio"""
        portfolio_cumulative, assets_cumulative = self.calculate_cumulative_returns(lookback_days)
        
        # Portfolio metrics
        portfolio_returns = portfolio_cumulative.pct_change().dropna()
        total_return = portfolio_cumulative.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252/lookback_days) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_weight': np.max(self.weights),
            'min_weight': np.min(self.weights),
            'num_assets': len(self.weights)
        }
        
        return metrics