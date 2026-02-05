"""
Black-Scholes Option Pricing Module
Includes Black-Scholes formula, Greeks, and market comparison
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')


class BlackScholesPricer:
    """Black-Scholes option pricing with market comparison"""
    
    def __init__(self, ticker, current_price=None):
        """
        Initialize option pricer for a specific asset
        
        Parameters:
        -----------
        ticker : str
            Asset ticker symbol
        current_price : float, optional
            Current price of the asset. If None, fetched from Yahoo Finance
        """
        self.ticker = ticker
        self.current_price = current_price
        self.option_chain = None
        self.risk_free_rate = 0.03  # 3% risk-free rate (10-year Treasury yield)
        
        if current_price is None:
            self.fetch_current_price()
    
    def fetch_current_price(self):
        """Fetch current price of the asset"""
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                self.current_price = hist['Close'].iloc[-1]
                print(f"Current price of {self.ticker}: ${self.current_price:.2f}")
            else:
                raise ValueError(f"Cannot fetch price for {self.ticker}")
        except Exception as e:
            print(f"Error fetching price for {self.ticker}: {e}")
            self.current_price = 100  # Default fallback
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """
        Black-Scholes option pricing formula
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float: Option price
        """
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # Put option
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)  # Ensure non-negative price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho)
        
        Returns:
        --------
        dict: Dictionary of Greek values
        """
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # Put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # Per day
        else:  # Put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Per day
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in interest rate
        else:  # Put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change in interest rate
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def fetch_option_chain(self, expiration_date=None):
        """
        Fetch option chain data from Yahoo Finance
        """
        try:
            stock = yf.Ticker(self.ticker)
            expirations = stock.options
            
            if not expirations:
                print(f"No option chain data available for {self.ticker}")
                return None
            
            if expiration_date:
                if expiration_date not in expirations:
                    print(f"Warning: {expiration_date} not in available expirations.")
                    print(f"Using nearest expiration: {expirations[0]}")
                    expiration_date = expirations[0]
            else:
                expiration_date = expirations[0]
            
            print(f"Fetching option chain for {self.ticker} with expiration {expiration_date}...")
            
            # Fetch option chain
            chain = stock.option_chain(expiration_date)
            
            # Combine calls and puts
            calls = chain.calls.copy()
            calls['type'] = 'call'
            puts = chain.puts.copy()
            puts['type'] = 'put'
            
            self.option_chain = pd.concat([calls, puts], ignore_index=True)
            
            # Clean and organize data
            self.option_chain['moneyness'] = self.current_price / self.option_chain['strike'] - 1
            
            # NEW: LESS RESTRICTIVE FILTERING
            # Keep options where EITHER bid OR ask is > 0, OR there's volume/open interest
            original_count = len(self.option_chain)
            
            # Calculate mid price (average of bid and ask)
            self.option_chain['mid_price'] = (self.option_chain['bid'] + self.option_chain['ask']) / 2
            
            # Filter: keep options with some indication of liquidity
            self.option_chain = self.option_chain[
                (self.option_chain['bid'] > 0) |  # Has a bid
                (self.option_chain['ask'] > 0) |  # Has an ask
                (self.option_chain['volume'] > 0) |  # Has traded volume
                (self.option_chain['openInterest'] > 10)  # Has open interest
            ]
            
            # Alternative: Keep ALL options and let user decide
            # self.option_chain = self.option_chain  # No filtering
            
            filtered_count = len(self.option_chain)
            
            print(f"✅ Successfully fetched {filtered_count} options")
            if filtered_count < original_count:
                print(f"   (Removed {original_count - filtered_count} completely illiquid options)")
            
            return self.option_chain
            
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None
    
    def calculate_historical_volatility(self, lookback_days=252):
        """
        Calculate historical volatility
        
        Parameters:
        -----------
        lookback_days : int
            Number of trading days to look back
            
        Returns:
        --------
        float: Annualized historical volatility
        """
        try:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period=f"{lookback_days}d")
            
            if len(hist) < 10:
                raise ValueError("Insufficient historical data")
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate annualized volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            print(f"Historical volatility ({lookback_days} days): {annual_vol:.2%}")
            
            return annual_vol
            
        except Exception as e:
            print(f"Error calculating historical volatility: {e}")
            # Return a reasonable default
            return 0.30  # 30% as default
    
    def calculate_black_scholes_prices(self, volatility=None, time_to_expiry=None):
        """
        Calculate Black-Scholes prices for all options in the chain
        
        Parameters:
        -----------
        volatility : float, optional
            Volatility to use. If None, uses historical volatility
        time_to_expiry : float, optional
            Time to expiry in years. If None, uses actual expiry from option chain
            
        Returns:
        --------
        pd.DataFrame: Option chain with Black-Scholes prices added
        """
        if self.option_chain is None:
            print("No option chain data. Fetching option chain first...")
            self.fetch_option_chain()
        
        if volatility is None:
            volatility = self.calculate_historical_volatility()
        
        chain = self.option_chain.copy()
        
        # Calculate time to expiry
        if time_to_expiry is None:
            # Use the actual expiry date from the option chain
            # Note: Yahoo Finance doesn't provide expiry date in the chain, so we'll use average
            # For now, we'll assume 30 days if not specified
            chain['time_to_expiry'] = 30 / 365  # 30 days in years
        else:
            chain['time_to_expiry'] = time_to_expiry
        
        # Calculate Black-Scholes prices
        bs_prices = []
        greeks_list = []
        
        for _, row in chain.iterrows():
            # Get parameters
            S = self.current_price
            K = row['strike']
            T = row['time_to_expiry']
            r = self.risk_free_rate
            sigma = volatility
            option_type = row['type']
            
            # Calculate Black-Scholes price
            bs_price = self.black_scholes(S, K, T, r, sigma, option_type)
            bs_prices.append(bs_price)
            
            # Calculate Greeks
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            greeks_list.append(greeks)
        
        # Add columns to DataFrame
        chain['bs_price'] = bs_prices
        
        # Add market price (midpoint of bid-ask)
        chain['market_price'] = (chain['bid'] + chain['ask']) / 2
        
        # Calculate difference
        chain['price_difference'] = chain['market_price'] - chain['bs_price']
        chain['price_error_pct'] = chain['price_difference'] / chain['market_price'] * 100
        
        # Add Greeks as separate columns
        greeks_df = pd.DataFrame(greeks_list)
        chain = pd.concat([chain, greeks_df], axis=1)
        
        return chain
    
    def plot_comparison(self, chain=None, option_type='all'):
        """
        Plot comparison between Black-Scholes and market prices
        
        Parameters:
        -----------
        chain : pd.DataFrame, optional
            Option chain data with BS prices. If None, calculates them
        option_type : str
            'call', 'put', or 'all'
        """
        if chain is None:
            chain = self.calculate_black_scholes_prices()
        
        # Filter by option type
        if option_type.lower() == 'call':
            data = chain[chain['type'] == 'call']
            title_suffix = 'Calls'
        elif option_type.lower() == 'put':
            data = chain[chain['type'] == 'put']
            title_suffix = 'Puts'
        else:
            data = chain
            title_suffix = 'Calls & Puts'
        
        if data.empty:
            print(f"No {option_type} options to plot")
            return
        
        # Sort by strike
        data = data.sort_values('strike')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Price comparison
        ax1 = axes[0, 0]
        ax1.plot(data['strike'], data['market_price'], 'bo-', label='Market Price', alpha=0.7, markersize=4)
        ax1.plot(data['strike'], data['bs_price'], 'r--', label='Black-Scholes', linewidth=2)
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Option Price')
        ax1.set_title(f'Price Comparison: {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price difference
        ax2 = axes[0, 1]
        colors = ['red' if diff < 0 else 'green' for diff in data['price_difference']]
        ax2.bar(data['strike'], data['price_difference'], color=colors, alpha=0.7)
        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Price Difference (Market - BS)')
        ax2.set_title('Price Difference')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Percentage error
        ax3 = axes[1, 0]
        ax3.scatter(data['strike'], data['price_error_pct'], alpha=0.7)
        ax3.set_xlabel('Strike Price')
        ax3.set_ylabel('Price Error (%)')
        ax3.set_title('Percentage Error (Market vs BS)')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Moneyness vs Error
        ax4 = axes[1, 1]
        scatter = ax4.scatter(data['strike'] / self.current_price, 
                            data['price_error_pct'], 
                            c=data['time_to_expiry'] * 365,
                            cmap='viridis',
                            alpha=0.7)
        ax4.set_xlabel('Strike / Current Price')
        ax4.set_ylabel('Price Error (%)')
        ax4.set_title('Moneyness vs Pricing Error')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Days to Expiry')
        
        plt.suptitle(f'Black-Scholes vs Market Prices: {self.ticker}', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_volatility_smile(self, chain=None):
        """
        Plot implied volatility smile/skew
        
        Parameters:
        -----------
        chain : pd.DataFrame, optional
            Option chain data with implied volatility
        """
        if chain is None:
            chain = self.option_chain
            if chain is None:
                chain = self.fetch_option_chain()
        
        # Separate calls and puts
        calls = chain[chain['type'] == 'call']
        puts = chain[chain['type'] == 'put']
        
        if calls.empty or puts.empty:
            print("Insufficient data for volatility smile plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot calls
        ax1 = axes[0]
        ax1.scatter(calls['strike'], calls['impliedVolatility'] * 100, 
                   alpha=0.7, color='blue', label='Calls')
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title(f'Implied Volatility Smile: Calls ({self.ticker})')
        ax1.axvline(x=self.current_price, color='red', linestyle='--', 
                   label=f'Current Price (${self.current_price:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot puts
        ax2 = axes[1]
        ax2.scatter(puts['strike'], puts['impliedVolatility'] * 100, 
                   alpha=0.7, color='red', label='Puts')
        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Implied Volatility (%)')
        ax2.set_title(f'Implied Volatility Smile: Puts ({self.ticker})')
        ax2.axvline(x=self.current_price, color='blue', linestyle='--', 
                   label=f'Current Price (${self.current_price:.2f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Volatility Smile Analysis: {self.ticker}', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_greeks_analysis(self, chain=None):
        """
        Plot Greeks analysis across strikes
        
        Parameters:
        -----------
        chain : pd.DataFrame, optional
            Option chain data with Greeks calculated
        """
        if chain is None:
            chain = self.calculate_black_scholes_prices()
        
        # Filter for near-the-money options (90-110% of current price)
        near_money = chain[
            (chain['strike'] >= 0.9 * self.current_price) & 
            (chain['strike'] <= 1.1 * self.current_price)
        ].sort_values('strike')
        
        if near_money.empty:
            print("No near-the-money options for Greeks analysis")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        greeks = ['delta', 'gamma', 'vega', 'theta', 'rho']
        titles = ['Delta', 'Gamma', 'Vega', 'Theta (per day)', 'Rho']
        
        for idx, (greek, title) in enumerate(zip(greeks, titles)):
            ax = axes[idx // 3, idx % 3]
            
            # Plot calls
            calls_data = near_money[near_money['type'] == 'call']
            if not calls_data.empty:
                ax.plot(calls_data['strike'], calls_data[greek], 'b-', 
                       label='Calls', linewidth=2)
            
            # Plot puts
            puts_data = near_money[near_money['type'] == 'put']
            if not puts_data.empty:
                ax.plot(puts_data['strike'], puts_data[greek], 'r-', 
                       label='Puts', linewidth=2)
            
            ax.set_xlabel('Strike Price')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Strike')
            ax.axvline(x=self.current_price, color='black', linestyle='--', 
                      label=f'Current Price', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot (2, 2)
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Greeks Analysis: {self.ticker}', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def summary_statistics(self, chain=None):
        """
        Calculate summary statistics for the comparison
        
        Parameters:
        -----------
        chain : pd.DataFrame, optional
            Option chain data with BS prices
        
        Returns:
        --------
        dict: Summary statistics
        """
        if chain is None:
            chain = self.calculate_black_scholes_prices()
        
        stats = {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'num_options': len(chain),
            'num_calls': len(chain[chain['type'] == 'call']),
            'num_puts': len(chain[chain['type'] == 'put']),
            'avg_price_difference': chain['price_difference'].mean(),
            'avg_abs_price_difference': chain['price_difference'].abs().mean(),
            'avg_percentage_error': chain['price_error_pct'].abs().mean(),
            'max_overpriced': chain['price_difference'].max(),
            'max_underpriced': chain['price_difference'].min(),
            'volatility_used': self.calculate_historical_volatility()
        }
        
        return stats