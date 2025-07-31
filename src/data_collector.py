"""
Data collection module for financial markets and Magnitsky Act sanctions data
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import yaml
import os

class DataCollector:
    """
    Class to collect financial data, sanctions data, and news sentiment
    """
    
    def __init__(self, config_path="../config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get_financial_data(self, ticker, start_date, end_date):
        """
        Collect financial data using yfinance
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Financial data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error collecting data for {ticker}: {e}")
            return None
    
    def get_ibovespa_data(self, start_date, end_date):
        """Get Ibovespa data specifically"""
        return self.get_financial_data("^BVSP", start_date, end_date)
    
    def get_sp500_data(self, start_date, end_date):
        """Get S&P 500 data for market model"""
        return self.get_financial_data("^GSPC", start_date, end_date)
    
    def get_usd_brl_data(self, start_date, end_date):
        """Get USD/BRL exchange rate data"""
        return self.get_financial_data("BRL=X", start_date, end_date)
    
    def get_vix_data(self, start_date, end_date):
        """Get VIX data for volatility analysis"""
        return self.get_financial_data("^VIX", start_date, end_date)
    
    def calculate_returns(self, data, price_column='Close'):
        """
        Calculate daily returns from price data
        
        Args:
            data (pd.DataFrame): Price data
            price_column (str): Column name for prices
            
        Returns:
            pd.Series: Daily returns
        """
        return data[price_column].pct_change().dropna()
    
    def get_all_market_data(self, start_date, end_date):
        """
        Collect all relevant market data for analysis
        
        Returns:
            dict: Dictionary with all market data
        """
        market_data = {}
        
        # Brazilian market
        market_data['ibovespa'] = self.get_ibovespa_data(start_date, end_date)
        market_data['usd_brl'] = self.get_usd_brl_data(start_date, end_date)
        
        # Global benchmarks
        market_data['sp500'] = self.get_sp500_data(start_date, end_date)
        market_data['vix'] = self.get_vix_data(start_date, end_date)
        
        return market_data
    
    def save_data(self, data, filename, data_dir="../data/raw"):
        """Save data to CSV file"""
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        
        if isinstance(data, dict):
            # Save multiple datasets
            for key, df in data.items():
                if df is not None:
                    df.to_csv(f"{filepath}_{key}.csv")
        else:
            # Save single dataset
            data.to_csv(f"{filepath}.csv")
        
        print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Collect data for the last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Get all market data
    market_data = collector.get_all_market_data(start_date, end_date)
    
    # Save the data
    collector.save_data(market_data, "market_data")
    
    print("Data collection completed!")
