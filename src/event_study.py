"""
Event study analysis module for measuring market impact of sanctions
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class EventStudy:
    """
    Class to perform event study analysis
    """
    
    def __init__(self, event_date, estimation_window=120, event_window_start=-10, event_window_end=30):
        """
        Initialize event study parameters
        
        Args:
            event_date (str): Date of the event in YYYY-MM-DD format
            estimation_window (int): Number of days for estimation period
            event_window_start (int): Start of event window (days before event)
            event_window_end (int): End of event window (days after event)
        """
        self.event_date = pd.to_datetime(event_date)
        self.estimation_window = estimation_window
        self.event_window_start = event_window_start
        self.event_window_end = event_window_end
        
        # Define estimation period (ends 11 days before event)
        self.estimation_end = self.event_date + timedelta(days=-11)
        self.estimation_start = self.estimation_end - timedelta(days=estimation_window)
        
        # Define event window
        self.event_start = self.event_date + timedelta(days=event_window_start)
        self.event_end = self.event_date + timedelta(days=event_window_end)
    
    def estimate_market_model(self, target_returns, market_returns):
        """
        Estimate market model (CAPM) parameters during estimation period
        
        Args:
            target_returns (pd.Series): Returns of target asset (e.g., Ibovespa)
            market_returns (pd.Series): Returns of market benchmark (e.g., S&P 500)
            
        Returns:
            dict: Dictionary with alpha, beta, and residuals
        """
        # Filter data for estimation period
        estimation_mask = (target_returns.index >= self.estimation_start) & \
                         (target_returns.index <= self.estimation_end)
        
        target_est = target_returns[estimation_mask]
        market_est = market_returns[estimation_mask]
        
        # Align the series
        aligned_data = pd.concat([target_est, market_est], axis=1, join='inner')
        aligned_data.columns = ['target', 'market']
        aligned_data = aligned_data.dropna()
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned_data['market'], aligned_data['target']
        )
        
        # Calculate residuals
        predicted = intercept + slope * aligned_data['market']
        residuals = aligned_data['target'] - predicted
        residual_std = residuals.std()
        
        return {
            'alpha': intercept,
            'beta': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'residual_std': residual_std,
            'residuals': residuals
        }
    
    def calculate_abnormal_returns(self, target_returns, market_returns, model_params):
        """
        Calculate abnormal returns during event window
        
        Args:
            target_returns (pd.Series): Target asset returns
            market_returns (pd.Series): Market benchmark returns
            model_params (dict): Market model parameters from estimation
            
        Returns:
            pd.Series: Abnormal returns during event window
        """
        # Filter data for event window
        event_mask = (target_returns.index >= self.event_start) & \
                    (target_returns.index <= self.event_end)
        
        target_event = target_returns[event_mask]
        market_event = market_returns[event_mask]
        
        # Align the series
        aligned_data = pd.concat([target_event, market_event], axis=1, join='inner')
        aligned_data.columns = ['target', 'market']
        aligned_data = aligned_data.dropna()
        
        # Calculate expected returns using market model
        expected_returns = model_params['alpha'] + model_params['beta'] * aligned_data['market']
        
        # Calculate abnormal returns
        abnormal_returns = aligned_data['target'] - expected_returns
        
        return abnormal_returns
    
    def calculate_cumulative_abnormal_returns(self, abnormal_returns):
        """
        Calculate cumulative abnormal returns (CAR)
        
        Args:
            abnormal_returns (pd.Series): Daily abnormal returns
            
        Returns:
            pd.Series: Cumulative abnormal returns
        """
        return abnormal_returns.cumsum()
    
    def test_significance(self, abnormal_returns, residual_std):
        """
        Test statistical significance of abnormal returns
        
        Args:
            abnormal_returns (pd.Series): Abnormal returns
            residual_std (float): Standard deviation of residuals from estimation
            
        Returns:
            dict: Test statistics and p-values
        """
        n_obs = len(abnormal_returns)
        
        # T-statistics for each day
        t_stats = abnormal_returns / residual_std
        p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=n_obs-2))
        
        # Cumulative abnormal returns
        car = self.calculate_cumulative_abnormal_returns(abnormal_returns)
        
        # T-statistic for CAR
        car_std = residual_std * np.sqrt(n_obs)
        car_t_stat = car.iloc[-1] / car_std
        car_p_value = 2 * (1 - stats.t.cdf(abs(car_t_stat), df=n_obs-2))
        
        return {
            'daily_t_stats': t_stats,
            'daily_p_values': p_values,
            'car_t_stat': car_t_stat,
            'car_p_value': car_p_value,
            'car_final': car.iloc[-1]
        }
    
    def run_full_analysis(self, target_returns, market_returns):
        """
        Run complete event study analysis
        
        Args:
            target_returns (pd.Series): Target asset returns
            market_returns (pd.Series): Market benchmark returns
            
        Returns:
            dict: Complete analysis results
        """
        # Step 1: Estimate market model
        model_params = self.estimate_market_model(target_returns, market_returns)
        
        # Step 2: Calculate abnormal returns
        abnormal_returns = self.calculate_abnormal_returns(
            target_returns, market_returns, model_params
        )
        
        # Step 3: Calculate cumulative abnormal returns
        car = self.calculate_cumulative_abnormal_returns(abnormal_returns)
        
        # Step 4: Test significance
        significance_tests = self.test_significance(
            abnormal_returns, model_params['residual_std']
        )
        
        return {
            'model_parameters': model_params,
            'abnormal_returns': abnormal_returns,
            'cumulative_abnormal_returns': car,
            'significance_tests': significance_tests
        }
    
    def plot_results(self, results):
        """
        Plot event study results
        
        Args:
            results (dict): Results from run_full_analysis
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot abnormal returns
        ar = results['abnormal_returns']
        ax1.bar(range(len(ar)), ar.values, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=-self.event_window_start, color='red', linestyle='--', alpha=0.7, label='Event Date')
        ax1.set_title('Daily Abnormal Returns')
        ax1.set_xlabel('Days Relative to Event')
        ax1.set_ylabel('Abnormal Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative abnormal returns
        car = results['cumulative_abnormal_returns']
        ax2.plot(range(len(car)), car.values, linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=-self.event_window_start, color='red', linestyle='--', alpha=0.7, label='Event Date')
        ax2.set_title('Cumulative Abnormal Returns (CAR)')
        ax2.set_xlabel('Days Relative to Event')
        ax2.set_ylabel('Cumulative Abnormal Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

if __name__ == "__main__":
    # Example usage would go here
    print("Event Study module loaded successfully")
