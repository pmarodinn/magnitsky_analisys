"""
Machine Learning module for clustering and prediction analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class MagnitskyMLAnalysis:
    """
    Machine Learning analysis for Magnitsky sanctions impact
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.prediction_model = None
    
    def prepare_clustering_features(self, sanctions_data):
        """
        Prepare features for clustering analysis
        
        Args:
            sanctions_data (pd.DataFrame): Historical sanctions data with features
            
        Returns:
            pd.DataFrame: Prepared features for clustering
        """
        # Define clustering features as described in the methodology
        clustering_features = [
            'CAR_magnitude',      # CAR[-5, +5] magnitude
            'volatility_spike',   # Volatility change
            'profile_score',      # Individual profile score (1-4)
            'country_risk',       # Political risk score
            'market_cap_gdp'      # Market importance ratio
        ]
        
        # Check if all features exist
        available_features = [f for f in clustering_features if f in sanctions_data.columns]
        
        if len(available_features) != len(clustering_features):
            missing = set(clustering_features) - set(available_features)
            print(f"Warning: Missing features for clustering: {missing}")
        
        # Use available features
        features_df = sanctions_data[available_features].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        return features_df
    
    def perform_clustering(self, features_df, n_clusters_range=[2, 3, 4, 5]):
        """
        Perform K-means clustering with different numbers of clusters
        
        Args:
            features_df (pd.DataFrame): Features for clustering
            n_clusters_range (list): Range of cluster numbers to test
            
        Returns:
            dict: Clustering results with different k values
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_df)
        
        clustering_results = {}
        
        for n_clusters in n_clusters_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            
            clustering_results[n_clusters] = {
                'model': kmeans,
                'labels': cluster_labels,
                'inertia': inertia,
                'cluster_centers': kmeans.cluster_centers_
            }
        
        # Find optimal number of clusters using elbow method
        inertias = [clustering_results[k]['inertia'] for k in n_clusters_range]
        optimal_k = self._find_elbow(n_clusters_range, inertias)
        
        # Store the best model
        self.clustering_model = clustering_results[optimal_k]['model']
        
        return clustering_results, optimal_k
    
    def _find_elbow(self, k_values, inertias):
        """
        Find elbow point for optimal number of clusters
        """
        # Simple elbow detection - can be improved with more sophisticated methods
        if len(k_values) < 3:
            return k_values[0]
        
        # Calculate second derivative
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # Find the point with maximum second derivative
        elbow_idx = np.argmax(second_derivatives) + 1
        return k_values[elbow_idx]
    
    def analyze_clusters(self, features_df, clustering_results, optimal_k):
        """
        Analyze the characteristics of each cluster
        
        Args:
            features_df (pd.DataFrame): Original features
            clustering_results (dict): Results from clustering
            optimal_k (int): Optimal number of clusters
            
        Returns:
            pd.DataFrame: Cluster analysis summary
        """
        labels = clustering_results[optimal_k]['labels']
        
        # Add cluster labels to features
        features_with_clusters = features_df.copy()
        features_with_clusters['cluster'] = labels
        
        # Calculate cluster statistics
        cluster_summary = features_with_clusters.groupby('cluster').agg({
            col: ['mean', 'std', 'count'] for col in features_df.columns
        }).round(3)
        
        return cluster_summary
    
    def prepare_prediction_features(self, sanctions_data):
        """
        Prepare features for prediction model
        
        Args:
            sanctions_data (pd.DataFrame): Historical sanctions data
            
        Returns:
            tuple: (X, y) features and target
        """
        # Prediction features (more comprehensive than clustering)
        prediction_features = [
            'profile_score',
            'country_risk', 
            'market_cap_gdp',
            'media_sentiment_score',
            'social_media_volume',
            'polarization_index',
            'vix_level',
            'usd_exchange_trend'
        ]
        
        # Target variable
        target = 'CAR_5_days'
        
        # Check available features
        available_features = [f for f in prediction_features if f in sanctions_data.columns]
        
        if target not in sanctions_data.columns:
            raise ValueError(f"Target variable '{target}' not found in data")
        
        # Prepare features and target
        X = sanctions_data[available_features].copy()
        y = sanctions_data[target].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove rows where target is missing
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train_prediction_model(self, X, y, model_type='random_forest'):
        """
        Train prediction model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            model_type (str): Type of model to use
            
        Returns:
            dict: Model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Store model
        self.prediction_model = model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
    
    def predict_brazil_scenario(self, brazil_features):
        """
        Predict impact for Brazil scenario
        
        Args:
            brazil_features (dict): Features for Brazil scenario
            
        Returns:
            dict: Prediction results for different scenarios
        """
        if self.prediction_model is None:
            raise ValueError("Prediction model not trained yet")
        
        # Create scenarios
        scenarios = {
            'optimistic': brazil_features.copy(),
            'base': brazil_features.copy(),
            'pessimistic': brazil_features.copy()
        }
        
        # Adjust sentiment features for scenarios
        scenarios['optimistic']['media_sentiment_score'] = 0.1   # Slightly positive
        scenarios['base']['media_sentiment_score'] = -0.3        # Negative
        scenarios['pessimistic']['media_sentiment_score'] = -0.8 # Very negative
        
        scenarios['optimistic']['polarization_index'] = 0.3     # Low polarization
        scenarios['base']['polarization_index'] = 0.6           # Moderate
        scenarios['pessimistic']['polarization_index'] = 0.9    # High polarization
        
        predictions = {}
        
        for scenario_name, features in scenarios.items():
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.prediction_model.predict(features_scaled)[0]
            predictions[scenario_name] = prediction
        
        return predictions
    
    def plot_clustering_results(self, features_df, clustering_results, optimal_k):
        """
        Plot clustering results
        """
        labels = clustering_results[optimal_k]['labels']
        
        # Create scatter plot of first two features colored by cluster
        plt.figure(figsize=(10, 8))
        
        feature_cols = features_df.columns[:2]  # Use first two features
        
        scatter = plt.scatter(
            features_df.iloc[:, 0], 
            features_df.iloc[:, 1], 
            c=labels, 
            cmap='viridis',
            alpha=0.7
        )
        
        plt.xlabel(feature_cols[0])
        plt.ylabel(feature_cols[1])
        plt.title(f'Clustering Results (k={optimal_k})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    print("Machine Learning module loaded successfully")
