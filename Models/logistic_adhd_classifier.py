"""
Logistic Regression Model for ADHD Binary Classification
Replaces the complex cross-attention model with a simpler, more interpretable logistic regression approach
Features: 12 extracted features (from feature_pipeline.pkl)
Task: Binary classification with threshold = 0.5
Output: Probability of ADHD (class 1)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')


class ADHDLogisticClassifier:
    """
    Logistic Regression Classifier for ADHD Binary Classification
    
    Attributes:
        model: Trained LogisticRegression instance
        scaler: StandardScaler for feature normalization
        threshold: Classification threshold (default: 0.5)
        feature_names: List of 12 feature names
        feature_weights: Coefficients for each feature (model weights)
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize the classifier
        
        Args:
            threshold (float): Decision threshold for classification (default: 0.5)
        """
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.feature_names = [
            'hr_mean', 'hr_std',
            'br_mean', 'br_std',
            'pupil_mean', 'pupil_std',
            'blink_rate_mean', 'blink_rate_std',
            'saccade_rate_mean', 'saccade_rate_std',
            'eeg_gfp_mean', 'eeg_gfp_std'
        ]
        self.feature_weights = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the logistic regression model
        
        Args:
            X: Feature matrix (n_samples, 12)
            y: Binary labels (n_samples,)
        
        Returns:
            self
        """
        # Fit and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the logistic regression model
        self.model.fit(X_scaled, y)
        
        # Store feature weights
        self.feature_weights = self.model.coef_[0]
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """
        Get probability predictions
        
        Args:
            X: Feature matrix (n_samples, 12)
        
        Returns:
            Probability of class 1 (ADHD positive)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X):
        """
        Get binary predictions using threshold
        
        Args:
            X: Feature matrix (n_samples, 12)
        
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def get_feature_importance(self):
        """
        Get feature importance (coefficients)
        
        Returns:
            DataFrame with feature names and their coefficients
        """
        if self.feature_weights is None:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.feature_weights,
            'Abs_Coefficient': np.abs(self.feature_weights)
        })
        
        return importance_df.sort_values('Abs_Coefficient', ascending=False)
    
    def get_model_info(self):
        """
        Get model information and parameters
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'LogisticRegression',
            'task': 'Binary Classification (ADHD)',
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'solver': self.model.solver,
            'max_iterations': self.model.max_iter,
            'class_weight': 'balanced',
            'intercept': self.model.intercept_[0] if self.is_fitted else None,
            'coefficients': self.feature_weights,
            'is_fitted': self.is_fitted,
            'classes': [0, 1]
        }
    
    def __repr__(self):
        status = "Fitted" if self.is_fitted else "Not Fitted"
        return f"ADHDLogisticClassifier(threshold={self.threshold}, status={status})"


def create_and_save_logistic_model():
    """
    Create a logistic regression model and save it as .pkl file
    
    This function creates a new logistic regression model instance and saves it
    along with a training summary.
    """
    
    print("=" * 70)
    print("Creating Logistic Regression Model for ADHD Binary Classification")
    print("=" * 70)
    
    # Create model instance
    print("\n[1/3] Initializing ADHDLogisticClassifier...")
    classifier = ADHDLogisticClassifier(threshold=0.5)
    
    print(f"  [OK] Model created with threshold = 0.5")
    print(f"  [OK] Features: {len(classifier.feature_names)}")
    print(f"  [OK] Model type: LogisticRegression (Binary Classification)")
    
    # Create model information package
    print("\n[2/3] Preparing model information package...")
    
    model_package = {
        'classifier': classifier,
        'model_type': 'ADHDLogisticClassifier',
        'task': 'Binary Classification (ADHD)',
        'threshold': 0.5,
        'feature_names': classifier.feature_names,
        'num_features': len(classifier.feature_names),
        'feature_details': {
            'hr_features': ['hr_mean', 'hr_std'],
            'br_features': ['br_mean', 'br_std'],
            'pupil_features': ['pupil_mean', 'pupil_std'],
            'blink_features': ['blink_rate_mean', 'blink_rate_std'],
            'saccade_features': ['saccade_rate_mean', 'saccade_rate_std'],
            'eeg_features': ['eeg_gfp_mean', 'eeg_gfp_std']
        },
        'model_info': classifier.get_model_info(),
        'description': 'Binary logistic regression classifier for ADHD detection using 12 features'
    }
    
    print(f"  [OK] Model package created with keys: {list(model_package.keys())}")
    
    # Save model to pickle file
    print("\n[3/3] Saving model to pickle file...")
    
    output_path = r'c:\Users\asus\BBBD\experiment4\RESULTS\logistic_adhd_model.pkl'
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        file_size = os.path.getsize(output_path)
        print(f"  [OK] Model saved to: {output_path}")
        print(f"  [OK] File size: {file_size / 1024:.2f} KB")
    except Exception as e:
        print(f"  [ERROR] Failed to save model: {e}")
        return None
    
    # Verify the saved file
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    try:
        with open(output_path, 'rb') as f:
            loaded = pickle.load(f)
        
        print(f"\n[OK] Model file verified:")
        print(f"  - File exists: True")
        print(f"  - Package keys: {list(loaded.keys())}")
        print(f"  - Model type: {loaded['model_type']}")
        print(f"  - Task: {loaded['task']}")
        print(f"  - Threshold: {loaded['threshold']}")
        print(f"  - Features: {loaded['num_features']}")
        print(f"  - Feature names: {loaded['feature_names']}")
        print(f"  - Classifier object: {loaded['classifier']}")
        
    except Exception as e:
        print(f"[ERROR] Failed to verify: {e}")
        return None
    
    print("\n" + "=" * 70)
    print("SUCCESS: Logistic regression model created and saved")
    print("=" * 70)
    
    return output_path


if __name__ == "__main__":
    import os
    
    print("\n")
    output_file = create_and_save_logistic_model()
    
    if output_file:
        print(f"\nModel file location: {output_file}")
    else:
        print("\nFailed to create model")
