import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='department', target_col='avg_training_score', new_col_name='relative_training_score'):
        self.group_col = group_col
        self.target_col = target_col
        self.new_col_name = new_col_name
        self.group_means_ = {}

    def fit(self, X, y=None):
        self.group_means_ = X.groupby(self.group_col)[self.target_col].mean().to_dict()
        return self

    def transform(self, X):
        X_new = X.copy()
        
        X_new['total_training_score'] = X_new['no_of_trainings'] * X_new['avg_training_score']

        dept_mean = X_new[self.group_col].map(self.group_means_).fillna(X_new[self.target_col].mean())
        X_new[self.new_col_name] = X_new[self.target_col] / (dept_mean + 1e-5)
        
        return X_new
    def set_output(self, transform=None):
        return self
        
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ + ['total_training_score', self.new_col_name]

