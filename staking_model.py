import numpy as np
from models import kernel_ridge_regression, kernel_spectral_clustering, kernel_logistic_regression

class StackingModel:

    def __init__(self, base_models, meta_model, **kwargs):
        self.base_models = base_models  # List of base models
        self.meta_model = meta_model  # Final model
        self.models_parameters = kwargs
    
    def fit(self, X, y_original, y_onehot):
        """
        Train base models and meta model.
        y_original : real labels.
        y_onehot : labels one hot encoded
        """
        meta_features = []
        
        # Train models and stpck predictions 
        for i, model in enumerate(self.base_models):
            if i == 0:
                y = y_onehot
            else:
                y = y_original
            model.fit(X, y)
            meta_features.append(model.predict(X))
        
       # Tranform prediction into a matrix for the meta model
        meta_X = np.column_stack(meta_features)
        
        # Train meta model
        y = y_onehot
        self.meta_model.fit(meta_X, y)
    
    def predict(self, X):
        """
        Predict with the meta model
        """
        meta_features = [model.predict(X) for model in self.base_models]
        meta_X = np.column_stack(meta_features)

        prediction = self.meta_model.predict(meta_X)

        return prediction