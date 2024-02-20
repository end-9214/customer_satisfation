import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract call for all Models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train: Training data
            y_train: Training labels

        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self,X_train,y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels

        Returns:
            None
        
        """
        try:
            reg= LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model trained successfully")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
    

