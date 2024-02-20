import logging
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE,R2,RMSE
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluation_model(model: RegressorMixin,
                     X_test:pd.DataFrame,
                     y_test: pd.DataFrame,
                     )-> Tuple[
                        Annotated[float, "R2 Score"],
                        Annotated[float, "RMSE"],
                     ]:
    """
    Evaluates the model on the ingested data.

    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return r2_score , rmse

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e