import numpy as np
import pandas as pd
from collections.abc import Callable
from FX.book import Book

class RiskMeasure:
    def __init__(self, measure_function: Callable[[dict], float]):
        self.measure_function = measure_function
    
    def evaluate_risk(self, states: pd.DataFrame) -> float:
        return self.measure_function(states)

def variance_risk(states: dict) -> float:
    """
    """
    market_data = states['market_data']
    position = states['position']
           
    # align columns
    w = np.array([position[currency] for currency in market_data.columns])
    
    # only need data for the currency pairs we have positions in
    # covariance matrix of dX
    cov = market_data[list(position.keys())].diff().cov()
    # This ought to be expected variance of portfolio w:  w cov wT
    return np.matmul(w.T, np.matmul(cov, w))