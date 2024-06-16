import numpy as np
import pandas as pd
import scipy
from collections.abc import Callable
from book import Book
from risk import RiskMeasure


class Trader:
    def __init__(self, book: Book):
        self.book = book
        # this dataframe contains past/desired trades to do
        self.trades=pd.DataFrame(columns=["time", "fx_cross", "amount"])

    def get_reaction(self, state: pd.DataFrame) -> pd.DataFrame:
        return self.strategy(state, self.internal_parameters)

    def assign_risk_measure(self, risk_measure: RiskMeasure) -> None:
        self.risk_measure = risk_measure
    
    def assign_strategy(self, strategy_func: Callable[[[pd.DataFrame], [dict]], pd.DataFrame]) -> None:
        self.strategy = strategy_func

    def set_internal_parameters(self, parameters: dict) -> None:
        self.internal_parameters = parameters

class RandomTrader(Trader):
    # At any time, we generate a number between [0-1], if below threshold we do a trade of size/dir normal dist
    def simulate_trades(self, time_table:list, fx_crosses:list, trade_probability:float, size:float, bias:float, variance:float) -> pd.DataFrame:
        # trigger parameters
        threshold = trade_probability
        turns = len(time_table)
        
        # trade parameters
            
        # trade triggers
        triggers = np.random.rand(turns)
            
        # trade size / cross
        trade_directions = np.random.normal(bias, variance, [turns, len(fx_crosses)])
        df = pd.DataFrame(trade_directions, columns=fx_crosses)
        
        
        trades = pd.DataFrame(time_table, columns=["time"])
        # find fx cross to do and what amount by looking at max
        
        trades["fx_cross"] = df.abs().idxmax(axis=1) # remember to look at max of absolute!!!
        # Get the amount by extracting the abs largest amount
        v = df.values
        trades['amount'] = v[range(len(v)), np.abs(v).argmax(axis=1)] # get the actual non-abs value of the argmax
        trades["triggers"] = triggers
        
        # filter lower than threshold
        trades = trades[triggers <= threshold]
        trades = trades.drop(['triggers'], axis=1)
        trades = trades.reset_index(drop=True)
        self.trades = trades
        return trades
    
    def get_reaction(self, state: pd.DataFrame) -> pd.DataFrame:
        """
        State dict has to include 'time' key: that is the only one used.
        """
        return self.trades[self.trades['time'] == state['time']]


class MarketMaker(Trader):
    pass



# STRATEGY I GUESS SHOULD BE A CLASS

# In theory the risk here should be generic function from its RISK MEASURE, but ok.
def risk_minimizer(state: pd.DataFrame, parameters: dict, return_optimizer: bool=False):
    # -> pd.DataFrame:
    # unpack internal parameters    
    scale_risk = 1.0
    scale_cost = 1.0
    scale_hedge = 1.0
    trade_theshold = 0.0
    options = {}
    tol = 10**(-15)
    method = None
    
    try:
        scale_risk = parameters['scale_risk']
    except:
        pass
    try:
        scale_cost = parameters['scale_cost']
    except:
        pass
    try:
        scale_hedge = parameters['scale_hedge']
    except:
        pass    
    try:
        trade_theshold = parameters['trade_theshold']
    except:
        pass
    try:
        options = parameters['options']
    except:
        pass
    try:
        tol = parameters['tol']
    except:
        pass
    try:
        method = parameters['minimzer_method']
    except:
        pass
    
    # unpack
    market_data = state['market_data']
    position = state['position']
    hedge_spreads = state['hedge_spreads']

    w = np.array([position[currency] for currency in market_data.columns])
    hedge_cost = np.array([hedge_spreads[currency] for currency in market_data.columns])

    cov = market_data[list(position.keys())].diff().cov()
    
    # save columns for later
    cols = list(cov.columns)

    # define loss function
    def hedger_loss_func(hedge: np.array, position: np.array, covariance: np.array, hedge_cost: np.array, scale_risk: float, scale_cost: float):
        position = position + hedge
        return np.matmul(position.T, np.matmul(covariance, position))*scale_risk + scale_cost*np.sum(hedge_cost * np.abs(hedge))
    
    # MINIMIZE
    optimizer = scipy.optimize.minimize(
        hedger_loss_func,
        x0 = w * np.random.rand(), #randomized start
        args=(w, cov.values, hedge_cost, scale_risk, scale_cost),
        tol=tol,
        options=options,
        method=method)
    
    hedge = optimizer['x']*scale_hedge
    loss = optimizer['fun']
    # format trades
    if return_optimizer:
        return optimizer
    
    if not optimizer['success']:
        return pd.DataFrame([], columns=['fx_cross', 'amount'])
    
    
    return pd.DataFrame([{'fx_cross': ccy, 'amount': amnt} for ccy, amnt in list(zip(cols, hedge)) if np.abs(amnt) > trade_theshold])
    #return trades[np.abs(trades['amount']) > 0]



def back_to_back(state: pd.DataFrame):
    market_data = state['market_data']
    position = state['position']
    hedge_spreads = state['hedge_spreads']

    return pd.DataFrame([{'fx_cross': ccy, 'amount': amnt} for ccy, amnt in position.items() if np.abs(amnt) > 0])


def pca_b2b(state: pd.DataFrame):
    market_data = state['market_data']
    position = state['position']
    hedge_spreads = state['hedge_spreads']
    
    # pca on cov?

    return pd.DataFrame([{'fx_cross': ccy, 'amount': amnt} for ccy, amnt in position.items() if np.abs(amnt) > 0])











if __name__ == "__main__":
    dt = 0.01  # Time step
    num_steps = 1000  # Number of steps
    time = np.arange(0, num_steps * dt, dt)
    
    fx_crosses = ["EURUSD", "EURGBP", "EURSEK"]
    start_rates = [1.11, 0.895, 9.2]


    # DERIVED

    value_dict = {"EUR": 1}
    value_dict.update({fx_cross[-3:]: 1.0/rate for fx_cross, rate in list(zip(fx_crosses, start_rates))})

    asset_names = list(value_dict)
    start_values = list(value_dict.values())
    
    # trigger parameters
    threshold = 0.2
    turns = num_steps
    
    # trade parameters
    bias = 0.0
    variance = 1.0
    size = 1
    
    random_trader = RandomTrader(book=Book(asset_names))
    random_trader.simulate_trades(time, fx_crosses, threshold, size, bias, variance)
    
    
        
    # trade triggers
    triggers = np.random.rand(turns)
          
    # trade size / fx_cross
    trade_directions = np.random.normal(bias, variance, [turns, len(fx_crosses)])
    df = pd.DataFrame(trade_directions, columns=fx_crosses)
    
    trades = pd.DataFrame(time, columns=["time"])
    
    # find fx cross to do and what amount by looking at max
    df["fx_cross"] = df.abs().idxmax(axis=1)
    # Get the amount by extracting the abs largest amount
    vals = df.values
    trades['amount'] = v[range(len(v)), np.abs(v).argmax(axis=1)]
    trades["triggers"] = triggers
    
    # filter lower than threshold
    trades = trades[triggers <= threshold]
    trades = trades.drop(['triggers'], axis=1)
    trades = trades.reset_index(drop=True)
    
    states = {'time': 0.04}
    trades[trades['time'] == states['time']]
    
    
    
    
    