import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from tqdm import tqdm
import scipy
import FX.Utils as utils
from FX.risk import RiskMeasure, variance_risk
from FX.generator import Generator
from FX.book import Book
from FX.trader import Trader, RandomTrader, MarketMaker, risk_minimizer
from FX.plotting import plot_trader_ccypair_pnl, plot_pnl_position_risk

sns.set_theme()

##########################
#
# WORLD ENVIRONMENT
#
##########################

# SIMULATION
dt = 0.001  # Time step
num_steps = 1000  # Number of steps
vol = 0.1
grace_period = 200
time_line = np.arange(0, num_steps * dt, dt)[grace_period:]

# volatilities * dt * N(t) is the expected distribution
# 97.5% Z * sigma / sqrt(n)
# 1.96 * vol * dt * np.sqrt(num_steps) / np.sqrt(num_steps)


# Correlation matrix
correlation_matrix = np.array([[1.0, 0.8, 0.0],
                               [0.8, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])


# CLIENT PARAMETERS
# trigger parameters
threshold = 0.2
turns = num_steps

# trade parameters
bias = 0.0
variance = 1.0
size = 1

# cost to trade
# spread = 10.0/10.0**6 # eur/mio
spread = 1.96 * vol * dt/ np.sqrt(num_steps)

# cost to hedge
hedge_spread = 1.96 * vol * dt/ np.sqrt(num_steps)

# ASSETS
fx_crosses = ["EURUSD", "EURGBP", "EURSEK"]
start_rates = [1.11, 0.895, 9.2]

# DERIVED
value_dict = {"EUR": 1}
value_dict.update({fx_cross[-3:]: 1.0/rate for fx_cross, rate in list(zip(fx_crosses, start_rates))})

asset_names = list(value_dict)
start_values = list(value_dict.values())


###################################################################################

# Generate X different assets with correlation matrix W

generator = Generator(fx_crosses)
data = generator.simulate_rates(start_rates, vol, dt, num_steps, correlation_matrix)

# Create a naive "client"
random_trader = RandomTrader(Book(asset_names))
random_trader.simulate_trades(time_line, fx_crosses, threshold, size, bias, variance)

# Create a naive LP/no hedging
market_maker = MarketMaker(Book(asset_names))
market_maker.assign_risk_measure(risk_measure=RiskMeasure(variance_risk))

market_maker.assign_strategy(risk_minimizer)
market_maker.set_internal_parameters({'scale_cost': 1, 'scale_risk': 1000000})

# Create world-book to dump risk into
world_maker = MarketMaker(Book(asset_names))


# START TESTING

# just a single trade
random_trader.trades = random_trader.trades.iloc[:1]

margins = list(np.zeros(grace_period))
margins_hedge = list(np.zeros(grace_period))
maker_risk = list(np.zeros(grace_period))

# execute a trade
for time in tqdm(time_line):
    
    time = time_line[4]
    
    # update market data
    market_data = data[data.index <= time]
    # the build current snap shot such that it is easier to send into trade functions
    current_state = market_data.reset_index().to_dict('records')[-1]
    # calc spreads   
    spreads = {key:spread*current_state[key] for key in list(current_state.keys())[1:]}    
    hedge_spreads = {key:hedge_spread*current_state[key] for key in list(current_state.keys())[1:]}    
    # IF HEDGER GOES FIRST, THAT MEANS A TURN HAS TO PASS FOR THE HEDGER TO REACT
    
    # #######################################################
    #
    #                   MARKET MAKER ACTION
    #
    # #######################################################
    
    # get hedges
    state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
            'market_data': market_data}

    acc_margin_hedger = 0
    #market_maker.book.get_position_log()
    hedges = market_maker.get_reaction(state=state).to_dict('records')
    
    # trade = hedges.iloc[0]
    for trade in hedges:
        # print('Trade/Hedger:')
        # print(trade)
        acc_margin_hedger += hedge_spreads[trade['fx_cross']] * np.abs(trade['amount'])
        # convert spd into rate
        market_maker.book.fx_trade(time=time
                                   ,target_book=world_maker.book
                                   ,fx_cross=trade['fx_cross']
                                   ,fx_rate=(current_state[trade['fx_cross']] + hedge_spreads[trade['fx_cross']] * np.sign(trade['amount'])) # fixed spread cost
                                   ,base_amount=trade['amount'])

    # #######################################################
    #
    #                   CLIENT ACTION
    #
    # #######################################################
    
    trades = random_trader.get_reaction(current_state).to_dict('records')
    acc_margin = 0
    # EXECUTE
    # trade = trades.iloc[0]
    for trade in trades:
        # print('Trade/Client:')
        # print(trade)
        acc_margin += spreads[trade['fx_cross']] * np.abs(trade['amount'])
        # convert spd into rate
        random_trader.book.fx_trade(time=time
                                    ,target_book=market_maker.book
                                    ,fx_cross=trade['fx_cross']
                                    ,fx_rate=(current_state[trade['fx_cross']] + spreads[trade['fx_cross']] * np.sign(trade['amount'])) # fixed spread cost
                                    ,base_amount=trade['amount'])

    market_maker.book.get_position()
    # get hedger risk measure
    state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
        'market_data': market_data}
    maker_risk.append(market_maker.risk_measure.evaluate_risk(state))
    margins.append(acc_margin)
    margins_hedge.append(acc_margin_hedger)
    # next turn

##########################################################################

# Get PnL at each time step
random_trader.book.get_PnL(market_data=market_data)
market_maker.book.get_PnL(market_data=market_data)


df = market_maker.book.get_PnL(market_data=market_data)
df['margin'] = np.cumsum(margins)
df['risk'] = maker_risk
df['risk'] = df['risk'].fillna(0)
df['trade_pnl'] = df['PnL'] - df['margin']


market_maker.book.get_trade_log()


plot_pnl_position_risk(pnl_data=df, position_log=market_maker.book.get_position_log())
# for hedger lets plot his portfolio variance/risk
plot_trader_ccypair_pnl(market_maker, 'EURUSD', market_data, asset_names=asset_names)