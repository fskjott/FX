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
correlation_matrix = np.array([[1.0, 0.8, 0.3],
                               [0.8, 1.0, 0.3],
                               [0.3, 0.3, 1.0]])


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

# np.percentile(data['EURUSD'].diff()[1:], 97.5)


# Create a naive "client"
random_trader = RandomTrader(Book(asset_names))
random_trader.simulate_trades(time_line, fx_crosses, threshold, size, bias, variance)

# Create a naive LP/no hedging
market_maker = MarketMaker(Book(asset_names))
market_maker.assign_risk_measure(risk_measure=RiskMeasure(variance_risk))

market_maker.assign_strategy(risk_minimizer)
market_maker.set_internal_parameters({'scale_risk': 10**5})

# Create world-book to dump risk into
world_maker = MarketMaker(Book(asset_names))
world_maker.assign_risk_measure(risk_measure=RiskMeasure(variance_risk))

# pad with 0s for grace period
margins = list(np.zeros(grace_period))
margins_hedge = list(np.zeros(grace_period))
maker_risk = list(np.zeros(grace_period))
hedged_risk = list(np.zeros(grace_period))

# execute a trade
for time in tqdm(time_line):
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
            'market_data': market_data,
            'hedge_spreads': hedge_spreads}

    acc_margin_hedger = 0
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

    # get hedger risk measure
    state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
             'market_data': market_data}
    maker_risk.append(market_maker.risk_measure.evaluate_risk(state))
    
    # get world_maker risk
    state = {'position': utils.currency_positions_to_pairs(positions=world_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
             'market_data': market_data}
    
    hedged_risk.append(world_maker.risk_measure.evaluate_risk(state))
    
    margins.append(acc_margin)
    margins_hedge.append(acc_margin_hedger)
    # next turn

##########################################################################

# Get PnL at each time step
random_trader.book.get_PnL(market_data=market_data)
market_maker.book.get_PnL(market_data=market_data)


# WE NEED TO SORT HOW TO FIX THE MULTIPLE PER TIME THINGY
# IN THEORY WE JUST NEED THE LATEST OF THE TIME
# SO IN BOOK WE NEED TO REPLACE IF TIME ALREADY EXISTS ? 

df = market_maker.book.get_PnL(market_data=market_data)
df['margin'] = np.cumsum(margins)
df['risk'] = maker_risk
df['risk'] = df['risk'].fillna(0)
df['trade_pnl'] = df['PnL'] - df['margin']

# formalize all this into a class that checks columns match?
# do same for current_state ? 
# derived class from pd.DataFrame? which basically checks columns are right and then returns pd.DataFrame()


#trades = market_maker.book.get_trade_log().groupby(['fx_cross']).cumsum()

# cool bit of cumsum-pivot/unpicot
# trades = market_maker.book.get_trade_log().drop('fx_rate', axis=1)
# unmelt into pivot to cumsum and then re-melt for plot
# pos = pd.pivot(trades, index='time', columns='fx_cross').fillna(0).cumsum()
# pos.columns = pos.columns.droplevel().rename(None)
# pos = pos.melt(ignore_index=False)

# or plot each currency holding
#data['EURUSD'].plot()
#plt.show()

plot_pnl_position_risk(pnl_data=df, position_log=market_maker.book.get_position_log())
# for hedger lets plot his portfolio variance/risk
plot_trader_ccypair_pnl(market_maker, 'EURSEK', market_data, asset_names=asset_names)



df = random_trader.book.get_PnL(market_data=market_data)
df['margin'] = -np.cumsum(margins)
df['risk'] = 0# maker_risk
df['risk'] = 0# df['risk'].fillna(0)
df['trade_pnl'] = df['PnL'] - df['margin']

plot_pnl_position_risk(pnl_data=df, position_log=random_trader.book.get_position_log())


maker_risk
world_maker.book.get_trade_log()


# Lets compare trader with world to see the risk split
df = market_maker.book.get_PnL(market_data=market_data)
w_df = world_maker.book.get_PnL(market_data=market_data)

df['mm_risk'] = maker_risk
df['hedged'] = hedged_risk


sns.lineplot(df, x='time', y='mm_risk')
sns.lineplot(df, x='time', y='hedged')
plt.show()

maker_risk
hedged_risk



# list(dict(market_maker.book.get_trade_log().groupby('fx_cross').sum()['amount']).values())

# get currency pair positions arranged similar to market data columns


# change positions in currency to currencyPairs



# plot_trader_ccypair_pnl(market_maker, 'EURGBP', market_data=data, asset_names=asset_names)

##################################################################################
#
#
#       THOUGHTS FOR FUTURE
#
#
##################################################################################


# CREATE RISK MEASURE FOR HEDGER SUCH THAT WE CAN SEE RISK
# But how do we find our "positions"? Do we do it by rewiring everything against EUR?
# or do we do movements XTX on value of each of our holding? I guess thats the same as EUR/whatever
# How do we do risk in the beginning ? Do we do it on the full data or just data so far? Data so far will look very funky in beginning.....


# Create function to extract correlation from the observed data (playground)
# Create a hedger using the estimated correlation matrix (see below)
# CREATE HEDGER FROM PLAYGROUND THAT WILL MINIMIZE THE RISK PORTFOLIO WITH CONSTRAINT (MARKOWITZ STYLE)

# Create a suite of plots that can highlight what we want to see
# Density plots with correlations? Heatmaps? Seaborn has a lot of juicy stuff

# Create a hedger with bath tub mode - empty out if position too large
# or a maximum risk (for some risk measure)


# Once all this is working, we should experiment with more difficult trade styles -- trade with the momentum?

# Work out the probability of profit/flat given the volatility vs spread. Say what is the probability that we go flat within X seconds?
# And what is the probability that we see a new trade that will offset current risk ?
# Basically how long until risk is internalized? Estimate this based on earlier trades.
# We probably have to upgrade the number of steps from 1k to 10k -- see how balance is
# Find out fair values for timesteps/volatility such that things make sense
# scale trade sizes/hedge cost



# Voila