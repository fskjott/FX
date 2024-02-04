#from importlib import reload
#import sys
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
num_steps = 2000  # Number of steps
vol = 0.1
grace_period = 1000
time_line = np.arange(0, num_steps * dt, dt)[grace_period:]

# volatilities * dt * N(t) is the expected distribution
# 97.5% Z * sigma / sqrt(n)
# 1.96 * vol * dt * np.sqrt(num_steps) / np.sqrt(num_steps)


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

# Correlation matrix
correlation_matrix = np.array([[1.0, 0.8, 0.3],
                               [0.8, 1.0, 0.3],
                               [0.3, 0.3, 1.0]])

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
market_maker.set_internal_parameters({'scale_risk': 10**2,
                                      'scale_cost': 10**2,
                                      'scale_hedge': 1.0,
                                      'minize_method': None, #'Nelder-Mead' for simplex
                                      'trade_theshold': .05})

# Create world-book to dump risk into
world_maker = MarketMaker(Book(asset_names))
world_maker.assign_risk_measure(risk_measure=RiskMeasure(variance_risk))

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

    hedges = market_maker.get_reaction(state=state).to_dict('records')
    # trade = hedges.iloc[0]
    for trade in hedges:
        # print('Trade/Hedger:')
        # print(trade)
        market_maker.book.fx_trade(time=time
                                   ,target_book=world_maker.book
                                   ,fx_cross=trade['fx_cross']
                                   ,fx_rate=(current_state[trade['fx_cross']] + hedge_spreads[trade['fx_cross']] * np.sign(trade['amount'])) # fixed spread cost
                                   ,base_amount=trade['amount']
                                   ,pay_margin=hedge_spreads[trade['fx_cross']] * np.abs(trade['amount']))

    # #######################################################
    #
    #                   CLIENT ACTION
    #
    # #######################################################
    
    trades = random_trader.get_reaction(current_state).to_dict('records')
    # EXECUTE
    # trade = trades.iloc[0]
    for trade in trades:
        # print('Trade/Client:')
        # print(trade)
        random_trader.book.fx_trade(time=time
                                    ,target_book=market_maker.book
                                    ,fx_cross=trade['fx_cross']
                                    ,fx_rate=(current_state[trade['fx_cross']] + spreads[trade['fx_cross']] * np.sign(trade['amount'])) # fixed spread cost
                                    ,base_amount=trade['amount']
                                    ,pay_margin=spreads[trade['fx_cross']] * np.abs(trade['amount']))

    # get hedger risk measure
    state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
             'market_data': market_data}
    market_maker.book.log_risk_entry(time=time, risk=market_maker.risk_measure.evaluate_risk(state))
    
    # get world_maker risk
    state = {'position': utils.currency_positions_to_pairs(positions=world_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
             'market_data': market_data}
    
    world_maker.book.log_risk_entry(time=time, risk=world_maker.risk_measure.evaluate_risk(state))
    # next turn

##########################################################################


plot_pnl_position_risk(trader=market_maker, market_data=market_data, asset_names=asset_names, scaling=utils.convert_to_eur(market_data.to_dict('records')[-1]))
# for hedger lets plot his portfolio variance/risk


plot_pnl_position_risk(trader=random_trader, market_data=market_data, asset_names=asset_names, scaling=utils.convert_to_eur(market_data.to_dict('records')[-1]))
plot_pnl_position_risk(trader=world_maker, market_data=market_data, asset_names=asset_names, scaling=utils.convert_to_eur(market_data.to_dict('records')[-1]))

# Lets compare trader with world to see the risk split
#df = market_maker.book.get_PnL(market_data=market_data)
plot_trader_ccypair_pnl(market_maker, 'EURSEK', market_data, asset_names=asset_names)
plot_trader_ccypair_pnl(world_maker, 'EURSEK', market_data, asset_names=asset_names)






# STOP HERE BELOW IS TO TINKER WITH OPTIMIZER

# INVESTIGATE WHY THIS OPTIMIZER ISNT HAPPY




market_data = data
plotspreads = pd.DataFrame([{key:hedge_spread*state[key] for key in list(state.keys())} for state in market_data.to_dict('records')])
market_maker.book.get_PnL(market_data=market_data)


# Create a naive LP/no hedging
market_maker = MarketMaker(Book(asset_names))
market_maker.assign_risk_measure(risk_measure=RiskMeasure(variance_risk))

market_maker.assign_strategy(risk_minimizer)
market_maker.set_internal_parameters({'scale_risk': 10**0,
                                      'scale_cost': 10**0,
                                      'scale_hedge': 1.0,
                                      'minize_method': 'Newton-CG', #'Nelder-Mead' for simplex
                                      'trade_theshold': .05})

current_state = market_data.reset_index().to_dict('records')[-1]
hedge_spreads = {key:hedge_spread*current_state[key] for key in list(current_state.keys())[1:]}
trade = {'fx_cross': fx_crosses[0], 'amount': 1}
# trade
market_maker.book.fx_trade(time=time
                            ,target_book=world_maker.book
                            ,fx_cross=trade['fx_cross']
                            ,fx_rate=(current_state[trade['fx_cross']] + hedge_spreads[trade['fx_cross']] * np.sign(trade['amount'])) # fixed spread cost
                            ,base_amount=trade['amount']
                            ,pay_margin=hedge_spreads[trade['fx_cross']] * np.abs(trade['amount']))



state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
            'market_data': market_data}

market_maker.risk_measure.evaluate_risk(state)

state = {'position': utils.currency_positions_to_pairs(positions=market_maker.book.get_position(), market_rates=dict(market_data.iloc[-1])),
        'market_data': market_data,
        'hedge_spreads': hedge_spreads}

market_maker.get_reaction(state=state)

risk_minimizer(state=state, parameters=market_maker.internal_parameters, return_optimizer=True)




# TODO

#[DONE] SIMULATE CORRELATED BROWNIANS/FXRATES
#[DONE] CREATE BOOK SETUP
#[DONE] CREATE RANDOM TRADER
#[DONE] CREATE PASSIVE MARKET MAKER
#[DONE] CREATE PLOT FOR PERFORMANCE PER CCY
#[DONE] CREATE PLOTS FOR PL/POSITION
#[DONE] CREATE MARGINS/COSTS FOR TRADING
#[DONE] CREATE RISK MEASURES AND HEDGING STRAT
#
#       CALCULATE MEANINGFUL PARAMETERS FOR SPREADS/RISK
#
#           - LOSS PROBABILITIES
#           - VAR? CVAR? RISK OF LOSING X?
#       GET MARKET MAKER TO HEDGE SMART (rip)
#           - portfolio variance minimization
#       MORE ASSETS
#
#       PLUMBING
#       - hide more functions under TRADER
#       - create environment class that holds all necessary info to be passed around?
#           (this should clean up code significantly as most things would just require trader
#           to call function using environment and self should cover the rest)
#               
# FUTURE
#       PCA TO REDUCE DIMENSIONS TO REDUCE HEDGING COSTS
#       REALISTIC FX RATE PATHS (MIMIC ACTUAL DATA?)
#       REAL FX RATE PATHS?
#
# POTENTIAL COOL STUFF
#       CREATE DASHBOARD TO SHOW THINGS?
#       USE DJANGO TO LET USERS TRADE AGAINST SYSTEM/OTHERS?



# PLUMBING:

# IT SEEMS THAT THE HEDGER WORKS FINE WITH THESE SETTINGS (or does it lol)
# 1: ANALYSE RATIO BETWEEN TRADER RISK, HEDGER RISK and WORLD RISK
# 2: HOW LONG DOES THE HEDGER ON AVG HOLD RISK?
# 3: CREATE PLOT WITH DOTS FOR TRADER AND X FOR HEDGER
# 4: CALCULATE (BOOTSTRAP?) EXPECTED RISK OF LOSING ONE SPD EQUIVALENT IN Y
# -- HOW LONG UNTIL 30% (or X%) OF PATHS PRODUCE LOSS GREATER THAN SPREAD (50% loss would be infinite because the distribution has zero mean)
# -- so do DIFFs and sample
# -- MAYBE EASIER TO DO GIVEN LAST X OBSERVATIONS, WHATS THE RISK OF LOSS GREATER THAN SPREAD ?? SHOULD BE MUCH EASIER MUCH EASIER

# SPREAD ANALYSIS AND LOSS PROBABILITIES
spread

data = generator.simulate_rates(start_rates, vol, dt, 50000, correlation_matrix)

dX = data.diff().iloc[1:]

dX.gt(0).mean()
# EURUSD
time_ticks = 10
samples = 100
df = pd.DataFrame([(dX.sample(time_ticks).sum() + 10*spread).to_dict() for x in range(0,samples)])

# Voila this is % of pl after 50 ticks -- maybe spreads should be increased until this gets up to like 75% for 1 tick like 10x spreads
# avg with profit


f, (ax1) = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
plotdata = df.melt(ignore_index=False, var_name='currency', value_name='amount').reset_index()
p1 = sns.scatterplot(data=plotdata, ax=ax1, x='index', y='amount', hue='currency')
ax1.axhline(0, color='black', alpha=0.5)
plt.show()


time_ticks = np.arange(0,50, 1)
samples = 1000


df = pd.DataFrame([pd.DataFrame([(dX.sample(ticks).sum() - 10.0*spread).to_dict() for x in range(0,samples)]).gt(0).mean() for ticks in time_ticks])
df['ticks'] = time_ticks
plotdata = df.set_index('ticks').melt(ignore_index=False, var_name='currency', value_name='probability').reset_index()
f, (ax1) = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
p1 = sns.scatterplot(data=plotdata, ax=ax1, x='ticks', y='probability', hue='currency')
ax1.axhline(0.5, color='black', alpha=0.5)
f.suptitle('Probability of loss')
plt.show()

# Get PnL at each time step
#random_trader.book.get_PnL(market_data=market_data)
#market_maker.book.get_PnL(market_data=market_data)

# formalize all this into a class that checks columns match?
# do same for current_state ? 
# derived class from pd.DataFrame? which basically checks columns are right and then returns pd.DataFrame()







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


#[DONE] CREATE RISK MEASURE FOR HEDGER SUCH THAT WE CAN SEE RISK
#[DONE] But how do we find our "positions"? Do we do it by rewiring everything against EUR?
#[DONE] or do we do movements XTX on value of each of our holding? I guess thats the same as EUR/whatever
#[DONE] How do we do risk in the beginning ? Do we do it on the full data or just data so far? Data so far will look very funky in beginning.....


#[DONE] Create function to extract correlation from the observed data (playground)
#[DONE] Create a hedger using the estimated correlation matrix (see below)
#[DONE] CREATE HEDGER FROM PLAYGROUND THAT WILL MINIMIZE THE RISK PORTFOLIO WITH CONSTRAINT (MARKOWITZ STYLE)

#[DONE] Create a suite of plots that can highlight what we want to see
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

