import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from FX.book import Book
from FX.trader import Trader
import FX.Utils as utils


#df = pd.melt(df.reset_index(), id_vars='time', value_vars=list(df.columns))
#sns.relplot(data=df, x="time", y="value", hue="variable")
#plt.show()

# plot XTX over time ?

# PLOTS RELATED TO MARKET DATA

def plot_market_dX(market_data: pd.DataFrame) -> None:
    df = data.diff()[1:]
    sns.pairplot(df)
    plt.show()


# CREATE A FIGURE OF THREE PLOTS WITH MARKET RATE-TRADES, POSITON AND PNL
def plot_trader_ccypair_pnl(trader: Trader, currency_pair:str, market_data: pd.DataFrame, asset_names: list[str]) -> None:
    trades = trader.book.get_trade_log()
    # trades
    trades = trades[trades['fx_cross'] == currency_pair]
    trades['direction'] = ['BUY' if np.sign(x) == 1 else 'SELL' for x in  trades['amount'].values]
    trades['position'] = trades['amount'].cumsum()
    market =  pd.DataFrame(market_data[currency_pair])

    # replay trades with dummy book    
    dummy_book = Book(asset_names=asset_names)
    dummy_book2 = Book(asset_names=asset_names)

    for trade in trades.to_dict('records'):   
        dummy_book.fx_trade(time=trade['time']
                            ,target_book=dummy_book2
                            ,fx_cross=trade['fx_cross']
                            ,fx_rate=trade['fx_rate']
                            ,base_amount=trade['amount'])

    pnl = dummy_book.get_PnL(market_data=market_data)[np.concatenate([asset_names,["PnL"]])]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    p1 = sns.lineplot(data=market, ax=ax1, x="time", y=currency_pair)
    p1.ticklabel_format(style='plain', axis='y',useOffset=False)
    p2 = sns.scatterplot(data=trades, ax=ax1, x="time", y="fx_rate", hue="direction",  palette={'BUY': 'green', 'SELL': 'red'})

    ax2.axhline(0, color='black', alpha=0.5)
    p3 = sns.lineplot(data=trades, ax=ax2, x='time', y='position', drawstyle='steps-post')
    ax3.axhline(0, color='black', alpha=0.5)
    p4 = sns.lineplot(data=pnl, ax=ax3, x='time', y='PnL', drawstyle='steps-pre')
    f.suptitle(currency_pair)
    plt.ticklabel_format(useOffset=False)
    plt.show()
    
# PNL POSITION AND RISK
def plot_pnl_position_risk(trader:Trader, market_data: pd.DataFrame, asset_names: list, scaling: dict={}) -> None:
    #scaling = utils.convert_to_eur(pnl_data[fx_crosses].to_dict('records')[-1])
    # asset names are used
    df = trader.book.get_PnL(market_data=market_data)
    #df = pnl_data
    position_log = df[asset_names]
    
    if len(scaling.items()) > 0:
        position_log = position_log * scaling
    
    pos = position_log.melt(ignore_index=False, var_name='currency', value_name='amount')
    

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    p1 = sns.lineplot(data=df, ax=ax1, x="time", y='PnL')
    p1.ticklabel_format(style='plain', axis='y',useOffset=False)
    p2 = sns.lineplot(data=df, ax=ax1, x='time', y='acc_margin', drawstyle='steps-post')
    # color gap between pnl and margin to highligh composition of pnl
    ax1.fill_between(x=df.index, y1=df['PnL'], y2=df['acc_margin'], where=df['PnL']>df['acc_margin'], facecolor='green', alpha=0.3, step='post')
    ax1.fill_between(x=df.index, y1=df['PnL'], y2=df['acc_margin'], where=df['PnL']<df['acc_margin'], facecolor='red', alpha=0.3, step='post')

    ax2.axhline(0, color='black', alpha=0.5)
    p3 = sns.lineplot(data=pos, ax=ax2, x='time', y='amount', hue='currency', drawstyle='steps-post')
    ax3.axhline(0, color='black', alpha=0.5)
    p4 = sns.lineplot(data=df, ax=ax3, x='time', y='risk', drawstyle='steps-post')
    f.suptitle('RISK')
    plt.ticklabel_format(useOffset=False)
    plt.show()

