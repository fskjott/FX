# future thingy is such that a class can typehint itself
from __future__ import annotations
import pandas as pd
import numpy as np

import Utils as utils

class Book:
    def __init__(self, asset_names:list):
        # self.id = random number
        self.asset_names = asset_names
        self.position = dict.fromkeys(["time"] + asset_names, 0)
        # log the empty book
        self.position_log = []#[self.position.copy()]
        self.trade_log = []
        self.margin_log = []
        self.risk_log = []

    def get_book_at_time(self, time):
        # get latest book where #yeah we need latest and not ==
        #return [entry for entry in book.position_log if entry['time'] == time]
        pass
    
    def get_position(self):
        position = self.position.copy()
        position.pop('time', None)
        return position
    
    def add(self, time, asset:str, value:float):
        # update time stamp & value
        self.position['time'] = time
        self.position[asset] += value

    def trade(self, time, target_book:Book, buy_asset:str, sell_asset:str, buy_amount:float, sell_amount:float):
        self.add(time, buy_asset, +buy_amount)
        target_book.add(time, buy_asset, -buy_amount)
        self.add(time, sell_asset, -sell_amount)
        target_book.add(time, sell_asset, +sell_amount)
        # save copy in the log
        self.log_position_entry()
        # save log in other bok
        target_book.log_position_entry()
    
    def fx_trade(self, time, target_book:Book, fx_cross:str, fx_rate:float, base_amount:float, pay_margin: float=None):
        # base=+ amnt, term=+ -amnt * fx_rate
        # find base/term
        base = fx_cross[:3]
        term = fx_cross[-3:]
        self.trade(time, target_book, base, term, base_amount, base_amount * fx_rate)
        # log in trade log
        self.log_trade_entry(time=time, fx_cross=fx_cross, fx_rate=fx_rate, base_amount=base_amount)
        target_book.log_trade_entry(time=time, fx_cross=fx_cross, fx_rate=fx_rate, base_amount=-base_amount)
        # assign margin if any
        if pay_margin is not None:
            self.log_margin_entry(time=time, margin=-pay_margin)
            target_book.log_margin_entry(time=time, margin=pay_margin)
            

    def log_position_entry(self):
        self.position_log.append(self.position.copy())
        
    def get_position_log(self):
        return pd.DataFrame.from_dict(self.position_log)
    
    def log_margin_entry(self, time: float, margin: float):
        self.margin_log.append({'time': time, 'margin': margin})
        
    def get_margin_log(self):
        return pd.DataFrame.from_dict(self.margin_log)

    def log_risk_entry(self, time: float, risk: float):
        self.risk_log.append({'time': time, 'risk': risk})
        
    def get_risk_log(self):
        return pd.DataFrame.from_dict(self.risk_log)

    def log_trade_entry(self, time, fx_cross:str, fx_rate:float, base_amount:float):
        self.trade_log.append({'time': time, 'fx_cross': fx_cross, 'amount': base_amount, 'fx_rate': fx_rate})
    
    def get_trade_log(self):
        return pd.DataFrame.from_dict(self.trade_log)

    def get_worth_by_currency(self, price_dict:dict):
        return {k: price_dict[k] * self.position[k] for k in self.asset_names}
    
    def get_worth(self, price_dict:dict):
        return sum(price_dict[k] * self.position[k] for k in self.asset_names)
    
    def get_PnL(self, market_data:pd.DataFrame):
        # record of all states       
        log = self.get_position_log().drop_duplicates('time', keep='last')
        overview = market_data.join(log.set_index('time')).ffill().fillna(0)
        
        # add risk if any
        if len(self.get_risk_log()) > 0:
            risk_log = self.get_risk_log().drop_duplicates('time', keep='last')
            overview = overview.join(risk_log.set_index('time')).ffill().fillna(0)
        else:
            overview['risk'] = 0.0
        
        if len(self.get_margin_log()) > 0:
            margin_log = self.get_margin_log()
            margin_log['acc_margin'] = margin_log['margin'].cumsum()
            margin_log = margin_log.drop_duplicates('time', keep='last')
            margin_log = margin_log.drop('margin', axis=1)
            overview = overview.join(margin_log.set_index('time')).ffill().fillna(0)
        else:
            overview['acc_margin'] = 0.0
            
        states = market_data.reset_index().to_dict('records')

        # get eur equivalent
        price_dict = pd.DataFrame([utils.convert_to_eur(state) for state in states])
        price_dict['time'] = market_data.index
        price_dict = price_dict.set_index('time')
        # multiply each row with its price to get PnL
        overview['PnL'] = overview[self.asset_names].mul(price_dict).sum(axis=1)
        return overview


if __name__ == "__main__":
    fx_crosses = ["EURUSD", "EURGBP", "EURSEK"]
    start_rates = [1.11, 0.895, 9.2]
    time = 0.00
    market_data = {'time': time}
    
    # idx to test
    idx = 0
   
    market_data.update({cross: rate for cross, rate in list(zip(fx_crosses, start_rates))})
    market_data = pd.DataFrame([market_data]).set_index('time')
    
    value_dict = utils.convert_to_eur(market_data.to_dict('records')[0])

    asset_names = list(value_dict)
    start_values = list(value_dict.values())

    # TESTS
    book = Book(asset_names)
    book2 = Book(asset_names)
    
    # TEST TRADE
    book.fx_trade(time=time, target_book=book2, fx_cross=fx_crosses[idx], fx_rate=start_rates[idx], base_amount=1)
    
    # TEST TRADE/LOG
    book.get_position_log()
    log = book.get_position_log()
    
    log['time'].values[0] == time
    log[fx_crosses[idx][:3]].values[0] == 1
    log[fx_crosses[idx][3:]].values[0] == -start_rates[0]
    
    # TEST 
    market_values = value_dict
    
    log = book.get_worth_by_currency(market_values)
    log[fx_crosses[idx][:3]] == 1
    log[fx_crosses[idx][3:]] == -1
    # check no value has been created from trade
    book.get_worth(market_values) == 0

    # TEST TWO TRADES SAME TIME
    book.fx_trade(time=time, target_book=book2, fx_cross=fx_crosses[idx], fx_rate=start_rates[idx], base_amount=1)
    log = book.get_position_log()
    
    log[fx_crosses[idx][:3]].values == [1, 2]
    log[fx_crosses[idx][3:]].values == [-start_rates[idx], -2*start_rates[idx]]

    # TEST PNL for TWO TRADES AT SAME TIME
    pnl = book.get_PnL(market_data=market_data)
    
    # expected
    exp = market_data.reset_index().to_dict('records')[0]
    exp.update({name: 0 for name in asset_names})
    # add the ones we traded
    exp[fx_crosses[idx][:3]] = 2
    exp[fx_crosses[idx][3:]] = -2*start_rates[idx]
    exp['PnL'] = 0

    # check that PnL is equal to expected    
    pnl.reset_index('time').to_dict('records')[0] == exp
    
    # check that length of log is still 0, so no overwriting has happened    
    len(book.get_position_log()) == 2
    len(book.get_trade_log()) == 2
    
    # lots of asserts
    # put into class and go