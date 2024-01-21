
# UTILITIES
def convert_to_eur(market_data:dict) -> dict:
    """
    dict with prices
    """
    # get rid of time
    df = market_data.copy()
    df.pop('time', None)
    value_dict = {"EUR": 1}
    value_dict.update({fx_cross[-3:]: 1.0/rate for fx_cross, rate in list(df.items())})
    return value_dict

def currency_positions_to_pairs(positions: dict, market_rates: dict) -> dict:
    rates = dict(market_rates.items())
    pos = positions
    # excess EUR would be just PnL
    pos.pop('EUR', None)
    ccy_dict = {}
    ccy_dict.update({'EUR' + currency: -amnt / rates['EUR'+currency] for currency, amnt in pos.items()})
    return ccy_dict