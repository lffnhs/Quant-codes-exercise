"""
this code implements simple trend following strategy (SMA) using Binance API
"""

from binance.client import Client
import time

api_key = "***"
api_secret = "***"

client = Client(api_key, api_secret)

#generate buy and sell signals using long/short moving average 
def signal_sma(symbol, period, short, long):
    try:
        data = client.futures_klines(symbol = symbol, interval = period, limit = 200)
        ma_short = sum([data[-short:][i][4] for i in range(len(data[-short:]))])/short
        ma_long = sum([data[-long:][i][4] for i in range(len(data[-long:]))])/long
        if ma_short > ma_long:
            return 'buy'
        if ma_short > ma_long:
            return 'sell'
    except Exception as e:
        print('fail to fetch data')
        return 'fail'

#generate current position
def get_position(symbol):
    position = client.futures_position_information(symbol = symbol)
    if position[0]['positionAmt'] > 0:
        return 'long'
    if position[0]['positionAmt'] < 0:
        return 'short'
    else:
        return 'zero'

#convert period to seconds
def to_sec(period):
    if period[-1] == 'm':
        return float(period[:-1]) * 60
    if period[-1] == 'h':
        return float(period[:-1]) * 60 * 60
    if period[-1] == 'd':
        return float(period[:-1]) * 60 * 60 * 24
    
#placing order
def create_order(symbol, side, quantity):
    try:
        order = client.futures_create_order(symbol = symbol, side = side, type = 'MARKET', quantity = quantity)
    except Exception as e :
        print('fail')

#run function that uses a while loop to place orders periodically
def run(symbol, period, short, long, quantity):
    while True:
        new_position = signal_sma(symbol, period, short, long)

        if new_position == 'buy':
            current = get_position(symbol)
            if current == 'zero':
                create_order(symbol, 'BUY', quantity)
            if current == 'long':
                pass
            if current == 'short':
                create_order(symbol, 'BUY', quantity * 2)
        if new_position == 'sell':
            current = get_position(symbol)
            if current == 'zero':
                create_order(symbol, 'SELL', quantity)
            if current == 'long':
                create_order(symbol, 'BUY', quantity * 2)
            if current == 'short':
                pass
        time.sleep(to_sec(period))

if __name__ == '__main__':
    run('ETHUSDT_230331', '15m', 10, 40, 0.5)

