"""
This code exploit future arbitrage strategy using Binance API
"""

from binance.client import Client
import time

class Arbitrage():

    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)

    def get_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')
    
    #calculate the difference between opening and closing positions
    def get_data(self, symbol1, symbol2):
        while True:
            try:
                spot = self.client.get_order_book(symbol = symbol1, limit = 5)
                future = self.client.futures_coin_order_book(symbol = symbol2, limit = 5)
                bid_spot = float(spot['bids'][0][0])
                ask_spot = float(spot['asks'][0][0])
                bid_future = float(future['bids'][0][0])
                ask_future = float(future['asks'][0][0])

                dif_buy = (bid_future - ask_spot) / ask_spot
                dif_sell = (ask_future - bid_spot) / bid_spot

                return [dif_buy, dif_sell]
                break
            except Exception as e:
                print(f'{e}')
                time.sleep(2)
                continue
    
    #create order based on the amount invested (amount) and value of each contract (amount_x)
    def create_order(self, symbol1, symbol2, side, amount, amount_x):
        if side == 'buy':
            try:
                order_spot = self.client.order_market_buy(symbol1, quantity = amount)
                order_future = self.client.futures_coin_create_order(symbol = symbol2, side = 'SELL', type = 'MARKET', quantity = amount / amount_x)
            except Exception as e:
                print(f'{e}')
        if side == 'sell':
            try:
                order_spot = self.client.order_market_sell(symbol1, quantity = amount)
                order_future = self.client.futures_coin_create_order(symbol = symbol2, side = 'BUY', type = 'MARKET', quantity = amount / amount_x)
            except Exception as e:
                print(f'{e}')
    
    def open_order(self, symbol1, symbol2, dif_open, side, amount, amount_x):
            diff = self.get_data(symbol1, symbol2)
            if dif_open > diff[0]:
                self.create_order(symbol1, symbol2, 'buy', amount, amount_x)
        
    def close_order(self, symbol1, symbol2, dif_close, side, amount, amount_x):
            diff = self.get_data(symbol1, symbol2)
            if dif_close < diff[1]:
                self.create_order(symbol1, symbol2, 'sell', amount, amount_x)

    #place order automatically once the price differences between spot and future exceeds dif_open and dif_close, maximum amount max_amount
    def circle_order(self, symbol1, symbol2, dif_open, dif_close, max_amount, amount, amount_x):
        number = 0
        n = max_amount / amount
        while True:
            diff = self.get_data(symbol1, symbol2)
            if dif_open > diff[0]:
                self.create_order(symbol1, symbol2, 'buy', amount)
            if dif_close < diff[1]:
                self.create_order(symbol1, symbol2, 'sell', amount)
            time.sleep(1)






