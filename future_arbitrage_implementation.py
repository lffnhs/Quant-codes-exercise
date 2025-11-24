#this code uses threading to implement futures arbitrage strategy
import time
import threading
from futures_arbitrage import Arbitrage as arb

api_key = ''
secret_key = ''

s = arb(api_key, secret_key)

#example symbols
symbol1 = 'BTCUSDT'
symbol2 = 'BTCUSD_230331'
dif_open = 0.025
dif_close = 0.01
max_amount = 6000
amount = 200
amount_x = 100

if __name__ == '__main__':
    threads = []
    t1 = threading.Thread(target = s.open_order, args = (symbol1, symbol2, dif_open, 'buy', amount, amount_x))
    t2 = threading.Thread(target = s.close_order, args = (symbol1, symbol2, dif_close, 'sell', amount, amount_x))
    t3 = threading.Thread(target = s.circle_order, args = (symbol1, symbol2, dif_open, dif_close, max_amount, amount, amount_x))

    threads.append(t1)
    threads.append(t2)
    threads.append(t3)

    for i in threads:
        i.start()
        time.sleep(2)