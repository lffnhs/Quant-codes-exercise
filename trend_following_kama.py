""""
this code reads historical stock/futures data from local files and tests a famous
trend following strategy (KAMA) using a simple backtest function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read from local files and save as a dataframe
test_df = pd.read_csv(
    "/Users/liufeifan/Desktop/day_20220611/R.CN.CZC.SF.0004.csv",
    parse_dates=["CLOCK"],
    index_col="CLOCK",
    date_format=lambda x: pd.to_datetime(x, format="%m/%d/%Y")
)

#calculate average true range over a period
def atr(df, period):
    df['H-L'] = df['HIGH'] - df['LOW']
    df['H-C'] = abs(df['HIGH'] - df['CLOSE'].shift(1))
    df['L-C'] = abs(df['LOW'] - df['CLOSE'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['Atr'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['H-L', 'H-C', 'L-C', 'TR'], inplace=True)
    
    return df

#calculate atr ratio to indicate short term volatility, used in determining noise
def get_atr_ratio(df, short_period=10, long_period=30):
    df = atr(df, period=short_period)
    short_atr = df['Atr']
    df = atr(df, period=long_period)
    long_atr = df['Atr']
    df['Atr_ratio'] = short_atr / long_atr
    
    return df

#calculate efficiency ratio
def er(df, period=14):
    price_change = df["CLOSE"].shift(period) - df["CLOSE"]
    price_change = price_change.abs()
    price_movements = df["CLOSE"].diff().abs()
    return price_change / price_movements.rolling(window=period).sum()

#calculate Kaufman Adaptive Moving Average (KAMA)
def get_kama(df, period=30, fast=2, slow=30):
    ER = er(df, period=period)
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    SC = (ER * (sc_fast - sc_slow) + sc_slow) ** 2
    kama_values = [np.nan] * len(df)
    kama_values[period] = df["CLOSE"].iloc[period]

    for i in range(period + 1, len(df)):
        prev_kama = kama_values[i - 1]
        price = df["CLOSE"].iloc[i]
        kama_values[i] = prev_kama + SC.iloc[i] * (price - prev_kama)

    df["Kama"] = kama_values

    return df

#return numbers from -1~1, where positive indicates buy and 1 indicates buy all positions
def signal(df, min_diff=0, cooldown=5, atr_mult_sl=2, atr_mult_tp=4):
    get_kama(df)
    get_atr_ratio(df)
    df["Signal"] = 0
    df["Position"] = 0.0
    df['entry'] = 0.0
    df['leave'] = 0.0
    df.at[df.index[0], "Signal"] = 0
    df.at[df.index[0], "Position"] = 0
    last_signal = 0
    cooldown_counter = 0
    entry_price = None

    for i in range(1, len(df)): 
        prev_close = df.iloc[i-1]["CLOSE"]
        prev_kama = df.iloc[i-1]["Kama"]
        price = df.iloc[i]["OPEN"]
        kama = df.iloc[i]["Kama"]
        atr = df.iloc[i]["Atr"]
        atr_ratio = df.iloc[i]["Atr_ratio"]

        if cooldown_counter > 0:
            cooldown_counter -= 1

        if last_signal == 0 and cooldown_counter == 0:
            if prev_kama - prev_close > min_diff and kama - price > min_diff:
                last_signal = 1
                df.at[df.index[i], "entry"] = price
                entry_price = price
                df.at[df.index[i], "Signal"] = last_signal
                cooldown_counter = cooldown
            elif prev_close - prev_kama > min_diff and price - kama > min_diff:  
                last_signal = -1
                df.at[df.index[i], "entry"] = price
                entry_price = price
                cooldown_counter = cooldown
        else:
            if last_signal == 1: 
                if df.iloc[i]["LOW"] <= entry_price - atr_mult_sl * atr: 
                    last_signal = 0
                    df.at[df.index[i], "leave"] = entry_price - atr_mult_sl * atr
                elif df.iloc[i]["HIGH"] >= entry_price + atr_mult_tp * atr:
                    last_signal = 0
                    df.at[df.index[i], "leave"] = entry_price + atr_mult_tp * atr
            elif last_signal == -1: 
                if df.iloc[i]["HIGH"] >= entry_price + atr_mult_sl * atr:
                    last_signal = 0
                    df.at[df.index[i], "leave"] = entry_price + atr_mult_sl * atr
                elif df.iloc[i]["LOW"] <= entry_price - atr_mult_tp * atr:
                    last_signal = 0
                    df.at[df.index[i], "leave"] = entry_price - atr_mult_tp * atr

        position_size = 1 / atr_ratio if atr_ratio > 0 else 0
        position_size = min(position_size, 1.0)

        df.at[df.index[i], "Signal"] = last_signal
        df.at[df.index[i], "Position"] = last_signal * position_size

    return df

#a backtest model showing market return, position signal and profit and loss (PnL)
def backtest_trend_follow(df, allocation=100000.0):
    """
    slippage cost and charge not included
    """
    df = df.copy()
    df['Returns'] = df['CLOSE'].pct_change().fillna(0)
    df['PnL'] = 0.0
    current = 0.0
    position = 0.0
    df["Equity"] = allocation

    for i in range(1, len(df)):
        dailyPnL = 0.0
        close = df.iloc[i]['CLOSE']
        pre_close = df.iloc[i-1]['CLOSE']
        leave = df.iloc[i]['leave']

        if df.iloc[i]['Signal'] == 1:
            if df.iloc[i]['entry'] != 0:
                current = df.iloc[i]['entry']
                position = allocation * df.iloc[i]['Position']
                dailyPnL = position * ((close - current) / current)
                position = position * (1 - (close - current) / current)
                current = df.iloc[i]['CLOSE']
            else:
                dailyPnL = position * ((close - current) / current)
                position = position * (1 - (close - current) / current)
                current = df.iloc[i]['CLOSE']
        elif df.iloc[i]['Signal'] == -1:
            if df.iloc[i]['entry'] != 0:
                current = df.iloc[i]['entry']
                position = allocation * df.iloc[i]['Position']
                dailyPnL = position * ((close - current) / current)
                current = df.iloc[i]['CLOSE']
                position = position * (1 - ((close - current) / current))
            else:
                dailyPnL = position * ((close - current) / current)
                position = position * (1 - (close - current) / current)
                current = df.iloc[i]['CLOSE']
        elif df.iloc[i]['Signal'] == 0 and df.iloc[i - 1]['Signal'] == 1:
            dailyPnL = position * ((leave - pre_close) /pre_close)
        elif df.iloc[i]['Signal'] == 0 and df.iloc[i - 1]['Signal'] == -1:
            dailyPnL = position * ((leave - pre_close) /pre_close)
        
        df.at[df.index[i], "Equity"] = allocation + dailyPnL
        df.at[df.index[i], 'PnL'] = df.at[df.index[i-1], 'PnL'] + dailyPnL
        allocation = allocation + dailyPnL
        
    df['Stategy_ret'] = df['Equity'].pct_change()
    df["PnL_o"] = (1 + df['Returns']).cumprod() * allocation - allocation

    plt.figure(figsize=(14,12))

    plt.subplot(3,1,1)
    ax1 = plt.gca()
    ax1.plot(df.index, df['CLOSE'], color='blue', label='Market Price')
    ax1.plot(df.index, df['Kama'], color='red', label='Kama')
    ax2 = ax1.twinx()
    ax2.bar(df.index, df['Position'], color='green', label='Position', width=1)

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)

    plt.title('Position')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.legend(handles=ax1.lines + ax2.collections)

    plt.subplot(3,1,2)
    ax1 = plt.gca()
    ax1.plot(df.index, df['CLOSE'], color='tab:blue', label='Market Price', alpha=0.7)
    ax1.set_ylabel('Market Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    ax1.scatter(buy_signals.index, buy_signals['CLOSE'], marker='^', color='g', label='Buy Signal', s=3)
    ax1.scatter(sell_signals.index, sell_signals['CLOSE'], marker='v', color='r', label='Sell Signal', s=3)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Equity'], color='tab:red', label='Equity Curve', alpha=0.7)
    ax2.set_ylabel('Equity', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.set_title('Buy/Sell signal and Equity curve')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.legend(handles=ax1.lines + ax2.lines + ax1.collections)

    plt.subplot(3,1,3)
    plt.plot(df.index, df['PnL'], label='Strategy', color='black')
    plt.plot(df.index, df['PnL_o'], label='Market', color='Red')
    plt.fill_between(df.index, df['PnL'], 0, where=(df['PnL']>=0), facecolor='blue', interpolate=True)
    plt.fill_between(df.index, df['PnL'], 0, where=(df['PnL']<0), facecolor='red', interpolate = True)
    plt.title('PnL')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    signal(test_df)
    backtest_trend_follow(test_df)


