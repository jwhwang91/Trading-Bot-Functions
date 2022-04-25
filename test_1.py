
from binance_functions import *
import pandas as pd
from ta import momentum, volume, volatility
from matplotlib import pyplot as plt

object = Binance_func("1_Binance.txt", 'BTCUSDT')

df = object.get_ohlcv(interval = '1m', limit = 100)

df_close = df['close']
df_volume = df['volume']

real = momentum.rsi(df_close, 14)
rsi_1 = object.GetRSI(df, period = 14)

real = volatility.average_true_range(df['high'], df['low'], df['close'], 14)

fig, ax1 = plt.subplots()

ax1.plot(real)
ax1.plot(rsi_1['RSI'])

ax2 = ax1.twinx()
ax2.plot(df['close'], color = 'tab:red')

plt.show()
