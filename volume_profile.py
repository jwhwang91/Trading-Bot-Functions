from ast import Num
import pandas as pd
import numpy as np
from scipy import stats, signal 
import plotly.express as px
import plotly.graph_objects as go
import binance_functions

object = binance_functions.Binance_func('1_Binance.txt', 'BTCUSDT')
def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig

def volume_profile(interval, num_samples, kde_factor, min_prom_factor):
    df = object.get_ohlcv(interval = interval, limit = num_samples)

    volume = df['volume']
    close = df['close']
    kde = stats.gaussian_kde(close, weights = volume, bw_method = kde_factor)
    xr = np.linspace(close.min(), close.max(), num_samples)
    kdy = kde(xr)
    ticks_per_sample = (xr.max() - xr.min())/num_samples
    min_prom_factor = 0.3
    min_prom = kdy.max() * min_prom_factor
    width_range=1
    peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=width_range)

    pkx = xr[peaks]
    pky = kdy[peaks]
    pk_marker_args=dict(size=10)

    left_ips = peak_props['left_ips']
    right_ips = peak_props['right_ips']
    width_x0 = xr.min() + (left_ips * ticks_per_sample)
    width_x1 = xr.min() + (right_ips * ticks_per_sample)
    width_y = peak_props['width_heights']

    return xr, kdy, pkx, pky, width_x0, width_x1, width_y

interval = '5m'
num_samples = 100
kde_factor = 0.05
pk_marker_args = dict(size=10)
df = object.get_ohlcv(interval = interval, limit = num_samples)
close = df['close']
volume = df['volume']
xr, kdy, pkx, pky, width_x0, width_x1, width_y = volume_profile(interval, num_samples, kde_factor, min_prom_factor = 0.3)

fig = get_dist_plot(close, volume, xr, kdy)
fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))

for x0, x1, y in zip(width_x0, width_x1, width_y):
    fig.add_shape(type='line',
        xref='x', yref='y',
        x0=x0, y0=y, x1=x1, y1=y,
        line=dict(
            color='red',
            width=2,
        )
    )
fig.show()

