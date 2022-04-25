# Stop Hunting Algorithm --
# 1. 변곡점을 찾는다 (볼린저 밴드의 상/하단을 뚫은 캔들이 변곡점이라고 가정하자)
#    - 조금더 확실하게하자면 --> 전 3-4개의 봉의 중간값이 꾸준히 올랐는지/내렸는지 확인하는 함수를 만들자
# 2. 변곡점의 캔들의 "고가" (또는 "저가")를 갱신하는 봉을 기다린다. 
# 3. 갱신봉을 찾으면 5-10분 내에 안긴 캔들이 뜨는지 확인
#    - 안긴 캔들이 안뜬다면 break
# 4. 안긴캔들 정의 = 현재 캔들 [-2]이 바로 전 캔들 [-3]의 고가~저가 사이에 위치한 캔들 
# 5. 만약 갱신봉이 계속 나타난다면 (볼린저 밴드 하/상 단을 계속해서 긁으면서 간다면) --> 시간 초기화 (연장)
# 6. 안긴캔들이 나오면, 방향결정 캔들을 기다린다
# 7. 방향결정 캔들 정의 = 방향결정 캔들의 현재가['close']가 안긴 캔들의 저/고가를 이탈한 캔들
# 8. 방향결정 캔들 확인 후 바로 진입 (방향결정 캔들의 방향에 따라 short/long 진입)
# 9. 손절가는 안긴캔들 의 저가 
# 10. 목표가는 최소 75틱 ~ 최대 150틱 
# 11. (진입가 - 손절가)가 75틱보다 작으면 목표가를 75틱으로 설정
# 12. (진입가 - 손절가)가 150틱 보다 많으면 목표가를 150틱으로 설정하거나 pass (차이가 200틱이 넘으면 그냥 break)

from os import stat
from re import T
from tracemalloc import stop
import ccxt
import time
from ccxt.base.exchange import Exchange
import pandas as pd
import pprint
import Coin_func
import calendar
from datetime import datetime
import numpy as np
import line_alert
# from mplfinance import candlestick_ohlcv

access = "jfbFgAVyVOMf6jNeIvLztWOOXZA1jjE883QRtVg4zvSJAqEQdoGAc1OxqokRNbPg"
secret = "qlmcYflp22OOLNNwnez7TSh5TUpDA5AB2o0Q1FQI0m64ensMVlPzhpCVBwws7Pqo"

binance = ccxt.binance(config={
    'apiKey' : access,
    'secret' : secret,
    'enableRateLimit' : True,
    'adjustForTimeDifference' : True,
    'options' : {
        'defaultType' : 'future'
    }
})

def get_ohlcv(symbol = 'BTCUSDT', interval = '1m', limit = 200): 

    now = datetime.utcnow()
    unixtime = calendar.timegm(now.utctimetuple())
    if interval == '1m':
        since = (unixtime - limit * 60) * 1000 # UTC timestamp in milliseconds
    elif interval == '5m':
        since = (unixtime - limit * 5 * 60) * 1000
    elif interval == '15m':
        since = (unixtime - limit * 15 * 60) * 1000
    elif interval == '30m':
        since = (unixtime - limit * 30 * 60) * 1000
    elif interval == '1h':
        since = (unixtime - limit * 60 * 60) * 1000
    elif interval == '1d':
        since = (unixtime - limit * 24 * 60 * 60) * 1000

    ohlcv = binance.fetch_ohlcv(symbol= symbol, timeframe= interval, since=since, limit=limit)

    df = pd.DataFrame(ohlcv, columns = ['Time', 'open', 'high', 'low', 'close', 'volume'])
    df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
    df.set_index('Time', inplace=True)
    return df

def bollingerband(df, window = 20): # data = close 
    sma = df['close'].rolling(window = window).mean()
    std = df['close'].rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    df['sma'] = sma
    df['upper_bb'] = upper_bb
    df['lower_bb'] = lower_bb
    df['bb_diff'] = upper_bb - lower_bb
    return df

def GetMA_Vol(df, period):
    volume = df['volume']
    ma = volume.rolling(period).mean()
    df['ma_vol'] = ma
    return df

def median(target_coin_symbol, interval, limit, window):
    period = 7
    df = get_ohlcv(target_coin_symbol, interval = interval, limit = limit)
    df = bollingerband(df, window = window)
    df['ma_vol'] = (df['volume']).rolling(period).mean()
    df['median'] = (df['high'] + df['low'])/2
    return df

def inflection_ctr(df, period):
    cnt_up, cnt_dwn = 0, 0
    # 최근 period분간 몇번이나 볼린저 밴드 상/하단을 뚫었는지 계산 함
    for i in range(2, period):
        i = i  *  -1
        print(i, df['close'][i], df['upper_bb'][i])
        if df.loc[df.index[i], 'close'] > df.loc[df.index[i], 'upper_bb']:
           
            cnt_up += 1
        elif df.loc[df.index[i], 'close'] < df.loc[df.index[i], 'lower_bb']:
            cnt_dwn += 1
    
    return cnt_up, cnt_dwn 


def det_direction(df):
    direction = None
    if df['bb_diff'].iloc[-2] > 100:
        if df['close'].iloc[-2] > df['upper_bb'].iloc[-2]:
        #df['median'].iloc[-2] > df['median'].iloc[-3] > df['median'].iloc[-4]:
            direction = 'upper'
        
        elif df['close'].iloc[-2] < df['lower_bb'].iloc[-2]:
            #df['median'].iloc[-2] < df['median'].iloc[-3] < df['median'].iloc[-4]:
            direction = 'lower'
    
    # if df['volume'].iloc[-2] > 4000 and (cnt_up != 0 or cnt_dwn != 0) \
    #    and df['volume'].iloc[-2] > df['volume'].iloc[-3] * 2:

    #    if df['close'].iloc[-2] > df['open'].iloc[-2]:
    #         direction = 'volume_burst_long'
    #    elif df['close'].iloc[-2] < df['open'].iloc[-2]:
    #         direction = 'volume_burst_short'
            
    return direction

def embrace_candle(df):
    embrace = False
    embrace_high, embrace_low = 0, 0

    if (df['high'].iloc[-3] > df['high'].iloc[-2] and\
       df['low'].iloc[-3] < df['low'].iloc[-2]):
    #    or\
    #    (df['high'].iloc[-4] > df['high'].iloc[-2] and\
    #    df['low'].iloc[-4] < df['low'].iloc[-2]):
       embrace = True
       embrace_high, embrace_low = df['high'].iloc[-2], df['low'].iloc[-2]
    
    return embrace, embrace_high, embrace_low

def profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin):
    # low, high list length must be greater than 2
    wait_flag, losscut_price, tgt_price = False, 0, 0

    if opposite_flag == False:
        if position == 'long':
            diff = abs(entry_price - embrace_low)
            if diff > max_margin:
                wait_flag = True
            elif diff < min_margin:
                if len(low) >= 2:
                    losscut_price = max(min(low[-1], low[-2], embrace_low), entry_price - min_margin)
                    tgt_price = entry_price + min_margin
                else:
                    losscut_price = max(min(low[-1], embrace_low), entry_price - min_margin)
                    tgt_price = entry_price + min_margin
            else:
                losscut_price = embrace_low
                tgt_price = entry_price + diff
        
        elif position == 'short':
            diff = abs(embrace_high - entry_price)
            if diff > max_margin:
                wait_flag = True
            elif diff < min_margin:
                if len(high) >= 2:
                    losscut_price = min(max(high[-1], high[-2], embrace_high), entry_price + min_margin)
                    tgt_price = entry_price - min_margin 
                else:
                    losscut_price = min(max(high[-1], embrace_high), entry_price + min_margin)
                    tgt_price = entry_price - min_margin
            else:
                losscut_price = embrace_high
                tgt_price = entry_price - diff
    
    elif opposite_flag == True:
        before_candle_high, before_candle_low = df['high'].iloc[-3], df['low'].iloc[-3]
        if position == 'long':
            diff = abs(embrace_high - entry_price)
            if diff > max_margin:
                wait_flag = True
            elif diff < min_margin:
                diff_before = abs(entry_price - before_candle_low)
                losscut_price = max(min(entry_price - min_margin, entry_price - diff_before), entry_price - diff)
                tgt_price = entry_price + profit_margin
            else:
                losscut_price = entry_price - diff
                tgt_price = max(entry_price + diff, entry_price + profit_margin)
        
        elif position == 'short':
            diff = abs(entry_price - embrace_low)
            if diff > max_margin:
                wait_flag = True
            elif diff < min_margin:
                diff_before = abs(before_candle_high - entry_price)
                losscut_price = min(max(entry_price + min_margin, entry_price + diff_before), entry_price + diff)
                tgt_price = entry_price - profit_margin
            else: 
                losscut_price = entry_price + diff
                tgt_price = min(entry_price - diff, entry_price - profit_margin)
    
    if position == 'volume_burst_long':
        diff = entry_price - df['low'].iloc[-2]
        if diff > 150:
            pass
                
    return wait_flag, losscut_price, tgt_price

def valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol):
    # liquidity = freeUSDT (당장 가용 가능한 USDT)

    if flag == False:
        leverage *= 2
        if leverage >= max_lev:
            leverage = max_lev
        try:
            binance.fapiPrivate_post_leverage({'symbol': Target_Coin_Symbol, 'leverage': leverage})
        except Exception as e:
            print("---:", e)
            #print(f"Last trading was loss --> leverage doubled to :{leverage}")
        amt = liquidity * leverage / entry_price 
        if amt < minimum_order_amt:
            amt = minimum_order_amt
            amt *= 2
            return amt, leverage
        else:
            try:
                binance.fapiPrivate_post_leverage({'symbol': Target_Coin_Symbol, 'leverage': leverage})
            except Exception as e:
                print("---:", e)
            #print(f"Last trading was loss --> leverage doubled to :{leverage}")
    if leverage >= max_lev:
        leverage = max_lev
    try:
        binance.fapiPrivate_post_leverage({'symbol': Target_Coin_Symbol, 'leverage': leverage})
    except Exception as e:
        print("---:", e)

    amt = liquidity * leverage / entry_price 
    
    if amt <= minimum_order_amt:
        amt = minimum_order_amt

    return amt, leverage

def init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker):
    
    timeout_end = time.time() + 60 * timeout_m
    tolerance_tick = 5
    tolerance = tolerance_tick * tick_size
    short, long = 'short', 'long'
    if short in position:
        binance.create_limit_sell_order(Target_Coin_Ticker, amt, entry_price)
        time.sleep(0.5)
        while True:
            try:
                df_temp = get_ohlcv(symbol = Target_Coin_Symbol, interval = '1m', limit = 30)
                order_l, status = Coin_func.b_checkopen_orders(Target_Coin_Symbol)
                time.sleep(0.5)
                if time.time() > timeout_end:
                    for order in order_l:
                        time.sleep(0.1)
                        binance.cancel_order(order, Target_Coin_Symbol)
                        time.sleep(0.1)
                    binance.create_market_sell_order(Target_Coin_Ticker, amt)
                    time.sleep(0.1)
                    break
                elif status == False:
                    break
                elif df_temp['close'].iloc[-1] <= entry_price - tolerance:
                    for order in order_l:
                        time.sleep(0.1)
                        binance.cancel_order(order, Target_Coin_Symbol)
                        time.sleep(0.1)
                    binance.create_market_sell_order(Target_Coin_Ticker, amt)
                    time.sleep(0.1)
                    break
            except Exception as e:
                continue
    
    elif long in position:
        binance.create_limit_buy_order(Target_Coin_Ticker, amt, entry_price)
        time.sleep(0.5)
        while True:
            try:
                df_temp = get_ohlcv(symbol = Target_Coin_Symbol, interval = '1m', limit = 30)
                order_l, status = Coin_func.b_checkopen_orders(Target_Coin_Symbol)
                time.sleep(0.5)
                if time.time() > timeout_end:
                    for order in order_l:
                        time.sleep(0.1)
                        binance.cancel_order(order, Target_Coin_Symbol)
                        time.sleep(0.1)
                    binance.create_market_buy_order(Target_Coin_Ticker, amt)
                    time.sleep(0.1)
                    break
                elif status == False:
                    break
                elif df_temp['close'].iloc[-1] >= entry_price + tolerance:
                    for order in order_l:
                        time.sleep(0.1)
                        binance.cancel_order(order, Target_Coin_Symbol)
                        time.sleep(0.1)
                    binance.create_market_buy_order(Target_Coin_Ticker, amt)
                    time.sleep(0.1)
                    break
            except Exception as e:
                continue

def calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df):
    short, long = 'short', 'long'
    amt = abs(amt)
    entry_usdt = entry_price * amt
    losscut_usdt = losscut_price * amt
    tgt_usdt = tgt_price * amt
    cur_usdt = df['close'].iloc[-1] * amt

    if short in position:
        loss_usdt = round(entry_usdt - losscut_usdt, 3)
        profit_usdt = round(entry_usdt - tgt_usdt, 3)
        loss_perc = round(loss_usdt / liquidity * 100, 3)
        profit_perc = round(profit_usdt / liquidity * 100, 3)
        cur_prof = round(entry_usdt - cur_usdt, 3)
   
    elif long in position:
        loss_usdt = round(losscut_usdt - entry_usdt, 3)
        profit_usdt = round(tgt_usdt - entry_usdt, 3)
        loss_perc = round(loss_usdt / liquidity * 100, 3)
        profit_perc = round(profit_usdt / liquidity * 100, 3)
        cur_prof = round(cur_usdt - entry_usdt, 3)

    cur_perc = round(cur_prof / entry_usdt * 100, 3)

    return loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc

def cur_position_check(balance, Target_Coin_Symbol):
    amt, entryPrice, leverage, unrealizedProfit = 0, 0, 1, 0
    for posi in balance['info']['positions']:
        if posi['symbol'] == Target_Coin_Symbol:
            leverage = float(posi['leverage'])
            entryPrice = float(posi['entryPrice'])
            unrealizedProfit = float(posi['unrealizedProfit'])
            amt = float(posi['positionAmt'])
    return amt, entryPrice, leverage, unrealizedProfit

def b_order_check(Target_Coin_Symbol, order_type):
    # order_type = "TAKE_PROFIT", "STOP_MARKET", "LIMIT"
    status = False
    order = binance.fetch_open_orders(Target_Coin_Symbol)
    time.sleep(0.3)
    order_l = []
    for each in order:
        order_l.append(each['info']['origType'])
    for order in order_l:
        if order == order_type:
            status = True
    return status

def SetStopLoss(binance, Ticker, stopPrice):
    time.sleep(0.1)
    orders = binance.fetch_orders(Ticker)

    StopLossOk = False
    for order in orders:

        if order['status'] == "open" and order['type'] == 'stop_market':
            #print(order)
            StopLossOk = True
            break

    #스탑로스 주문이 없다면 주문을 건다!
    if StopLossOk == False:

        time.sleep(1.0)

        #잔고 데이타를 가지고 온다.
        balance = binance.fetch_balance(params={"type": "future"})
        time.sleep(0.1)
                                
        amt = 0
        entryPrice = 0
        leverage = 0
        #평균 매입단가와 수량을 가지고 온다.
        for posi in balance['info']['positions']:
            if posi['symbol'] == Ticker.replace("/", ""):
                entryPrice = float(posi['entryPrice'])
                amt = float(posi['positionAmt'])
                leverage = float(posi['leverage'])


        #롱일땐 숏을 잡아야 되고
        side = "sell"
        #숏일땐 롱을 잡아야 한다.
        if amt < 0:
            side = "buy"

        params = {
            'stopPrice': stopPrice,
            'closePosition' : True
        }

        print("side:",side,"   stopPrice:",stopPrice, "   entryPrice:",entryPrice)
        #스탑 로스 주문을 걸어 놓는다.
        binance.create_order(Ticker,'STOP_MARKET',side,abs(amt),stopPrice,params)
        if b_order_check(Ticker, "STOP_MARKET") == False:
            SetStopLoss(binance, Ticker, stopPrice)

        print("####STOPLOSS SETTING DONE ######################")

def SetTakeProfit(binance, Ticker, target_price):
    time.sleep(0.1)
    orders = binance.fetch_orders(Ticker)

    TakeProfitOk = False
    for order in orders:

        if order['status'] == "open" and order['type'] == 'take_profit':
            #print(order)
            TakeProfitOk = True
            break

    #스탑로스 주문이 없다면 주문을 건다!
    if TakeProfitOk == False:

        time.sleep(1.0)

        #잔고 데이타를 가지고 온다.
        balance = binance.fetch_balance(params={"type": "future"})
        time.sleep(0.1)
                                
        amt = 0
        entryPrice = 0
        leverage = 0
        #평균 매입단가와 수량을 가지고 온다.
        for posi in balance['info']['positions']:
            if posi['symbol'] == Ticker.replace("/", ""):
                entryPrice = float(posi['entryPrice'])
                amt = float(posi['positionAmt'])
                leverage = float(posi['leverage'])


        #롱일땐 숏을 잡아야 되고
        side = "sell"
        #숏일땐 롱을 잡아야 한다.
        if amt < 0:
            side = "buy"

        params = {
            'stopPrice': target_price,
        }

        print("side:",side,"   stopPrice:",target_price, "   entryPrice:",entryPrice)
        #스탑 로스 주문을 걸어 놓는다.
        binance.create_order(Ticker,'TAKE_PROFIT',side,abs(amt),target_price,params)
        if b_order_check(Ticker, "TAKE_PROFIT") == False:
            SetTakeProfit(binance, Ticker, target_price)

        print("####TAKE_PROFIT SETTING DONE ######################")

def save_df_to_csv(df, position, path, result): # 
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    file_name = f"{cur_time}_{position}_{result}"
    df.to_csv(path + file_name, sep=',')

def Cancel_AllOrder(binance, Target_Coin_Symbol):
    order = binance.fetch_open_orders(Target_Coin_Symbol)
    time.sleep(0.5)
    for each in order:
        binance.cancel_order(each['id'], Target_Coin_Symbol)

def Close_AllPositions_whenprofit(binance, Target_Coin_Symbol, Target_Coin_Ticker):
    balance = binance.fetch_balance(params={"type":"future"})
    time.sleep(0.5)
    amt = cur_position_check(balance, Target_Coin_Symbol)[0]
    if amt < 0:
        binance.create_market_buy_order(Target_Coin_Ticker, amt)
    elif amt > 0:
        binance.create_market_sell_order(Target_Coin_Ticker, amt)
    elif amt == 0:
        print("All Position Already Closed")
    print("All Position Closed")

def condition_det(df, opposite_flag, embrace_high, embrace_low):
    position = None
    if df['close'].iloc[-2] > embrace_high:
        if opposite_flag == False:
            position = 'long'
        elif opposite_flag == True:
            position = 'short'
    elif df['close'].iloc[-2] < embrace_low: 
        if opposite_flag == False:
            position = 'short'
        elif opposite_flag == True: 
            position = 'long'
    return position 


# ================= Constants =======================
# for BTCUSDT Perpetual : may vary for other coins
flag = None
Target_Coin_Ticker = "BTC/USDT"
Target_Coin_Symbol = "BTCUSDT"
tick_size = 1
tolerance_tick = 2
tolerance = tolerance_tick * tick_size
timeout_m = 0.5
profit_margin = 100 
profit_margin_tolerance = 2 * tick_size # tick
loss_margin_tolerance = 1 * tick_size # tick
initial_leverage = 2
current_leverage = 1
minimum_order_amt = 0.001
sleep_time_m = 2
save_path = r"/Users/mac/Desktop/Binance_Autobot/CSV_Files_Saved" #r"C:\csv_file_saved/" # needs to be adjusted for window
amt_init = 0
# ====================
leverage = 5 # 시작 레버리지
default_leverage = 5 # 평소 거래 레버리지
# ====================
stophunting_timeout = 30
embrace_timeout = 6
min_margin = 75
max_margin = 150
max_lev = 10
liquidity_ratio = 2 # (if it is 1 --> using your all freeUSDT for one position)
trade_fee = 0
stop_hunting_activate = True
opposite_flag = False

# ================= ========= =======================

while True:
    try:
        df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
        time.sleep(0.2)
        # ====== Initialize ========
        direction = det_direction(df)
        position_closed = False
        embrace_timeout_flag = False
        # cnt_up, cnt_dwn = inflection_ctr(df, 10) # 10분
        high_val, low_val = 0, 0
        high, low = [], []
        # =========================

        print(f"Trading Pending... {Target_Coin_Symbol} current price at : {df['close'].iloc[-1]}, direction = {direction}")

        if direction == 'upper' and stop_hunting_activate == True:
            print(f"변곡점 발생 : {direction}")
            high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
            high.append(high_val)
            low.append(low_val)
            time.sleep(60)
            timeout_end = time.time() + 60 * stophunting_timeout

            while True:
                try: 
                    df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                    time.sleep(0.2)
                    direction = det_direction(df)
                    # print(f"Waiting for 갱신봉 one step before price is {df['close'].iloc[-2]}")

                    if time.time() > timeout_end:
                        print(f"갱신봉이 {timeout_end} 안에 발생하지 않음 --> 종료")
                        break

                    # elif direction == 'lower':
                    #     print(f"변곡점 스위칭 발생 : {direction} --> break")
                    #     break

                    elif position_closed == True:
                        print(f"All positions closed : 2")
                        break

                    elif embrace_timeout_flag == True:
                        print(f"Embrace candle time out")
                        break

                    elif df['close'].iloc[-2] >= high_val:
                        print(f"갱신봉 발생 {df['close'].iloc[-2]} > {high_val}")
                        # 갱신봉 고가/저가 update
                        high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
                        high.append(high_val)
                        low.append(low_val)
                        time.sleep(60) # wait until [-2] updates
                        embrace_find_timeout = time.time() + 60 * embrace_timeout

                        while True:
                            try:
                                df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                time.sleep(0.2)
                                direction = det_direction(df)
                                embrace, embrace_high, embrace_low = embrace_candle(df)
                                # print(f"안김 캔들 기다리는 중...{embrace}, {embrace_high}, {embrace_low}")

                                if df['close'].iloc[-2] >= high_val:
                                    print(f"갱신봉 발생 {df['close'].iloc[-2]} > {high_val} --> 타임아웃 시간 리셋 : {embrace_timeout}")
                                    high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
                                    high.append(high_val)
                                    low.append(low_val)
                                    time.sleep(60)
                                    embrace_find_timeout = time.time() + 60 * embrace_timeout
                                    continue

                                # elif direction == 'lower':
                                #     print(f"변곡점 스위칭 발생 : {direction} --> break")
                                #     break

                                elif position_closed == True:
                                    print(f"All positions closed : 1")
                                    break

                                elif time.time() > embrace_find_timeout:
                                    print(f"안김 캔들 발생안함 {time.time()} > {embrace_find_timeout}")
                                    embrace_timeout_flag = True
                                    break

                                elif embrace == True and embrace_high != 0 and embrace_low != 0:
                                    print(f"Embracing Candle occured high : {embrace_high}, low : {embrace_low}")
                                    time.sleep(60) # wait until [-2] updates
                                    while True:
                                        try:
                                            df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                            time.sleep(0.2)
                                            balance = binance.fetch_balance(params={"type":"future"})
                                            time.sleep(0.2)
                                            free_USDT = balance['USDT']['free']
                                            before_free_USDT = balance['USDT']['free']
                                            liquidity = free_USDT/liquidity_ratio # / 100

                                            position = condition_det(df, opposite_flag, embrace_high, embrace_low)

                                            if position == 'long': 
                                                # Below is a sequence of safely placing the initial order : This sequence would be used for both positions (short/long)
                                                # ==========================================================================================================================
                                                # =========================== Position Determined ==========================
                                                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
                                                msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
                                                print(msg)
                                                # line_alert.Sendmessage(msg) # Send Message
                                                # ==========================================================================
                                                # ========================= Making Initial Position ========================
                                                entry_price = df['close'].iloc[-1] # Getting Current Price
                                                # amt and leverage update
                                                amt, leverage = \
                                                    valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol)
                                                # Actually Buying the Coin at determined amt/leverage
                                                init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker)
                                                # ==========================================================================
                                                # =================== Rechecking entry price from balance ==================
                                                # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
                                                balance = binance.fetch_balance(params={"type":"future"})
                                                time.sleep(0.3) 
                                                # amt_act is the amt that actually successfully ordered
                                                amt_act, entryPrice, leverage, unrealizedProfit = cur_position_check(balance, Target_Coin_Symbol)
                                                if entryPrice != entry_price:
                                                    entry_price = entryPrice
                                                else:
                                                    entry_price = entry_price
                                                # ==========================================================================
                                                # ================ StopLoss/TakeProfit Calc and place order ================
                                                wait_flag, losscut_price, tgt_price \
                                                    = profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
                                                if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
                                                    position_closed = True
                                                    break
                                                SetStopLoss(binance, Target_Coin_Symbol, losscut_price) # Actually Making stoploss order
                                                SetTakeProfit(binance, Target_Coin_Symbol, tgt_price) # Actually Making takeProfit order
                                                # ==========================================================================
                                                msg = f"진입 포지션 : {position} \
                                                        진입 수량: {amt_act}, \
                                                        진입 가격: {entry_price}, \
                                                        손절 가: {losscut_price}, \
                                                        목표 가: {tgt_price} 설정 완료, \
                                                        현재 자산: {before_free_USDT}"
                                                print(msg)
                                                line_alert.Sendmessage(msg)
                                                # ==========================================================================================================================
                                                

                                                while True:
                                                    try:
                                                        df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                                        time.sleep(0.2)
                                                        status_sl = b_order_check(Target_Coin_Symbol, "STOP_MARKET")
                                                        status_tl = b_order_check(Target_Coin_Symbol, "TAKE_PROFIT")
                                                        time.sleep(0.1)

                                                        loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc = \
                                                            calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df)
                                                        print(f"Current Price is at {df['close'].iloc[-1]}, current profit is {cur_prof} USDT {cur_perc} %")

                                                        # =======================================
                                                        # Current price reached LosscutPrice 
                                                        # =======================================
                                                        if status_sl == False:
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with loss of {loss_usdt} USDT, {loss_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = False
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Loss')
                                                            time.sleep(30)
                                                            break
                                                        # =======================================
                                                        # Current price reached TgtPrice 
                                                        # =======================================
                                                        elif status_tl == False:
                                                            Close_AllPositions_whenprofit(binance, Target_Coin_Symbol, Target_Coin_Ticker)
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with profit of {profit_usdt} USDT, {profit_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = True
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Profit')
                                                            time.sleep(30)
                                                            break
                                                    except Exception as e:
                                                        print("Error : ", e)
                                                        continue

                                                print(f"{position} Position Completed")
                                                position_closed = True
                                                break

                                            elif position == 'short':
                                                # Below is a sequence of safely placing the initial order : This sequence would be used for both positions (short/long)
                                                # ==========================================================================================================================
                                                # =========================== Position Determined ==========================
                                                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
                                                msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
                                                print(msg)
                                                # line_alert.Sendmessage(msg) # Send Message
                                                # ==========================================================================
                                                # ========================= Making Initial Position ========================
                                                entry_price = df['close'].iloc[-1] # Getting Current Price
                                                # amt and leverage update
                                                amt, leverage = \
                                                    valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol)
                                                # Actually Buying the Coin at determined amt/leverage
                                                init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker)
                                                # ==========================================================================
                                                # =================== Rechecking entry price from balance ==================
                                                # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
                                                balance = binance.fetch_balance(params={"type":"future"})
                                                time.sleep(0.3) 
                                                # amt_act is the amt that actually successfully ordered
                                                amt_act, entryPrice, leverage, unrealizedProfit = cur_position_check(balance, Target_Coin_Symbol)
                                                if entryPrice != entry_price:
                                                    entry_price = entryPrice
                                                else:
                                                    entry_price = entry_price
                                                # ==========================================================================
                                                # ================ StopLoss/TakeProfit Calc and place order ================
                                                wait_flag, losscut_price, tgt_price \
                                                    = profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
                                                if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
                                                    position_closed = True
                                                    break
                                                SetStopLoss(binance, Target_Coin_Symbol, losscut_price) # Actually Making stoploss order
                                                SetTakeProfit(binance, Target_Coin_Symbol, tgt_price) # Actually Making takeProfit order
                                                # ==========================================================================
                                                msg = f"진입 포지션 : {position} \
                                                        진입 수량: {amt_act}, \
                                                        진입 가격: {entry_price}, \
                                                        손절 가: {losscut_price}, \
                                                        목표 가: {tgt_price} 설정 완료, \
                                                        현재 자산: {before_free_USDT}"
                                                print(msg)
                                                line_alert.Sendmessage(msg)
                                                # ==========================================================================================================================


                                                while True:
                                                    try:
                                                        df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                                        time.sleep(0.2)
                                                        status_sl = b_order_check(Target_Coin_Symbol, "STOP_MARKET")
                                                        status_tl = b_order_check(Target_Coin_Symbol, "TAKE_PROFIT")
                                                        time.sleep(0.1)

                                                        loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc = \
                                                            calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df)
                                                        print(f"Current Price is at {df['close'].iloc[-1]}, current profit is {cur_prof} USDT {cur_perc} %")

                                                        # =======================================
                                                        # Current price reached LosscutPrice 
                                                        # =======================================
                                                        if status_sl == False:
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with loss of {loss_usdt} USDT, {loss_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = False
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Loss')
                                                            time.sleep(30)
                                                            break
                                                        # =======================================
                                                        # Current price reached TgtPrice 
                                                        # =======================================
                                                        elif status_tl == False:
                                                            Close_AllPositions_whenprofit(binance, Target_Coin_Symbol, Target_Coin_Ticker)
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with profit of {profit_usdt} USDT, {profit_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = True
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Profit')
                                                            time.sleep(30)
                                                            break
                                                    except Exception as e:
                                                        print("Error :", e)
                                                        continue

                                                position_closed = True
                                                print("Short Trading Completed")
                                                break
                                        
                                        except Exception as e:
                                            print("Error : ", e)
                                            break

                            except Exception as e:
                                print("Error : ", e)
                                continue

                except Exception as e:
                    print("Error : ", e)
                    continue

        elif direction == 'lower' and stop_hunting_activate == True:
            print(f"변곡점 발생 : {direction}")
            high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
            high.append(high_val)
            low.append(low_val)
            time.sleep(60)
            timeout_end = time.time() + 60 * stophunting_timeout

            while True:
                try:
                    df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                    time.sleep(0.2)
                    direction = det_direction(df)
                    # print(f"Waiting for 갱신봉 one step before price is {df['close'].iloc[-2]}")
                    
                    if time.time() > timeout_end:
                        print(f"갱신봉이 {timeout_end} 안에 발생하지 않음 --> 종료")
                        break

                    # elif direction == 'upper':
                    #     print(f"변곡점 스위칭 발생 : {direction} --> break")
                    #     break

                    elif position_closed == True:
                        print(f"All positions closed : 2")
                        break

                    elif embrace_timeout_flag == True:
                        print(f"Embrace candle time out")
                        break

                    elif df['close'].iloc[-2] <= low_val:
                        print(f"갱신봉 발생 {df['close'].iloc[-2]} < {low_val}")
                        # 갱신봉 고가/저가 update
                        high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
                        high.append(high_val)
                        low.append(low_val)
                        time.sleep(60) # wait until [-2] updates
                        embrace_find_timeout = time.time() + 60 * embrace_timeout
                        

                        while True:
                            try: 
                                df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                time.sleep(0.2)
                                direction = det_direction(df)
                                embrace, embrace_high, embrace_low = embrace_candle(df)
                                # print(f"안김 캔들 기다리는 중...{embrace}, {embrace_high}, {embrace_low}")

                                if df['close'].iloc[-2] <= low_val:
                                    print(f"갱신봉 발생 {df['close'].iloc[-2]} < {low_val} --> 타임아웃 시간 리셋 : {embrace_timeout}")
                                    high_val, low_val = df['high'].iloc[-2], df['low'].iloc[-2]
                                    high.append(high_val)
                                    low.append(low_val)
                                    time.sleep(60)
                                    embrace_find_timeout = time.time() + 60 * embrace_timeout
                                    continue
                                
                                # elif direction == 'upper':
                                #     print(f"변곡점 스위칭 발생 : {direction} --> break")
                                #     break

                                elif position_closed == True:
                                    print(f"All positions closd : 1")
                                    break

                                elif time.time() > embrace_find_timeout:
                                    print(f"안김 캔들 발생안함 {time.time()} > {embrace_find_timeout}")
                                    embrace_timeout_flag = True
                                    break

                                elif embrace == True and embrace_high != 0 and embrace_low != 0:
                                    print(f"Embracing Candle occured high : {embrace_high}, low : {embrace_low}")
                                    time.sleep(60) # wait until [-2] updates
                                    while True:
                                        try: 
                                            df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                            time.sleep(0.2)
                                            balance = binance.fetch_balance(params={"type":"future"})
                                            time.sleep(0.2)
                                            free_USDT = balance['USDT']['free']
                                            before_free_USDT = balance['USDT']['free']
                                            liquidity = free_USDT/liquidity_ratio # / 100

                                            position = condition_det(df, opposite_flag, embrace_high, embrace_low)

                                            if position == 'long':
                                                # Below is a sequence of safely placing the initial order : This sequence would be used for both positions (short/long)
                                                # ==========================================================================================================================
                                                # =========================== Position Determined ==========================
                                                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
                                                msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
                                                print(msg)
                                                # line_alert.Sendmessage(msg) # Send Message
                                                # ==========================================================================
                                                # ========================= Making Initial Position ========================
                                                entry_price = df['close'].iloc[-1] # Getting Current Price
                                                # amt and leverage update
                                                amt, leverage = \
                                                    valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol)
                                                # Actually Buying the Coin at determined amt/leverage
                                                init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker)
                                                # ==========================================================================
                                                # =================== Rechecking entry price from balance ==================
                                                # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
                                                balance = binance.fetch_balance(params={"type":"future"})
                                                time.sleep(0.3) 
                                                # amt_act is the amt that actually successfully ordered
                                                amt_act, entryPrice, leverage, unrealizedProfit = cur_position_check(balance, Target_Coin_Symbol)
                                                if entryPrice != entry_price:
                                                    entry_price = entryPrice
                                                else:
                                                    entry_price = entry_price
                                                # ==========================================================================
                                                # ================ StopLoss/TakeProfit Calc and place order ================
                                                wait_flag, losscut_price, tgt_price \
                                                    = profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
                                                if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
                                                    position_closed = True
                                                    break
                                                SetStopLoss(binance, Target_Coin_Symbol, losscut_price) # Actually Making stoploss order
                                                SetTakeProfit(binance, Target_Coin_Symbol, tgt_price) # Actually Making takeProfit order
                                                # ==========================================================================
                                                msg = f"진입 포지션 : {position} \
                                                        진입 수량: {amt_act}, \
                                                        진입 가격: {entry_price}, \
                                                        손절 가: {losscut_price}, \
                                                        목표 가: {tgt_price} 설정 완료, \
                                                        현재 자산: {before_free_USDT}"
                                                print(msg)
                                                line_alert.Sendmessage(msg)
                                                # ==========================================================================================================================

                                                while True:
                                                    try:
                                                        df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                                        time.sleep(0.2)
                                                        status_sl = b_order_check(Target_Coin_Symbol, "STOP_MARKET")
                                                        status_tl = b_order_check(Target_Coin_Symbol, "TAKE_PROFIT")
                                                        time.sleep(0.1)

                                                        loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc = \
                                                            calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df)
                                                        print(f"Current Price is at {df['close'].iloc[-1]}, current profit is {cur_prof} USDT {cur_perc} %")

                                                        # =======================================
                                                        # Current price reached LosscutPrice 
                                                        # =======================================
                                                        if status_sl == False:
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with loss of {loss_usdt} USDT, {loss_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = False
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Loss')
                                                            time.sleep(30)
                                                            break
                                                        # =======================================
                                                        # Current price reached TgtPrice 
                                                        # =======================================
                                                        elif status_tl == False:
                                                            Close_AllPositions_whenprofit(binance, Target_Coin_Symbol, Target_Coin_Ticker)
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with profit of {profit_usdt} USDT, {profit_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = True
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Profit')
                                                            time.sleep(30)
                                                            break
                                                    
                                                    except Exception as e:
                                                        print("Error : ", e)
                                                        continue
                                                    
                                                print(f"{position} Position Completed")
                                                position_closed = True
                                                break

                                            elif position == 'short':
                                                # Below is a sequence of safely placing the initial order : This sequence would be used for both positions (short/long)
                                                # ==========================================================================================================================
                                                # =========================== Position Determined ==========================
                                                cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
                                                msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
                                                print(msg)
                                                # line_alert.Sendmessage(msg) # Send Message
                                                # ==========================================================================
                                                # ========================= Making Initial Position ========================
                                                entry_price = df['close'].iloc[-1] # Getting Current Price
                                                # amt and leverage update
                                                amt, leverage = \
                                                    valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol)
                                                # Actually Buying the Coin at determined amt/leverage
                                                init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker)
                                                # ==========================================================================
                                                # =================== Rechecking entry price from balance ==================
                                                # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
                                                balance = binance.fetch_balance(params={"type":"future"})
                                                time.sleep(0.3) 
                                                # amt_act is the amt that actually successfully ordered
                                                amt_act, entryPrice, leverage, unrealizedProfit = cur_position_check(balance, Target_Coin_Symbol)
                                                if entryPrice != entry_price:
                                                    entry_price = entryPrice
                                                else:
                                                    entry_price = entry_price
                                                # ==========================================================================
                                                # ================ StopLoss/TakeProfit Calc and place order ================
                                                wait_flag, losscut_price, tgt_price \
                                                    = profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
                                                if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
                                                    position_closed = True
                                                    break
                                                SetStopLoss(binance, Target_Coin_Symbol, losscut_price) # Actually Making stoploss order
                                                SetTakeProfit(binance, Target_Coin_Symbol, tgt_price) # Actually Making takeProfit order
                                                # ==========================================================================
                                                msg = f"진입 포지션 : {position} \
                                                        진입 수량: {amt_act}, \
                                                        진입 가격: {entry_price}, \
                                                        손절 가: {losscut_price}, \
                                                        목표 가: {tgt_price} 설정 완료, \
                                                        현재 자산: {before_free_USDT}"
                                                print(msg)
                                                line_alert.Sendmessage(msg)
                                                # ==========================================================================================================================

                                                while True:
                                                    try:
                                                        df = median(Target_Coin_Symbol, interval = '1m', limit = 50, window = 20)
                                                        time.sleep(0.2)
                                                        status_sl = b_order_check(Target_Coin_Symbol, "STOP_MARKET")
                                                        status_tl = b_order_check(Target_Coin_Symbol, "TAKE_PROFIT")
                                                        time.sleep(0.1)

                                                        loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc = \
                                                            calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df)
                                                        print(f"Current Price is at {df['close'].iloc[-1]}, current profit is {cur_prof} USDT {cur_perc} %")

                                                        # =======================================
                                                        # Current price reached LosscutPrice 
                                                        # =======================================
                                                        if status_sl == False:
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with loss of {loss_usdt} USDT, {loss_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = False
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Loss')
                                                            time.sleep(30)
                                                            break
                                                        # =======================================
                                                        # Current price reached TgtPrice 
                                                        # =======================================
                                                        elif status_tl == False:
                                                            Close_AllPositions_whenprofit(binance, Target_Coin_Symbol, Target_Coin_Ticker)
                                                            Cancel_AllOrder(binance, Target_Coin_Symbol)
                                                            balance = binance.fetch_balance(params={"type":"future"})
                                                            time.sleep(0.2)
                                                            after_free_USDT = balance['USDT']['free']
                                                            msg = f"{Target_Coin_Symbol} : All positions closed with profit of {profit_usdt} USDT, {profit_perc}%, \
                                                                remaining balance : {after_free_USDT}"
                                                            print(msg)
                                                            line_alert.Sendmessage(msg)
                                                            flag = True
                                                            # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                                                            # save_df_to_csv(df_temp, position, save_path, result = 'Profit')
                                                            time.sleep(30)
                                                            break
                                                    
                                                    except Exception as e:
                                                        print("Error : ", e)
                                                        continue

                                                position_closed = True
                                                print("Short Trading Completed")
                                                break

                                        except Exception as e:
                                            print("Error :", e)
                                            break

                            except Exception as e:
                                print("Error :", e)
                                continue

                except Exception as e:
                    print("Error :", e)
                    continue

        elif direction == 'volume_burst_long':
            position = direction
            pass
            # # Below is a sequence of safely placing the initial order : This sequence would be used for both positions (short/long)
            # # ==========================================================================================================================
            # # =========================== Position Determined ==========================
            # cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
            # msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
            # print(msg)
            # # line_alert.Sendmessage(msg) # Send Message
            # # ==========================================================================
            # # ========================= Making Initial Position ========================
            # entry_price = df['close'].iloc[-1] # Getting Current Price
            # # amt and leverage update
            # amt, leverage = \
            #     valid_amt_check(flag, liquidity, leverage, entry_price, max_lev, minimum_order_amt, Target_Coin_Symbol)
            # # Actually Buying the Coin at determined amt/leverage
            # init_make_position(amt, entry_price, position, tolerance_tick, tick_size, timeout_m, Target_Coin_Symbol, Target_Coin_Ticker)
            # # ==========================================================================
            # # =================== Rechecking entry price from balance ==================
            # # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
            # balance = binance.fetch_balance(params={"type":"future"})
            # time.sleep(0.3) 
            # # amt_act is the amt that actually successfully ordered
            # amt_act, entryPrice, leverage, unrealizedProfit = cur_position_check(balance, Target_Coin_Symbol)
            # if entryPrice != entry_price:
            #     entry_price = entryPrice
            # else:
            #     entry_price = entry_price
            # # ==========================================================================
            # # ================ StopLoss/TakeProfit Calc and place order ================
            # wait_flag, losscut_price, tgt_price \
            #     = profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
            # if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
            #     position_closed = True
            #     break
            # SetStopLoss(binance, Target_Coin_Symbol, losscut_price) # Actually Making stoploss order
            # SetTakeProfit(binance, Target_Coin_Symbol, tgt_price) # Actually Making takeProfit order
            # # ==========================================================================
            # msg = f"진입 포지션 : {position} \
            #         진입 수량: {amt_act}, \
            #         진입 가격: {entry_price}, \
            #         손절 가: {losscut_price}, \
            #         목표 가: {tgt_price} 설정 완료, \
            #         현재 자산: {before_free_USDT}"
            # print(msg)
            # line_alert.Sendmessage(msg)
            # # ==========================================================================================================================
            


    except Exception as e:
        print("Error :", e)
        continue





