
import re
import pyupbit
import ccxt
import time
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import line_alert
# import ende_key
# import my_key

class SimpleEnDecrypt:
    def __init__(self, key = True):
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        self.f = Fernet(self.key)
    
    def encrypt(self, data, is_our_string = True):
        if isinstance(data, bytes):
            ou = self.f.encrypt(data)
        else:
            ou = self.f.encrypt(data.encode('utf-8'))
        if is_our_string is True:
            return ou.decode('utf-8')
        else:
            return ou
        
    def decrypt(self, data, is_our_string = True):
        if isinstance(data,bytes):
            ou = self.f.decrypt(data)
        else:
            ou = self.f.decrypt(data.encode('utf-8'))
        if is_our_string is True:
            return ou.decode('utf-8')
        else:
            return ou

access = "jfbFgAVyVOMf6jNeIvLztWOOXZA1jjE883QRtVg4zvSJAqEQdoGAc1OxqokRNbPg"
secret = "qlmcYflp22OOLNNwnez7TSh5TUpDA5AB2o0Q1FQI0m64ensMVlPzhpCVBwws7Pqo"

binance = ccxt.binance(config={
    'apiKey' : access,
    'secret' : secret,
    'enableRateLimit' : True,
    # 'adjustForTimeDifference' : True,
    'options' : {
        'defaultType' : 'future'
    }
})

# SimpleEnDecrypt = SimpleEnDecrypt(ende_key.ende_key)
# Upbit_AccessKey = SimpleEnDecrypt.decrypt(my_key.upbit_access)
# Upbit_Secretkey = SimpleEnDecrypt.decrypt(my_key.upbit_secret)

# upbit = pyupbit.Upbit(Upbit_AccessKey, Upbit_Secretkey)


#ohlc = df 형식, period = 설정가능, st = 기준날짜 예) st = -1 (오늘), -2 (어제)
def GetRSI(ohlcv, period, st): 
    ohlcv["close"] = ohlcv["close"]
    delta = ohlcv["close"].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period - 1), min_periods = period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods = period).mean()
    RS = _gain / _loss
    return float(pd.Series(100 - (100 / (1 + RS)), name = "RSI").iloc[st])

# 미분치 구하기
def differentiate(xlist, ylist):
    yprime = np.diff(ylist)/np.diff(xlist)
    xprime = []
    for i in range(len(yprime)):
        xtemp = (xlist[i+1] + xlist[i])/2
        xprime = np.append(xprime, xtemp)
    return xprime, yprime

# 이평선 구하기
def GetMA_cur(ohlcv, period, st):
    close = ohlcv["close"]
    ma = close.rolling(period).mean()
    return float(ma[st])

# 종가 
def GetMA_per(ohlcv, period):
    close = ohlcv["close"]
    ma = close.rolling(period).mean()
    return float(ma)

# 거래량 이동평균 선
def GetMA_per_Vol(ohlcv, period, st):
    volume = ohlcv["volume"]
    ma = volume.rolling(period).mean()
    return float(ma[st])

def moving_avg (a, n):
    ret = np.cumsum(a, dtype = float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 거래대금 상위 코인 리스트 구하기
def GetTopCoinList(interval,top):
    print("--------------GetTopCoinList Start-------------------")
    Tickers = pyupbit.get_tickers("KRW")

    dic_coin_money = dict()
    for ticker in Tickers:
        try:
            df = pyupbit.get_ohlcv(ticker,interval)
            #최근 2개 캔들의 종가와 거래량을 곱하여 대략의 거래대금을 구합니다.
            volume_money = (df['close'][-1] * df['volume'][-1]) + (df['close'][-2] * df['volume'][-2])
            #volume_money = float(df['value'][-1]) + float(df['value'][-2]) #거래대금! value가 거래대금이었네요.. 이걸 이제야 알다니 ㅎ
            #이걸 위에서 만든 딕셔너리에 넣어줍니다. Key는 코인의 티커, Value는 위에서 구한 거래대금 
            dic_coin_money[ticker] = volume_money
            #출력해 봅니다.
            print(ticker, dic_coin_money[ticker])
            #반드시 이렇게 쉬어줘야 합니다. 안그럼 에러가.. 시간조절을 해보시며 최적의 시간을 찾아보세요 전 일단 0.1로 수정했어요!
            time.sleep(0.1)

        except Exception as e:
            print("exception:",e)

    #딕셔너리를 값으로 정렬하되 숫자가 큰 순서대로 정렬합니다.
    dic_sorted_coin_money = sorted(dic_coin_money.items(), key = lambda x : x[1], reverse= True)

    #빈 리스트를 만듭니다.
    coin_list = list()

    #코인을 셀 변수를 만들어요.
    cnt = 0

    #티커와 거래대금 많은 순으로 정렬된 딕셔너리를 순회하면서 
    for coin_data in dic_sorted_coin_money:
        #코인 개수를 증가시켜주는데..
        cnt += 1

        #파라메타로 넘어온 top의 수보다 작으면 코인 리스트에 코인 티커를 넣어줍니다.
        #즉 top에 10이 들어갔다면 결과적으로 top 10에 해당하는 코인 티커가 coin_list에 들어갑니다.
        if cnt <= top:
            coin_list.append(coin_data[0])
        else:
            break

    print("--------------GetTopCoinList End-------------------")

    #코인 리스트를 리턴해 줍니다.
    return coin_list

# 해당되는 리스트안에 해당 코인이 있는 확인하기
def CheckCoinInList(CoinList, Ticker):
    InCoinOk = False

    for coinTicker in CoinList:
        if coinTicker == Ticker:
            InCoinOk = True
            break
    
    return InCoinOk

# 티커에 해당하는 코인의 수익률 구하기
def GetRevenueRate(balances,Ticker):
    revenue_rate = 0.0
    for value in balances:
        try:
            realTicker = value['unit_currency'] + "-" + value['currency']
            if Ticker == realTicker:
                time.sleep(0.05)
                
                #현재 가격을 가져옵니다.
                nowPrice = pyupbit.get_current_price(realTicker)

                #수익율을 구해서 넣어줍니다
                revenue_rate = (float(nowPrice) - float(value['avg_buy_price'])) * 100.0 / float(value['avg_buy_price'])
                break

        except Exception as e:
            print("GetRevenueRate error:", e)

    return revenue_rate

# 티커에 해당하는 코인이 매수된 상태면 참을 리턴하기
def IsHasCoin(balances,Ticker):
    HasCoin = False
    for value in balances:
        realTicker = value['unit_currency'] + "-" + value['currency']
        if Ticker == realTicker:
            HasCoin = True
    return HasCoin

# 내가 매수한 코인 개수 구하기
def GetHasCoinCnt(balances):
    CoinCnt = 0
    for value in balances:
        avg_buy_price = float(value['avg_buy_price'])
        if avg_buy_price != 0: # 드랍받은 코인은 avg_buy_price = 0
            CoinCnt += 1
    return CoinCnt

def GetHasCoin(balances):
    hasCoin_list = []
    for value in balances:
        ticker = value['currency']
        if ticker == "KRW":
            continue
        else:
            hasCoin_list.append("KRW-" + ticker)
    return hasCoin_list

# 총원금 구하기
def GetTotalMoney(balances):
    total = 0.0
    for value in balances:
        try:
            ticker = value['currency']
            if ticker == "KRW":
                total += (float(value['balance']) + float(value['locked']))
            else:
                avg_buy_price = float(value['avg_buy_price'])
            
                if avg_buy_price != 0 and (float(value['balance']) != 0 or float(value['locked']) != 0):
                    total += (avg_buy_price * (float(value['balance']) + float(value['locked'])))
        except Exception as e:
            print("GetTotalMoney error : ", e)
    
    return total

# 총 평가금액 구하기
def GetTotalRealMoney(balances):
    total = 0.0
    for value in balances:
        try:
            ticker = value['currency']
            if ticker == "KRW":
                total += (float(value['balance']) + float(value['locked']))
            else: 

                avg_buy_price = float(value['avg_buy_price'])
                if avg_buy_price != 0 and (float(value['balance']) != 0 or float(value['locked']) != 0): 
                    realTicker = value['unit_currency'] + "-" + value['currency']

                    time.sleep(0.1)
                    nowPrice = pyupbit.get_current_price(realTicker)
                    total += (float(nowPrice) * (float(value['balance']) + float(value['locked'])))
        except Exception as e:
            print("GetTotalRealMoney : ", e)
    
    return total

def eachcoinprice_revenue(balances, ticker):
    df = pyupbit.get_ohlcv(ticker, interval = "minute1", count = 10) 
    ticker = ticker.replace("KRW-","")
    for value in balances:
        if value['currency'] == ticker:
            each_coin_pos = float(value['balance'])
            each_coin_avg_price = float(value['avg_buy_price'])
            mesu_avg_price = round(each_coin_avg_price * each_coin_pos)

            my_amt_money = each_coin_pos * df["close"].iloc[-1]
            my_revenue = (my_amt_money - mesu_avg_price)/mesu_avg_price * 100
    return my_amt_money, my_revenue


# 가용 가능한 금액 구하기 (매수 가능한 금액)
def spendable_KRW(balances):
    for value in balances:
        try:
            ticker = value['currency']
            if ticker == "KRW":
                total = float(value['balance'])
        except Exception as e:
            continue
    return int(total)

# 총 수익률 구하기 (총금액 대비 수익률임 : 업비트에서 보여지는 수익률과 다름)
def GetTotalRevenue(balances):
    TotalMoney = GetTotalMoney(balances)
    TotalRealMoney = GetTotalRealMoney(balances)
    TotalRevenue = (TotalRealMoney - TotalMoney) * 100.0 / TotalMoney
    return TotalRevenue

def counter(signal, cnt):
    if signal > 0:
        cnt += 1
    elif signal == 0:
        cnt += 0
    else:
        cnt -= 1
    return cnt

def differ(signal, signal_before, cnt):
    if signal > signal_before:
        cnt += 1
    elif signal == signal_before:
        cnt += 0
    else:
        cnt -= 0

def GetCoinHalfMaxPrice(Tickers, fluc = 0.5, interval = "day", count = 365):
    invest_list = []
    for ticker in Tickers:
        df = pyupbit.get_ohlcv(ticker, interval, count)
        df_m = max(df["close"].tolist())
        df_m_half = df_m * fluc
        cur_price = df["close"].iloc[-1]
        if cur_price < df_m_half:
            invest_list.append(ticker)
            #print("Coin :", ticker, "1 year max price :", df_m, "current price :", cur_price)
    return invest_list

# txt 파일로 저장된 data processing
def split_list(list):
    length = len(list)
    a_list = list[0:int(length/2)]
    b_list = list[int(length/2):]
    for i in range(len(b_list)):
        b_list[i] = float(b_list[i])
    return a_list, b_list

# 투자 대상 코인 중 balances에 있는 코인 제거
def split_remove_coin(list, balances):
    a_list, b_list = split_list(list)
    coin_list = []
    price_list = []
    for i in range(len(a_list)):
        #print(a_list[i])
        if IsHasCoin(balances, a_list[i]) == True:
            continue
        else:
            coin_list.append(a_list[i])
            price_list.append(b_list[i])
    return coin_list, price_list

def split_txt(list):
    a_list, b_list = split_list(list)
    coin_list = []
    price_list = []
    for i in range(len(a_list)):

        coin_list.append(a_list[i])
        price_list.append(b_list[i])
    return coin_list, price_list

def txttolist(filename):
    file = open(filename, "r")
    file_string = file.readlines()
    file.close()
    resulting_list = []
    for item in file_string:
        item = item.replace("\n", "")
        resulting_list.append(item)
    return resulting_list

def save_list(file, list):
    textfile = open(file, "w")
    for element in list:
        textfile.write(str(element) + "\n")
    textfile.close()

def bollingerband(data, window): # data = close 
    sma = data.rolling(window = window).mean()
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return sma, upper_bb, lower_bb

def ichimoku_cloud(df):
    nine_period_high = df['high'].rolling(window=9).max()
    nine_period_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low)/2

    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low)/2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen'])/2).shift(26)

    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high+period52_low)/2).shift(26)

    df['chikou_span'] = df['close'].shift(-26)
    return df

# balances = upbit.get_balances()

Tickers = pyupbit.get_tickers("KRW")

#print(balances)



# ====================== Binance Future Functions ========================================
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

def ichimoku_cloud(df):
    nine_period_high = df['high'].rolling(window=9).max()
    nine_period_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low)/2

    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low)/2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen'])/2).shift(26)

    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high+period52_low)/2).shift(26)

    df['chikou_span'] = df['close'].shift(-26)
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

def df_comb(target_coin_symbol, interval, limit, window, period):
    # target_coin_symbol = trading coin symbol
    # interval = delta time (1m, 5m, 15m, 30m, 1h, 1d ...)
    # limit = number of data 
    # window = bollingerband window (normally 20)
    # period = period of Moving Average for volume
    df = get_ohlcv(symbol = target_coin_symbol, interval = interval, limit = limit)
    df = ichimoku_cloud(df)
    df = bollingerband(df, window)
    df = GetMA_Vol(df, period)
    return df

def median(target_coin_symbol, interval, limit, window):
    df = get_ohlcv(target_coin_symbol, interval = interval, limit = limit)
    df = bollingerband(df, window = window)
    df['median'] = (df['high'] + df['low'])/2
    return df

def GetMA_Vol(df, period):
    volume = df['volume']
    ma = volume.rolling(period).mean()
    df['ma_vol'] = ma
    return df

def det_direction(df):
    direction = None
    if df['bb_diff'].iloc[-2] > 100:
        if df['close'].iloc[-2] > df['upper_bb'].iloc[-2]:
        #df['median'].iloc[-2] > df['median'].iloc[-3] > df['median'].iloc[-4]:
            direction = 'upper'
        
        elif df['close'].iloc[-2] < df['lower_bb'].iloc[-2]:
            #df['median'].iloc[-2] < df['median'].iloc[-3] < df['median'].iloc[-4]:
            direction = 'lower'
            
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

def profit_loss_calc(position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin):
    # low, high list length must be greater than 2
    wait_flag, losscut_price, tgt_price = False, 0, 0
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

    return wait_flag, losscut_price, tgt_price

def Loss_tgt_price_calc(entry_price, position, tick_size, water = 50, tgt = 100):
    # water_price
    water_l = []
    water_price = entry_price
    web = 4
    init_tgt_price = 0 
    short, long = 'short', 'long'
    if long in position:
        init_tgt_price = entry_price + tgt * tick_size
        for _ in range(web):
            water_price -= water * tick_size
            water_l.append(water_price)
        losscut_price = water_l[-1] - water * tick_size
    
    elif short in position:
        init_tgt_price = entry_price - tgt * tick_size
        for _ in range(web):
            water_price += water * tick_size
            water_l.append(water_price)
        losscut_price = water_l[-1] + water * tick_size
    
    return init_tgt_price, losscut_price, water_l

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

def b_checkopen_orders(Target_Coin_Symbol):
    time.sleep(0.1)
    order = binance.fetch_open_orders(Target_Coin_Symbol)
    order_l = []
    for each in order:
        order_l.append(each['info']['orderId'])
    
    if len(order_l) == 0:
        status = False
        return order_l, status
    else: 
        status = True
        return order_l, status

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
                order_l, status = b_checkopen_orders(Target_Coin_Symbol)
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
                order_l, status = b_checkopen_orders(Target_Coin_Symbol)
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

def daily_cumulative_profit(trades):
    # 오늘 총 거래의 이득 값 계산
    cur_time = time.strftime('%m-%d', time.localtime(time.time()))
    cumulative_profit = 0
    for i in range(len(trades)):
        if cur_time in trades[i]['datetime']:
            cumulative_profit += float(trades[i]['info']['realizedPnl'])
    return cumulative_profit, cur_time

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

def trade_sendmsg(binance, Target_Coin_Symbol):
    trades = binance.fetch_my_trades(Target_Coin_Symbol)
    time.sleep(0.2)
    cumulative_profit = daily_cumulative_profit(trades)[0]
    msg_cum_profit = f"Cumulative_profit from the last trade is {round(cumulative_profit,3)} USDT"
    line_alert.Sendmessage(msg_cum_profit)

def Cancel_AllOrder(binance, Target_Coin_Symbol):
    order = binance.fetch_open_orders(Target_Coin_Symbol)
    time.sleep(0.5)
    for each in order:
        binance.cancel_order(each['id'], Target_Coin_Symbol)

def Cancel_TLOrder(binance, Target_Coin_Symbol):
    order = binance.fetch_open_orders(Target_Coin_Symbol)
    time.sleep(0.5)
    for each in order:
        if each['info']['origType'] == 'TAKE_PROFIT':
            binance.cancel_order(each['id'], Target_Coin_Symbol)
    print("Canceled TL_Order")

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



'''

def dataSendLoop(addData_callbackFunc):

    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    while True:
        try:
            ticker = "KRW-BTC"
            time_step = 1
            time.sleep(time_step)
            df_60 = pyupbit.get_ohlcv(ticker, interval = "minute1")
            rsi60_min = GetRSI(df_60, 14, -1)
            _movavg_5_cur = GetMA(df_60, 5, -1)
            _movavg_5_cur_before = GetMA(df_60, 5, -2)
            slope = (_movavg_5_cur - _movavg_5_cur_before)/time_step
            print("Slope : ", slope)
            mySrc.data_signal.emit(slope)

        except Exception as e:
            print("Error :", e)

if __name__== '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())


#내가 가진 잔고 데이터를 다 가져온다.
balances = upbit.get_balances()


#거래대금이 많은 탑코인 10개의 리스트
#여러분의 전략대로 마음껏 바꾸세요. 
#첫번째 파라메타에 넣을 수 있는 값 day/minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/week/month
#ex) TopCoinList = GetTopCoinList("minute10",30) <- 최근 10여분 동안 거래대금이 많은 코인 30개를 리스트로 리턴
TopCoinList = GetTopCoinList("week",10)

#제외할 코인들을 넣어두세요. 상폐예정이나 유의뜬 코인등등 원하는 코인을 넣어요! 
DangerCoinList = ['KRW-MARO','KRW-TSHP','KRW-PXL']

#만약 나는 내가 원하는 코인만 지정해서 사고 싶다면 여기에 코인 티커를 넣고 아래 for문에서 LovelyCoinList를 활용하시면 되요!
LovelyCoinList = ['KRW-BTC','KRW-ETH','KRW-DOGE','KRW-DOT']

Tickers = pyupbit.get_tickers("KRW")

for ticker in Tickers:
    try:
        #거래량 많은 탑코인 리스트안의 코인이 아니라면 스킵! 탑코인에 해당하는 코인만 이후 로직을 수행한다.
        if CheckCoinInList(TopCoinList,ticker) == False:
            continue
        #위험한 코인이라면 스킵!!!
        if CheckCoinInList(DangerCoinList,ticker) == True:
            continue
        #나만의 러블리만 사겠다면 여기 주석을 풀고 위의 2부분을 주석처리 한다 
        #if CheckCoinInList(LovelyCoinList,ticker) == False:
        #    continue

        #이렇게 쉬어주는거 잊지 마세요!
        time.sleep(0.1)

        #60분봉 1시간봉 기준의 캔들 정보를 가져온다 
        df_60 = pyupbit.get_ohlcv(ticker,interval="minute60")

        #RSI지표를 구한다
        rsi60_min_before = GetRSI(df_60,14,-2) #이전 캔들 RSI지표
        rsi60_min = GetRSI(df_60,14,-1) #현재 캔들 RSI지표

        #수익율을 구해준다
        revenu_rate = GetRevenueRate(balances,ticker)
        print(ticker , ", RSI :", rsi60_min_before, " -> ", rsi60_min)
        print("revenu_rate : ",revenu_rate)

        #보유하고 있는 코인들 즉 매수 상태인 코인들
        if IsHasCoin(balances,ticker) == True:
            print("HasCoin")
        #아직 매수하기 전인 코인들 즉 매수 대상
        else:
            print("No have")
            
        #if rsi60_min <= 30.0 and revenu_rate < -5.0:
            #분할 매수를 진행한다.




    except Exception as e:
        print("error:", e)

'''


