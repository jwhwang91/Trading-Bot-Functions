import ccxt
import time
import pandas as pd
import numpy as np
from scipy import stats, signal 
import calendar
from datetime import datetime
import pyupbit
from pyupbit.exchange_api import Upbit
import line_alert

class Binance_func:
    def __init__(self, file, ticker):
        self.file = file
        self.ticker = ticker
        # This conditions may be added for different coins : conditions never change while trading
        if 'BTC' in ticker:
            self.tick_size = 1 
            self.minimum_order_amt = 0.001

        # Reads *.txt file for access/secret keys 
        with open(self.file) as f:
            lines = f.read().splitlines()
        f.close()
        self.access, self.secret = lines[0], lines[1]

        # Creates ccxt binance object 
        self.binance = ccxt.binance(config ={
            'apiKey' : self.access,
            'secret' : self.secret,
            'enableRateLimit' : True,
            'adjustForTimeDifference' : True,
            'options' : {
                'defaultType' : 'future'
            }
        })

    def get_ohlcv(self, interval = '1m', limit = 50):
        # This function converts aquired ohlcv data into pd Dateframe
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

        ohlcv = self.binance.fetch_ohlcv(symbol= self.ticker, timeframe= interval, since=since, limit=limit)

        df = pd.DataFrame(ohlcv, columns = ['Time', 'open', 'high', 'low', 'close', 'volume'])
        df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
        df.set_index('Time', inplace=True)
        df['median'] = (df['high'] + df['low'])/2
        return df

    def bollingerband(self, df, window = 20): # data = close 
        # This function calculates bollingerband
        sma = df['close'].rolling(window = window).mean()
        std = df['close'].rolling(window = window).std()
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        df['sma'] = sma
        df['upper_bb'] = upper_bb
        df['lower_bb'] = lower_bb
        df['bb_diff'] = upper_bb - lower_bb #Difference between upper/lower band
        return df

    def GetMA_Vol(self, df, period):
        # This function computes moving average of volume 
        volume = df['volume']
        ma = volume.rolling(period).mean()
        df['ma_vol'] = ma
        return df

    def GetRSI(self, df, period):
        # This function computes Relative Strength Index
        # Returing dataframe contains RSI colume
        df['close'] = df['close']
        delta = df['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(com=(period-1), min_periods = period).mean()
        _loss = down.abs().ewm(com=(period-1), min_periods = period).mean()
        RS = _gain / _loss
        df['RSI'] = 100 - (100 / (1 + RS))
        return df

    def volume_profile(self, interval, num_samples, kde_factor, min_prom_factor):
        # This function returns price range where most volumes are crowded.
        # interval 
        df = self.get_ohlcv(interval = interval, limit = num_samples)
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

        left_ips = peak_props['left_ips']
        right_ips = peak_props['right_ips']
        width_x0 = xr.min() + (left_ips * ticks_per_sample)
        width_x1 = xr.min() + (right_ips * ticks_per_sample)
        width_y = peak_props['width_heights']

        return pkx, width_x0, width_x1

    def b_order_check(self, order_type):
        # order_type = "TAKE_PROFIT", "STOP_MARKET", "LIMIT"
        status = False
        order = self.binance.fetch_open_orders(self.ticker)
        time.sleep(0.3)
        order_l = []
        for each in order:
            order_l.append(each['info']['origType'])
        for order in order_l:
            if order == order_type:
                status = True
        return status

    def SetStopLoss(self, stopPrice):
        time.sleep(0.1)
        orders = self.binance.fetch_orders(self.ticker)

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
            balance = self.binance.fetch_balance(params={"type": "future"})
            time.sleep(0.1)
                                    
            amt = 0
            entryPrice = 0
            leverage = 0
            #평균 매입단가와 수량을 가지고 온다.
            for posi in balance['info']['positions']:
                if posi['symbol'] == self.ticker.replace("/", ""):
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
            self.binance.create_order(self.ticker,'STOP_MARKET',side,abs(amt),stopPrice,params)
            if self.b_order_check("STOP_MARKET") == False:
                self.SetStopLoss(stopPrice)

            print("####STOPLOSS SETTING DONE ######################")

    def SetTakeProfit(self, target_price):
        time.sleep(0.1)
        orders = self.binance.fetch_orders(self.ticker)

        TakeProfitOk = False
        for order in orders:

            if order['status'] == "open" and order['type'] == 'take_profit':
                #print(order)
                TakeProfitOk = True
                break

        #스탑로스 주문이 없다면 주문을 건다
        if TakeProfitOk == False:

            time.sleep(1.0)

            #잔고 데이타를 가지고 온다.
            balance = self.binance.fetch_balance(params={"type": "future"})
            time.sleep(0.1)
                                    
            amt = 0
            entryPrice = 0
            leverage = 0
            #평균 매입단가와 수량을 가지고 온다.
            for posi in balance['info']['positions']:
                if posi['symbol'] == self.ticker.replace("/", ""):
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
            self.binance.create_order(self.ticker,'TAKE_PROFIT',side,abs(amt),target_price,params)
            if self.b_order_check("TAKE_PROFIT") == False:
                self.SetTakeProfit(target_price)

            print("####TAKE_PROFIT SETTING DONE ######################")

    def Cancel_AllOrder(self):
        order = self.binance.fetch_open_orders(self.ticker)
        time.sleep(0.5)
        for each in order:
            self.binance.cancel_order(each['id'], self.ticker)

    def cur_position_check(self):
        balance = self.binance.fetch_balance(params={"type":"future"})
        amt, entryPrice, leverage, unrealizedProfit = 0, 0, 1, 0
        for posi in balance['info']['positions']:
            if posi['symbol'] == self.ticker:
                leverage = float(posi['leverage'])
                entryPrice = float(posi['entryPrice'])
                unrealizedProfit = float(posi['unrealizedProfit'])
                amt = float(posi['positionAmt'])
        return amt, entryPrice, leverage, unrealizedProfit

    def Close_AllPositions_whenprofit(self):
        #balance = self.binance.fetch_balance(params={"type":"future"})
        #Target_Coin_Ticker = self.ticker[:self.ticker.find('USD')] + '/' + self.ticker[self.ticker.find('USD'):]
        time.sleep(0.5)
        amt = self.cur_position_check()[0]
        if amt < 0:
            self.binance.create_market_buy_order(self.ticker, amt)
        elif amt > 0:
            self.binance.create_market_sell_order(self.ticker, amt)
        elif amt == 0:
            print("All Position Already Closed")
        print("All Position Closed")

    def b_checkopen_orders(self):
        time.sleep(0.1)
        order = self.binance.fetch_open_orders(self.ticker)
        order_l = []
        for each in order:
            order_l.append(each['info']['orderId'])
        
        if len(order_l) == 0:
            status = False
            return order_l, status
        else: 
            status = True
            return order_l, status

    def init_make_position(self, amt, entry_price, position, tolerance_tick, timeout_m):
        timeout_end = time.time() + 60 * timeout_m
        tolerance_tick = 5
        tolerance = tolerance_tick * self.tick_size
        short, long = 'short', 'long'
        if short in position:
            self.binance.create_limit_sell_order(self.ticker, amt, entry_price)
            time.sleep(0.5)
            while True:
                try:
                    df_temp = self.get_ohlcv(interval = '1m', limit = 30)
                    order_l, status = self.b_checkopen_orders()
                    time.sleep(0.5)
                    if time.time() > timeout_end:
                        for order in order_l:
                            time.sleep(0.1)
                            self.binance.cancel_order(order, self.ticker)
                            time.sleep(0.1)
                        self.binance.create_market_sell_order(self.ticker, amt)
                        time.sleep(0.1)
                        break
                    elif status == False:
                        break
                    elif df_temp['close'].iloc[-1] <= entry_price - tolerance:
                        for order in order_l:
                            time.sleep(0.1)
                            self.binance.cancel_order(order, self.ticker)
                            time.sleep(0.1)
                        self.binance.create_market_sell_order(self.ticker, amt)
                        time.sleep(0.1)
                        break
                except Exception as e:
                    continue
        
        elif long in position:
            self.binance.create_limit_buy_order(self.ticker, amt, entry_price)
            time.sleep(0.5)
            while True:
                try:
                    df_temp = self.get_ohlcv(interval = '1m', limit = 30)
                    order_l, status = self.b_checkopen_orders()
                    time.sleep(0.5)
                    if time.time() > timeout_end:
                        for order in order_l:
                            time.sleep(0.1)
                            self.binance.cancel_order(order, self.ticker)
                            time.sleep(0.1)
                        self.binance.create_market_buy_order(self.ticker, amt)
                        time.sleep(0.1)
                        break
                    elif status == False:
                        break
                    elif df_temp['close'].iloc[-1] >= entry_price + tolerance:
                        for order in order_l:
                            time.sleep(0.1)
                            self.binance.cancel_order(order, self.ticker)
                            time.sleep(0.1)
                        self.binance.create_market_buy_order(self.ticker, amt)
                        time.sleep(0.1)
                        break
                except Exception as e:
                    continue

    def valid_amt_check(self, flag, liquidity, leverage, entry_price, max_lev):
        # liquidity = freeUSDT (당장 가용 가능한 USDT)

        if flag == False:
            leverage *= 2
            if leverage >= max_lev:
                leverage = max_lev
            try:
                self.binance.fapiPrivate_post_leverage({'symbol': self.ticker, 'leverage': leverage})
            except Exception as e:
                print("---:", e)
                #print(f"Last trading was loss --> leverage doubled to :{leverage}")
            amt = liquidity * leverage / entry_price 
            if amt < self.minimum_order_amt:
                amt = self.minimum_order_amt
                amt *= 2
                return amt, leverage
            else:
                try:
                    self.binance.fapiPrivate_post_leverage({'symbol': self.ticker, 'leverage': leverage})
                except Exception as e:
                    print("---:", e)
                #print(f"Last trading was loss --> leverage doubled to :{leverage}")
        if leverage >= max_lev:
            leverage = max_lev
        try:
            self.binance.fapiPrivate_post_leverage({'symbol': self.ticker, 'leverage': leverage})
        except Exception as e:
            print("---:", e)

        amt = liquidity * leverage / entry_price 
        
        if amt <= self.minimum_order_amt:
            amt = self.minimum_order_amt

        return amt, leverage

    def save_df_to_csv(df, position, path, result): # 
        cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        file_name = f"{cur_time}_{position}_{result}"
        df.to_csv(path + file_name, sep=',')

class stophunting_algo(Binance_func):
    def profit_loss_calc(self, df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin):
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

    def calc_profit(self, liquidity, amt, entry_price, losscut_price, tgt_price, position, df):
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

    def condition_det(self, df, opposite_flag, embrace_high, embrace_low):
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

    def embrace_candle(self, df):
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

    def trading_sequence(self, df, position, flag, opposite_flag, embrace_low, embrace_high, \
        low, high, min_margin, max_margin, profit_margin, liquidity, before_free_USDT, leverage, max_lev, tolerance_tick, timeout_m):
        # =========================== Position Determined =======================================
        position_closed = False
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # Current Time 
        msg = f"{position} Position Activated at time of {cur_time}" # Message Creation
        print(msg)
        # ========================= Making Initial Position =====================================
        entry_price = df['close'].iloc[-1] # Getting Current Price
        # amt and leverage update
        amt, leverage = \
            self.valid_amt_check(flag, liquidity, leverage, entry_price, max_lev)
        # Actually Buying the Coin at determined amt/leverage
        self.init_make_position(amt, entry_price, position, tolerance_tick, timeout_m)
        # =================== Rechecking entry price from balance ===============================
        # Note : This step is neccessary b/c actual: entry price != df['close'].iloc[-1]
        balance = self.binance.fetch_balance(params={"type":"future"})
        time.sleep(0.3)
        amt_act, entryPrice, leverage, unrealizedProfit = self.cur_position_check()
        if entryPrice != entry_price:
            entry_price = entryPrice
        else:
            entry_price = entry_price
        # ================ StopLoss/TakeProfit Calc and place order =============================
        wait_flag, losscut_price, tgt_price \
            = self.profit_loss_calc(df, opposite_flag, position, entry_price, embrace_low, embrace_high, low, high, min_margin, max_margin, profit_margin)
        if wait_flag == True: # if waitflag = True --> diff > 150 tick --> break
            position_closed = True
            return flag, position_closed
        self.SetStopLoss(losscut_price) # Actually Making stoploss order
        self.SetTakeProfit(tgt_price) # Actually Making takeProfit order
        msg = f"진입 포지션 : {position} \
                진입 수량: {amt_act}, \
                진입 가격: {entry_price}, \
                손절 가: {losscut_price}, \
                목표 가: {tgt_price} 설정 완료, \
                현재 자산: {before_free_USDT}"
        print(msg)
        line_alert.Sendmessage(msg)
        # ========================================================================================

        while True:
            try:
                df = self.get_ohlcv()
                time.sleep(0.2)
                status_sl = self.b_order_check("STOP_MARKET")
                status_tl = self.b_order_check("TAKE_PROFIT")
                time.sleep(0.2)
                loss_usdt, profit_usdt, loss_perc, profit_perc, cur_prof, cur_perc = \
                    self.calc_profit(liquidity, amt, entry_price, losscut_price, tgt_price, position, df)
                print(f"Current Price is at {df['close'].iloc[-1]}, current profit is {cur_prof} USDT {cur_perc} %")
                # =======================================
                # Current price reached LosscutPrice 
                # =======================================
                if status_sl == False:
                    self.Cancel_AllOrder()
                    balance = self.binance.fetch_balance(params={"type":"future"})
                    time.sleep(0.2)
                    after_free_USDT = balance['USDT']['free']
                    msg = f"{self.ticker} : All positions closed with loss of {loss_usdt} USDT, {loss_perc}%, \
                        remaining balance : {after_free_USDT}"
                    print(msg)
                    line_alert.Sendmessage(msg)
                    flag = False
                    # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                    # save_df_to_csv(df_temp, position, save_path, result = 'Loss')
                    time.sleep(30)
                    position_closed = True
                    return flag, position_closed
                # =======================================
                # Current price reached TgtPrice 
                # =======================================
                elif status_tl == False:
                    self.Close_AllPositions_whenprofit()
                    self.Cancel_AllOrder()
                    balance = self.binance.fetch_balance(params={"type":"future"})
                    time.sleep(0.2)
                    after_free_USDT = balance['USDT']['free']
                    msg = f"{self.ticker} : All positions closed with profit of {profit_usdt} USDT, {profit_perc}%, \
                        remaining balance : {after_free_USDT}"
                    print(msg)
                    line_alert.Sendmessage(msg)
                    flag = True
                    # df_temp = median(Target_Coin_Symbol, interval = '1m', limit = 100, window = 20)
                    # save_df_to_csv(df_temp, position, save_path, result = 'Profit')
                    time.sleep(30)
                    position_closed = True
                    return flag, position_closed
            
            except Exception as e:
                print("Error : ", e)
                continue

class other_algo(Binance_func):
    pass


class Upbit_func: # private API Functions
    def __init__(self, file):
        self.file = file
        with open(self.file) as f:
            lines = f.read().splitlines()
        f.close()
        self.access, self.secret = lines[0], lines[1]

        upbit = Upbit(self.access, self.secret)
        






