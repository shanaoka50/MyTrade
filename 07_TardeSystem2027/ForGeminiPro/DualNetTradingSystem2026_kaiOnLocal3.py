#!/usr/bin/env python
# coding: utf-8
'''
DualNetTradingSystem2026_kaiOnLocal3.py
本番・推論環境用システムモジュール (TradingCore2027 連携版)
'''
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import glob
import configparser
import math
import sys
from collections import namedtuple
import os
import shutil
import csv
import requests
import gc
import MetaTrader5 as mt5
import pytz

# 共通コアモジュールのインポート
from TradingCore2027 import (
    Brain, CoreEnvironment, BaseAccount, GAMMA,
    NO_ACTION, LONG_ENTRY, SHORT_ENTRY, POSITION_CLOSE, CLOSE_AND_LONG, CLOSE_AND_SHORT
)

EXEC_ENV = "LOCAL"

inifile = configparser.ConfigParser()
ENC = 'UTF-8'
inifile.read(r'D:\ColabNotebooks\00_Common\settings.ini', ENC)

TIMEFRAME_DICT = eval(inifile.get('LOCAL', 'timrframe_dict'))
portfolio = configparser.ConfigParser()
LOGGING_INIFILE_PATH = inifile.get(EXEC_ENV, 'LOGGING_INIFILE_PATH')

PRICEDATA_PATH_REAL = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_REAL')
PRICEDATA_PATH_DEMO = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMO')
PRICEDATA_PATH_DEMOXM = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMOXM')

TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_REAL')
TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMO')
TRAIN_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMOXM')

TRADE_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMO')
TRADE_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMOXM')

TRADE_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMO')
TRADE_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMOXM')

SYMBOL_5DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_5DIGITS'))
SYMBOL_4DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_4DIGITS'))
SYMBOL_3DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_3DIGITS'))
DIGIT_MAGNIFICATION = eval(inifile.get('COMMOM', 'DIGIT_MAGNIFICATION'))
MARGIN_DICT = eval(inifile.get('COMMOM', 'MARGIN_DICT'))

LONG_PERIOD = int(inifile.get('COMMOM', 'LONG_PERIOD'))
SHORT_PERIOD = int(inifile.get('COMMOM', 'SHORT_PERIOD'))
TICKVOL_PERIOD = int(inifile.get('COMMOM', 'TICKVOL_PERIOD'))

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2
ACCOUNT_TRADE_MODE_DEMOXM = 3

MT5_PATH = None
MT5_REAL_PATH = inifile.get('LOCAL', 'mt5_real_path')
MT5_DEMO_PATH = inifile.get('LOCAL', 'mt5_demo_path')
MT5_DEMOXM_PATH = inifile.get('LOCAL', 'mt5_demoxm_path')
MAX_RETRY_NUM = eval(inifile.get('LOCAL', 'MAX_RETRY_NUM'))

logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH, ENC)
import logging.config
logging.config.fileConfig(logconfigfile)
logger = logging.getLogger('DRLLogging')
logger_agent = logging.getLogger('DRLAgent')
logger_trader = logging.getLogger('DRLTrader')


class Environment:
    def __init__(self, trdmd):
        global MT5_PATH
        self.account_trademode = trdmd
        if self.account_trademode == ACCOUNT_TRADE_MODE_DEMO:
            portfolio.read(TRADE_MODEL_PATH_DEMO + 'ModelPortfolio.ini', 'UTF-8')
            MT5_PATH = MT5_DEMO_PATH
        elif self.account_trademode == ACCOUNT_TRADE_MODE_REAL:
            portfolio.read(TRADE_MODEL_PATH_REAL + 'ModelPortfolio.ini', 'UTF-8')
            MT5_PATH = MT5_REAL_PATH
        elif self.account_trademode == ACCOUNT_TRADE_MODE_DEMOXM:
            portfolio.read(TRADE_MODEL_PATH_DEMOXM + 'ModelPortfolio.ini', 'UTF-8')
            MT5_PATH = MT5_DEMOXM_PATH
        else:
            raise RuntimeError("Environment: Account Trade Mode is not defined.")

    def GetDigitMagnification(self, symbol):
        if symbol in SYMBOL_5DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_5DIGITS']
        elif symbol in SYMBOL_3DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_3DIGITS']
        elif symbol in SYMBOL_4DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_4DIGITS']
        else:
            raise RuntimeError(f'Symbol {symbol} is not defined.')

    def GetPeriodTimeDelta(self, period):
        return CoreEnvironment.GetPeriodTimeDelta(period)

    def getAvailableAction(self, profit_loss, has_long, has_short, countdown):
        return CoreEnvironment.getAvailableAction(profit_loss, has_long, has_short, countdown)

    def GetTradeData(self, dt, timeframe, sbl, lprd=LONG_PERIOD, sprd=SHORT_PERIOD, tvprd=TICKVOL_PERIOD, tsl=None):
        dt = dt.replace(tzinfo=pytz.timezone("Etc/UTC"))
        train_symbols = tsl if tsl is not None else [sbl]
        max_period = max([lprd, sprd, tvprd])
        prices_list = []

        if not mt5.initialize(MT5_PATH):
            ecd, emsg = mt5.last_error()
            raise RuntimeError(f"Environment:initialize() failed, error code={ecd} msg={emsg}")

        for symbol in train_symbols:
            for l in range(5):
                rates = mt5.copy_rates_from(symbol, TIMEFRAME_DICT[timeframe], dt, max_period)
                df_rates = pd.DataFrame(rates)
                if df_rates.isna().any().any():
                    if l >= 4:
                        raise RuntimeError('Environment.GetTradeData: Copy rates from MT5 failed.')
                    time.sleep(1)
                else:
                    break

            digits = mt5.symbol_info(symbol).digits
            df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
            df_rates = df_rates.set_index('time')
            df_rates.index.tz_localize('Etc/UTC')

            df_rates[['open', 'high', 'low', 'close']] *= 10 ** (digits - 3)

            vol_mean = df_rates['tick_volume'].rolling(tvprd).mean() + 1e-8
            vol_ratio = (df_rates['tick_volume'] + 1e-8) / vol_mean
            df_rates['tick_volume'] = np.tanh(np.log10(np.maximum(vol_ratio, 1e-8)))

            df_rates['close-open'] = df_rates['close'] - df_rates['open']
            df_rates['high-low'] = df_rates['high'] - df_rates['low']
            df_rates['sma_close_short'] = df_rates['close'].rolling(sprd).mean()
            df_rates['sma_close_long'] = df_rates['close'].rolling(lprd).mean()
            df_rates['sma_open_short'] = df_rates['open'].rolling(sprd).mean()
            df_rates['sma_open_long'] = df_rates['open'].rolling(lprd).mean()
            df_rates['std_short'] = df_rates['close'].rolling(sprd).std()
            df_rates['std_long'] = df_rates['close'].rolling(lprd).std()
            df_rates['z_score_short'] = (df_rates['close'] - df_rates['sma_close_short']) / (df_rates['std_short'] + 1e-8)
            df_rates['z_score_long'] = (df_rates['close'] - df_rates['sma_close_long']) / (df_rates['std_long'] + 1e-8)
            df_rates['sma_close_short-long'] = df_rates['sma_close_short'] - df_rates['sma_close_long']
            df_rates['std_short-long'] = df_rates['std_short'] - df_rates['std_long']
            df_rates['z_score_short-long'] = df_rates['z_score_short'] - df_rates['z_score_long']
            df_rates['sma_close-open_short'] = df_rates['sma_close_short'] - df_rates['sma_open_short']
            df_rates['sma_close-open_long'] = df_rates['sma_close_long'] - df_rates['sma_open_long']

            df_rates.drop(columns=['open', 'high', 'low', 'close', 'spread', 'real_volume',
                                   'sma_close_short', 'sma_close_long', 'sma_open_short', 'sma_open_long'], inplace=True)
            df_rates.rename(columns=lambda s: f"{symbol}_{s}", inplace=True)
            prices_list.append(df_rates)

        df_prices = pd.concat(prices_list, axis=1, join='outer')
        return df_prices.iloc[-1]

    def CalcCountdown(self, dt, tf, sbl):
        now_datetime = datetime.now(timezone.utc)
        start_period = (now_datetime + timedelta(days=1 - (now_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0)
        end_period = (now_datetime + timedelta(days=6 - (now_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0)
        delta_p = self.GetPeriodTimeDelta(tf)
        end_period -= delta_p
        end_trade = end_period - delta_p
        countdown = (dt - start_period).total_seconds() / (end_trade - start_period).total_seconds()
        return countdown, sbl, tf

    def CalcOrderLots(self, mname, tf, sbl, trdmd):
        if trdmd == ACCOUNT_TRADE_MODE_DEMO:
            mt5_p = inifile.get('LOCAL', 'mt5_demo_path')
            growth_rate = float(inifile.get('LOCAL', 'GROWTH_RATE_DEMO'))
        elif trdmd == ACCOUNT_TRADE_MODE_REAL:
            mt5_p = inifile.get('LOCAL', 'mt5_real_path')
            growth_rate = float(inifile.get('LOCAL', 'GROWTH_RATE_REAL'))
        else:
            mt5_p = inifile.get('LOCAL', 'mt5_demoxm_path')
            growth_rate = float(inifile.get('LOCAL', 'GROWTH_RATE_DEMOXM'))

        if not mt5.initialize(mt5_p):
            raise RuntimeError(f"Environment:initialize() failed, error code = {mt5.last_error()}")

        account_info = mt5.account_info()
        equity = account_info.equity
        investment_rate = float(portfolio.get('LOCAL', mname))
        expected_value = float(portfolio.get('LOCAL', 'mu'))

        if account_info.currency == 'USD':
            rates = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_H1, 0, 1)
            order_lots = (rates[0][4] * equity / 100000) * (growth_rate / expected_value) * investment_rate
        else:
            order_lots = (equity / 100000) * (growth_rate / expected_value) * investment_rate

        return round(order_lots, 2), sbl, tf

    def GetTickData(self, sbl, dt_from, dt_to):
        if not mt5.initialize(MT5_PATH):
            raise RuntimeError(f"Environment.GetTickData.initialize() failed: {mt5.last_error()}")

        dt_from = dt_from.replace(tzinfo=pytz.timezone("Etc/UTC"))
        dt_to = dt_to.replace(tzinfo=pytz.timezone("Etc/UTC"))
        ticks_period = mt5.copy_ticks_range(sbl, dt_from, dt_to, mt5.COPY_TICKS_ALL)

        df_ticks_period = pd.DataFrame(ticks_period)
        if not df_ticks_period.empty:
            df_ticks_period['bid'] = df_ticks_period['bid'].astype(float)
            df_ticks_period['ask'] = df_ticks_period['ask'].astype(float)
            df_ticks_period['time'] = pd.to_datetime(df_ticks_period['time'], unit='s')
            df_ticks_period['time_msc'] = pd.to_datetime(df_ticks_period['time_msc'], unit='ms')
            df_ticks_period = df_ticks_period.set_index('time_msc')
            df_ticks_period.sort_index(inplace=True)
            df_ticks_period.index.tz_localize('Etc/UTC')

        return df_ticks_period

    def GetPriceData(self, sb, tf, trddt):
        trddt = trddt.replace(tzinfo=pytz.timezone("Etc/UTC"))
        if not mt5.initialize(MT5_PATH):
            ecd, emsg = mt5.last_error()
            raise RuntimeError(f"Environment:initialize() failed, error code={ecd} msg={emsg}")

        rates = mt5.copy_rates_from(sb, TIMEFRAME_DICT[tf], trddt, 1)
        df_rates = pd.DataFrame(rates)
        df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
        df_rates = df_rates.set_index('time')
        df_rates.index.tz_localize('Etc/UTC')
        return df_rates


class Account(BaseAccount):
    def __init__(self, sbl, tf, rsl, lcl, tpl, ps, ptw):
        margin_val = MARGIN_DICT.get(sbl, 0.02)
        super().__init__(sbl, tf, rsl, lcl, tpl, ps, ptw, margin=margin_val)
        self.env = None
        self.magnification = None

    def SetEnv(self, env):
        self.env = env
        self.magnification = self.env.GetDigitMagnification(self.symbol)

    def SetDeltaPeriod(self, dltprd):
        self.delta_period = dltprd

    def CheckTpLc(self, lstdt, lstprc):
        self.last_close_price = 0.0
        utc_from = lstdt
        utc_to = lstdt + self.delta_period

        df_ticks_period = self.env.GetTickData(self.symbol, utc_from, utc_to)
        if len(df_ticks_period) == 0:
            return

        self.last_open_ask = df_ticks_period.iloc[0]['ask']
        self.last_open_bid = df_ticks_period.iloc[0]['bid']
        self.last_close_ask = df_ticks_period.iloc[-1]['ask']
        self.last_close_bid = df_ticks_period.iloc[-1]['bid']

        self.pos_open_price, self.has_long, self.has_short, self.close_pl = self.df_account.loc[
            utc_from, ['pos_open_price', 'has_long', 'has_short', 'close_pl']
        ]

        if self.pos_open_price > 0:
            if self.has_long > 0.0:
                takeprofit_price = self.pos_open_price + self.tp_level / self.magnification
                losscut_price = self.pos_open_price - self.lc_level / self.magnification
                self.last_close_price = self.last_close_bid
                self.float_pl = (self.last_close_price - self.pos_open_price) * self.magnification

                bids = df_ticks_period['bid'].to_numpy()
                hit_tp = np.where(bids >= takeprofit_price)[0]
                hit_lc = np.where(bids <= losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    self.close_pl -= self.lc_level
                    self.pos_open_price, self.has_long, self.float_pl = 0.0, 0.0, 0.0
                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.pos_open_price, self.has_long, self.float_pl = 0.0, 0.0, 0.0

            elif self.has_short > 0.0:
                takeprofit_price = self.pos_open_price - self.tp_level / self.magnification
                losscut_price = self.pos_open_price + self.lc_level / self.magnification
                self.last_close_price = self.last_close_ask
                self.float_pl = (self.pos_open_price - self.last_close_price) * self.magnification

                asks = df_ticks_period['ask'].to_numpy()
                hit_tp = np.where(asks <= takeprofit_price)[0]
                hit_lc = np.where(asks >= losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    self.close_pl -= self.lc_level
                    self.pos_open_price, self.has_short, self.float_pl = 0.0, 0.0, 0.0
                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.pos_open_price, self.has_short, self.float_pl = 0.0, 0.0, 0.0

        self.df_account.loc[utc_from, ['pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = (
            self.pos_open_price, self.has_long, self.has_short, self.float_pl, self.close_pl
        )

    def EvaluateRewrd(self, actn_idx, test_dt, last_dt, df_price_data):
        utc_from = test_dt
        utc_to = utc_from + timedelta(minutes=1)
        sell_open_price, buy_open_price = 0.0, 0.0
        sell_close_price, buy_close_price = 0.0, 0.0

        for n in range(MAX_RETRY_NUM[self.period]):
            df_ticks_first = self.env.GetTickData(self.symbol, utc_from, utc_to)
            if len(df_ticks_first) == 0:
                if n == MAX_RETRY_NUM[self.period] - 1:
                    if actn_idx in [LONG_ENTRY, SHORT_ENTRY]:
                        actn_idx = NO_ACTION
                    elif actn_idx in [CLOSE_AND_LONG, CLOSE_AND_SHORT]:
                        actn_idx = POSITION_CLOSE
                utc_from += timedelta(minutes=1)
                utc_to += timedelta(minutes=1)
                continue

            sell_price = float(df_ticks_first.iloc[0]['bid'])
            buy_price = float(df_ticks_first.iloc[0]['ask'])
            real_spread = abs(buy_price - sell_price)

            if real_spread >= self.real_spread_limit:
                if n == MAX_RETRY_NUM[self.period] - 1:
                    if actn_idx in [LONG_ENTRY, SHORT_ENTRY]:
                        actn_idx = NO_ACTION
                    elif actn_idx in [CLOSE_AND_LONG, CLOSE_AND_SHORT]:
                        actn_idx = POSITION_CLOSE
                    sell_close_price = buy_price
                    buy_close_price = sell_price
                utc_from += timedelta(minutes=1)
                utc_to += timedelta(minutes=1)
            else:
                sell_open_price, buy_open_price = sell_price, buy_price
                sell_close_price, buy_close_price = buy_price, sell_price
                break

        self.df_account.loc[test_dt] = self.df_account.loc[last_dt]

        if actn_idx == NO_ACTION:
            self.df_account.loc[test_dt, 'close_pl'] = 0.0
        elif actn_idx == LONG_ENTRY:
            self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = (
                [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0, 0.0] if buy_open_price > 0.0 else [None, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
        elif actn_idx == SHORT_ENTRY:
            self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = (
                [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0, 0.0] if sell_open_price > 0.0 else [None, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
        elif actn_idx == POSITION_CLOSE:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if buy_close_price == 0.0 else (buy_close_price - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if sell_close_price == 0.0 else (self.df_account.loc[last_dt, 'pos_open_price'] - sell_close_price) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
        elif actn_idx == CLOSE_AND_LONG:
            if self.df_account.loc[last_dt, 'has_short'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if sell_close_price == 0.0 else (self.df_account.loc[last_dt, 'pos_open_price'] - sell_close_price) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            if buy_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']] = [test_dt, buy_open_price, self.pos_scale]
        elif actn_idx == CLOSE_AND_SHORT:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if buy_close_price == 0.0 else (buy_close_price - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            if sell_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']] = [test_dt, sell_open_price, self.pos_scale]

    def CalcCountdown(self, cntdt):
        countdown = (cntdt - self.start_period).total_seconds() / (self.end_trade - self.start_period).total_seconds()
        self.df_account.loc[cntdt, ['countdown']] = countdown

    def GetAccountInfo(self, idx_dt, cols=None):
        return self.df_account.loc[idx_dt, cols] if cols else self.df_account.loc[idx_dt]


class Agent():
    def __init__(self, input_num, hidden_num, output_num, model_name, trade_mode):
        self.main_brain = Brain(input_num, output_num, hidden_num)
        trade_model_path = TRADE_MODEL_PATH_REAL if trade_mode == ACCOUNT_TRADE_MODE_REAL else (
            TRADE_MODEL_PATH_DEMO if trade_mode == ACCOUNT_TRADE_MODE_DEMO else TRADE_MODEL_PATH_DEMOXM
        )
        self.main_brain.loadModel(f"{trade_model_path}Model_{model_name}.pth")
        self.device = self.main_brain.device

        agt_inifile = configparser.ConfigParser()
        agt_inifile.read(f"{trade_model_path}Agent_{model_name}.ini")
        self.est_float_pl = float(agt_inifile.get('COMMOM', 'ESTIMATE_FLOAT_PL'))

    def DecideAction(self, set_datetime, input_data, action_mask):
        self.main_state_action_values = self.main_brain(input_data)
        self.acition_index, _ = self.main_brain.decideAction(action_mask, self.main_state_action_values, episode=-2, train_mode=False)
        return self.acition_index

    def GetMainHiddenCellState(self):
        return self.main_brain.getHiddenCellState()


class Trader():
    def __init__(self, sbl, tf, tdur, prds, rsl, lcl, tpl, ps, ptw, ipn, opn, hdn, trdmd, mdlnm, tsl):
        self.symbol = sbl
        self.period = tf
        self.delta_period = None
        self.test_duration = tdur
        self.periods = prds
        self.real_spread_limit = rsl
        self.lc_level = lcl
        self.tp_level = tpl
        self.pos_scale = ps
        self.pre_trade_weeks = ptw
        self.input_num = ipn
        self.output_num = opn
        self.hidden_num = hdn
        self.trade_mode = trdmd
        self.model_name = mdlnm
        self.trade_symbol_list = tsl
        self.env = None

        self.df_result = pd.DataFrame(columns=['tradenum', 'sum', 'mean', 'sd'])
        self.acnt = Account(self.symbol, self.period, self.real_spread_limit, self.lc_level, self.tp_level, self.pos_scale, self.pre_trade_weeks)
        self.agnt = Agent(self.input_num, self.hidden_num, self.output_num, self.model_name, self.trade_mode)

    def SetTradePeriod(self, now_dt):
        if self.test_duration == 'W':
            q, r = divmod(now_dt.isoweekday(), 6)
            self.start_period = (now_dt + timedelta(days=1 * q + 1 - r)).replace(hour=0, minute=0, second=0, microsecond=0)
            self.end_period = (self.start_period + timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0) - self.delta_period
            self.end_trade = self.end_period - self.delta_period

    def ExecPreTrade(self, now_dt):
        self.SetTradePeriod(now_dt)
        from_date = self.start_period - timedelta(weeks=self.pre_trade_weeks)
        to_date = self.start_period - timedelta(weeks=1)
        every_monday = pd.date_range(from_date, to_date, freq='W-MON')
        for w in every_monday:
            self._ExecTrade(w, pretrade_flg=True)

    def _ExecTrade(self, stdt, pretrade_flg):
        self.SetTradePeriod(stdt)
        lstdt = self._GetLastDateTime(stdt, self.delta_period)
        self.acnt.SetAccount(lstdt)

        pd_freq = PERIOD_FREQ_DICT[self.period]
        dtrng = pd.date_range(self.start_period, self.end_period, freq=pd_freq)
        for tstdt in dtrng:
            lstdt = self._GetLastDateTime(tstdt, self.delta_period)
            df_price_data = self.env.GetPriceData(self.symbol, self.period, lstdt)
            if df_price_data.empty:
                continue

            self.acnt.CheckTpLc(lstdt, df_price_data)
            self.acnt.start_period = self.start_period
            self.acnt.end_trade = self.end_trade
            self.acnt.CalcCountdown(lstdt)

            df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown'])
            action_mask = self.env.getAvailableAction(df_acnt.float_pl, df_acnt.has_long, df_acnt.has_short, df_acnt.countdown)
            df_trade = self.env.GetTradeData(lstdt, self.period, self.symbol, self.periods[0], self.periods[1], self.periods[2], self.trade_symbol_list)

            df_acnt_train = pd.concat([df_acnt, df_trade])
            action_index = self.agnt.DecideAction(tstdt, df_acnt_train.to_list(), action_mask)
            self.acnt.EvaluateRewrd(action_index, tstdt, lstdt, df_price_data)

        self.acnt.DropAccount()

    def _GetLastDateTime(self, cur_dt, dlt_dt):
        tmp_last_dt = (cur_dt - dlt_dt).replace(tzinfo=pytz.timezone("Etc/UTC"))
        if not mt5.initialize(MT5_PATH):
            raise RuntimeError(f"Trader._GetLastDateTime initialize failed: {mt5.last_error()}")

        rates = mt5.copy_rates_from(self.symbol, TIMEFRAME_DICT[self.period], tmp_last_dt, 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Trader._GetLastDateTime: rates is empty for {self.symbol} at {tmp_last_dt}")

        return datetime.fromtimestamp(rates[0][0]).astimezone(pytz.timezone("Etc/UTC"))

    def setEnv(self, env):
        self.env = env
        self.acnt.SetEnv(env)

    def setPeriodTimeDelta(self):
        self.delta_period = self.env.GetPeriodTimeDelta(self.period)
        self.acnt.SetDeltaPeriod(self.delta_period)