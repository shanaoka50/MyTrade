#!/usr/bin/env python
# coding: utf-8
'''
TradingSystem2027_OnLocal.py
本番・推論環境用システムモジュール (TradingCore2027 連携・System2027対応版)
- 8指標 (36次元入力)
- マルチタイムフレーム [W1, D1, H4, H1]
- 7行動 (トレーリングストップ対応)
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
import time

# 共通コアモジュールのインポート
from TradingCore2027 import (
    Brain2027, CoreEnvironment, BaseAccount, GAMMA,
    NO_ACTION, LONG_ENTRY, SHORT_ENTRY, POSITION_CLOSE, CLOSE_AND_LONG, CLOSE_AND_SHORT, TRAILING_STOP
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

TRADE_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_REAL')
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

PERIOD_FREQ_DICT = eval(inifile.get('COMMOM', 'PERIOD_FREQ_DICT'))
MTF_TIMEFRAMES = ['W1', 'D1', 'H4', 'H1']
BASE_TIMEFRAME = 'H1'

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

    def _calc_tf_features(self, df_rates, symbol, tf, lprd, sprd, tvprd):
        """1時間足分の8指標特徴量を算出"""
        digits = mt5.symbol_info(symbol).digits
        magnification = 10 ** (digits - 3)

        close_p = df_rates['close'] * magnification
        open_p  = df_rates['open'] * magnification
        high_p  = df_rates['high'] * magnification
        low_p   = df_rates['low'] * magnification

        signal_p = max(3, sprd // 2)

        df_feat = pd.DataFrame(index=df_rates.index)

        # 1. tick_volume
        vol_mean = df_rates['tick_volume'].rolling(tvprd).mean()
        vol_std  = df_rates['tick_volume'].rolling(tvprd).std() + 1e-8
        vol_z    = (df_rates['tick_volume'] - vol_mean) / vol_std
        df_feat['tick_volume'] = np.tanh(vol_z / 2.0)

        # 2. close-open / 3. high-low
        hl_raw  = high_p - low_p
        co_raw  = close_p - open_p
        mean_hl = hl_raw.rolling(sprd).mean() + 1e-8
        df_feat['close-open'] = co_raw / (hl_raw + 1e-8)
        df_feat['high-low']   = np.tanh(np.log(np.maximum(hl_raw / mean_hl, 1e-8)))

        # 4. z_score_short / 5. z_score_long
        sma_close_short = close_p.rolling(sprd).mean()
        sma_close_long  = close_p.rolling(lprd).mean()
        std_short       = close_p.rolling(sprd).std()
        std_long        = close_p.rolling(lprd).std()
        df_feat['z_score_short'] = np.tanh((close_p - sma_close_short) / (std_short + 1e-8) / 2.0)
        df_feat['z_score_long']  = np.tanh((close_p - sma_close_long) / (std_long + 1e-8) / 2.0)

        # 6. macd / 7. macd_hist
        macd_raw   = sma_close_short - sma_close_long
        signal_raw = macd_raw.rolling(signal_p).mean()
        hist_raw   = macd_raw - signal_raw
        df_feat['macd']      = np.tanh(macd_raw / (std_long + 1e-8))
        df_feat['macd_hist'] = np.tanh(hist_raw / (std_long + 1e-8))

        # 8. rsi
        delta     = close_p.diff()
        gain      = np.where(delta > 0, delta, 0.0)
        loss      = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=close_p.index).rolling(sprd).mean()
        roll_loss = pd.Series(loss, index=close_p.index).rolling(sprd).mean()
        rs        = roll_gain / (roll_loss + 1e-8)
        rsi       = 100.0 - (100.0 / (1.0 + rs))
        df_feat['rsi'] = (rsi - 50.0) / 50.0

        df_feat.rename(columns=lambda s: f"{symbol}_{tf}_{s}", inplace=True)
        return df_feat

    def GetMTFTradeData(self, dt, symbol, lprd=LONG_PERIOD, sprd=SHORT_PERIOD, tvprd=TICKVOL_PERIOD):
        """[W1, D1, H4, H1] の4時間足をMT5から取得・アライメント結合 (32次元)"""
        dt = dt.replace(tzinfo=pytz.timezone("Etc/UTC"))
        max_period = max([lprd, sprd, tvprd]) + 50

        if not mt5.initialize(MT5_PATH):
            ecd, emsg = mt5.last_error()
            raise RuntimeError(f"Environment:initialize() failed, error code={ecd} msg={emsg}")

        tf_dfs = {}
        for tf in MTF_TIMEFRAMES:
            for l in range(5):
                rates = mt5.copy_rates_from(symbol, TIMEFRAME_DICT[tf], dt, max_period)
                if rates is not None and len(rates) > 0:
                    df_rates = pd.DataFrame(rates)
                    if not df_rates.isna().any().any():
                        break
                if l >= 4:
                    raise RuntimeError(f'Environment.GetMTFTradeData: Copy rates from MT5 failed for {symbol} {tf}.')
                time.sleep(1)

            df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
            df_rates = df_rates.set_index('time')
            df_rates.index = df_rates.index.tz_localize('Etc/UTC')
            tf_dfs[tf] = self._calc_tf_features(df_rates, symbol, tf, lprd, sprd, tvprd)

        # H1 をベースに H4, D1, W1 を結合
        base_df = tf_dfs['H1'].copy()
        for higher_tf in ['H4', 'D1', 'W1']:
            base_df = pd.merge_asof(
                base_df, tf_dfs[higher_tf],
                left_index=True, right_index=True,
                direction='backward'
            )
        base_df.bfill(inplace=True)
        base_df.ffill(inplace=True)
        return base_df.iloc[-1]

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

        logger.debug(f'[CalcOrderLots]:equity {equity},investment_rate {investment_rate}, expected_value {expected_value}')
        if account_info.currency == 'USD':
            rates = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_H1, 0, 1)
            order_lots = (rates[0][4] * equity / 100000) * (growth_rate / expected_value) * investment_rate
        else:
            order_lots = (equity / 100000) * (growth_rate / expected_value) * investment_rate
        logger.debug(f'[CalcOrderLots]:order_lots {order_lots}')

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
    # def __init__(self, sbl, tf, rsl, lcl, tpl, ps, ptw):
    def __init__(self, sbl, tf, rsl, lcl, tpl, ptw):
        margin_val = MARGIN_DICT.get(sbl, 0.02)
        # super().__init__(sbl, tf, rsl, lcl, tpl, ps, ptw, margin=margin_val)
        # super().__init__(sbl, tf, rsl, lcl, tpl, ptw, margin=margin_val)
        super().__init__(sbl, tf, rsl, lcl, tpl, ptw)
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
                if self.losscut_price == 0.0:
                    self.losscut_price = self.pos_open_price - self.lc_level / self.magnification
                self.takeprofit_price = self.pos_open_price + self.tp_level / self.magnification
                self.last_close_price = self.last_close_bid
                self.float_pl = (self.last_close_price - self.pos_open_price) * self.magnification

                bids = df_ticks_period['bid'].to_numpy()
                hit_tp = np.where(bids >= self.takeprofit_price)[0]
                hit_lc = np.where(bids <= self.losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    actual_loss_gain = (self.losscut_price - self.pos_open_price) * self.magnification
                    self.close_pl += actual_loss_gain
                    self.pos_open_price, self.has_long, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0
                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.pos_open_price, self.has_long, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

            elif self.has_short > 0.0:
                if self.losscut_price == 0.0:
                    self.losscut_price = self.pos_open_price + self.lc_level / self.magnification
                self.takeprofit_price = self.pos_open_price - self.tp_level / self.magnification
                self.last_close_price = self.last_close_ask
                self.float_pl = (self.pos_open_price - self.last_close_price) * self.magnification

                asks = df_ticks_period['ask'].to_numpy()
                hit_tp = np.where(asks <= self.takeprofit_price)[0]
                hit_lc = np.where(asks >= self.losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    actual_loss_gain = (self.pos_open_price - self.losscut_price) * self.magnification
                    self.close_pl += actual_loss_gain
                    self.pos_open_price, self.has_short, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0
                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.pos_open_price, self.has_short, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

        self.df_account.loc[utc_from, ['pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = (
            self.pos_open_price, self.has_long, self.has_short, self.float_pl, self.close_pl
        )

    def EvaluateRewrd(self, actn_idx, test_dt, last_dt, df_price_data):
        utc_from = test_dt
        utc_to = utc_from + timedelta(minutes=1)
        sell_open_price, buy_open_price = 0.0, 0.0
        sell_close_price, buy_close_price = 0.0, 0.0
        # ★修正1: sell_price, buy_price を初期化 (UnboundLocalError 防止)
        sell_price, buy_price = 0.0, 0.0

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
            if buy_open_price == 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, buy_open_price, 1.0, 0.0, 0.0, 0.0]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
        elif actn_idx == SHORT_ENTRY:
            if sell_open_price == 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, sell_open_price, 0.0, 1.0, 0.0, 0.0]
                self.losscut_price = sell_open_price + (self.lc_level / self.magnification)
        elif actn_idx == POSITION_CLOSE:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if buy_close_price == 0.0 else (buy_close_price - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if sell_close_price == 0.0 else (self.df_account.loc[last_dt, 'pos_open_price'] - sell_close_price) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            self.losscut_price = 0.0
        elif actn_idx == CLOSE_AND_LONG:
            if self.df_account.loc[last_dt, 'has_short'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if sell_close_price == 0.0 else (self.df_account.loc[last_dt, 'pos_open_price'] - sell_close_price) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            if buy_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']] = [test_dt, buy_open_price, 1.0]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
        elif actn_idx == CLOSE_AND_SHORT:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if buy_close_price == 0.0 else (buy_close_price - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            if sell_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']] = [test_dt, sell_open_price, 1.0]
                self.losscut_price = sell_open_price + (self.lc_level / self.magnification)
        elif actn_idx == TRAILING_STOP:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                # 2026/9/27 sell_price が取得できた場合にトレイル実行
                if sell_price > 0.0:
                    new_lc = sell_price - (self.lc_level / self.magnification)
                    if new_lc > self.losscut_price:
                        self.losscut_price = new_lc
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                if buy_price > 0.0:
                    new_lc = buy_price + (self.lc_level / self.magnification)
                    if self.losscut_price == 0.0 or new_lc < self.losscut_price:
                        self.losscut_price = new_lc
            self.df_account.loc[test_dt, 'close_pl'] = 0.0

    def CalcCountdown(self, cntdt):
        countdown = (cntdt - self.start_period).total_seconds() / (self.end_trade - self.start_period).total_seconds()
        self.df_account.loc[cntdt, ['countdown']] = countdown

    def GetAccountInfo(self, idx_dt, cols=None):
        return self.df_account.loc[idx_dt, cols] if cols else self.df_account.loc[idx_dt]


class Agent():
    def __init__(self, input_num, hidden_num, output_num, model_name, trade_mode):
        self.main_brain = Brain2027(input_num, output_num, hidden_num)
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
    def __init__(self, sbl, tf, tdur, prds, rsl, lcl, tpl, ptw, ipn, opn, hdn, trdmd, mdlnm):
        self.symbol = sbl
        self.period = tf
        self.delta_period = None
        self.test_duration = tdur
        self.periods = prds
        self.real_spread_limit = rsl
        self.lc_level = lcl
        self.tp_level = tpl
        # self.pos_scale = ps
        self.pre_trade_weeks = ptw
        self.input_num = ipn
        self.output_num = opn
        self.hidden_num = hdn
        self.trade_mode = trdmd
        self.model_name = mdlnm
        self.env = None

        self.df_result = pd.DataFrame(columns=['tradenum', 'sum', 'mean', 'sd'])
        # self.acnt = Account(self.symbol, self.period, self.real_spread_limit, self.lc_level, self.tp_level, self.pos_scale, self.pre_trade_weeks)
        self.acnt = Account(self.symbol, self.period, self.real_spread_limit, self.lc_level, self.tp_level, self.pre_trade_weeks)
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

            # df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown'])
            df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown']).copy()
            action_mask = self.env.getAvailableAction(df_acnt.float_pl, df_acnt.has_long, df_acnt.has_short, df_acnt.countdown)
            
            # System2027 は MTF (32次元) データを取得
            df_trade = self.env.GetMTFTradeData(lstdt, self.symbol, self.periods[0], self.periods[1], self.periods[2])

            # モデル入力用に float_pl を正規化
            raw_float_pl = float(df_acnt['float_pl'])
            norm_float_pl = np.tanh(raw_float_pl / (self.acnt.lc_level + 1e-8))
            df_acnt_input = df_acnt.copy()
            df_acnt_input['float_pl'] = norm_float_pl            
            
            # df_acnt_train = pd.concat([df_acnt, df_trade])
            df_acnt_train = pd.concat([df_acnt_input, df_trade])
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