#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import torch
import glob
import configparser
import random
import math
import sys
from collections import namedtuple
from scipy import stats
import os
import shutil
import csv
import requests
import gc
import connectorx as cx
import logging
import logging.config

# 共通コアモジュールのインポート (7行動)
from TradingCore2027 import (
    Brain2027, CoreEnvironment, BaseAccount, GAMMA,
    NO_ACTION, LONG_ENTRY, SHORT_ENTRY, POSITION_CLOSE, CLOSE_AND_LONG, CLOSE_AND_SHORT, TRAILING_STOP
)

# =====================================================================
# 設定・クラス変数
# =====================================================================
EXEC_ENV = "LOCAL"

inifile = configparser.ConfigParser()
ENC = 'UTF-8'
inifile.read(r'D:\ColabNotebooks\06_DuelNetTradingSystem2025\settings.ini', ENC)

LOGGING_INIFILE_PATH = inifile.get(EXEC_ENV, 'LOGGING_INIFILE_PATH')

PRICEDATA_PATH_REAL = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_REAL')
PRICEDATA_PATH_DEMO = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMO')
PRICEDATA_PATH_DEMOXM = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMOXM')

TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_REAL')
TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMO')
TRAIN_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMOXM')

TRAIN_MODEL_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_REAL')
TRAIN_MODEL_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMO')
TRAIN_MODEL_PATH_VOID_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMOXM')

TMP_TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_REAL')
TMP_TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMO')
TMP_TRAIN_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMOXM')

TRADE_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMO')
TRADE_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMOXM')

TRADE_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMO')
TRADE_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMOXM')

TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_REAL')
TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMO')
TRAIN_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMOXM')

TRAIN_RESULT_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_REAL')
TRAIN_RESULT_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMO')
TRAIN_RESULT_PATH_VOID_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMOXM')

TMP_TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_REAL')
TMP_TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMO')
TMP_TRAIN_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMOXM')

SYMBOL_5DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_5DIGITS'))
SYMBOL_4DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_4DIGITS'))
SYMBOL_3DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_3DIGITS'))

DIGIT_MAGNIFICATION = eval(inifile.get('COMMOM', 'DIGIT_MAGNIFICATION'))
MARGIN_DICT = eval(inifile.get('COMMOM', 'MARGIN_DICT'))

TICKVOL_PERIOD = int(inifile.get('COMMOM', 'TICKVOL_PERIOD'))
LONG_PERIOD = int(inifile.get('COMMOM', 'LONG_PERIOD'))
SHORT_PERIOD = int(inifile.get('COMMOM', 'SHORT_PERIOD'))
PERIOD_FREQ_DICT = eval(inifile.get('COMMOM', 'PERIOD_FREQ_DICT'))

TP_LC_MIN_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MIN_LEVEL'))
TP_LC_MAX_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MAX_LEVEL'))
TP_LC_STEP = int(inifile.get('COMMOM', 'TP_LC_STEP'))

URL_TOPIC_NAME = inifile.get('COMMOM', 'URL_TOPIC_NAME')
TRADE_SYSTEM = 'System2027'

# マルチタイムフレーム構成 [W1, D1, H4, H1]
MTF_TIMEFRAMES = ['W1', 'D1', 'H4', 'H1']
BASE_TIMEFRAME = 'H1'

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2
ACCOUNT_TRADE_MODE_DEMOXM = 3
ACCOUNT_TRADE_MODE_STR = ['DEMO', 'CONTEST', 'REAL', 'DEMOXM']

REAL_SPREAD_LIMIT_LIST = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST'))
MAX_RETRY_NUM = eval(inifile.get('LOCAL', 'MAX_RETRY_NUM'))

MODEL_INPUT_NUM = 0
MODEL_HIDDEN_NUM = 0
MODEL_OUTPUT_NUM = 7  # 7行動

RDBMS = 'postgresql'
HOST = 'localhost'
PORT = '5432'
USER = 'appop'
PASSWORD = 'appop'
SCHEMA = 'public'
CONN_URL = None

# =====================================================================
# Logger 設定 (コンソール StreamHandler を追加)
# =====================================================================
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH, ENC)
logging.config.fileConfig(logconfigfile)

logger = logging.getLogger('DRLLogging')
logger_agent = logging.getLogger('DRLAgent')
logger_trainer = logging.getLogger('DRLTrainer')
logger_root = logging.getLogger('DRLRoot')

# コンソール出力用のハンドラが存在しない場合に追加
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

for log_instance in [logger, logger_agent, logger_trainer, logger_root]:
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in log_instance.handlers):
        log_instance.addHandler(console_handler)


# =====================================================================
# トレーニング特化ヘルパークラス
# =====================================================================
class TrainEnvHelper:
    @classmethod
    def GetDigitMagnification(cls, symbol):
        if symbol in SYMBOL_5DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_5DIGITS']
        elif symbol in SYMBOL_3DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_3DIGITS']
        elif symbol in SYMBOL_4DIGITS:
            return DIGIT_MAGNIFICATION['SYMBOL_4DIGITS']
        else:
            print(999)
            raise RuntimeError(f'Symbol {symbol} is not defined.')

    @classmethod
    def periodRandomSelect(cls):
        prime_number = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        period_numbers = random.sample(prime_number, 2)
        return max(period_numbers), min(period_numbers), random.choice(prime_number)

    @classmethod
    def setTradeModePath(cls, trdmd):
        global PRICEDATA_PATH, TRAIN_MODEL_PATH, TRAIN_MODEL_PATH_VOID, TMP_TRAIN_MODEL_PATH
        global TRADE_MODEL_PATH, TRADE_RESULT_PATH, TRAIN_RESULT_PATH, TRAIN_RESULT_PATH_VOID, TMP_TRAIN_RESULT_PATH, DBNAME
        if trdmd == ACCOUNT_TRADE_MODE_REAL:
            PRICEDATA_PATH = PRICEDATA_PATH_REAL
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_REAL
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_REAL
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_REAL
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_REAL
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_REAL
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_REAL
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_REAL
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_REAL
            DBNAME = 'xmtradingreal'
        elif trdmd == ACCOUNT_TRADE_MODE_DEMO:
            PRICEDATA_PATH = PRICEDATA_PATH_DEMO
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_DEMO
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_DEMO
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_DEMO
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMO
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_DEMO
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_DEMO
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_DEMO
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_DEMO
            DBNAME = 'metaquotesdemo'
        else:
            PRICEDATA_PATH = PRICEDATA_PATH_DEMOXM
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_DEMOXM
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_DEMOXM
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_DEMOXM
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMOXM
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_DEMOXM
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_DEMOXM
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_DEMOXM
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_DEMOXM
            DBNAME = 'xmtradingdemo'

    @classmethod
    def connectDB(cls):
        global CONN_URL
        CONN_URL = f"{RDBMS}://{USER}:{PASSWORD}@{HOST}/{DBNAME}"

    @classmethod
    def send_ntfy_message(cls, ttl, msg):
        try:
            requests.post(URL_TOPIC_NAME, data=msg.encode('utf-8'), headers={"Title": ttl.encode('utf-8')}, timeout=5)
        except Exception as e:
            logger.warning(f"send_ntfy_message failed: {e}")

    @classmethod
    def compareEvalValue(cls, sbl, eval_value):
        # 2026/9/24 通貨ペア×時間足で1つずつの生き残りを通貨ペア毎に1つの生き残りにする
        # filelist = glob.glob(f"{TRAIN_MODEL_PATH}Agent_{sbl}_{TRADE_SYSTEM}_*.ini")
        filelist = glob.glob(f"{TRAIN_MODEL_PATH}Agent_{sbl}_*_*.ini")
        win_file = None

        for l in filelist:
            agentfile = configparser.ConfigParser()
            agentfile.read(l, ENC)
            ev = float(agentfile.get('COMMOM', 'EVAL_VALUE'))
            if ev >= eval_value:
                eval_value = ev
                win_file = l
        # 2026/9/24 通貨ペア×時間足で1つずつの生き残りを通貨ペア毎に1つの生き残りにする
        # tm_filelist = glob.glob(f"{TRAIN_MODEL_PATH}*_{sbl}_{TRADE_SYSTEM}_*")
        # tr_filelist = glob.glob(f"{TRAIN_RESULT_PATH}*_{sbl}_{TRADE_SYSTEM}_*")
        tm_filelist = glob.glob(f"{TRAIN_MODEL_PATH}*_{sbl}_*_*")
        tr_filelist = glob.glob(f"{TRAIN_RESULT_PATH}*_{sbl}_*_*")

        os.makedirs(TRAIN_MODEL_PATH_VOID, exist_ok=True)
        os.makedirs(TRAIN_RESULT_PATH_VOID, exist_ok=True)

        if win_file is None:
            for l in tm_filelist:
                shutil.move(l, TRAIN_MODEL_PATH_VOID)
            for l in tr_filelist:
                shutil.move(l, TRAIN_RESULT_PATH_VOID)
            return True
        else:
            MAGIC = win_file.split('\\')[-1].split('_')[4].split('.')[0]
            for l in tm_filelist:
                if MAGIC not in l:
                    shutil.move(l, TRAIN_MODEL_PATH_VOID)
            for l in tr_filelist:
                if MAGIC not in l:
                    shutil.move(l, TRAIN_RESULT_PATH_VOID)
            return False


# =====================================================================
# PriceData
# =====================================================================
class PriceData:
    def __init__(self, sbl, prd):
        self.symbol = sbl
        self.period = prd
        self.price_list = None
        self.static_data = None

    def ConvertPriceDataFileToDataFrame(self):
        filenames = glob.glob(f"{PRICEDATA_PATH}{self.symbol}_{self.period}_*.csv")
        list_ = []
        for file in filenames:
            df = pd.read_csv(file)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            list_.append(df)
        self.price_list = pd.concat(list_)
        self.price_list.sort_index(inplace=True)
        self.price_list.drop_duplicates(inplace=True)

    def AddStaticInfo(self, periods=None):
        self.static_data = pd.DataFrame(index=self.price_list.index)
        magnification = TrainEnvHelper.GetDigitMagnification(self.symbol)
        self.static_data[['open', 'high', 'low', 'close']] = self.price_list[['open', 'high', 'low', 'close']] * magnification

        long_p = periods[0] if periods else LONG_PERIOD
        short_p = periods[1] if periods else SHORT_PERIOD
        tickvol_p = periods[2] if periods else TICKVOL_PERIOD

        # >>'tick_volume'
        # vol_mean = self.price_list['tick_volume'].rolling(tickvol_p).mean() + 1e-8
        # self.static_data['tick_volume'] = np.tanh(np.log10(np.maximum((self.price_list['tick_volume'] + 1e-8) / vol_mean, 1e-8)))
        vol_mean = self.price_list['tick_volume'].rolling(tickvol_p).mean()
        vol_std  = self.price_list['tick_volume'].rolling(tickvol_p).std() + 1e-8
        vol_z    = (self.price_list['tick_volume'] - vol_mean) / vol_std        
        # ±2σ 程度でなだらかに飽和させ、[-1, 1] に収める
        self.static_data['tick_volume'] = np.tanh(vol_z / 2.0)
        
        # >> 'high-low', 'close-open'
        # self.static_data['close-open'] = self.static_data['close'] - self.static_data['open']
        # self.static_data['high-low'] = self.static_data['high'] - self.static_data['low']
        hl_raw = self.static_data['high'] - self.static_data['low']
        co_raw = self.static_data['close'] - self.static_data['open']
        # 1. high-low: 平均値幅に対する比率を tanh で [-1, 1] に変換
        mean_hl = hl_raw.rolling(self.short_period).mean() + 1e-8
        self.static_data['high-low'] = np.tanh(np.log(np.maximum(hl_raw / mean_hl, 1e-8)))
        # 2. close-open: 当該バーの値幅に対する実体の割合 [-1, 1]
        self.static_data['close-open'] = co_raw / (hl_raw + 1e-8)
        
        self.static_data['sma_close_short'] = self.static_data['close'].rolling(short_p).mean()
        self.static_data['sma_close_long'] = self.static_data['close'].rolling(long_p).mean()
        self.static_data['sma_open_short'] = self.static_data['open'].rolling(short_p).mean()
        self.static_data['sma_open_long'] = self.static_data['open'].rolling(long_p).mean()
        self.static_data['std_short'] = self.static_data['close'].rolling(short_p).std()
        self.static_data['std_long'] = self.static_data['close'].rolling(long_p).std()
        self.static_data['z_score_short'] = (self.static_data['close'] - self.static_data['sma_close_short']) / (self.static_data['std_short'] + 1e-8)
        self.static_data['z_score_long'] = (self.static_data['close'] - self.static_data['sma_close_long']) / (self.static_data['std_long'] + 1e-8)
        self.static_data['sma_close_short-long'] = self.static_data['sma_close_short'] - self.static_data['sma_close_long']
        self.static_data['std_short-long'] = self.static_data['std_short'] - self.static_data['std_long']
        self.static_data['z_score_short-long'] = self.static_data['z_score_short'] - self.static_data['z_score_long']
        self.static_data['sma_close-open_short'] = self.static_data['sma_close_short'] - self.static_data['sma_open_short']
        self.static_data['sma_close-open_long'] = self.static_data['sma_close_long'] - self.static_data['sma_open_long']

        self.static_data.drop(columns=['open', 'high', 'low', 'close', 'sma_close_short', 'sma_close_long', 'sma_open_short', 'sma_open_long'], inplace=True)
        self.static_data.rename(columns=lambda s: f"{self.symbol}_{self.period}_{s}", inplace=True)

    def GetPriceData(self, dt):
        return self.price_list[self.price_list.index == dt]


# =====================================================================
# マルチタイムフレーム TrainData (W1, D1, H4, H1 の結合)
# =====================================================================
class MTFTrainData:
    def __init__(self, symbol, periods_dict, periods):
        self.symbol = symbol
        dfs = []
        for tf in MTF_TIMEFRAMES:  # ['W1', 'D1', 'H4', 'H1']
            p_obj = periods_dict[tf][symbol]
            p_obj.AddStaticInfo(periods)
            dfs.append(p_obj.static_data)

        base_df = dfs[3].copy()
        for higher_df in [dfs[2], dfs[1], dfs[0]]:
            base_df = pd.merge_asof(
                base_df, higher_df,
                left_index=True, right_index=True,
                direction='backward'
            )
        base_df.bfill(inplace=True)
        base_df.ffill(inplace=True)
        self.train_data = base_df

    def GetTrainDataPeriod(self, start_dt, end_dt):
        return self.train_data.loc[start_dt:end_dt]

    def GetTrainDataColLen(self):
        return len(self.train_data.columns)


# =====================================================================
# Account クラス (トレーリングストップ対応)
# =====================================================================
class Account(BaseAccount):
    def __init__(self, sbl, tf, inidt, trndrt, load_flg=False, file_name=None, periods=None, real_spread_flg=False):
        margin_val = MARGIN_DICT.get(sbl, 0.02)
        if load_flg and (file_name is not None) and (file_name.account is not None):
            acc_inifile = configparser.ConfigParser()
            acc_inifile.read(file_name.account)
            tpl = float(acc_inifile.get('COMMOM', 'TAKEPROFIT_LEVEL'))
            lcl = float(acc_inifile.get('COMMOM', 'LOSSCUT_LEVEL'))
            ps = float(acc_inifile.get('COMMOM', 'POS_SCALE'))
            rsl = float(acc_inifile.get('COMMOM', 'REAL_SPREAD_LIMIT'))
            ptw = int(acc_inifile.get('COMMOM', 'PRE_TRADE_WEEKS'))
        else:
            tpl = random.choices([random.randrange(TP_LC_MIN_LEVEL, TP_LC_MAX_LEVEL + 1, TP_LC_STEP), 0],
                                 [math.floor((TP_LC_MAX_LEVEL - TP_LC_MIN_LEVEL + 1) / TP_LC_STEP), 1])[0] / 1000.0
            lcl = random.choices([random.randrange(TP_LC_MIN_LEVEL, TP_LC_MAX_LEVEL + 1, TP_LC_STEP), 0],
                                 [math.floor((TP_LC_MAX_LEVEL - TP_LC_MIN_LEVEL + 1) / TP_LC_STEP), 1])[0] / 1000.0
            ps = random.randrange(1, 11, 1) / 10.0
            rsl = REAL_SPREAD_LIMIT_LIST.get(sbl, 1.0) if real_spread_flg else 1.0
            ptw = random.choice([0, 1, 2, 4])

        super().__init__(sbl, tf, rsl, lcl, tpl, ps, ptw, margin=margin_val)
        self.magnification = TrainEnvHelper.GetDigitMagnification(self.symbol)
        self.long_period = periods[0] if periods else LONG_PERIOD
        self.short_period = periods[1] if periods else SHORT_PERIOD
        self.tickvol_period = periods[2] if periods else TICKVOL_PERIOD

        self.ticks_frame = None
        self.bar_to_tick = {}
        self.bar_to_tick_minute = {}
        self.SetTradePeriod(inidt, trndrt)

    def SetTradePeriod(self, stdt, trdl):
        self.ini_datetime = stdt
        self.train_duration = trdl
        if self.train_duration == 'W':
            self.start_period = (self.ini_datetime + timedelta(days=1 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0)
            self.end_period = (self.ini_datetime + timedelta(days=6 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0) - self.delta_period
            self.end_trade = self.end_period - self.delta_period

    def CheckTpLc(self, lstdt, lstprc):
        self.last_close_price = 0.0
        utc_from = lstdt
        utc_to = lstdt + self.delta_period

        if utc_from not in self.df_account.index:
            last_dt = self.df_account.iloc[-1].name
            for bd in pd.date_range(start=last_dt + self.delta_period, end=utc_from, freq=PERIOD_FREQ_DICT[self.period]):
                if bd.weekday() < 5:
                    self.df_account.loc[bd] = self.df_account.loc[last_dt]
                    self.df_account.at[bd, 'close_pl'] = 0.0

        self.pos_open_price, self.has_long, self.has_short, self.close_pl = self.df_account.loc[
            utc_from, ['pos_open_price', 'has_long', 'has_short', 'close_pl']
        ]

        start_idx, end_idx = self.bar_to_tick.get(utc_from, (0, 0))
        if start_idx == end_idx:
            return

        _ticks_period = self.ticks_frame.iloc[start_idx:end_idx]
        self.last_open_ask = _ticks_period.iloc[0]['ask']
        self.last_open_bid = _ticks_period.iloc[0]['bid']
        self.last_close_ask = _ticks_period.iloc[-1]['ask']
        self.last_close_bid = _ticks_period.iloc[-1]['bid']

        if self.pos_open_price > 0:
            if self.has_long > 0.0:
                if self.losscut_price == 0.0:
                    self.losscut_price = self.pos_open_price - self.lc_level / self.magnification
                self.takeprofit_price = self.pos_open_price + self.tp_level / self.magnification
                self.last_close_price = self.last_close_bid
                self.float_pl = (self.last_close_price - self.pos_open_price) * self.magnification

                bids = _ticks_period['bid'].to_numpy()
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

                asks = _ticks_period['ask'].to_numpy()
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

    def CalcCountdown(self, cntdt):
        countdown = (cntdt - self.start_period).total_seconds() / (self.end_trade - self.start_period).total_seconds()
        self.df_account.loc[cntdt, ['countdown']] = countdown

    def GetAccountInfo(self, idx_dt, cols=None):
        return self.df_account.loc[idx_dt, cols] if cols else self.df_account.loc[idx_dt]

    def EvaluateRewrd(self, actn_idx, test_dt, last_dt, df_price_data):
        utc_from = test_dt
        utc_to = utc_from + timedelta(minutes=1)
        sell_open_price, buy_open_price = 0.0, 0.0
        sell_close_price, buy_close_price = 0.0, 0.0

        for n in range(MAX_RETRY_NUM[self.period]):
            if utc_from in self.bar_to_tick_minute:
                start_idx, end_idx = self.bar_to_tick_minute[utc_from]
            else:
                utc_from += timedelta(minutes=1)
                utc_to += timedelta(minutes=1)
                continue

            if start_idx == end_idx:
                if n == MAX_RETRY_NUM[self.period] - 1:
                    if actn_idx in [LONG_ENTRY, SHORT_ENTRY]:
                        actn_idx = NO_ACTION
                    elif actn_idx in [CLOSE_AND_LONG, CLOSE_AND_SHORT]:
                        actn_idx = POSITION_CLOSE
                utc_from += timedelta(minutes=1)
                utc_to += timedelta(minutes=1)
                continue

            _ticks_first = self.ticks_frame.iloc[start_idx:end_idx]
            sell_price = float(_ticks_first.iloc[0]['bid'])
            buy_price = float(_ticks_first.iloc[0]['ask'])
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
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0, 0.0]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
        elif actn_idx == SHORT_ENTRY:
            if sell_open_price == 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0, 0.0]
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
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']] = [test_dt, buy_open_price, self.pos_scale]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
        elif actn_idx == CLOSE_AND_SHORT:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                c_pl = self.df_account.loc[last_dt, 'float_pl'] if buy_close_price == 0.0 else (buy_close_price - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl]
            if sell_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']] = [test_dt, sell_open_price, self.pos_scale]
                self.losscut_price = sell_open_price + (self.lc_level / self.magnification)
        elif actn_idx == TRAILING_STOP:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                new_lc = sell_price - (self.lc_level / self.magnification)
                if new_lc > self.losscut_price:
                    self.losscut_price = new_lc
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                new_lc = buy_price + (self.lc_level / self.magnification)
                if self.losscut_price == 0.0 or new_lc < self.losscut_price:
                    self.losscut_price = new_lc
            self.df_account.loc[test_dt, 'close_pl'] = 0.0

    def SetTickDataPeriod(self, start_date, end_date):
        schema_tbl = f"{SCHEMA}.pricedata_{self.symbol.lower()}_tick"
        sql = f"SELECT bid, ask, time_msc FROM {schema_tbl} WHERE time_msc>='{start_date.strftime('%Y-%m-%d %H:%M')}' AND time_msc<'{end_date.strftime('%Y-%m-%d %H:%M')}' ORDER BY time_msc ASC"
        for _ in range(5):
            self.ticks_frame = cx.read_sql(query=sql, conn=CONN_URL)
            if not self.ticks_frame.empty:
                self.ticks_frame['bid'] = self.ticks_frame['bid'].astype(float)
                self.ticks_frame['ask'] = self.ticks_frame['ask'].astype(float)
                self.ticks_frame = self.ticks_frame.set_index('time_msc')
                return True
        return False

    def DropTickDataPeriod(self):
        if self.ticks_frame is not None:
            self.ticks_frame.drop(self.ticks_frame.index, inplace=True)

    def SaveIniFile(self, filepath):
        with open(filepath, mode='w') as f:
            f.write(f"[COMMOM]\nLOSSCUT_LEVEL={self.lc_level}\nTAKEPROFIT_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nPOS_SCALE={self.pos_scale}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")

    def SaveSetFile(self, mdlname):
        with open(mdlname.set_file_path, mode='w') as f:
            f.write(f"MODEL_NAME={mdlname.model_name}\nLC_LEVEL={self.lc_level}\nTP_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nPOS_SCALE={self.pos_scale}\nMODEL_INPUT_NUM={MODEL_INPUT_NUM}\nMODEL_HIDDEN_NUM={MODEL_HIDDEN_NUM}\nMODEL_OUTPUT_NUM={MODEL_OUTPUT_NUM}\nMAGIC={mdlname.magic}\nTRADE_SYSTEM={TRADE_SYSTEM}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")


# =====================================================================
# Agent クラス (7行動対応)
# =====================================================================
class Agent():
    def __init__(self, sbl, input_num, output_num, LoadModel=False, FilePath=None, train_mode=True):
        self.hidden_num = input_num
        self.symbol = sbl
        self.train_mode = train_mode

        self.main_brain = Brain2027(input_num, output_num, self.hidden_num)
        self.target_brain = Brain2027(input_num, output_num, self.hidden_num)
        if LoadModel and FilePath and FilePath.brain:
            self.main_brain.loadModel(FilePath.brain)

        self.target_brain.copyNN(self.main_brain.state_dict())
        self.device = self.main_brain.device

        self.df_reward = pd.DataFrame(columns=['symbol', 'reward', 'q_taken', 'target_q'])
        self.q_taken_list = []
        self.target_q_list = []
        self.last_transition = None
        self.prev_float_pl = 0.0

        if LoadModel and FilePath and FilePath.agent:
            agt_inifile = configparser.ConfigParser()
            agt_inifile.read(FilePath.agent)
            self.est_float_pl = float(agt_inifile.get('COMMOM', 'ESTIMATE_FLOAT_PL'))
        else:
            self.est_float_pl = random.random()

    def Step(self, set_datetime, input_data, action_mask, df_account_info, epi_num, is_terminal=False):
        main_q = self.main_brain(input_data)
        target_q = self.target_brain(input_data)

        action_index, _ = self.main_brain.decideAction(action_mask, main_q, epi_num, self.train_mode)

        if self.last_transition is not None and self.train_mode:
            prev_dt = self.last_transition['datetime']
            prev_q_taken = self.last_transition['q_taken']
            prev_reward = self.last_transition['reward']

            masked_next_main_q = torch.where(
                torch.BoolTensor(action_mask).to(self.device),
                main_q,
                torch.full_like(main_q, float('-inf'))
            )
            best_next_action = masked_next_main_q.argmax(dim=1)
            next_val = target_q[0, best_next_action].view(1)
            target_value = prev_reward + GAMMA * next_val

            self.q_taken_list.append(prev_q_taken)
            self.target_q_list.append(target_value.detach())
            self.df_reward.loc[prev_dt, ['symbol', 'reward', 'q_taken', 'target_q']] = [
                self.symbol, prev_reward.item(), prev_q_taken.item(), target_value.item()
            ]

        curr_float_pl = float(df_account_info.float_pl)
        curr_close_pl = float(df_account_info.close_pl)
        delta_float_pl = curr_float_pl - self.prev_float_pl
        self.prev_float_pl = curr_float_pl

        step_reward = curr_close_pl + (delta_float_pl * self.est_float_pl)
        step_reward_tensor = torch.tensor([step_reward], dtype=torch.float32, device=self.device)
        current_q_taken = main_q[0, action_index].view(1)

        if is_terminal and self.train_mode:
            self.q_taken_list.append(current_q_taken)
            self.target_q_list.append(step_reward_tensor.detach())
            self.df_reward.loc[set_datetime, ['symbol', 'reward', 'q_taken', 'target_q']] = [
                self.symbol, step_reward_tensor.item(), current_q_taken.item(), step_reward_tensor.item()
            ]
            self.last_transition = None
        else:
            self.last_transition = {
                'datetime': set_datetime,
                'q_taken': current_q_taken,
                'reward': step_reward_tensor
            }

        return action_index

    def GetEpisodeTensors(self):
        if not self.q_taken_list:
            return None, None
        return torch.cat(self.q_taken_list).view(-1, 1), torch.cat(self.target_q_list).view(-1, 1)

    def ClearEpisode(self):
        self.df_reward = self.df_reward.drop(self.df_reward.index)
        self.q_taken_list.clear()
        self.target_q_list.clear()
        self.last_transition = None
        self.prev_float_pl = 0.0

    def UpdateMainNN(self, output, target):
        loss = self.main_brain.evaluateLossFunction(output, target)
        self.main_brain.updateNN(loss)
        return loss.item()

    def CopyMainNNToTargetNN(self):
        self.target_brain.copyNN(self.main_brain.state_dict())

    def ResetHiddenCellState(self):
        self.main_brain.resetHiddenCellState()
        self.target_brain.resetHiddenCellState()

    def SaveAgentModel(self, AgentFilePath, BrainFilePath, eval_value):
        with open(AgentFilePath, mode='w') as f:
            f.write(f"[COMMOM]\nESTIMATE_FLOAT_PL = {self.est_float_pl}\nEVAL_VALUE = {eval_value}\n")
        self.main_brain.saveModel(BrainFilePath)

    def SetModelMode(self, train_mode):
        self.train_mode = train_mode
        self.main_brain.train(train_mode)
        self.target_brain.train(train_mode)

    def GetMainHiddenCellState(self):
        return self.main_brain.getHiddenCellState()

    def GetOutput(self):
        return self.main_brain.action_score, self.main_brain.masked_action_scores


# =====================================================================
# Trainer クラス
# =====================================================================
class Trainer():
    def __init__(self, epinum, sbl, start_dt, tdur, gnum, anum, mtf_train_data, periods_dict, load_flg, filepth=None, prds=None, cp_frq=2, rsf=False):
        self.epi_num = epinum
        self.symbol = sbl
        self.period = BASE_TIMEFRAME
        self.start_date_time = start_dt
        self.test_duration = tdur
        self.gen_num = gnum
        self.agent_num = anum
        self.load_flg = load_flg
        self.file_path = filepth
        self.periods = prds
        self.copy_frq = cp_frq
        self.real_spread_flg = rsf
        self.periods_dict = periods_dict
        self.mtf_train_data = mtf_train_data

        self.df_result = pd.DataFrame(columns=['tradenum', 'sum', 'mean', 'sd'])
        self.acnt = Account(self.symbol, self.period, self.start_date_time, self.test_duration, self.load_flg, self.file_path, self.periods, self.real_spread_flg)
        self.pre_trade_weeks = self.acnt.pre_trade_weeks

        self.start_date_time, self.end_date_time = self.acnt.start_period, self.acnt.end_period
        self.delta_time = CoreEnvironment.GetPeriodTimeDelta(self.period)

        self.input_num = mtf_train_data.GetTrainDataColLen() + 4
        self.output_num = 7
        self.agnt = Agent(self.symbol, self.input_num, self.output_num, self.load_flg, self.file_path)

        global MODEL_INPUT_NUM, MODEL_HIDDEN_NUM, MODEL_OUTPUT_NUM
        MODEL_INPUT_NUM = self.input_num
        MODEL_HIDDEN_NUM = self.input_num
        MODEL_OUTPUT_NUM = self.output_num

        if not self.UpdateTradePeriod(self.start_date_time, self.test_duration):
            print(999)
            raise RuntimeError('UpdateTradePeriod failed during Trainer init.')

    def TrainAgent(self, epinum=-1, train_mode=True):
        if epinum > -1:
            self.epi_num = epinum
        self.agnt.SetModelMode(train_mode)

        for self.enum in range(self.epi_num):
            from_date = self.start_date_time - timedelta(weeks=self.pre_trade_weeks)
            to_date = self.start_date_time - timedelta(weeks=1)
            every_monday = pd.date_range(from_date, to_date, freq='W-MON')

            for w in every_monday:
                self._ExecTrade(w, train_mode, pretrade_flg=True)
            self._ExecTrade(self.start_date_time, train_mode, pretrade_flg=False)

        return self.df_result

    def _ExecTrade(self, stdt, train_mode, pretrade_flg):
        if not self.UpdateTradePeriod(stdt, self.test_duration):
            print(999)
            raise RuntimeError('UpdateTradePeriod failed in _ExecTrade.')

        lstdt = self._GetLastDateTime(stdt, self.delta_time)
        tstdt = stdt
        self.acnt.SetAccount(lstdt)

        price_obj = self.periods_dict[self.period][self.symbol]
        bar_list = list(self.df_trade_data_period.iterrows())
        total_bars = len(bar_list)

        action_counts = {code: 0 for code in range(7)}

        for idx, (lstdt, df_train_row) in enumerate(bar_list):
            is_last_bar = (idx == total_bars - 1)

            if price_obj.GetPriceData(lstdt).empty:
                continue

            self.acnt.CheckTpLc(lstdt, price_obj.GetPriceData(lstdt))
            self.acnt.CalcCountdown(lstdt)
            self.df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown'])

            self.action_mask = CoreEnvironment.getAvailableAction(
                self.df_acnt.float_pl, self.df_acnt.has_long, self.df_acnt.has_short, self.df_acnt.countdown
            )

            df_input = pd.concat([self.df_acnt, df_train_row])
            account_pl_info = self.acnt.GetAccountInfo(lstdt, ['float_pl', 'close_pl'])

            self.action_index = self.agnt.Step(
                tstdt, df_input.to_list(), self.action_mask, account_pl_info, self.enum, is_terminal=is_last_bar
            )
            act_val = self.action_index[0].item() if hasattr(self.action_index, '__getitem__') else self.action_index
            action_counts[act_val] = action_counts.get(act_val, 0) + 1

            self.acnt.EvaluateRewrd(act_val, tstdt, lstdt, price_obj.GetPriceData(tstdt))
            tstdt += self.delta_time

        if not pretrade_flg:
            trade_count = (self.acnt.df_account.close_pl != 0).sum()
            total_pl = self.acnt.df_account.close_pl.sum()
            self.df_result.loc[self.enum] = [
                trade_count,
                total_pl,
                self.acnt.df_account.close_pl.mean(),
                self.acnt.df_account.close_pl.std()
            ]

            loss_val = 0.0
            q_taken_tensor, target_q_tensor = self.agnt.GetEpisodeTensors()
            if q_taken_tensor is not None and train_mode and (self.enum + 1 < self.epi_num):
                loss_val = self.agnt.UpdateMainNN(q_taken_tensor, target_q_tensor)
                if (self.enum + 1) % self.copy_frq == 0:
                    self.agnt.CopyMainNNToTargetNN()

            self.agnt.ResetHiddenCellState()

            act_str = f"None:{action_counts[0]} L:{action_counts[1]} S:{action_counts[2]} Cls:{action_counts[3]} C&L:{action_counts[4]} C&S:{action_counts[5]} Trl:{action_counts[6]}"

            # ★バックテスト時 (train_mode == False) のログ出力
            if not train_mode:
                start_date_str = stdt.strftime('%Y-%m-%d')
                logger_trainer.info(
                    f"  [Backtest] Start: {start_date_str} | Trades: {trade_count:02d} | "
                    f"Week P/L: {total_pl:+7.2f} | {act_str}"
                )
            # ★通常トレーニング時 (train_mode == True) のログ出力
            else:
                if (self.enum + 1) % 10 == 0 or self.enum == 0 or (self.enum + 1) == self.epi_num:
                    eps = 0.5 * (1.0 / (self.enum + 1))
                    logger_trainer.info(
                        f"[Gen {self.gen_num} | Agt {self.agent_num}] Epi {self.enum + 1:03d}/{self.epi_num} | "
                        f"Loss: {loss_val:.5f} | Trades: {trade_count:02d} | P/L: {total_pl:+7.2f} | eps: {eps:.3f} | {act_str}"
                    )

        self.acnt.DropAccount()
        self.agnt.ClearEpisode()

    def ExecBacktest(self, start_year, end_year, test_duration):
        self.df_backtest_result = pd.DataFrame(columns=['tradenum', 'sum', 'mean', 'sd', 'cum_sum'])
        for y in range(start_year, end_year + 1):
            test_end_week = datetime.datetime(y, 12, 28).isocalendar().week
            for w in range(2, test_end_week):
                self.start_date_time = datetime.datetime.fromisocalendar(y, w, 1)
                if not self.UpdateTradePeriod(self.start_date_time, test_duration):
                    continue
                self.TrainAgent(1, train_mode=False)
                self.df_backtest_result.loc[self.start_date_time] = self.df_result.iloc[-1]
                self.df_backtest_result.loc[self.start_date_time, 'cum_sum'] = self.df_backtest_result['sum'].sum()
                self.df_result = self.df_result.drop(self.df_result.index)
        return self.df_backtest_result

    def UpdateTradePeriod(self, stdt, tstdl):
        self.acnt.SetTradePeriod(stdt, tstdl)
        tmp_start_date_time, tmp_end_date_time = self.acnt.start_period, self.acnt.end_period
        self.end_date_time = tmp_end_date_time

        previous_start_datetime = self._GetLastDateTime(tmp_start_date_time, self.delta_time)
        previous_end_date_time = self._GetLastDateTime(tmp_end_date_time, self.delta_time)
        self.df_trade_data_period = self.mtf_train_data.GetTrainDataPeriod(previous_start_datetime, previous_end_date_time)

        self.acnt.DropTickDataPeriod()
        if not self.acnt.SetTickDataPeriod(previous_start_datetime, tmp_end_date_time + self.delta_time):
            return False

        ticks = self.acnt.ticks_frame
        tick_index = ticks.index.to_numpy()

        self.bar_to_tick = {}
        bar_dt = previous_start_datetime
        while bar_dt <= tmp_end_date_time:
            next_bar_dt = bar_dt + self.delta_time
            start_idx = np.searchsorted(tick_index, np.datetime64(bar_dt), side='left')
            end_idx = np.searchsorted(tick_index, np.datetime64(next_bar_dt), side='left')
            self.bar_to_tick[bar_dt] = (start_idx, end_idx)
            bar_dt = next_bar_dt

        self.acnt.bar_to_tick = self.bar_to_tick

        self.bar_to_tick_minute = {}
        bar_dt = previous_start_datetime
        while bar_dt <= tmp_end_date_time + self.delta_time:
            next_bar_dt = bar_dt + timedelta(minutes=1)
            start_idx = np.searchsorted(tick_index, np.datetime64(bar_dt), side='left')
            end_idx = np.searchsorted(tick_index, np.datetime64(next_bar_dt), side='left')
            self.bar_to_tick_minute[bar_dt] = (start_idx, end_idx)
            bar_dt = next_bar_dt

        self.acnt.bar_to_tick_minute = self.bar_to_tick_minute
        return True

    def SaveParameters(self, Parameters, Modelname=None):
        self.acnt.SaveIniFile(Parameters.account)
        self.agnt.SaveAgentModel(Parameters.agent, Parameters.brain, Parameters.eval_value)
        if Modelname:
            self.acnt.SaveSetFile(Modelname)

    def _GetLastDateTime(self, cur_dt, dlt_dt):
        last_dt = cur_dt - dlt_dt
        while (self.periods_dict[self.period][self.symbol].GetPriceData(last_dt).empty or
               self.mtf_train_data.GetTrainDataPeriod(last_dt, last_dt).empty):
            last_dt -= dlt_dt
        return last_dt


# =====================================================================
# Main 実行ブロック
# =====================================================================
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print(999)
        sys.exit('Usage: python TradingSystem2027_comandline.py [trade_mode(REAL/DEMO/DEMOXM)] [symbol] [max_train_num] [train_num]')

    if args[1] == 'REAL':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_REAL
    elif args[1] == 'DEMO':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMO
    elif args[1] == 'DEMOXM':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMOXM
    else:
        print(999)
        sys.exit('trade_mode must be REAL/DEMO/DEMOXM.')
    TRADE_SYMBOL = args[2]

    TrainEnvHelper.setTradeModePath(ACCOUNT_TRADE_MODE)
    TrainEnvHelper.connectDB()

    trainnum_file_path = f"{TRAIN_MODEL_PATH}TrainNumList_System2027.csv"
    if os.path.isfile(trainnum_file_path):
        dfTrainNumList = pd.read_csv(trainnum_file_path, index_col=0)
    else:
        symbols = ['EURUSD', 'USDJPY', 'GBPUSD', 'EURJPY', 'EURGBP', 'GBPJPY']
        dfTrainNumList = pd.DataFrame(np.zeros((len(symbols), 1), dtype='int16'), columns=['System2027'], index=symbols)

    already_train_num = dfTrainNumList.at[TRADE_SYMBOL, 'System2027']
    if already_train_num >= int(args[3]):
        logger_root.info(f"Training already completed ({already_train_num} >= {args[3]}). Exit.")
        print(int(already_train_num))
        sys.exit(0)

    LOAD_FLAG = True
    PERIOD_RANDOM_SELECT = True
    TRAIN_AGENT_NUM = 5
    AGENT_TOGO_NEXT_NUM = 3
    MAX_GEN_NUM = int(already_train_num) + int(args[4])
    EPISODE_NUM = 201
    TEST_DURATION = 'W'
    BACKTEST_START_YEAR = 2022
    BACKTEST_END_YEAR = 2025
    BACKTEST_DURATION = 'W'
    COPY_FREQ_TARGET_Q_NN = 3
    REAL_SPREAD_FLG = True

    df_gen_rank = pd.DataFrame(columns=['gen', 'agt_num', 'result', 'trade_avg', 'std', 'cdp_0', 'eval_value', 'agt_obj'])
    TrainParameters = namedtuple('TrainParameters', ['account', 'agent', 'brain', 'eval_value'])
    TrainParametersList = [TrainParameters(None, None, None, None)] * TRAIN_AGENT_NUM
    AgentList = [None] * TRAIN_AGENT_NUM
    ModelName = namedtuple('ModelName', ['model_name', 'magic', 'set_file_path'])

    logger_root.info(f"=== {TRADE_SYSTEM} Training Start: {TRADE_SYMBOL} (Target Gens: {already_train_num + 1} to {MAX_GEN_NUM}) ===")

    # 全MTF [W1, D1, H4, H1] の価格CSVを展開
    periods_dict = {tf: {} for tf in MTF_TIMEFRAMES}
    for tf in MTF_TIMEFRAMES:
        p_obj = PriceData(TRADE_SYMBOL, tf)
        p_obj.ConvertPriceDataFileToDataFrame()
        periods_dict[tf][TRADE_SYMBOL] = p_obj

    for gnum in range(MAX_GEN_NUM):
        # 訓練する世代のNN Paramファイルが存在していれば、それを取得し
        # 当該世代の訓練をスキップする
        # ParameterFileは、AccountのINIファイルとAgentのINIファイル、BrainのPTHファイルがある
        # 各世代において、それぞれのファイルの数が同じかを確認し、同じであればRank毎にNamedTupleを生成する
        tmp_account_params=glob.glob('%s%s_%s_Account_Gen%s_Rank*.ini'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
        tmp_agent_params=glob.glob('%s%s_%s_Agent_Gen%s_Rank*.ini'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
        tmp_brain_params=glob.glob('%s%s_%s_Brain_Gen%s_Rank*.pth'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
    
        # AccounのINIファイルと、AgentのINIファイル、BrainのPTHファイルの数があっている場合
        # ランクごとにパラメータファイルを読み込む
        if len(tmp_account_params) > 0 and len(tmp_account_params)==len(tmp_agent_params) and len(tmp_brain_params)==len(tmp_account_params):
          TrainParametersList.clear()
          TrainParametersList=[TrainParameters(None,None,None,None)]*TRAIN_AGENT_NUM
          # 当該世代のモデルファイルを読み込む
          # 2026/02/23 TrainParametersにeval_valueを追加したため、初期化するがこの値はダミー
          for rnk in range(len(tmp_account_params)):
            TrainParametersList[rnk] = TrainParameters(account=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Account_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.ini',
                                          agent=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Agent_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.ini',
                                          brain=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Brain_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.pth',
                                          eval_value=0.0)
          logger_root.info('TrainModel:Generation %s Parameters Loaded.' %(str(gnum+1)))
    
          # 次の世代へ
          ## For Comandline
          # TrainNumListにこれまでに完了したトレーニング数を記録する
          dfTrainNumList.at[TRADE_SYMBOL,'System2027'] = gnum+1
          continue
        
        YEAR = random.randint(2021, 2024)
        WEEKNUM = random.randint(2, 51)
        WEEKDAY = 1
        test_date_time = datetime.datetime.fromisocalendar(YEAR, WEEKNUM, WEEKDAY)

        logger_root.info(f"\n>>>>> {TRADE_SYMBOL} [Generation {gnum + 1}/{MAX_GEN_NUM}] Target Week: {YEAR}-W{WEEKNUM:02d} <<<<<")

        for p, prmlst in enumerate(TrainParametersList):
            if AgentList[p] is None:
                periods = [LONG_PERIOD, SHORT_PERIOD, TICKVOL_PERIOD]
                if prmlst and prmlst.account:
                    acc_inifile = configparser.ConfigParser()
                    acc_inifile.read(prmlst.account)
                    periods = [
                        int(acc_inifile.get('COMMOM', 'LONG_PERIOD')),
                        int(acc_inifile.get('COMMOM', 'SHORT_PERIOD')),
                        int(acc_inifile.get('COMMOM', 'TICKVOL_PERIOD'))
                    ]
                elif PERIOD_RANDOM_SELECT:
                    periods = list(TrainEnvHelper.periodRandomSelect())

                mtf_data = MTFTrainData(TRADE_SYMBOL, periods_dict, periods)

                AgentList[p] = Trainer(
                    EPISODE_NUM, TRADE_SYMBOL, test_date_time, TEST_DURATION,
                    gnum + 1, p + 1, mtf_data, periods_dict, LOAD_FLAG,
                    filepth=prmlst, prds=periods, cp_frq=COPY_FREQ_TARGET_Q_NN, rsf=REAL_SPREAD_FLG
                )
            else:
                # 2026/9/23 ログに世代数が正しく表示されるようにする
                AgentList[p].gen_num = gnum + 1
                if not AgentList[p].UpdateTradePeriod(test_date_time, TEST_DURATION):
                    print(999)
                    sys.exit('UpdateTradePeriod failed.')

            logger_trainer.info(f"--- {TRADE_SYMBOL} Gen {gnum + 1} | Agent {p + 1}/{TRAIN_AGENT_NUM} Training Start ---")
            df_result = AgentList[p].TrainAgent(epinum=EPISODE_NUM, train_mode=True)
            
            logger_trainer.info(f"--- {TRADE_SYMBOL} Gen {gnum + 1} | Agent {p + 1}/{TRAIN_AGENT_NUM} Backtest Running ({BACKTEST_START_YEAR}-{BACKTEST_END_YEAR}) ---")
            df_backtest_result = AgentList[p].ExecBacktest(BACKTEST_START_YEAR, BACKTEST_END_YEAR, BACKTEST_DURATION)

            result = df_backtest_result['sum'].sum()
            trade_avg = df_backtest_result['sum'].mean()
            std = df_backtest_result['sum'].std()
            cdp_0 = stats.norm(loc=trade_avg, scale=std).cdf(0)
            eval_value = trade_avg * cdp_0 if trade_avg < 0 else trade_avg * (1 - cdp_0)
            df_gen_rank.loc[f"{gnum+1}{p+1}"] = [gnum + 1, p + 1, result, trade_avg, std, cdp_0, eval_value, AgentList[p]]

            now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            is_better = TrainEnvHelper.compareEvalValue(TRADE_SYMBOL, eval_value)
            model_file_path = TRAIN_MODEL_PATH if is_better else TRAIN_MODEL_PATH_VOID
            result_file_path = TRAIN_RESULT_PATH if is_better else TRAIN_RESULT_PATH_VOID

            model_grade = None
            if eval_value > 0.95:
                model_grade = 'G'
            elif eval_value > 0.3:
                model_grade = 'S'
            elif eval_value > 0.15:
                model_grade = 'B'
            elif eval_value > 0.08:
                model_grade = 'T'

            # ★バックテストサマリーのコンソール表示
            logger_root.info(
                f"[Agent {p + 1} Result] Total: {result:+7.1f}p | Mean: {trade_avg:+5.2f}p | "
                f"Std: {std:5.2f} | Shortfall(CDP0): {cdp_0*100:4.1f}% | Eval: {eval_value:.4f} | "
                f"Grade: {model_grade if model_grade else 'None'} ({'NEW BEST' if is_better and model_grade else 'Keep'})"
            )

            if model_grade:
                model_params = TrainParameters(
                    account=f"{model_file_path}Account_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.ini",
                    agent=f"{model_file_path}Agent_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.ini",
                    brain=f"{model_file_path}Model_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.pth",
                    eval_value=eval_value
                )
                model_name_obj = ModelName(
                    model_name=f"{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}",
                    magic=now_str,
                    set_file_path=f"{model_file_path}EA_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.set"
                )
                AgentList[p].SaveParameters(model_params, model_name_obj)
                # CSVファイルを保存する
                df_gen_rank.loc['%s%s'%(gnum+1, p+1)].to_csv(result_file_path+'Result_'+TRADE_SYMBOL+'_'+TRADE_SYSTEM+'_'+model_grade+'_'+now_str+'.csv')
                df_result.to_csv(result_file_path+'TrainResult_'+TRADE_SYMBOL+'_'+TRADE_SYSTEM+'_'+model_grade+'_'+now_str+'.csv')
                df_backtest_result.to_csv(result_file_path+'BacktestResult_'+TRADE_SYMBOL+'_'+TRADE_SYSTEM+'_'+model_grade+'_'+now_str+'.csv')

                TrainEnvHelper.send_ntfy_message(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE], f"{TRADE_SYMBOL} {TRADE_SYSTEM} Grade: {model_grade} Eval: {eval_value:.3f}")

        df_gen_rank.sort_values(['gen', 'eval_value'], ascending=[True, False], inplace=True)
        df_cur_gen_rank = df_gen_rank.query(f"gen == {gnum + 1}")

        logger_root.info(f"\n=== {TRADE_SYMBOL} Gen {gnum + 1} Summary Ranking ===")
        for rank_idx in range(len(df_cur_gen_rank)):
            row = df_cur_gen_rank.iloc[rank_idx]
            logger_root.info(f"Rank {rank_idx + 1}: Agent {int(row['agt_num'])} | Eval: {row['eval_value']:.4f} | Sum: {row['result']:+7.1f}p")

        del AgentList
        AgentList = [None] * TRAIN_AGENT_NUM
        del TrainParametersList
        TrainParametersList = [TrainParameters(None, None, None, None)] * TRAIN_AGENT_NUM

        for rnk in range(AGENT_TOGO_NEXT_NUM):
            AgentList[rnk] = df_cur_gen_rank.iloc[rnk]['agt_obj']
            TrainParametersList[rnk] = TrainParameters(
                account=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Account_Gen{gnum+1}_Rank{rnk+1}.ini",
                agent=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Agent_Gen{gnum+1}_Rank{rnk+1}.ini",
                brain=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Brain_Gen{gnum+1}_Rank{rnk+1}.pth",
                eval_value=df_cur_gen_rank.iloc[rnk]['eval_value']
            )
            AgentList[rnk].SaveParameters(TrainParametersList[rnk], None)

        gc.collect()
        dfTrainNumList.at[TRADE_SYMBOL, 'System2027'] = gnum + 1
        dfTrainNumList.to_csv(trainnum_file_path)
        
        EnvironmentCommon.send_ntfy_message(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
                                           TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\nGen '+str(gnum+1)
                                           +' Train has finished.\n'+str(df_cur_gen_rank.iloc[0,[1,2,6]]))
        
    logger_root.info(f"\nAll Training Completed. Final Completed Gens: {int(dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM])}")
    print(int(dfTrainNumList.at[TRADE_SYMBOL, 'System2027']))
    sys.exit(0)