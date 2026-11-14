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
import optuna

# 共通コアモジュールのインポート (7行動)
from TradingCore2027 import (
    Brain2027, CoreEnvironment, BaseAccount, GAMMA,
    NO_ACTION, LONG_ENTRY, SHORT_ENTRY, POSITION_CLOSE, CLOSE_AND_LONG, CLOSE_AND_SHORT, TRAILING_STOP,
    MODEL_INPUT_NUM, MODEL_HIDDEN_NUM, MODEL_OUTPUT_NUM
)

# =====================================================================
# 設定・クラス変数
# =====================================================================
EXEC_ENV = "LOCAL"

cwd = os.getcwd()
inifile = configparser.ConfigParser()
ENC = 'UTF-8'
# inifile.read(r'D:\ColabNotebooks\07_TardeSystem2027\settings.ini', ENC)
inifile.read(os.path.join(cwd, "settings.ini"), ENC)

LOGGING_INIFILE_PATH = inifile.get(EXEC_ENV, 'LOGGING_INIFILE_PATH')

PRICEDATA_PATH_REAL = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_REAL')
PRICEDATA_PATH_DEMO = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMO')
PRICEDATA_PATH_DEMOXM = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMOXM')
PRICEDATA_PATH_REALXMK = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_REALXMK')
PRICEDATA_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMOXMK')

TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_REAL')
TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMO')
TRAIN_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMOXM')
TRAIN_MODEL_PATH_REALXMK = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_REALXMK')
TRAIN_MODEL_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMOXMK')

TRAIN_MODEL_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_REAL')
TRAIN_MODEL_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMO')
TRAIN_MODEL_PATH_VOID_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMOXM')
TRAIN_MODEL_PATH_VOID_REALXMK = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_REALXMK')
TRAIN_MODEL_PATH_VOID_DEMOXMK = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMOXMK')

TMP_TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_REAL')
TMP_TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMO')
TMP_TRAIN_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMOXM')
TMP_TRAIN_MODEL_PATH_REALXMK = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_REALXMK')
TMP_TRAIN_MODEL_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMOXMK')

TRADE_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMO')
TRADE_MODEL_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMOXM')
TRADE_MODEL_PATH_REALXMK = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REALXMK')
TRADE_MODEL_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMOXMK')

TRADE_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_REAL')
TRADE_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMO')
TRADE_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMOXM')
TRADE_RESULT_PATH_REALXMK = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_REALXMK')
TRADE_RESULT_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMOXMK')

TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_REAL')
TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMO')
TRAIN_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMOXM')
TRAIN_RESULT_PATH_REALXMK = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_REALXMK')
TRAIN_RESULT_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMOXMK')

TRAIN_RESULT_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_REAL')
TRAIN_RESULT_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMO')
TRAIN_RESULT_PATH_VOID_DEMOXM = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMOXM')
TRAIN_RESULT_PATH_VOID_REALXMK = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_REALXMK')
TRAIN_RESULT_PATH_VOID_DEMOXMK = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMOXMK')

TMP_TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_REAL')
TMP_TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMO')
TMP_TRAIN_RESULT_PATH_DEMOXM = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMOXM')
TMP_TRAIN_RESULT_PATH_REALXMK = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_REALXMK')
TMP_TRAIN_RESULT_PATH_DEMOXMK = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMOXMK')

BACKTEST_RESULT_ALL_FILE_NAME = inifile.get('COMMOM', 'BACKTEST_RESULT_ALL_FILE_NAME')
BACKTEST_ALL_FILE_NAME = inifile.get('COMMOM', 'BACKTEST_ALL_FILE_NAME')
MODEL_PORTFOLIO_FILE_NAME = inifile.get('COMMOM', 'MODEL_PORTFOLIO_FILE_NAME')

SYMBOL_5DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_5DIGITS'))
SYMBOL_4DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_4DIGITS'))
SYMBOL_3DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_3DIGITS'))
SYMBOL_2DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_2DIGITS'))

DIGIT_MAGNIFICATION = eval(inifile.get('COMMOM', 'DIGIT_MAGNIFICATION'))

TICKVOL_PERIOD = int(inifile.get('COMMOM', 'TICKVOL_PERIOD'))
LONG_PERIOD = int(inifile.get('COMMOM', 'LONG_PERIOD'))
SHORT_PERIOD = int(inifile.get('COMMOM', 'SHORT_PERIOD'))
PERIOD_FREQ_DICT = eval(inifile.get('COMMOM', 'PERIOD_FREQ_DICT'))

TP_LC_MIN_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MIN_LEVEL'))
TP_LC_MAX_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MAX_LEVEL'))
TP_LC_STEP = int(inifile.get('COMMOM', 'TP_LC_STEP'))

URL_TOPIC_NAME = inifile.get('COMMOM', 'URL_TOPIC_NAME')
TRADE_SYSTEM = inifile.get('COMMOM', 'TRADE_SYSTEM') # System2027

# マルチタイムフレーム構成 [W1, D1, H4, H1]
MTF_TIMEFRAMES = ['W1', 'D1', 'H4', 'H1']
BASE_TIMEFRAME = 'H1'

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2
ACCOUNT_TRADE_MODE_DEMOXM = 3
ACCOUNT_TRADE_MODE_REALXMK = 4
ACCOUNT_TRADE_MODE_DEMOXMK = 5
ACCOUNT_TRADE_MODE_STR = ['DEMO', 'CONTEST', 'REAL', 'DEMOXM', 'REALXMK', 'DEMOXMK']

#REAL_SPREAD_LIMIT_LIST = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST'))
REAL_SPREAD_LIMIT_LIST_STD = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST_STD'))
REAL_SPREAD_LIMIT_LIST_KWM = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST_KWM'))
REAL_SPREAD_LIMIT_LIST_MTQ = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST_MTQ'))

MAX_RETRY_NUM = eval(inifile.get('LOCAL', 'MAX_RETRY_NUM'))

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
        # 2026/10/24 Kiwami口座対応
        # if symbol in SYMBOL_5DIGITS:
        if any(s in symbol for s in SYMBOL_5DIGITS):
            return DIGIT_MAGNIFICATION['SYMBOL_5DIGITS']
        # elif symbol in SYMBOL_3DIGITS:
        elif any(s in symbol for s in SYMBOL_3DIGITS):
            return DIGIT_MAGNIFICATION['SYMBOL_3DIGITS']
        # elif symbol in SYMBOL_4DIGITS:
        elif any(s in symbol for s in SYMBOL_4DIGITS):
            return DIGIT_MAGNIFICATION['SYMBOL_4DIGITS']
        # GOLD/XAUUSD 対応
        elif any(s in symbol for s in SYMBOL_2DIGITS):
            return DIGIT_MAGNIFICATION['SYMBOL_2DIGITS']
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
        global PRICEDATA_PATH, TRAIN_MODEL_PATH, TRAIN_MODEL_PATH_VOID, TMP_TRAIN_MODEL_PATH,REAL_SPREAD_LIMIT_LIST
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
            REAL_SPREAD_LIMIT_LIST = REAL_SPREAD_LIMIT_LIST_STD
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
            REAL_SPREAD_LIMIT_LIST = REAL_SPREAD_LIMIT_LIST_MTQ
            DBNAME = 'metaquotesdemo'
        elif trdmd == ACCOUNT_TRADE_MODE_DEMOXM:
            PRICEDATA_PATH = PRICEDATA_PATH_DEMOXM
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_DEMOXM
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_DEMOXM
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_DEMOXM
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMOXM
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_DEMOXM
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_DEMOXM
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_DEMOXM
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_DEMOXM
            REAL_SPREAD_LIMIT_LIST = REAL_SPREAD_LIMIT_LIST_STD
            DBNAME = 'xmtradingdemo'
        elif trdmd == ACCOUNT_TRADE_MODE_REALXMK:
            PRICEDATA_PATH = PRICEDATA_PATH_REALXMK
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_REALXMK
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_REALXMK
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_REALXMK
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_REALXMK
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_REALXMK
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_REALXMK
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_REALXMK
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_REALXMK
            REAL_SPREAD_LIMIT_LIST = REAL_SPREAD_LIMIT_LIST_KWM
            DBNAME = 'xmkiwamireal'
        elif trdmd == ACCOUNT_TRADE_MODE_DEMOXMK:
            PRICEDATA_PATH = PRICEDATA_PATH_DEMOXMK
            TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_DEMOXMK
            TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_DEMOXMK
            TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_DEMOXMK
            TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMOXMK
            TRADE_RESULT_PATH = TRADE_RESULT_PATH_DEMOXMK
            TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_DEMOXMK
            TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_DEMOXMK
            TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_DEMOXMK
            REAL_SPREAD_LIMIT_LIST = REAL_SPREAD_LIMIT_LIST_KWM # kiwami口座で同じものを使う
            DBNAME = 'xmkiwamireal' # kiwami口座で同じものを使う

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
        """新しい eval_value が既存ファイルの最高値より優れているか判定のみを行う"""
        filelist = glob.glob(f"{TRAIN_MODEL_PATH}Agent_{sbl}_*_*.ini")
        win_file = None

        for l in filelist:
            agentfile = configparser.ConfigParser()
            agentfile.read(l, ENC)
            try:
                ev = float(agentfile.get('COMMOM', 'EVAL_VALUE'))
                if ev >= eval_value:
                    eval_value = ev
                    win_file = l
            except Exception:
                continue

        # 既存より良ければ True（更新候補）、既存の方が良ければ False
        return (win_file is None), win_file

    @classmethod
    def moveToVoid(cls, sbl, win_file=None):
        """実際に新モデルを保存する直前にのみ呼び出して退避させる"""
        tm_filelist = glob.glob(f"{TRAIN_MODEL_PATH}*_{sbl}_*_*")
        tr_filelist = glob.glob(f"{TRAIN_RESULT_PATH}*_{sbl}_*_*")

        os.makedirs(TRAIN_MODEL_PATH_VOID, exist_ok=True)
        os.makedirs(TRAIN_RESULT_PATH_VOID, exist_ok=True)

        if win_file is None:
            # 既存全ファイルを VOID へ退避
            for l in tm_filelist:
                f = os.path.basename(l)
                shutil.move(l, os.path.join(TRAIN_MODEL_PATH_VOID, f))
            for l in tr_filelist:
                f = os.path.basename(l)
                shutil.move(l, os.path.join(TRAIN_RESULT_PATH_VOID, f))
        else:
            MAGIC = win_file.split('\\')[-1].split('_')[4].split('.')[0]
            for l in tm_filelist:
                if MAGIC not in l:
                    f = os.path.basename(l)
                    shutil.move(l, os.path.join(TRAIN_MODEL_PATH_VOID, f))
            for l in tr_filelist:
                if MAGIC not in l:
                    f = os.path.basename(l)
                    shutil.move(l, os.path.join(TRAIN_RESULT_PATH_VOID, f))
    
    @staticmethod
    def calc_long_term_atr(df_rates, period=120):
        """
        H1足データから長期ATR (120期間 ≒ 約2週間分) を算出
        True Range = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
        """
        high = df_rates['high']
        low = df_rates['low']
        close = df_rates['close']
        close_prev = close.shift(1)
    
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
        # ボラティリティの急変を平滑化するため長めの期間で移動平均
        atr_series = tr.rolling(window=period).mean().dropna()
        
        # 期間全体の代表的な中央値（または直近平均値）を取得
        base_atr = float(atr_series.median())
        return base_atr
        
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
        # 2026/10/24 kiwami用シンボル(UERUSD#)対応
        filenames = glob.glob(f"{PRICEDATA_PATH}{self.symbol}*_{self.period}_*.csv")
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
        """8指標による無次元化・正規化特徴量生成"""
        self.static_data = pd.DataFrame(index=self.price_list.index)
        magnification = TrainEnvHelper.GetDigitMagnification(self.symbol)

        close_p = self.price_list['close'] * magnification
        open_p  = self.price_list['open'] * magnification
        high_p  = self.price_list['high'] * magnification
        low_p   = self.price_list['low'] * magnification

        self.long_period = periods[0] if periods else LONG_PERIOD
        self.short_period = periods[1] if periods else SHORT_PERIOD
        self.tickvol_period = periods[2] if periods else TICKVOL_PERIOD
        self.signal_period = max(3, self.short_period // 2)

        # 1. tick_volume: 出来高 Z-Score + tanh
        vol_mean = self.price_list['tick_volume'].rolling(self.tickvol_period).mean()
        vol_std  = self.price_list['tick_volume'].rolling(self.tickvol_period).std() + 1e-8
        vol_z    = (self.price_list['tick_volume'] - vol_mean) / vol_std
        self.static_data['tick_volume'] = np.tanh(vol_z / 2.0)

        # 2. close-open: 実体 / レンジ 比率 [-1, 1]
        # 3. high-low: レンジ / 平均レンジ比率の ln + tanh
        hl_raw  = high_p - low_p
        co_raw  = close_p - open_p
        mean_hl = hl_raw.rolling(self.short_period).mean() + 1e-8

        self.static_data['close-open'] = co_raw / (hl_raw + 1e-8)
        self.static_data['high-low']   = np.tanh(np.log(np.maximum(hl_raw / mean_hl, 1e-8)))

        # 4. z_score_short / 5. z_score_long: ボリンジャーバンド位置 + tanh
        sma_close_short = close_p.rolling(self.short_period).mean()
        sma_close_long  = close_p.rolling(self.long_period).mean()
        std_short       = close_p.rolling(self.short_period).std()
        std_long        = close_p.rolling(self.long_period).std()

        z_short = (close_p - sma_close_short) / (std_short + 1e-8)
        z_long  = (close_p - sma_close_long) / (std_long + 1e-8)
        self.static_data['z_score_short'] = np.tanh(z_short / 2.0)
        self.static_data['z_score_long']  = np.tanh(z_long / 2.0)

        # 6. macd: トレンド方向 (MACD線)
        # 7. macd_hist: トレンド勢い・反転初動 (ヒストグラム)
        macd_raw   = sma_close_short - sma_close_long
        signal_raw = macd_raw.rolling(self.signal_period).mean()
        hist_raw   = macd_raw - signal_raw

        self.static_data['macd']      = np.tanh(macd_raw / (std_long + 1e-8))
        self.static_data['macd_hist'] = np.tanh(hist_raw / (std_long + 1e-8))

        # 8. rsi: 相対力指数 (買われすぎ/売られすぎ) [-1, 1]
        delta     = close_p.diff()
        gain      = np.where(delta > 0, delta, 0.0)
        loss      = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=close_p.index).rolling(self.short_period).mean()
        roll_loss = pd.Series(loss, index=close_p.index).rolling(self.short_period).mean()
        rs        = roll_gain / (roll_loss + 1e-8)
        rsi       = 100.0 - (100.0 / (1.0 + rs))
        self.static_data['rsi'] = (rsi - 50.0) / 50.0

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
    def __init__(self, sbl, tf, inidt, trndrt, load_flg=False, file_name=None, periods=None, real_spread_flg=False, base_atr=None):
        if load_flg and (file_name is not None) and (file_name.account is not None):
            acc_inifile = configparser.ConfigParser()
            acc_inifile.read(file_name.account)
            tpl = float(acc_inifile.get('COMMOM', 'TAKEPROFIT_LEVEL'))
            lcl = float(acc_inifile.get('COMMOM', 'LOSSCUT_LEVEL'))
            rsl = float(acc_inifile.get('COMMOM', 'REAL_SPREAD_LIMIT'))
            ptw = int(acc_inifile.get('COMMOM', 'PRE_TRADE_WEEKS'))
        else:
            # 内部スケール倍率の取得 (EURUSD:100, USDJPY:1, GOLD:0.1 など)
            magnification = TrainEnvHelper.GetDigitMagnification(sbl)

            if base_atr is not None and base_atr > 0:
                # ATR (価格差) を システム内部単位 (1000pips = 1.0) に変換
                # 例: EURUSD ATR=0.0070 (70pips) -> 0.0070 * 100 = 0.7
                # 例: USDJPY ATR=0.85 (85pips)   -> 0.85 * 1.0 = 0.85
                # 例: GOLD   ATR=25.0 (25ドル幅) -> 25.0 * 0.1 = 2.5
                atr_scaled = base_atr * magnification

                # LC: ATR 1.0倍〜3.0倍 (0.0なし・必須設定)
                min_lc = atr_scaled * 1.0
                max_lc = atr_scaled * 3.0
                step_lc = (max_lc - min_lc) / 20.0
                lc_range = np.arange(min_lc, max_lc + step_lc, step_lc)
                lcl = float(random.choice(lc_range))

                # TP: ATR 1.5倍〜4.0倍、もしくは 0.0 (トレーリングにお任せ)
                min_tp = atr_scaled * 1.5
                max_tp = atr_scaled * 4.0
                step_tp = (max_tp - min_tp) / 20.0
                tp_range = np.arange(min_tp, max_tp + step_tp, step_tp)

                # 例: 30%の確率で TP=0.0 (トレール狙い)、70%の確率で固定TP
                is_unlimited_tp = (random.random() < 0.3)
                tpl = 0.0 if is_unlimited_tp else float(random.choice(tp_range))
            else:
                # フォールバック時
                lcl = float(random.randrange(1000, 3001, 50)) / 1000.0
                tpl = random.choices([float(random.randrange(1500, 4001, 50)) / 1000.0, 0.0], weights=[7, 3])[0]

            rsl = (REAL_SPREAD_LIMIT_LIST.get(sbl, 1.0) if real_spread_flg else 1.0)/magnification
            # ptw = random.choice([0, 1, 2, 4])
            ptw = random.choice([2, 4])

        super().__init__(sbl, tf, rsl, lcl, tpl, ptw)
        self.magnification = TrainEnvHelper.GetDigitMagnification(self.symbol)
        self.long_period = periods[0] if periods else LONG_PERIOD
        self.short_period = periods[1] if periods else SHORT_PERIOD
        self.tickvol_period = periods[2] if periods else TICKVOL_PERIOD

        self.ticks_frame = None
        self.bar_to_tick = {}
        self.bar_to_tick_minute = {}
        self.SetTradePeriod(inidt, trndrt)

        # ポジション単位のトレード履歴リスト
        self.trade_history = []
        
        # 現在保有中ポジションの属性バッファ
        self.cur_pos_type = None        # 'buy' or 'sell'
        self.cur_pos_open_dt = None     # エントリー日時
        self.cur_pos_open_price = 0.0   # エントリー価格
        self.cur_pos_sl = 0.0           # 最終S/L
        self.cur_pos_tp = 0.0           # 最終T/P

    def SetTradePeriod(self, stdt, trdl):
        self.ini_datetime = stdt
        self.train_duration = trdl
        if self.train_duration == 'W':
            self.start_period = (self.ini_datetime + timedelta(days=1 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0)
            self.end_period = (self.ini_datetime + timedelta(days=6 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0, minute=0, second=0, microsecond=0) - self.delta_period
            self.end_trade = self.end_period - self.delta_period

    def CheckTpLc(self, lstdt, lstprc):
        self.last_close_price = 0.0
        self.close_type = 0  # ★ バーごとの初期化
        utc_from = lstdt
        utc_to = lstdt + self.delta_period

        if utc_from not in self.df_account.index:
            last_dt = self.df_account.iloc[-1].name
            for bd in pd.date_range(start=last_dt + self.delta_period, end=utc_from, freq=PERIOD_FREQ_DICT[self.period]):
                if bd.weekday() < 5:
                    self.df_account.loc[bd] = self.df_account.loc[last_dt]
                    self.df_account.at[bd, 'close_pl'] = 0.0
                    self.df_account.at[bd, 'close_type'] = 0  # ★ 追加

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
                self.cur_pos_sl = self.losscut_price
                self.cur_pos_tp = self.takeprofit_price

                bids = _ticks_period['bid'].to_numpy()
                hit_tp = np.where(bids >= self.takeprofit_price)[0]
                hit_lc = np.where(bids <= self.losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    actual_loss_gain = (self.losscut_price - self.pos_open_price) * self.magnification
                    self.close_pl += actual_loss_gain
                    self.close_type = 1
                    # ★ トレード履歴記録
                    self._RecordClosePosition(lstdt, self.losscut_price, actual_loss_gain)
                    self.pos_open_price, self.has_long, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.close_type = 1
                    # ★ トレード履歴記録
                    self._RecordClosePosition(lstdt, self.takeprofit_price, self.tp_level)
                    self.pos_open_price, self.has_long, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

            elif self.has_short > 0.0:
                if self.losscut_price == 0.0:
                    self.losscut_price = self.pos_open_price + self.lc_level / self.magnification
                self.takeprofit_price = self.pos_open_price - self.tp_level / self.magnification
                self.cur_pos_sl = self.losscut_price
                self.cur_pos_tp = self.takeprofit_price

                asks = _ticks_period['ask'].to_numpy()
                hit_tp = np.where(asks <= self.takeprofit_price)[0]
                hit_lc = np.where(asks >= self.losscut_price)[0]

                if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
                    actual_loss_gain = (self.pos_open_price - self.losscut_price) * self.magnification
                    self.close_pl += actual_loss_gain
                    self.close_type = -1
                    # ★ トレード履歴記録
                    self._RecordClosePosition(lstdt, self.losscut_price, actual_loss_gain)
                    self.pos_open_price, self.has_short, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

                elif len(hit_tp) > 0:
                    self.close_pl += self.tp_level
                    self.close_type = -1
                    # ★ トレード履歴記録
                    self._RecordClosePosition(lstdt, self.takeprofit_price, self.tp_level)
                    self.pos_open_price, self.has_short, self.float_pl, self.losscut_price = 0.0, 0.0, 0.0, 0.0

        # ★ close_type を含めて保存
        self.df_account.loc[utc_from, ['pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'close_type']] = (
            self.pos_open_price, self.has_long, self.has_short, self.float_pl, self.close_pl, self.close_type
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
        # ★修正1: sell_price, buy_price を初期化 (UnboundLocalError 防止)
        sell_price, buy_price = 0.0, 0.0

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

        # ★ ティックデータが完全に欠損していた場合のバーデータフォールバック
        if buy_price == 0.0 or sell_price == 0.0:
            if df_price_data is not None and not df_price_data.empty:
                fallback_price = float(df_price_data.iloc[-1]['close'])
            else:
                fallback_price = float(self.df_account.loc[last_dt, 'pos_open_price'])
            sell_price = fallback_price
            buy_price = fallback_price
            sell_open_price, buy_open_price = fallback_price, fallback_price
            sell_close_price, buy_close_price = fallback_price, fallback_price
        
        self.df_account.loc[test_dt] = self.df_account.loc[last_dt]
        self.df_account.loc[test_dt, 'close_type'] = 0  # ★ デフォルトは0
        
        if actn_idx == NO_ACTION:
            self.df_account.loc[test_dt, 'close_pl'] = 0.0
            self.df_account.loc[test_dt, 'close_type'] = 0
        elif actn_idx == LONG_ENTRY:
            if buy_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, buy_open_price, 1.0, 0.0, 0.0, 0.0]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
                # ★ 新規買い記録
                tp_p = buy_open_price + (self.tp_level / self.magnification) if self.tp_level > 0 else 0.0
                self._RecordOpenPosition(test_dt, 'buy', buy_open_price, self.losscut_price, tp_p)

        elif actn_idx == SHORT_ENTRY:
            if sell_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl']] = [test_dt, sell_open_price, 0.0, 1.0, 0.0, 0.0]
                self.losscut_price = sell_open_price + (self.lc_level / self.magnification)
                # ★ 新規売り記録
                tp_p = sell_open_price - (self.tp_level / self.magnification) if self.tp_level > 0 else 0.0
                self._RecordOpenPosition(test_dt, 'sell', sell_open_price, self.losscut_price, tp_p)

        elif actn_idx == POSITION_CLOSE:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                actual_close = buy_close_price if buy_close_price > 0.0 else float(df_price_data.iloc[-1]['close'])
                c_pl = (actual_close - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'close_type']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl, 1]
                self._RecordClosePosition(test_dt, actual_close, c_pl)
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                actual_close = sell_close_price if sell_close_price > 0.0 else float(df_price_data.iloc[-1]['close'])
                c_pl = (self.df_account.loc[last_dt, 'pos_open_price'] - actual_close) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'close_type']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl, -1]
                self._RecordClosePosition(test_dt, actual_close, c_pl)
            self.losscut_price = 0.0

        elif actn_idx == CLOSE_AND_LONG:
            if self.df_account.loc[last_dt, 'has_short'] > 0.0:
                actual_close = sell_close_price if sell_close_price > 0.0 else float(df_price_data.iloc[-1]['close'])
                c_pl = (self.df_account.loc[last_dt, 'pos_open_price'] - actual_close) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'close_type']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl, -1]
                self._RecordClosePosition(test_dt, actual_close, c_pl)
            if buy_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']] = [test_dt, buy_open_price, 1.0]
                self.losscut_price = buy_open_price - (self.lc_level / self.magnification)
                tp_p = buy_open_price + (self.tp_level / self.magnification) if self.tp_level > 0 else 0.0
                self._RecordOpenPosition(test_dt, 'buy', buy_open_price, self.losscut_price, tp_p)

        elif actn_idx == CLOSE_AND_SHORT:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                actual_close = buy_close_price if buy_close_price > 0.0 else float(df_price_data.iloc[-1]['close'])
                c_pl = (actual_close - self.df_account.loc[last_dt, 'pos_open_price']) * self.magnification
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'close_type']] = [None, 0.0, 0.0, 0.0, 0.0, c_pl, 1]
                self._RecordClosePosition(test_dt, actual_close, c_pl)
            if sell_open_price > 0.0:
                self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']] = [test_dt, sell_open_price, 1.0]
                self.losscut_price = sell_open_price + (self.lc_level / self.magnification)
                tp_p = sell_open_price - (self.tp_level / self.magnification) if self.tp_level > 0 else 0.0
                self._RecordOpenPosition(test_dt, 'sell', sell_open_price, self.losscut_price, tp_p)

        elif actn_idx == TRAILING_STOP:
            if self.df_account.loc[last_dt, 'has_long'] > 0.0:
                if sell_price > 0.0:
                    new_lc = sell_price - (self.lc_level / self.magnification)
                    if new_lc > self.losscut_price:
                        self.losscut_price = new_lc
                        self.cur_pos_sl = new_lc  # ★ SL更新
            elif self.df_account.loc[last_dt, 'has_short'] > 0.0:
                if buy_price > 0.0:
                    new_lc = buy_price + (self.lc_level / self.magnification)
                    if self.losscut_price == 0.0 or new_lc < self.losscut_price:
                        self.losscut_price = new_lc
                        self.cur_pos_sl = new_lc  # ★ SL更新
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
            # f.write(f"[COMMOM]\nLOSSCUT_LEVEL={self.lc_level}\nTAKEPROFIT_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nPOS_SCALE={self.pos_scale}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")
            f.write(f"[COMMOM]\nLOSSCUT_LEVEL={self.lc_level}\nTAKEPROFIT_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")

    def SaveSetFile(self, mdlname):
        with open(mdlname.set_file_path, mode='w') as f:
            # f.write(f"MODEL_NAME={mdlname.model_name}\nLC_LEVEL={self.lc_level}\nTP_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nPOS_SCALE={self.pos_scale}\nMODEL_INPUT_NUM={MODEL_INPUT_NUM}\nMODEL_HIDDEN_NUM={MODEL_HIDDEN_NUM}\nMODEL_OUTPUT_NUM={MODEL_OUTPUT_NUM}\nMAGIC={mdlname.magic}\nTRADE_SYSTEM={TRADE_SYSTEM}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")
            f.write(f"MODEL_NAME={mdlname.model_name}\nLC_LEVEL={self.lc_level}\nTP_LEVEL={self.tp_level}\nLONG_PERIOD={self.long_period}\nSHORT_PERIOD={self.short_period}\nTICKVOL_PERIOD={self.tickvol_period}\nMODEL_INPUT_NUM={MODEL_INPUT_NUM}\nMODEL_HIDDEN_NUM={MODEL_HIDDEN_NUM}\nMODEL_OUTPUT_NUM={MODEL_OUTPUT_NUM}\nMAGIC={mdlname.magic}\nTRADE_SYSTEM={TRADE_SYSTEM}\nREAL_SPREAD_LIMIT={self.real_spread_limit}\nPRE_TRADE_WEEKS={self.pre_trade_weeks}\n")

    def _RecordOpenPosition(self, open_dt, pos_type, open_price, sl_price, tp_price):
        """新規エントリー時にポジション情報をバッファに保持"""
        self.cur_pos_type = pos_type
        self.cur_pos_open_dt = open_dt
        self.cur_pos_open_price = open_price
        self.cur_pos_sl = sl_price
        self.cur_pos_tp = tp_price

    def _RecordClosePosition(self, close_dt, close_price, pl_internal):
        # ★ 空トレード期間のポジションはCSVに記録しない
        if getattr(self, 'is_pretrade', False):
            return
        """ポジション決済時に1トレードのレコードを確定して履歴リストに追加"""
        if self.cur_pos_open_price <= 0.0 or self.cur_pos_type is None:
            return

        # P/L (pips) 換算 (内部スケール 1.0 = 1,000 pips)
        pl_pips = pl_internal * 1000.0

        # 価格変化率 (%)[RateOfChange]
        if self.cur_pos_type == 'buy':
            change_pct = (close_price - self.cur_pos_open_price) / self.cur_pos_open_price * 100.0
        else:
            change_pct = (self.cur_pos_open_price - close_price) / self.cur_pos_open_price * 100.0

        record = {
            'OpenDateTime': self.cur_pos_open_dt.strftime('%Y.%m.%d %H:%M:%S') if hasattr(self.cur_pos_open_dt, 'strftime') else str(self.cur_pos_open_dt),
            'Symbol': self.symbol,
            'Type(Position)': self.cur_pos_type,
            'OpenPrice': f"{self.cur_pos_open_price:.5f}",
            'Stop(S/L)': f"{self.cur_pos_sl:.5f}" if self.cur_pos_sl > 0 else "0.00000",
            'Limit(T/P)': f"{self.cur_pos_tp:.5f}" if self.cur_pos_tp > 0 else "0.00000",
            'CloseDateTime': close_dt.strftime('%Y.%m.%d %H:%M:%S') if hasattr(close_dt, 'strftime') else str(close_dt),
            'ClosePrice': f"{close_price:.5f}",
            'P/L(pips)': f"{pl_pips:.1f}",
            'RateOfChange': f"{change_pct:+.3f}%"
        }
        self.trade_history.append(record)

        # バッファの初期化
        self.cur_pos_type = None
        self.cur_pos_open_dt = None
        self.cur_pos_open_price = 0.0
        self.cur_pos_sl = 0.0
        self.cur_pos_tp = 0.0

# =====================================================================
# Agent クラス (7行動対応)
# =====================================================================
class Agent():
    def __init__(self, sbl, input_num, output_num, LoadModel=False, FilePath=None, train_mode=True, 
                 lr=0.0001, gamma=0.99, dropout=0.1, hidden_num=36):
        # ★ 引数で hidden_num を指定できるように変更
        self.hidden_num = hidden_num
        self.symbol = sbl
        self.train_mode = train_mode
        self.lr = lr
        self.gamma = gamma  # ★ 動的GAMMA

        self.main_brain = Brain2027(input_num, output_num, self.hidden_num, lr=self.lr, dropout=dropout)
        self.target_brain = Brain2027(input_num, output_num, self.hidden_num, lr=self.lr, dropout=dropout)
        # 重みロード時の形状不一致を安全にハンドリング
        if LoadModel and FilePath and FilePath.brain:
            try:
                self.main_brain.loadModel(FilePath.brain)
            except RuntimeError as e:
                # hidden_num が変更された場合は新規初期化として継続
                logger_agent.warning(f"Architecture mismatch (hidden_num: {self.hidden_num}). Re-initializing weights: {e}")

        self.target_brain.copyNN(self.main_brain.state_dict())
        self.device = self.main_brain.device

        self.df_reward = pd.DataFrame(columns=['symbol', 'reward', 'q_taken', 'target_q'])
        self.q_taken_list = []
        self.target_q_list = []
        self.last_transition = None
        self.prev_float_pl = 0.0

        # ★ ESTIMATE_FLOAT_PL 関連の処理を完全削除
        '''
        if LoadModel and FilePath and FilePath.agent:
            agt_inifile = configparser.ConfigParser()
            agt_inifile.read(FilePath.agent)
            self.est_float_pl = float(agt_inifile.get('COMMOM', 'ESTIMATE_FLOAT_PL'))
        else:
            self.est_float_pl = random.random()
        '''

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
            # ★ 定数 GAMMA ではなく self.gamma を使用
            target_value = prev_reward + self.gamma * next_val

            self.q_taken_list.append(prev_q_taken)
            self.target_q_list.append(target_value.detach())
            self.df_reward.loc[prev_dt, ['symbol', 'reward', 'q_taken', 'target_q']] = [
                self.symbol, prev_reward.item(), prev_q_taken.item(), target_value.item()
            ]

        curr_float_pl = float(df_account_info.float_pl)
        curr_close_pl = float(df_account_info.close_pl)
        delta_float_pl = curr_float_pl - self.prev_float_pl
        self.prev_float_pl = curr_float_pl

        # ★ 案B: 確定損益 ＋ 含み損益純増減（1.0倍固定）
        step_reward = curr_close_pl + delta_float_pl
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
            # ★ hidden_num も保存して推論環境と同期
            # おそらく書き込んだ HIDDEN_NUM が使われることはない
            f.write(f"[COMMOM]\nEVAL_VALUE = {eval_value}\nHIDDEN_NUM = {self.hidden_num}\n")
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
    def __init__(self, epinum, sbl, start_dt, tdur, gnum, anum, mtf_train_data, periods_dict, load_flg, 
                 filepth=None, prds=None, cp_frq=2, rsf=False, lr=0.0001, gamma=0.99, dropout=0.1, hidden_num=36):
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
        self.copy_frq = cp_frq # ★ 動的Target同期頻度
        self.real_spread_flg = rsf
        self.periods_dict = periods_dict
        self.mtf_train_data = mtf_train_data

        # H1 価格データから長期ATRを事前計算
        h1_df = self.periods_dict[self.period][self.symbol].price_list
        base_atr = TrainEnvHelper.calc_long_term_atr(h1_df, period=120)
        
        self.df_result = pd.DataFrame(columns=['tradenum', 'sum', 'mean', 'sd', 'long_tradenum', 'short_tradenum'])
        # self.acnt = Account(self.symbol, self.period, self.start_date_time, self.test_duration, self.load_flg, self.file_path, self.periods, self.real_spread_flg)
        # base_atr を Account に渡す
        self.acnt = Account(
            self.symbol, self.period, self.start_date_time, self.test_duration,
            self.load_flg, self.file_path, self.periods, self.real_spread_flg,
            base_atr=base_atr
        )
        self.pre_trade_weeks = self.acnt.pre_trade_weeks

        self.start_date_time, self.end_date_time = self.acnt.start_period, self.acnt.end_period
        self.delta_time = CoreEnvironment.GetPeriodTimeDelta(self.period)

        # 8指標 * 4時間足 + 4アカウント = 36次元
        self.input_num = mtf_train_data.GetTrainDataColLen() + 4
        self.output_num = 7
        self.hidden_num = hidden_num
        self.lr = lr
        self.gamma = gamma
        self.dropout = dropout
        # Agent に hidden_num を渡す
        self.agnt = Agent(
            self.symbol, self.input_num, self.output_num, self.load_flg, self.file_path,
            lr=self.lr, gamma=self.gamma, dropout=self.dropout, hidden_num=self.hidden_num
        )

        global MODEL_INPUT_NUM, MODEL_HIDDEN_NUM, MODEL_OUTPUT_NUM
        MODEL_INPUT_NUM = self.input_num
        MODEL_HIDDEN_NUM = self.hidden_num
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
        # _ExecTrade の冒頭付近
        self.acnt.is_pretrade = pretrade_flg
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
            # 変更前:
            # self.df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown'])
            # 変更後:
            self.df_acnt = self.acnt.GetAccountInfo(lstdt, ['has_long', 'has_short', 'float_pl', 'countdown']).copy()
            
            # 生の含み損益 (float_pl) を lc_level 基準で [-1, 1] に正規化してモデルへ入力
            raw_float_pl = float(self.df_acnt['float_pl'])
            norm_float_pl = np.tanh(raw_float_pl / (self.acnt.lc_level + 1e-8))
            
            df_acnt_input = self.df_acnt.copy()
            df_acnt_input['float_pl'] = norm_float_pl

            self.action_mask = CoreEnvironment.getAvailableAction(
                self.df_acnt.float_pl, self.df_acnt.has_long, self.df_acnt.has_short, self.df_acnt.countdown
            )

            # 変更前:
            # df_input = pd.concat([self.df_acnt, df_train_row])
            df_input = pd.concat([df_acnt_input, df_train_row])

            account_pl_info = self.acnt.GetAccountInfo(lstdt, ['float_pl', 'close_pl'])

            # 週末の最終バーでは、保有ポジションがあれば強制成行決済 (Action 3 相当)
            # NNモデルに任せても決済を選択するはずだが、確実に決済させるためのガード措置
            if is_last_bar and (self.df_acnt.has_long > 0.0 or self.df_acnt.has_short > 0.0):
                act_val = POSITION_CLOSE
            else:
                self.action_index = self.agnt.Step(
                    tstdt, df_input.to_list(), self.action_mask, account_pl_info, self.enum, is_terminal=is_last_bar
                )
                act_val = self.action_index[0].item() if hasattr(self.action_index, '__getitem__') else self.action_index
            action_counts[act_val] = action_counts.get(act_val, 0) + 1

            self.acnt.EvaluateRewrd(act_val, tstdt, lstdt, price_obj.GetPriceData(tstdt))
            tstdt += self.delta_time

        if not pretrade_flg:
            long_trade_count = int((self.acnt.df_account['close_type'] == 1).sum())
            short_trade_count = int((self.acnt.df_account['close_type'] == -1).sum())
            trade_count = long_trade_count + short_trade_count
            total_pl = float(self.acnt.df_account.close_pl.sum())
            
            self.df_result.loc[self.enum] = [
                trade_count,
                total_pl,
                self.acnt.df_account.close_pl.mean(),
                self.acnt.df_account.close_pl.std(),
                long_trade_count,
                short_trade_count
            ]

            loss_val = 0.0
            q_taken_tensor, target_q_tensor = self.agnt.GetEpisodeTensors()
            if q_taken_tensor is not None and train_mode and (self.enum + 1 < self.epi_num):
                loss_val = self.agnt.UpdateMainNN(q_taken_tensor, target_q_tensor)
                if (self.enum + 1) % self.copy_frq == 0:
                    self.agnt.CopyMainNNToTargetNN()

            self.agnt.ResetHiddenCellState()

            act_str = f"None:{action_counts[0]} L:{action_counts[1]} S:{action_counts[2]} Cls:{action_counts[3]} C&L:{action_counts[4]} C&S:{action_counts[5]} Trl:{action_counts[6]}"

            if not train_mode:
                start_date_str = stdt.strftime('%Y-%m-%d')
                logger_trainer.info(
                    f"  [Backtest] Start: {start_date_str} | Trades: {trade_count:02d} | "
                    f"Week P/L: {total_pl:+7.2f} | {act_str}"
                )
            else:
                if (self.enum + 1) % 10 == 0 or self.enum == 0 or (self.enum + 1) == self.epi_num:
                    eps = 0.5 * (1.0 / (self.enum + 1))
                    logger_trainer.info(
                        f"[Gen {self.gen_num} | Agt {self.agent_num}] Epi {self.enum + 1:03d}/{self.epi_num} | "
                        f"Loss: {loss_val:.5f} | Trades: {trade_count:02d} | P/L: {total_pl:+7.2f} | eps: {eps:.3f} | {act_str}"
                    )

        self.acnt.DropAccount()
        self.agnt.ClearEpisode()

    def ExecBacktest(self, test_duration, start_year=None, end_year=None, target_mondays=None):
        """
        target_mondays が与えられた場合はその週リストを直接実行（年跨ぎ対応）。
        指定がない場合は従来の start_year 〜 end_year の通年バックテストを実行。
        """
        self.df_backtest_result = pd.DataFrame(
            columns=['tradenum', 'sum', 'mean', 'sd', 'cum_sum', 'long_tradenum', 'short_tradenum']
        )
        # 1. 走らせる対象の月曜日リストを確定
        if target_mondays is None:
            target_mondays = []
            for y in range(start_year, end_year + 1):
                max_week = datetime.datetime(y, 12, 28).isocalendar().week
                # 2022年のみ空トレード4週を考慮して第6週からスタート、それ以外の年は第2週から
                start_w = 6 if y == 2022 else 2
                for w in range(start_w, max_week):
                    try:
                        monday_dt = datetime.datetime.fromisocalendar(y, w, 1)
                        target_mondays.append(monday_dt)
                    except ValueError:
                        continue

        # 2. 週ごとに順次バックテストを実行
        for monday_dt in target_mondays:
            self.start_date_time = monday_dt

            if not self.UpdateTradePeriod(self.start_date_time, test_duration):
                continue

            self.TrainAgent(1, train_mode=False)

            if not self.df_result.empty:
                # df_result の最新行を格納
                res_series = self.df_result.iloc[-1]
                self.df_backtest_result.loc[self.start_date_time] = [
                    res_series['tradenum'],
                    res_series['sum'],
                    res_series['mean'],
                    res_series['sd'],
                    0.0, # cum_sum (直後で計算)
                    res_series['long_tradenum'],
                    res_series['short_tradenum']
                ]
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

    def GetTradeHistoryDataFrame(self):
        """蓄積されたポジション履歴を DataFrame として取得"""
        if not self.acnt.trade_history:
                return pd.DataFrame(columns=[
                'OpenDateTime', 'Symbol', 'Type(Position)', 'OpenPrice', 
                'Stop(S/L)', 'Limit(T/P)', 'CloseDateTime', 'ClosePrice', 'P/L(pips)', 'RateOfChange'
            ])
        return pd.DataFrame(self.acnt.trade_history)

    def ClearTradeHistory(self):
        """トレード履歴リストをリセット"""
        self.acnt.trade_history.clear()

# =====================================================================
# Main 実行ブロック
# =====================================================================
if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print(999)
        sys.exit('Usage: python TradingSystem2027_comandline.py [trade_mode(REAL/DEMO/DEMOXM)] [symbol] [max_train_num] [train_num]')
    
    symbols = ['EURUSD', 'USDJPY', 'GBPUSD', 'EURJPY', 'EURGBP', 'GBPJPY', 'GOLD', 'XAUUSD']
    if args[1] == 'REAL':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_REAL
    elif args[1] == 'DEMO':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMO
    elif args[1] == 'DEMOXM':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMOXM
    elif args[1] == 'REALXMK':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_REALXMK
    elif args[1] == 'DEMOXMK':
        ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMOXMK
    else:
        print(999)
        sys.exit('trade_mode must be REAL/DEMO/DEMOXM/REALXMK/DEMOXMK.')
    TRADE_SYMBOL = args[2]

    TrainEnvHelper.setTradeModePath(ACCOUNT_TRADE_MODE)
    TrainEnvHelper.connectDB()

    trainnum_file_path = f"{TRAIN_MODEL_PATH}TrainNumList_{TRADE_SYSTEM}.csv"
    if os.path.isfile(trainnum_file_path):
        dfTrainNumList = pd.read_csv(trainnum_file_path, index_col=0)
    else:
        dfTrainNumList = pd.DataFrame(np.zeros((len(symbols), 1), dtype='int16'), columns=[TRADE_SYSTEM], index=symbols)

    already_train_num = dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM]
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
    BACKTEST_DURATION = 'W'
    COPY_FREQ_TARGET_Q_NN = 3
    REAL_SPREAD_FLG = True

    df_gen_rank = pd.DataFrame(columns=['gen', 'agt_num', 'result', 'trade_avg', 'std', 'cdp_0', 'eval_value', 'trnr_obj'])
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

    # 3期間の定義
    TRAIN_START_YEAR = 2022
    TRAIN_END_YEAR   = 2024
    
    VALID_START_YEAR = 2025
    VALID_END_YEAR   = 2025

    # 1. H1データの最新日時から直近月曜日を特定
    latest_dt = periods_dict[BASE_TIMEFRAME][TRADE_SYMBOL].price_list.index[-1]
    latest_monday = (latest_dt - timedelta(days=latest_dt.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    
    # 2. 直近24週分の月曜日リストを作成（年跨ぎ対応）
    TEST_WEEKS_NUM = 24
    test_mondays = pd.date_range(end=latest_monday, periods=TEST_WEEKS_NUM, freq='W-MON').to_pydatetime().tolist()
    test_start_str = test_mondays[0].strftime('%Y-W%W (%Y-%m-%d)')
    test_end_str   = test_mondays[-1].strftime('%Y-W%W (%Y-%m-%d)')
    logger_root.info(f"Configured Out-of-Sample Test Window: {test_start_str} to {test_end_str} ({len(test_mondays)} weeks)")
    
    # Test期間の合格基準（足切りライン）
    TEST_PASS_MIN_PF     = 1.15
    TEST_PASS_MIN_TRADES = 20   # 24週（約半年）なら20〜25回程度が現実的でおすすめです
    TEST_PASS_MAX_DD     = 2.0    # 最大ドローダウン許容値 (2,000 pips相当)
    TEST_MIN_SIDE_RATIO    = 0.15   # ロング・ショートの最小比率 (各20%以上)→15%へ緩和

    # Optuna Study の初期化
    # ログが冗長にならないよう Optuna 内部ログレベルを WARNING に設定
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # SQLite に保存することでプロセス再起動時にも探索知見が引き継がれる
    study_name = f"{TRADE_SYSTEM}_{TRADE_SYMBOL}_HyperParam"
    storage_url = f"sqlite:///{TMP_TRAIN_MODEL_PATH}optuna_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",          # eval_value を最大化
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    for gnum in range(MAX_GEN_NUM):
        # パラメータファイルが存在していればスキップ
        tmp_account_params = glob.glob(f'{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Account_Gen{gnum+1}_Rank*.ini')
        tmp_agent_params   = glob.glob(f'{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Agent_Gen{gnum+1}_Rank*.ini')
        tmp_brain_params   = glob.glob(f'{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Brain_Gen{gnum+1}_Rank*.pth')

        if len(tmp_account_params) > 0 and len(tmp_account_params) == len(tmp_agent_params) and len(tmp_brain_params) == len(tmp_account_params):
            TrainParametersList.clear()
            TrainParametersList = [TrainParameters(None, None, None, None)] * TRAIN_AGENT_NUM
            for rnk in range(len(tmp_account_params)):
                TrainParametersList[rnk] = TrainParameters(
                    account=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Account_Gen{gnum+1}_Rank{rnk+1}.ini",
                    agent=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Agent_Gen{gnum+1}_Rank{rnk+1}.ini",
                    brain=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Brain_Gen{gnum+1}_Rank{rnk+1}.pth",
                    eval_value=0.0
                )
            logger_root.info(f'TrainModel: Generation {gnum+1} Parameters Loaded.')
            dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM] = gnum + 1
            continue

        # 修正後: Train 期間のみからサンプリング (2022年のみ空トレード4週分を考慮して6週目以降)
        YEAR = random.randint(TRAIN_START_YEAR, TRAIN_END_YEAR)
        min_week = 6 if YEAR == 2022 else 2
        WEEKNUM = random.randint(min_week, 51)
        WEEKDAY = 1
        test_date_time = datetime.datetime.fromisocalendar(YEAR, WEEKNUM, WEEKDAY)

        logger_root.info(f"\n>>>>> {TRADE_SYMBOL} [Generation {gnum + 1}/{MAX_GEN_NUM}] Target Week: {YEAR}-W{WEEKNUM:02d} <<<<<")

        for p, prmlst in enumerate(TrainParametersList):
            # 1. Optuna から複数パラメータを同時にサンプリング (ask)
            trial = study.ask()
            
            p_lr         = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
            p_gamma      = trial.suggest_float("gamma", 0.95, 0.99, step=0.01)
            p_dropout    = trial.suggest_float("dropout", 0.05, 0.25, step=0.05)
            p_episodes   = trial.suggest_int("episode_num", 100, 250, step=25)
            p_copy_frq   = trial.suggest_int("copy_frq", 2, 6, step=1)
            # ★ hidden_num の探索候補（24, 32, 48, 64）
            p_hidden_num = trial.suggest_categorical("hidden_num", [24, 32, 48, 64])

            logger_trainer.info(
                f"--- {TRADE_SYMBOL} Gen {gnum + 1} | Agent {p + 1}/{TRAIN_AGENT_NUM} [Trial {trial.number}] ---\n"
                f"    Params: lr={p_lr:.6f}, gamma={p_gamma:.2f}, dropout={p_dropout:.2f}, "
                f"episodes={p_episodes}, copy_frq={p_copy_frq}, hidden={p_hidden_num}"
            )
            
            # インジケータ期間の決定
            periods = [LONG_PERIOD, SHORT_PERIOD, TICKVOL_PERIOD]
            load_flag_agent = False
            if prmlst and prmlst.account:
                acc_inifile = configparser.ConfigParser()
                acc_inifile.read(prmlst.account)
                periods = [
                    int(acc_inifile.get('COMMOM', 'LONG_PERIOD')),
                    int(acc_inifile.get('COMMOM', 'SHORT_PERIOD')),
                    int(acc_inifile.get('COMMOM', 'TICKVOL_PERIOD'))
                ]
                load_flag_agent = True
            elif PERIOD_RANDOM_SELECT:
                periods = list(TrainEnvHelper.periodRandomSelect())

            mtf_data = MTFTrainData(TRADE_SYMBOL, periods_dict, periods)
            
            # ★ 毎世代新規インスタンス化し、重みは prmlst 経由で継承
            # ★ hidden_num=p_hidden_num を引数に追加
            AgentList[p] = Trainer(
                p_episodes, TRADE_SYMBOL, test_date_time, TEST_DURATION,
                gnum + 1, p + 1, mtf_data, periods_dict, load_flag_agent,
                filepth=prmlst, prds=periods, cp_frq=p_copy_frq, rsf=REAL_SPREAD_FLG,
                lr=p_lr, gamma=p_gamma, dropout=p_dropout, hidden_num=p_hidden_num
            )

            # エージェント学習（サンプリングされた episode_num で実行）
            logger_trainer.info(f"--- Training Start (Epi: {p_episodes}) ---")
            df_result = AgentList[p].TrainAgent(epinum=p_episodes, train_mode=True)

            # Validation期間（2025年）バックテスト
            logger_trainer.info(f"--- Validation Running ({VALID_START_YEAR}) ---")
            df_valid_result = AgentList[p].ExecBacktest(
                BACKTEST_DURATION, 
                start_year=VALID_START_YEAR, 
                end_year=VALID_END_YEAR
            )

            result = df_valid_result['sum'].sum()
            trade_avg = df_valid_result['sum'].mean()
            std = df_valid_result['sum'].std()
            # ★ 標準偏差が0（トレードなし等）の場合のゼロ除算ガード
            if std <= 1e-8:
                cdp_0 = 1.0 if trade_avg < 0 else 0.5
                eval_value = 0.0
            else:
                cdp_0 = stats.norm(loc=trade_avg, scale=std).cdf(0)
                eval_value = trade_avg * cdp_0 if trade_avg < 0 else trade_avg * (1 - cdp_0)

            # 2. Optuna へ結果報告 (tell)
            report_score = eval_value if not np.isnan(eval_value) else -99.0
            study.tell(trial, report_score)

            logger_trainer.info(
                f"  [Optuna Feedback] Trial {trial.number} finished -> eval_value: {report_score:.4f} "
                f"(Best so far: {study.best_value:.4f})"
            )
            
            df_gen_rank.loc[f"{gnum+1}{p+1}"] = [gnum + 1, p + 1, result, trade_avg, std, cdp_0, eval_value, AgentList[p]]
            
            # (以降の Out-of-Sample Test 判定やモデル保存処理はそのまま)            
            now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            is_better, win_file = TrainEnvHelper.compareEvalValue(TRADE_SYMBOL, eval_value)
            
            model_grade = None
            if eval_value > 5.5:
                model_grade = 'C'  # Celestial (天上の領域：eval 5.5 超)
            elif eval_value > 3.5:
                model_grade = 'M'  # Mythic (神話級：eval 3.5 以上)
            elif eval_value > 2.0:
                model_grade = 'P'  # Philosopher (賢者の石：eval 2.0 以上)
            elif eval_value > 0.95:
                model_grade = 'G'  # Grail (聖杯：eval 0.95 以上)
            elif eval_value > 0.3:
                model_grade = 'S'  # Silver
            elif eval_value > 0.15:
                model_grade = 'B'  # Bronze
            elif eval_value > 0.08:
                model_grade = 'T'  # Trial

            # 有望モデル（Grade付与 ＆ 記録更新）の場合のみ Test 期間（Out-of-Sample）を実行
            test_passed = False
            df_test_result = None
            
            if model_grade is not None and is_better:
                logger_root.info(f"★ Candidate Found! Running Out-of-Sample Test ({test_start_str} to {test_end_str}) for Agent {p + 1}...")
                
                # ★ 履歴をクリアして Test 期間のみのトレードを蓄積
                AgentList[p].ClearTradeHistory()

                df_test_result = AgentList[p].ExecBacktest(
                    BACKTEST_DURATION, 
                    target_mondays=test_mondays
                )

                # ★ ポジション一覧 DataFrame を取得
                df_positions = AgentList[p].GetTradeHistoryDataFrame()

                # Test 期間の成績評価 (PF、トレード数、損益)
                # 修正後: int で整数型にキャスト
                test_trades = int(df_test_result['tradenum'].sum())
                test_sum = float(df_test_result['sum'].sum())

                # プロフィットファクター (PF) の計算
                pos_gains = df_test_result.loc[df_test_result['sum'] > 0, 'sum'].sum()
                neg_losses = abs(df_test_result.loc[df_test_result['sum'] < 0, 'sum'].sum())
                test_pf = (pos_gains / neg_losses) if neg_losses > 0 else (99.0 if pos_gains > 0 else 0.0)

                # 最大ドローダウン (Max Drawdown) 計算
                cum_pnl = df_test_result['cum_sum']
                running_max = np.maximum.accumulate(cum_pnl)
                drawdowns = running_max - cum_pnl
                test_max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
                
                logger_root.info(
                    f"  [Test Result] Trades: {int(test_trades):02d} | Profit: {test_sum:+7.2f}p | "
                    f"PF: {test_pf:5.2f} | MaxDD: {test_max_dd:5.2f}p (Limit: <= {TEST_PASS_MAX_DD})"
                )

                # ロング・ショートの取引比率の算出
                # acnt.df_account または週次ログから long_trades / short_trades を集計
                long_trades = int(df_test_result['long_tradenum'].sum())
                short_trades = int(df_test_result['short_tradenum'].sum())
                total_trades = int(df_test_result['tradenum'].sum())
                long_ratio = (long_trades / total_trades) if total_trades > 0 else 0.0

                logger_root.info(
                    f"  [Trade Balance] Long: {long_trades:02d} ({long_ratio*100:4.1f}%) | "
                    f"Short: {short_trades:02d} ({(1-long_ratio)*100:4.1f}%) | "
                    f"Required: {TEST_MIN_SIDE_RATIO*100:.0f}% - {(1-TEST_MIN_SIDE_RATIO)*100:.0f}%"
                )

                # 合否判定（PF、取引数、利益、Max DD、L/Sバランスの5条件）
                is_balanced = (TEST_MIN_SIDE_RATIO <= long_ratio <= (1.0 - TEST_MIN_SIDE_RATIO))

                if (test_pf >= TEST_PASS_MIN_PF and 
                    test_trades >= TEST_PASS_MIN_TRADES and 
                    test_sum > 0 and 
                    test_max_dd <= TEST_PASS_MAX_DD and 
                    is_balanced):
                    test_passed = True
                    logger_root.info("  >>> TEST PASSED! Balanced and robust on out-of-sample data. <<<")
                else:
                    logger_root.info("  >>> TEST FAILED (Violated Risk, DD, or L/S Balance Criteria). Discarding to VOID. <<<")

            # 保存先の決定: Test に合格した場合のみ TRAIN_MODEL_PATH (正式採用) へ保存
            if test_passed:
                TrainEnvHelper.moveToVoid(TRADE_SYMBOL, win_file=None)
                model_file_path = TRAIN_MODEL_PATH
                result_file_path = TRAIN_RESULT_PATH
            else:
                model_file_path = TRAIN_MODEL_PATH_VOID
                result_file_path = TRAIN_RESULT_PATH_VOID

            # パラメータ・モデル・結果CSVの保存
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
                df_gen_rank.loc[f"{gnum+1}{p+1}"].to_csv(f"{result_file_path}Result_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.csv")
                df_result.to_csv(f"{result_file_path}TrainResult_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.csv")
                df_valid_result.to_csv(f"{result_file_path}ValidResult_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.csv")
                
                if df_test_result is not None:
                    df_test_result.to_csv(f"{result_file_path}TestResult_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.csv")
                    
                    # ★ MT5レポート比較用「ポジション一覧」CSVの出力
                    positions_file_path = f"{result_file_path}Positions_Test_{TRADE_SYMBOL}_{TRADE_SYSTEM}_{model_grade}_{now_str}.csv"
                    df_positions.to_csv(positions_file_path, index=False, encoding='utf-8-sig')
                    logger_root.info(f"  [Report] Position history exported to: {os.path.basename(positions_file_path)} ({len(df_positions)} positions)")

                status_msg = "PASSED" if test_passed else "REJECTED"
                TrainEnvHelper.send_ntfy_message(
                    ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
                    f"[{status_msg}] {TRADE_SYMBOL} {TRADE_SYSTEM} Grade: {model_grade} ValidEval: {eval_value:.3f}"
                )

        df_gen_rank.sort_values(['gen', 'eval_value'], ascending=[True, False], inplace=True)
        df_cur_gen_rank = df_gen_rank.query(f"gen == {gnum + 1}")

        logger_root.info(f"\n=== {TRADE_SYMBOL} Gen {gnum + 1} Summary Ranking ===")
        for rank_idx in range(len(df_cur_gen_rank)):
            row = df_cur_gen_rank.iloc[rank_idx]
            logger_root.info(f"Rank {rank_idx + 1}: Agent {int(row['agt_num'])} | Eval: {row['eval_value']:.4f} | Sum: {row['result']:+7.1f}p")

        # 世代交代処理: インスタンスは保持せず、ファイルパス情報(重み・パラメータ)のみ次世代へ引き継ぐ
        del AgentList
        AgentList = [None] * TRAIN_AGENT_NUM
        del TrainParametersList
        TrainParametersList = [TrainParameters(None, None, None, None)] * TRAIN_AGENT_NUM

        for rnk in range(AGENT_TOGO_NEXT_NUM):
            best_trainer = df_cur_gen_rank.iloc[rnk]['trnr_obj']
            TrainParametersList[rnk] = TrainParameters(
                account=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Account_Gen{gnum+1}_Rank{rnk+1}.ini",
                agent=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Agent_Gen{gnum+1}_Rank{rnk+1}.ini",
                brain=f"{TMP_TRAIN_MODEL_PATH}{TRADE_SYMBOL}_{TRADE_SYSTEM}_Brain_Gen{gnum+1}_Rank{rnk+1}.pth",
                eval_value=df_cur_gen_rank.iloc[rnk]['eval_value']
            )
            best_trainer.SaveParameters(TrainParametersList[rnk], None)

        gc.collect()
        dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM] = gnum + 1
        dfTrainNumList.to_csv(trainnum_file_path)

        TrainEnvHelper.send_ntfy_message(
            ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
            f"{TRADE_SYSTEM}\n{TRADE_SYMBOL}_{BASE_TIMEFRAME}\nGen {gnum+1} Train has finished.\n{str(df_cur_gen_rank.iloc[0, [1, 2, 6]])}"
        )

    logger_root.info(f"\nAll Training Completed. Final Completed Gens: {int(dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM])}")
    print(int(dfTrainNumList.at[TRADE_SYMBOL, TRADE_SYSTEM]))
    sys.exit(0)