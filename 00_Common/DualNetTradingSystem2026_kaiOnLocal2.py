#!/usr/bin/env python
# coding: utf-8

# # DuelNetTradingSystem2026改OnLocal2
# DualNetTradingSystem2026_kai.ipynbよりコピー
# 20260606 06_DuelNetTradingSystem配下のDualNetTradingSystem2026_kai.ipynbはこれよりさらに変更を加えている  
# →SQLAlchemyの代わりにConnectorXを使うように変更  
# →EveluateRewardの不具合を改修  
# →position close時にもreal spread を評価するように改修
# 
# ## What's new (backlog)
# ### OnLocalへの改修
# - Environmentをオブジェクト化する
# - トレードで使わないクラス、スタブの削除
#   -  PriceData, TrainData, Account, Agent, Trainer, TrainDataMaker, PortfolioConstructer
#   -  Brainは残す
# - 不要なSandBoxを削除
# - 空トレードをするときにMT5からTick情報を取得する
# - トレードの結果を検証するため、BackTestDetailを実装
#   - input及びtick dataはMetaTrader5erverから取得する
# 
# ### .pyファイルへの変換
# pthon仮想環境がアクティベートされているコマンドライン上で  
# `jupyter nbconvert --to python app.ipynb`
# を実行
# ### メモ
# - 古い世代の.iniファイル.pthファイルはいらない？
# - 最重要：tick → bar のマッピングを前計算して、pandas.query を完全に排除する
# - サーバ側でAgentファイルやModelファイルにアクセスするのであれば、クライアントからパラメータを送ってもらう必要なないのでは？<br>
#   →リファクタリング対象

# # Package-Environment
# - 価格や損益の情報を提供するパッケージ。以下の機能を実装する
#  - 価格CSVファイルを結合して1つのpandas dataframeにする

# ## Dependency

# In[1]:


#@title Dependency
import pandas as pd
import numpy as np
## For OnLocal
# import datetime
from datetime import datetime,timezone,timedelta
import glob
import configparser
import math
import sys
# on Stab
from collections import namedtuple
from scipy import stats
import os
import shutil
import csv
import requests
import gc
from sqlalchemy import create_engine, text
## Fot OnLocal
import MetaTrader5 as mt5
import pytz


# ## クラス変数
# - SYMBOL_5DIGITS：小数点以下が5桁の通貨ペア(EURUSD等)の配列
# - SYMBOL_4DIGITS：小数点以下が4桁の通貨ペア(USDZAR等)の配列
# - SYMBOL_3DIGITS：小数点以下が3桁の通貨ペア(USDJPY等)の配列
# - DIGIT_MAGNIFICATION：通貨ペアごとの倍率辞書(5桁：100倍、4桁10倍、3桁1倍)

# In[2]:


# @title Class Valiables

EXEC_ENV = "LOCAL" #@param ["COLABO", "LOCAL"]

#
inifile = configparser.ConfigParser()

# .iniファイルのエンコード
ENC = None
if EXEC_ENV == 'COLABO':
  inifile.read('./drive/MyDrive/Colab Notebooks/00_Common/settings.ini',ENC)
else:
  ENC = 'UTF-8'
  inifile.read(r'D:\ColabNotebooks\00_Common\settings.ini',ENC)

## For OnLocal
# MT5関連
TIMEFRAME_DICT = eval(inifile.get('LOCAL', 'timrframe_dict'))
# Protfolioファイルの読み込み
portfolio = configparser.ConfigParser()

# ログ出力セッティングファイル
LOGGING_INIFILE_PATH = inifile.get(EXEC_ENV, 'LOGGING_INIFILE_PATH')

# PRICEDATA_PATH = inifile.get('COLABO', 'PRICEDATA_PATH_COLABO')
PRICEDATA_PATH = None
PRICEDATA_PATH_REAL = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_REAL')
PRICEDATA_PATH_DEMO = inifile.get(EXEC_ENV, 'PRICEDATA_PATH_DEMO')
# TRAIN_MODEL_PATH = inifile.get('COLABO', 'TRAIN_MODEL_PATH_COLABO')
TRAIN_MODEL_PATH = None
TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_REAL')
TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_DEMO')
# TRAIN_MODEL_PATH_VOID = inifile.get('COLABO', 'TRAIN_MODEL_PATH_COLABO')
TRAIN_MODEL_PATH_VOID = None
TRAIN_MODEL_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_REAL')
TRAIN_MODEL_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_MODEL_PATH_VOID_DEMO')
# TMP_TRAIN_MODEL_PATH = inifile.get('COLABO', 'TMP_TRAIN_MODEL_PARTH_COLABO')
TMP_TRAIN_MODEL_PATH = None
TMP_TRAIN_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_REAL')
TMP_TRAIN_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_MODEL_PATH_DEMO')
# TRADE_MODEL_PATH = inifile.get('COLABO', 'TRADE_MODEL_PATH_COLABO')
TRADE_MODEL_PATH = None
TRADE_MODEL_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_MODEL_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_DEMO')
# TRADE_RESULT_PATH
TRADE_RESULT_PATH = None
TRADE_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRADE_MODEL_PATH_REAL')
TRADE_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRADE_RESULT_PATH_DEMO')
# TRAIN_RESULT_PATH
TRAIN_RESULT_PATH = None
TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_REAL')
TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_DEMO')
# TRAIN_RESULT_PATH_VOID
TRAIN_RESULT_PATH_VOID = None
TRAIN_RESULT_PATH_VOID_REAL = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_REAL')
TRAIN_RESULT_PATH_VOID_DEMO = inifile.get(EXEC_ENV, 'TRAIN_RESULT_PATH_VOID_DEMO')
# TMP_TRAIN_RESULT_PATH
TMP_TRAIN_RESULT_PATH = None
TMP_TRAIN_RESULT_PATH_REAL = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_REAL')
TMP_TRAIN_RESULT_PATH_DEMO = inifile.get(EXEC_ENV, 'TMP_TRAIN_RESULT_PATH_DEMO')

BACKTEST_RESULT_ALL_FILE_NAME = inifile.get('COMMOM', 'BACKTEST_RESULT_ALL_FILE_NAME')
BACKTEST_ALL_FILE_NAME = inifile.get('COMMOM', 'BACKTEST_ALL_FILE_NAME')
MODEL_PORTFOLIO_FILE_NAME = inifile.get('COMMOM', 'MODEL_PORTFOLIO_FILE_NAME')

SYMBOL_5DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_5DIGITS'))
SYMBOL_4DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_4DIGITS'))
SYMBOL_3DIGITS = eval(inifile.get('COMMOM', 'SYMBOL_3DIGITS'))

DIGIT_MAGNIFICATION = eval(inifile.get('COMMOM', 'DIGIT_MAGNIFICATION'))
TICKVOL_MAGNIFICATION = int(inifile.get('COMMOM', 'TICKVOL_MAGNIFICATION'))

# TICKVOL_PERIOD,LONG_PERIOD,SHORT_PERIODはPraceDataを初期化する都度
# ランダムに取得することも可能とする。
TICKVOL_PERIOD = int(inifile.get('COMMOM', 'TICKVOL_PERIOD'))
LONG_PERIOD  = int(inifile.get('COMMOM', 'LONG_PERIOD'))
SHORT_PERIOD  = int(inifile.get('COMMOM', 'SHORT_PERIOD'))

PERIOD_FREQ_DICT = eval(inifile.get('COMMOM', 'PERIOD_FREQ_DICT'))
# 2023/5/27 マージンはスプレッドを反映させる
MARGIN_DICT = eval(inifile.get('COMMOM', 'MARGIN_DICT'))

TP_LC_MIN_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MIN_LEVEL'))
TP_LC_MAX_LEVEL = int(inifile.get('COMMOM', 'TP_LC_MAX_LEVEL'))
TP_LC_STEP = int(inifile.get('COMMOM', 'TP_LC_STEP'))

# LINE Notifyを利用するためのトークン
# 2025/3/31 Line Notify 終了に伴い、Line Message APIに切り替える
#LINE_NOTIFY_TOKEN = inifile.get('COMMOM', 'LINE_NOTIFY_TOKEN')
LINE_MESSAGING_API_TOKEN = inifile.get('COMMOM', 'LINE_MESSAGING_API_TOKEN')

# TradeSystemの名称
# TRADE_SYSTEM = 'DuelNet_2026'
TRADE_SYSTEM = inifile.get('COMMOM', 'TRADE_SYSTEM')

GAMMA = 0.99

# トレードモード(MT5より借用)
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2
# ログ出力用トレードモード文字列
ACCOUNT_TRADE_MODE_STR=['DEMO','CONTEST','REAL']
# MT5実行ファイルの場所
MT5_PATH = None
MT5_REAL_PATH = inifile.get('LOCAL', 'mt5_real_path')
MT5_DEMO_PATH = inifile.get('LOCAL', 'mt5_demo_path')
# ask-bidの差の閾値
REAL_SPREAD_LIMIT_LIST = eval(inifile.get('LOCAL', 'REAL_SPREAD_LIMIT_LIST'))
# RealSpreadが閾値を超えていた場合の再試行回数
MAX_RETRY_NUM = eval(inifile.get('LOCAL', 'MAX_RETRY_NUM'))

# action index に対応した行動
# output_data(行動A[seq,6])
# a0:何もしない
# a1:LongEntry
# a2:ShortEntry
# a3:PositionClose
# a4:PositionClose&LongEntry
# a5:PositionClose&ShortEntry
no_action = 0
long_entry = 1
short_entry = 2
position_close = 3
close_and_long = 4
close_and_short = 5

# トレーニングの詳細を記録するファイル名
TRAIN_DETAIL_INPUT_FILE_NAME = 'TrainDetailInput'
TRAIN_DETAIL_HIDDEN_FILE_NAME = 'TrainDetailHidden'
TRAIN_DETAIL_CELLSTATE_FILE_NAME = 'TrainDetailCellState'
TRAIN_DETAIL_OUTPUT_FILE_NAME = 'TrainDetailOutput'
TRAIN_DETAIL_MASKOUTPUT_FILE_NAME = 'TrainDetailMaskOutput'

# .set ファイルに書きこむNNモデルの素子数
MODEL_INPUT_NUM = 0
MODEL_HIDDEN_NUM = 0
MODEL_OUTPUT_NUM = 0

# 接続先DB情報
RDBMS='postgresql'
HOST='localhost'
DBNAME=None
DBNAME_DEMO='metaquotesdemo'
DBNAME_REAL='xmtradingreal'
PORT='5432'
USER='appop'
PASSWORD='appop'
SCHEMA='public'
ENGINE=None


# ## Logger
# ロギングを設定する。ログレベルは以下
# 1. CRITICAL
# 1. ERROR
# 1. WARNING
# 1. INFO
# 1. DEBUG
# 
# Environment package では、ログ空間を"DRL.Environment"とする

# In[3]:


#@title Logger
import logging
import logging.config

# logging.config.fileConfig('./drive/MyDrive/Colab Notebooks/05_DuelNetTradingSystem2024/Logging.ini')
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger = logging.getLogger('DRLLogging')
logger.debug('Debug level massage.')
logger.info('Info level massage.')
logger.warning('Warning level massage.')
logger.error('Error level massage.')
logger.critical('Critical level massage.')


# ## クラスEnvironment
# Environmentパッケージで使用するクラス関数を定義する
# ### クラス関数 GetDigitMagnification()
# - 通貨ペアの最少桁数によって値を何倍にするかを返す。1,000pips=1.000となるように調整する
#   - 100倍：小数5桁(EURUSD等)
#   - 10倍：小数4桁(USDZAR等)
#   - 1倍：小数3桁(USDJPY等)
# 
# ### クラス関数getAvailableAction(self,(plofit_loss, long_position, short_position),CountDown)
# 現在のポジションの状況から、可能な行動をマスク配列で返す。<br>
# 
# #### output_data(行動A[seq,6])
# a0:何もしない<br>
# a1:LongEntry<br>
# a2:ShortEntry<br>
# a3:PositionClose<br>
# a4:PositionClose&LongEntry<br>
# a5:PositionClose&ShortEntry<br>
# ※両建てはしない。ドテンを想定する。
# 
# <b>取引可能時間帯：COUNTDOWN < 1 </b>
# 
# <li>パターン1：ポジションなし(position = 0)<br>
#   新規ポジションエントリー(long,shot)と何もしないが選択可能。<br>
#  action_mask[0] = [a0=True, a1=True, a2=True, a3=False, a4=False, a5=False]<br>
# <li>パターン2：ロングポジション(position = 1)<br>
#   ポジションクローズと、クローズ＆追加エントリ、何もしないが選択可能。<br>
#  action_mask[1] = [a0=True, a1=False, a2=False, a3=True, a4=True, a5=True]<br>
# <li>パターン3：ショートポジション(position = -1)<br>
#   ポジションクローズと、クローズ＆追加エントリ、何もしないが選択可能。<br>
#  action_mask[2] =[a0=True, a1=False, a2=False, a3=True, a4=True, a5=True]<br>
# <li>パターン4：両建て(定義なし)<br>
#   ポジションクローズ(全決済)が選択可能。両建ては禁止とするため、このパターンにはならない。<br>
#  action_mask[3] =[a0=False, a1=False, a2=False, a3=True, a4=False, a5=False]<br>
# 
#   <b>取引可能時間外：COUNTDOWN \>= 1 </b>
# <li>パターン5：ポジションなし(position = 0)<br>
#   何もしないが選択可能。<br>
#  action_mask[4] = [a0=True, a1=False, a2=False, a3=False, a4=False, a5=False]<br>
# <li>パターン6：ロングポジション(position = 1)<br>
#   ポジションクローズが選択可能。<br>
#  action_mask[5] = [a0=False, a1=False, a2=False, a3=True, a4=False, a5=False]<br>
# <li>パターン7：ショートポジション(position = -1)<br>
#   ポジションクローズが選択可能。<br>
#  action_mask[6] = [a0=False, a1=False, a2=False, a3=True, a4=False, a5=False]<br>
# <li>パターン8：両建て(定義なし)<br>
#   ポジションクローズ(全決済)が選択可能。両建ては禁止とするため、このパターンにはならない。<br>
#  action_mask[7] =[a0=False, a1=False, a2=False, a3=True, a4=False, a5=False]<br>
# 
# 

# In[4]:


#@title class Environment
## For OnLocal
# クラスメソッドはなく、インスタンスを生成するように修正
class Environment:

  def __init__(self,trdmd):
    global MT5_PATH
    self.account_trademode = trdmd
    # 0:ACCOUNT_TRADE_MODE_DEMO,1:ACCOUNT_TRADE_MODE_CONTEST,2:ACCOUNT_TRADE_MODE_REAL
    # MT5のトレードモードにより起動するクライアントを分ける
    if self.account_trademode == 0:
      portfolio.read(TRADE_MODEL_PATH_DEMO+'ModelPortfolio.ini','UTF-8')
      MT5_PATH = MT5_DEMO_PATH
    elif self.account_trademode == 2:
      portfolio.read(TRADE_MODEL_PATH_REAL+'ModelPortfolio.ini','UTF-8')
      MT5_PATH = MT5_REAL_PATH
    else:
      sys.exit("Environment:Account Trade Mode is not defined.")


  def GetTradeData(self,dt,timeframe,sbl,lprd=LONG_PERIOD,sprd=SHORT_PERIOD,tvprd=TICKVOL_PERIOD,tsl=None):
    self.short_period = sprd
    self.long_period = lprd
    self.tickvol_period = tvprd

    # TimeZoneをUTCに変更する
    timezone = pytz.timezone("Etc/UTC")
    # TimeZoneをMataTraderサーバの稼働しているキプロス（ニコシア）に設定する
    # timezone = pytz.timezone("Asia/Nicosia")
    dt.replace(tzinfo=timezone)

    # TrainSymbolListを引数で取得する
    if tsl is None:
      self.train_symbol_list = TRAIN_SYMBOL_LIST
    else:
      self.train_symbol_list = tsl

    # MT5に接続して、LONG_PERIOD(lprd)、SHORT_PERIOD(sprd)、TICKVOL_PERIOD(tvprd)のうちで最長の期間分の
    # 時間足データを取得する
    self.max_period = max([self.long_period,self.short_period,self.tickvol_period])

    # 通貨ペア毎のinput_dataを格納
    prices_list = []

    if not mt5.initialize(MT5_PATH):
      # MetaTrader5のエラーコードとメッセージを受け取る
      (ecd,emsg) = mt5.last_error()
      sys.exit("Environment:initialize() failed, error code="+str(ecd)+" msg="+emsg)

    for i,symbol in enumerate(self.train_symbol_list):
      # 2024/4/14 MT5から取得したデータにNaNが含まれているか確認
      # NaNが含まれていたら1分おきに5回再試行し、それでもNaNの場合はエラーとする
      for l in range(5):
        rates = mt5.copy_rates_from(symbol, TIMEFRAME_DICT[timeframe], dt, self.max_period)
        # 取得したデータはnumpy.ndarrayなので、pandasのdataframeに変換する
        df_rates = pd.DataFrame(rates)
        logger.debug('Environment.GetTradeDate:df_rates.isna(): %s' %(df_rates.isna().any().any()))
        # 2024/4/16 np.isnanを実行するとエラーとなる(文字列が混ざっている?)
        if df_rates.isna().any().any():
          logger.debug('Environment.GetTradeData:df_rates.isna():NG %d time.' %(l+1))
          logger.debug('Environment.GetTradeData:MT5 rates contains NaN. %s'%(rates))
          if l >= 4:
            logger.error('Environment.GetTradeData:Copy rates from MT5 failed.')
            sys.exit('Environment.GetTradeData:Copy rates from MT5 failed.')
          else:
            # 1分待つ
            time.sleep(60)
        else:
          #NaNがなかった場合は、forループを抜ける
          logger.debug('Environment.GetTradeData:df_rates.isna():OK.')
          break

      digits = mt5.symbol_info(symbol).digits

      # 秒での時間をdatetime形式に変換する
      df_rates['time']=pd.to_datetime(df_rates['time'], unit='s')
      df_rates = df_rates.set_index('time')
      # 取得したdatetimeはnaiveなので、タイムゾーンを設定する
      df_rates.index.tz_localize('Etc/UTC')

      # インプットデータに加工していく
      # 取引ペアに応じて価格データを1,10,100倍する
      df_rates[['open','high','low','close']]\
        = df_rates[['open','high','low','close']]*10**(digits-3)
      # 統計情報の追加するものは上記の説明参照
      df_rates['tick_volume'] = np.tanh(np.log10(df_rates['tick_volume']/df_rates['tick_volume'].rolling(self.tickvol_period).mean()))
      df_rates['close-open'] = df_rates['close'] - df_rates['open']
      df_rates['high-low'] = df_rates['high'] - df_rates['low']
      df_rates['sma_close_short'] = df_rates['close'].rolling(self.short_period).mean()
      df_rates['sma_close_long'] = df_rates['close'].rolling(self.long_period).mean()
      df_rates['sma_open_short'] = df_rates['open'].rolling(self.short_period).mean()
      df_rates['sma_open_long'] = df_rates['open'].rolling(self.long_period).mean()
      df_rates['std_short'] = df_rates['close'].rolling(self.short_period).std()
      df_rates['std_long'] = df_rates['close'].rolling(self.long_period).std()
      df_rates['z_score_short'] = (df_rates['close'] - df_rates['sma_close_short']) / df_rates['std_short']
      df_rates['z_score_long'] = (df_rates['close'] - df_rates['sma_close_long']) / df_rates['std_long']
      df_rates['sma_close_short-long'] = df_rates['sma_close_short'] - df_rates['sma_close_long']
      df_rates['std_short-long'] = df_rates['std_short'] - df_rates['std_long']
      df_rates['z_score_short-long'] = df_rates['z_score_short'] - df_rates['z_score_long']
      df_rates['sma_close-open_short'] = df_rates['sma_close_short'] - df_rates['sma_open_short']
      df_rates['sma_close-open_long'] = df_rates['sma_close_long'] - df_rates['sma_open_long']

      # NNに投入しないカラム(open,high,low,close)は削除する
      df_rates.drop(columns=['open','high','low','close','spread','real_volume','sma_close_short','sma_close_long','sma_open_short','sma_open_long']
                            , inplace=True)
      # カラム名に一括で通貨ペアを付加する
      df_rates.rename(columns=lambda s: symbol + '_' + s, inplace = True)
      logger.debug('Environment.GetTradeData:\n%s' %(df_rates))
      prices_list.append(df_rates)

    # mt5.shutdown()
    df_prices = pd.concat([p for p in prices_list], axis=1, join='outer')

    # return df_prices.iloc[-1].to_list(), sbl, timeframe
    return df_prices.iloc[-1]

  # このメソッドは使わない
  def GetDigitMagnification(cls,symbol):
    # EURUSDのような下5桁通貨は100倍する
    if(symbol in SYMBOL_5DIGITS):
      return DIGIT_MAGNIFICATION['SYMBOL_5DIGITS']
    # USDJPYのような下3桁通貨は1倍する
    elif(symbol in SYMBOL_3DIGITS):
      return DIGIT_MAGNIFICATION['SYMBOL_3DIGITS']
    # USDZARのような下4桁通貨は10倍する
    elif(symbol in SYMBOL_4DIGITS):
      return DIGIT_MAGNIFICATION['SYMBOL_4DIGITS']
    else:
      EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                        +'EnvironmentCommon.GetDigitMagnification:Symbol is not defined.')
      sys.exit('Symbol is not defined.')

  # このメソッドは使わない
  def SetLimitStopLevel(cls,period):
    # Losscut,Takeprofitのレベルを時間足に対応して決める
    # 基準はH1の時に1.0、D1の時に2.0となるようにする
    # y = (1/23)x + 22/23
    if(period == 'M1'):
      period_value = (1/60)
    elif(period == 'M5'):
      period_value = (5/60)
    elif(period == 'M15'):
      period_value = (15/60)
    elif(period == 'M30'):
      period_value = (30/60)
    elif(period == 'H1'):
      period_value = 1
    elif(period == 'H4'):
      period_value = 4
    elif(period == 'H6'):
      period_value = 6
    elif(period == 'H8'):
      period_value = 8
    elif(period == 'H12'):
      period_value = 12
    elif(period == 'D1'):
      period_value = 24
    else:
      logger.warning('■■Period is not Implemented')
      # period_value = 1
      EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                        +'EnvironmentCommon.SetLimitStopLevel:Period is not Implemented.')
      sys.exit('EnvironmentCommon.SetLimitStopLevel:Period is not Implemented.')

    losscut_level =  (1/23)*period_value + (22/23)
    logger.debug('## Losscut Level: %.2f' %(losscut_level))
    return losscut_level

  # このメソッドは使う
  def GetPeriodTimeDelta(self,period):
    if(period == 'M1'):
      delta_period = timedelta(minutes=1)
    elif(period == 'M5'):
      delta_period = timedelta(minutes=5)
    elif(period == 'M15'):
      delta_period = timedelta(minutes=15)
    elif(period == 'M30'):
      delta_period = timedelta(minutes=30)
    elif(period == 'H1'):
      delta_period = timedelta(hours=1)
    elif(period == 'H4'):
      delta_period = timedelta(hours=4)
    elif(period == 'H6'):
      delta_period = timedelta(hours=6)
    elif(period == 'H8'):
      delta_period = timedelta(hours=8)
    elif(period == 'H12'):
      delta_period = timedelta(hours=12)
    elif(period == 'D1'):
      delta_period = timedelta(days=1)
    else:
      logger.warning('■■Period Delta is not Implemented')
      # delta_period = datetime.timedelta(hours=1)
      EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                        +'EnvironmentCommon.GetPeriodTimeDelta:Period Delta is not Implemented.')
      sys.exit('EnvironmentCommon.GetPeriodTimeDelta:Period Delta is not Implemented.')

    return delta_period

  # longとshortで分ける
  # このメソッドは使う
  def getAvailableAction(self,pl_, lg_, st_, cd_):
    action_mask = np.empty(6, dtype='bool')
    # logger.debug('EnvironmentCommon.getAvailableAction:plofit_loss=%.3f position=%d countwdown=%.3f' %(pl_,pos_,cd_))
    # plofit_lossは使っていない?
    logger.debug('EnvironmentCommon.getAvailableAction:plofit_loss=%.3f long_pos=%.1f shrot_pos=%.1f countwdown=%.3f' %(pl_,lg_,st_,cd_))

    #現在の状況によって、選択できる行動を制限する。詳細は上の説明参照。
    # 今回は両建てを想定しないため、パターン4,8は定義しない
    # 取引時間内 (CountDown < 1)
    if(cd_ < 1):
      # パターン2：ロングポジション(position = 1)
      # if(pos_ == 1):
      if(lg_ > 0.0 and st_ == 0.0):
        action_mask = [True, False, False, True, True, True]
        logger.debug('EnvironmentCommon.getAvailableAction:pattern 2 %s' %(action_mask))
      # パターン3：ショートポジション(position = -1)
      # elif(pos_ == -1):
      elif(st_ > 0.0 and lg_ == 0.0):
        action_mask = [True, False, False, True, True, True]
        logger.debug('EnvironmentCommon.getAvailableAction:pattern 3 %s' %(action_mask))
      # パターン1：ポジションなし(position = 0)
      # elif(pos_ == 0):
      elif(lg_ == 0 and st_ == 0.0):
        action_mask = [True, True, True, False, False, False]
        logger.debug('EnvironmentCommon.getAvailableAction:pattern 1 %s' %(action_mask))
      # エラー：ポジションクローズ
      else:
        action_mask = [False, False, False, True, False, False]
        logger.waring('EnvironmentCommon.getAvailableAction:pattern 4 error %s' %(action_mask))
        # for debug エラーとする
        EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                  +'EnvironmentCommon.getAvailableAction:pattern 4 error %s' %(action_mask))
        sys.exit('EnvironmentCommon.getAvailableAction:pattern 4 error ', str(action_mask))
    # 取引時間外 (CountDown >= 1)
    else:
      # パターン6：ロングポジション(position = 1)
      # if(pos_ == 1):
      if(lg_ > 0.0 and st_ == 0.0):
        action_mask = [False, False, False, True, False, False]
        logger.debug('EnvironmentCommon.getAvailableAction:patten 6 %s' %(action_mask))
      # パターン7：ショートポジション(position = -1)
      # elif(pos_ == -1):
      elif(st_ > 0.0 and lg_ == 0.0):
        action_mask = [False, False, False, True, False, False]
        logger.debug('EnvironmentCommon.getAvailableAction:patten 7 %s' %(action_mask))
      # パターン5：ポジションなし(position = 0)
      # elif(pos_ == 0):
      elif(lg_ == 0 and st_ == 0.0):
        action_mask = [True, False, False, False, False, False]
        logger.debug('EnvironmentCommon.getAvailableAction:patten 5 %s' %(action_mask))
      # エラー：ポジションクローズ
      else:
        action_mask = [False, False, False, True, False, False]
        logger.waring('EnvironmentCommon.getAvailableAction:pattern 8 error %s' %(action_mask))
        # for debug エラーとする
        EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                  +'EnvironmentCommon.getAvailableAction:pattern 8 error %s' %(action_mask))
        sys.exit('EnvironmentCommon.getAvailableAction:pattern 8 error ' ,str(action_mask))

    return action_mask

  # このメソッドは使わない
  def periodRandomSelect(cls):
    # 各Periodの値をランダムに選択する
    prime_number = [5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97\
                    ,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191]
    period_numbers = random.sample(prime_number,2)
    long_period = max(period_numbers)
    short_period = min(period_numbers)
    tickvol_period = random.choice(prime_number)

    return (long_period,short_period,tickvol_period)

  # このメソッドは使わない？
  def setTradeModePath(self, trdmd):
    # トレードモードによってパスを変える
    # global PRICEDATA_PATH,TRAIN_MODEL_PATH,TMP_TRAIN_MODEL_PATH,TRADE_MODEL_PATH,TRADE_RESULT_PATH,TRAIN_RESULT_PATH,TMP_TRAIN_RESULT_PATH,MT5_PATH
    global PRICEDATA_PATH,TRAIN_MODEL_PATH,TRAIN_MODEL_PATH_VOID,TMP_TRAIN_MODEL_PATH,TRADE_MODEL_PATH\
      ,TRADE_RESULT_PATH,TRAIN_RESULT_PATH,TRAIN_RESULT_PATH_VOID,TMP_TRAIN_RESULT_PATH,DBNAME
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
      DBNAME = DBNAME_REAL

    else:
      PRICEDATA_PATH = PRICEDATA_PATH_DEMO
      TRAIN_MODEL_PATH = TRAIN_MODEL_PATH_DEMO
      TRAIN_MODEL_PATH_VOID = TRAIN_MODEL_PATH_VOID_DEMO
      TMP_TRAIN_MODEL_PATH = TMP_TRAIN_MODEL_PATH_DEMO
      TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMO
      TRADE_RESULT_PATH = TRADE_RESULT_PATH_DEMO
      TRAIN_RESULT_PATH = TRAIN_RESULT_PATH_DEMO
      TRAIN_RESULT_PATH_VOID = TRAIN_RESULT_PATH_VOID_DEMO
      TMP_TRAIN_RESULT_PATH = TMP_TRAIN_RESULT_PATH_DEMO
      DBNAME = DBNAME_DEMO

  # このメソッドは使わない
  def send_line_notify(cls,notification_message):
    headers = {"Content_Type": "application/json","Authorization": "Bearer " + LINE_MESSAGING_API_TOKEN}
    requests.post("https://api.line.me/v2/bot/message/broadcast",headers=headers,json={"messages": [{"type": "text","text": notification_message}]}).json()

  # このメソッドは使う？
  def getRealSpreadLimit(self,sbl,flg=False):
      if flg:
          return REAL_SPREAD_LIMIT_LIST[sbl]
      else:
          return 1.0

  # このメソッドは使わない
  def connectDB(csl):
      global ENGINE
      ENGINE = create_engine(RDBMS+"://"+USER+":"+PASSWORD+"@"+HOST+"/"+DBNAME)

  # このメソッドは使わない
  def compareEvalValue(cls,sbl,tf,eval_value):
      # 新しく取得したeval_valueと比較して、新しいほうの値が良ければ、既存のファイルをvoidファイルに移動する
      # 戻り値：true→新規作成　false→変更なし
      # 同じ通貨ペアと時間足のファイルAgentファイルがあるかを確認する
      # フォルダにある同じ通貨ペア、時間足のファイルを取得する
      filelist = glob.glob(TRAIN_MODEL_PATH+'Agent_'+sbl+'_'+tf+'_*.ini')

      # 既存ファイルのほうが結果が良い場合は1位のファイル名が入る
      win_file=None

      for l in filelist:
          logger.debug('EnvironmentCommon.compareEvalValue:AgentFileName:%s' %(l))
          # AgentFileの内容を読みだしてeval_valueの値を取り出す
          agentfile = configparser.ConfigParser()
          agentfile.read(l,ENC)
          # eval_valueを取得する
          ev=float(agentfile.get('COMMOM', 'EVAL_VALUE'))
          logger.info('EnvironmentCommon.compareEvalValue:Evalvale:%s[%s]' %(str(ev),l))

          #新しいeval_valueと既存のevを比較する
          if ev >= eval_value:
              # 既存のほうが大きい場合は残すファイル名とeval_valueを更新する
              eval_value=ev
              win_file=l

      # TRAIN_MODEL_PATH内のファイルリスト
      tm_filelist = glob.glob(TRAIN_MODEL_PATH+'*_'+sbl+'_'+tf+'_*')
      # TRAIN_RESULT_PATH内のファイルリスト
      #   BacktestResult_CHFJPY_D1_B_20240721000428.csv,Result_CHFJPY_D1_B_20240721000428.csv,TrainResult_CHFJPY_D1_B_20240721000428.csv
      tr_filelist = glob.glob(TRAIN_RESULT_PATH+'*_'+sbl+'_'+tf+'_*')

      # Train_Model_Voidディレクトリが存在していないときはディレクトリを作成する
      if not os.path.isdir(TRAIN_MODEL_PATH_VOID):
          logger.info('EnvironmentCommon.compareEvalValue:Make Train_Model_Void Directry.')
          os.makedirs(TRAIN_MODEL_PATH_VOID)
      # Train_Result_Voidディレクトリが存在していないときはディレクトリを作成する
      if not os.path.isdir(TRAIN_RESULT_PATH_VOID):
          logger.info('EnvironmentCommon.compareEvalValue:Make Train_Result_Void Directry.')
          os.makedirs(TRAIN_RESULT_PATH_VOID)

      # win_fileがNoneのであれば、既存ファイルをすべてvoidへ移して、新規ファイルを作成する
      # win_fileに名前があれば、そのファイル以外はvoidへ移す
      if win_file is None:
          for l in tm_filelist:
              # voidへ移す。
              new_path = shutil.move(l, TRAIN_MODEL_PATH_VOID)
              logger.info('EnvironmentCommon.compareEvalValue:%s is moved to %s'%(l,new_path))

          for l in tr_filelist:
              # voidへ移す。
              new_path = shutil.move(l, TRAIN_RESULT_PATH_VOID)
              logger.info('EnvironmentCommon.compareEvalValue:%s is moved to %s'%(l,new_path))
          logger.info('EnvironmentCommon.compareEvalValue:AccountFile Need Update')
          return True
      else:
          #win_file以外をvoidへ移す
          logger.debug('EnvironmentCommon.compareEvalValue:WinFile:%s' %(win_file))
          # D:\ColabNotebooks\06_DuelNetTradingSystem2025\Demo\Model\Train\Agent_USDJPY_D1_B_20260125225640.ini
          MAGIC = win_file.split('\\')[-1].split('_')[4].split('.')[0] 
          logger.debug('EnvironmentCommon.compareEvalValue:MAGIC:%s' %(MAGIC))
          for l in tm_filelist:
              logger.debug('EnvironmentCommon.compareEvalValue:File:%s' %(l))
              # MAGICが同じファイルは移動しない
              if MAGIC not in l:
                  new_path = shutil.move(l, TRAIN_MODEL_PATH_VOID)
                  logger.info('EnvironmentCommon.compareEvalValue:%s is moved to %s' %(l,new_path))
          for l in tr_filelist:
              logger.debug('EnvironmentCommon.compareEvalValue:File:%s' %(l))
              # MAGICが同じファイルは移動しない
              if MAGIC not in l:
                  new_path = shutil.move(l, TRAIN_RESULT_PATH_VOID)
                  logger.info('EnvironmentCommon.compareEvalValue:%s is moved to %s' %(l,new_path))
          logger.info('EnvironmentCommon.compareEvalValue:AccountFile NOT Need Update')
          return False

## For OnLocal
# Accountクラスからの移植。ただし中身は違う
  def CalcCountdown(self, dt,tf,sbl):
      # トレード期間が1週の場合のみ
      # トレード開始はその週の月曜日の0:00(datetime.isoweekday()は、月曜日=1,日曜日=7)
      # 日曜日は次の週とするため剰余を使う
      # dt(=MT5クライアントから送られてくる一つ前の足の開始時間Time[1])は、
      # 先週金曜日の場合もあるため、現在時刻からstart_periodとend_periodを取得する
      now_datetime = datetime.now(timezone.utc)
      start_period = (now_datetime + timedelta(days=1 - (now_datetime.isoweekday() % 7))).replace(hour=0,minute=0,second=0,microsecond=0)

      # 引数のdatetimeからstart_periodとend_periodを取得する(これは間違い)
      # start_period = (dt + timedelta(days=1 - (dt.isoweekday() % 7))).replace(hour=0,minute=0)
      # トレードの終了は、その週の土曜日の0:00の2つ前(*)の時間足
      # (*)金曜日の23:00(=self.end_period)の時点で、22:00(=self.end_trade)の時間足を参照するので、df_acount(22:00)のcountdownが1.0となるように計算する
      # end_period = (dt + timedelta(days=6 - (dt.isoweekday() % 7))).replace(hour=0,minute=0)
      end_period = (now_datetime + timedelta(days=6 - (now_datetime.isoweekday() % 7))).replace(hour=0,minute=0,second=0,microsecond=0)
      end_period -= self.GetPeriodTimeDelta(tf)
      end_trade = end_period - self.GetPeriodTimeDelta(tf)

      countdown = (dt - start_period).total_seconds() / (end_trade - start_period).total_seconds()
      logger.debug('Environment.CalcCountdown:Countown=%.4f' %(countdown))
      return countdown,sbl,tf

## For OnLocal
# 新たに実装
  def CalcOrderLots(self,mname,tf,sbl,trdmd):
    # MT5クライアントに接続して、口座情報(残高)を取得する
    if trdmd == 0:
      MT5_PATH = inifile.get('LOCAL', 'mt5_demo_path')
      growth_rate = float(inifile.get('LOCAL', 'GROWTH_RATE_DEMO'))
    elif trdmd == 2:
      MT5_PATH = inifile.get('LOCAL', 'mt5_real_path')
      growth_rate = float(inifile.get('LOCAL', 'GROWTH_RATE_REAL'))
    else:
      sys.exit("Environment:Account Trade Mode is not defined.")
    logger.debug("Environment:MT5_PATH=%s" %(MT5_PATH))

    # 取引ロットを計算する
    if not mt5.initialize(MT5_PATH):
      sys.exit("Environment:initialize() failed, error code =", mt5.last_error())

    account_info = mt5.account_info()
    # mt5.shutdown()

    logger.debug("Environment:account_info=%s" %(str(account_info)))
    equity = account_info.equity
    logger.info("Environment:account_info.equity=%s" %(str(equity)))

    # ロット計算に必要な情報を取得する
    investment_rate = float(portfolio.get('LOCAL',mname))
    expected_value = float(portfolio.get('LOCAL','mu'))
    # logger_trader.debug("Environment:%s growth_rate=%s, investment_rate=%s" %(mname,str(growth_rate),str(investment_rate)))
    logger.debug("Environment:%s growth_rate=%s, investment_rate=%s, expected_value=%s" %(mname,str(growth_rate),str(investment_rate),str(expected_value)))
    # order_lots = (equity/100000)*growth_rate*investment_rate
    order_lots = (equity/100000)*(growth_rate/expected_value)*investment_rate
    # OrderLotsを少数第3位を四捨五入する
    order_lots = round(order_lots,2)
    logger.debug("Environment:order_lots=%s" %(str(order_lots)))

    return order_lots,sbl,tf

## For OnLocal
# System2026から実装。エントリーをする際のティック情報を取得する
  def GetTickData(self,sbl,dt_from,dt_to):
    # MetaTrader 5ターミナルとの接続を確立する
    if not mt5.initialize(MT5_PATH):
      logger.debug("Environment.GetTickData.initialize() failed, error code =",mt5.last_error())
      exit()
    # TimeZoneをUTCに変更する
    timezone = pytz.timezone("Etc/UTC")
    # TimeZoneをMataTraderサーバの稼働しているキプロス（ニコシア）に設定する
    # timezone = pytz.timezone("Asia/Nicosia")
    dt_from.replace(tzinfo=timezone)
    dt_to.replace(tzinfo=timezone)
    # dt_fromからdt_toまでのティックをリクエストする
    ticks_period = mt5.copy_ticks_range(sbl, dt_from, dt_to, mt5.COPY_TICKS_ALL)
    logger.debug("Ticks received:%d" %(len(ticks_period)))
    # 秒での時間をdatetime形式に変換する
    df_ticks_period = pd.DataFrame(ticks_period)
    df_ticks_period['time']=pd.to_datetime(df_ticks_period['time'], unit='s')
    df_ticks_period['time_msc']=pd.to_datetime(df_ticks_period['time_msc'], unit='ms')
    df_ticks_period = df_ticks_period.set_index('time_msc')
    df_ticks_period.sort_index(inplace=True)
    # 取得したdatetimeはnaiveなので、タイムゾーンを設定する
    df_ticks_period.index.tz_localize('Etc/UTC')

    return df_ticks_period

  def GenerateSysExit(self):
    # テスト用にsys.exit発生させる。
    logger.debug("Environment.GenerateSysExit:Start")
    sys.exit('Environment.GenerateSysExit:Test sys.exit()')

  # System2026用：新たに実装
  def GetPriceData(self,sb,tf,trddt):
    # MataTrader5サーバへアクセスして、当該時間足の価格情報を取得する
    # PriceDataを格納する
    prices_data = None
    # TimeZoneをUTCに変更する
    timezone = pytz.timezone("Etc/UTC")
    # TimeZoneをMataTraderサーバの稼働しているキプロス（ニコシア）に設定する
    # timezone = pytz.timezone("Asia/Nicosia")
    trddt.replace(tzinfo=timezone)

    if not mt5.initialize(MT5_PATH):
      # MetaTrader5のエラーコードとメッセージを受け取る
      (ecd,emsg) = mt5.last_error()
      sys.exit("Environment:initialize() failed, error code="+str(ecd)+" msg="+emsg)

    count=1
    rates = mt5.copy_rates_from(sb,TIMEFRAME_DICT[tf],trddt,count)
    df_rates = pd.DataFrame(rates)
    df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
    df_rates = df_rates.set_index('time')
    # 取得したdatetimeはnaiveなので、タイムゾーンを設定する
    df_rates.index.tz_localize('Etc/UTC')    
    return df_rates


# ## クラス Account

# In[5]:


#@title class Account
class Account:

  # def __init__(self,sbl,tf,inidt,trndrt,load_flg=False,file_name=None,periods=None,real_spread_flg=False):
  # self.acnt = Account(self.symbol,self.period,self.start_date_time, self.test_duration,self.real_spread_limit,
  #                     self.lc_level,self.tp_level,self.pos_scale)
  def __init__(self,sbl,tf,rsl,lcl,tpl,ps,ptw):

    # accountを格納する。pandas dataframe
    self.df_account = pd.DataFrame(columns=['symbol','period','pos_open_datetime','pos_open_price','has_long','has_short','float_pl','close_pl','countdown'])
    self.symbol = sbl
    self.period = tf
    # self.ini_datetime = inidt
    # self.train_duration = trndrt

    self.tp_level = tpl
    self.lc_level = lcl
    self.pos_scale = ps
    self.real_sprad_limit = rsl
    self.pre_trade_weeks = ptw

    logger.debug("Account.__init__:RealSpreadLimit:%.3f" %(self.real_sprad_limit))
    logger.debug("Account.__init__:pre_trade_weeks:%d" %(self.pre_trade_weeks))
    logger.debug('Account.__init__:tp_level=%.3f, lc_level=%.3f, pos_scale=%.1f' %(self.tp_level, self.lc_level ,self.pos_scale))

    self.margin = MARGIN_DICT[self.symbol]
		# self.magnificationはEnvironmentが初期化されるまで取得できない
    # self.magnification = EnvironmentCommon.GetDigitMagnification(self.symbol)
    self.magnification = None

    self.pos_open_price = 0.0
    # LongとShortを別々に入力する
    self.has_long = 0.0
    self.has_short = 0.0
    self.float_pl = 0.0
    self.close_pl = 0.0
    # self.train_duration = trndrt
    # 時間足の長さに応じたTimeDeltaを取得する
    # self.delta_period = EnvironmentCommon.GetPeriodTimeDelta(self.period)
    # self.delta_periodは後でセットする
    self.delta_period = None
    self.countdown = 0.0

    # self.SetTradePeriod(self.ini_datetime, self.train_duration)
  ''' 使わないかも
    self.ticks_frame = None

    # 2026/3/15 Copilotによるパフォーマンスチューニング
    # bar_to_tick = {
    #     bar_datetime: (tick_start_idx, tick_end_idx)
    # }
    self.bar_to_tick = {}
    self.bar_to_tick_minute = {}
  '''
  # start_period,end_period,end_tradeを格納する
  # count_downの計算で必要
  def SetTradePeriod(self, stpd, edpd, edtd):
      self.start_period = stpd
      self.end_period = edpd
      self.end_trade = edtd

# 特定のindexのdatetimeが存在していない場合は、空の行を追加する
  def SetAccount(self, set_datetime, symbol=None, period=None, pos_open_datetime=None, pos_open_price=0.0
                 , has_long=0.0, has_short=0.0, float_pl=0.0, close_pl=0.0, countdown=0.0):

    # 格納するdatetimeの行がdf_rewardになければ行を作成する
    if set_datetime in self.df_account.index:
      # 何もしない
      pass
    else:
      # 行を新たに作成する
      if symbol == None:
        symbol = self.symbol
      if period == None:
        period = self.period

      self.df_account.loc[set_datetime] = [symbol, period, pos_open_datetime, pos_open_price,
                                         has_long, has_short, float_pl, close_pl, countdown]

  def CheckTpLc(self, lstdt, lstprc):
    self.last_close_price = 0.0
    self.magnification = self.env.GetDigitMagnification(self.symbol)

    self.utc_from = lstdt
    self.utc_to = lstdt + self.delta_period
    # 直近足のすべてのtickを取得する
    # 2025/02/11 MT5ではなく、DBから取得するようにする
    # 2025/2/22 DBからではなく、あらかじめDBから取得したDataFrameから直近の時間足だけ取得する
    # 2026/3/15 Copilotによるパフォーマンスチューニング　pandsのqueryは致命的に遅い(らしい)ので、使わない
    df_ticks_period = self.env.GetTickData(self.symbol,self.utc_from,self.utc_to)

    if len(df_ticks_period) == 0:
      # tickが取得できなかった場合は、その期間は取引がなかったため、TpLcの評価はしない
      logger.warning('Account.CheckTpLc:Due to No ticks TpLc Skipped. From %s To %s' 
                     %(self.utc_from.strftime('%Y-%m-%d %H:%M'),self.utc_to.strftime('%Y-%m-%d %H:%M')))
      return

    logger.debug('Account.CheckTpLc:From %s To %s Count of _ticks_period %d' 
                 %(self.utc_from.strftime('%Y-%m-%d %H:%M'),self.utc_to.strftime('%Y-%m-%d %H:%M'),len(df_ticks_period)))

    # 直近足のopen(ask/bid)、close(ask/bid)を取得する
    self.last_open_ask  = df_ticks_period.iloc[0]['ask']
    self.last_open_bid  = df_ticks_period.iloc[0]['bid']
    self.last_close_ask = df_ticks_period.iloc[-1]['ask']
    self.last_close_bid = df_ticks_period.iloc[-1]['bid']

    # 2024/12/9 pos_open_priceに対するTP/LC priceを格納する
    self.takeprofit_price = 0.0
    self.losscut_price = 0.0
    # 直近のAccount情報を取得する
    self.pos_open_price,self.has_long,self.has_short,self.close_pl\
      = self.df_account.loc[self.utc_from,['pos_open_price','has_long','has_short','close_pl']]

    logger.debug('Account.CheckTpLc:tp_level=%.3f, lc_level=%.3f' %(self.tp_level, self.lc_level))
    logger.debug('Account.CheckTpLc:%s last_open_ask=%.3f, last_open_bid=%.3f, last_close_ask=%.3f, last_close_bid=%.3f' \
                 %(self.utc_from, self.last_open_ask, self.last_open_bid, self.last_close_ask, self.last_close_bid))
    logger.debug('Account.CheckTpLc:pos_open_price=%.3f, has_long=%.3f, has_short=%.3f, close_pl=%.3f' \
                 %(self.pos_open_price, self.has_long,self.has_short, self.close_pl))

    # ポジションを持っている場合は、TPLCの評価を行う
    if(self.pos_open_price > 0):
      # Account.has_position の状況により、直近時間足(n-1:00)の最大利益と最大損失を算出する
      # has_positionはhas_longとhas_shortに分ける。また、ポジションの倍率を掛け、0.1～1.0の範囲とする
      # 通貨ペアによらず、1,000pips=1.000となるように調整する
      # スプレッド分をマージンとして差し引く
      # ↑2023/5/27 スプレッドはprice_dataのspreadで評価する

      # 2024/12/10 pos_open_priceからTPpriceとLCpricrを計算する
      # ---- Long の場合 ----
      if(self.has_long > 0.0):
        # Long positionを持っている場合
        # last_close_priceはbidで評価する
        self.takeprofit_price = self.pos_open_price + self.tp_level/self.magnification
        self.losscut_price = self.pos_open_price - self.lc_level/self.magnification
        self.last_close_price = self.last_close_bid
        self.float_pl = (self.last_close_price - self.pos_open_price) * self.magnification

        # ---- pandas.query を使わず numpy で高速判定 ----
        # 1角時間足分のtick dataframe(_ticks_period)からbidの値をnumpyに抜き出す
        # TP\LCの評価
        bids = df_ticks_period['bid'].to_numpy()
        # takeprofit_priceを超えた行番号(index)を早い順に書き出す
        hit_tp = np.where(bids >= self.takeprofit_price)[0]
        # losscut_priceを下回った行番号(index)を早い順に書き出す
        hit_lc = np.where(bids <= self.losscut_price)[0]
        logger.debug('Account.CheckTpLc:%s Count of hit_tp:%d hit_lc:%d' %(self.utc_from,len(hit_tp),len(hit_lc)))

        if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
            self.__execLossCut()
            logger.debug('Account.CheckTpLc:Loss cut was executed.(10)')
        elif len(hit_tp) > 0:
            self.__execTakeProfit()
            logger.debug('Account.CheckTpLc:Take profit was executed.(10)')

      # ---- Short の場合 ----
      elif(self.has_short > 0.0):
        # Short positionを持っている場合
        # open_priceはbid last_priceはask(=bid+spread)で評価する
        self.takeprofit_price = self.pos_open_price - self.tp_level/self.magnification
        self.losscut_price = self.pos_open_price + self.lc_level/self.magnification
        self.last_close_price = self.last_close_ask
        self.float_pl = (self.pos_open_price - self.last_close_price) * self.magnification

        # TP\LCの評価
        asks = df_ticks_period['ask'].to_numpy()
        hit_tp = np.where(asks <= self.takeprofit_price)[0]
        hit_lc = np.where(asks >= self.losscut_price)[0]
        logger.debug('Account.CheckTpLc:%s Count of hit_tp:%d hit_lc:%d' %(self.utc_from,len(hit_tp),len(hit_lc)))

        if len(hit_lc) > 0 and (len(hit_tp) == 0 or hit_lc[0] < hit_tp[0]):
            self.__execLossCut()
            logger.debug('Account.CheckTpLc:Loss cut was executed.(20)')
        elif len(hit_tp) > 0:
            self.__execTakeProfit()
            logger.debug('Account.CheckTpLc:Take profit was executed.(20)')

      else:
        # ここに入ったらエラー
        self.env.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                          +'Account.CheckTpLc:Position opened but no flags.')
        sys.exit("Account.CheckTpLc:Position opened but no flags.")

      logger.debug("Account.CheckTpLc:%s takeprofit_price=%.3f, losscut_price=%.3f, last_close_price=%.3f, float_pl=%.3f" \
                   %(self.utc_to, self.takeprofit_price, self.losscut_price, self.last_close_price, self.float_pl))
      # logger.info("Account.CheckTpLc:max_profit=%.3f, max_loss=%.3f, float_pl=%.3f" %(self.max_profit, self.max_loss,self.float_pl))

    # 直近足を更新する
    self.df_account.loc[self.utc_from,['pos_open_price','has_long','has_short','float_pl','close_pl']]\
    = (self.pos_open_price, self.has_long, self.has_short, self.float_pl, self.close_pl)
    # logger.debug('Account.CheckTpLc:df_account(%s)\n%s\n-----'%(self.utc_from,self.df_account.loc[self.utc_from]))

  def EvaluateRewrd(self, actn_idx, test_dt, last_dt, df_price_data):

    no_action = 0
    long_entry = 1
    short_entry = 2
    position_close = 3
    close_and_long = 4
    close_and_short = 5

    utc_from = test_dt
    utc_to = utc_from + timedelta(minutes=1)
    # PositionOpenPrice(SellOpenPrice/BuyOpenPrice)を初期化する。
    # MAX_RETRY_NUM回試行してもRealSpreadが閾値未満にならなかったときは、PositionOpenPriceを0.0としてオーダー不成立とする
    sell_open_price = 0.0
    buy_open_price = 0.0
    # 2026/8/9 クローズの値を格納する
    sell_close_price = 0.0
    buy_close_price = 0.0

    for n in range(MAX_RETRY_NUM[self.period]):
      # 2025/2/22 都度DBから取得せずにあらかじめDBから取得したpandasDataFrameから取得する
      # 2026/3/15 Copilotによるパフォーマンスチューニング
      df_ticks_first = self.env.GetTickData(self.symbol,utc_from,utc_to)

      if len(df_ticks_first) == 0:
        # tickが取得できなかった場合は、１分進める
        logger.warning('Account.EvaluateRewrd:No ticks. From %s To %s' 
                       %(utc_from.strftime('%Y-%m-%d %H:%M'),utc_to.strftime('%Y-%m-%d %H:%M')))
        utc_from = utc_from + timedelta(minutes=1)
        utc_to = utc_from + timedelta(minutes=1)
        continue

      logger.debug('Account.EvaluateRewrd:From %s To %s Count of ticks_first %d' 
                   %(utc_from.strftime('%Y-%m-%d %H:%M'),utc_to.strftime('%Y-%m-%d %H:%M'),len(df_ticks_first)))

			# 1分ごとのtickの最初ほ１つのスプレッドを計算する
      [sell_price,buy_price]=df_ticks_first.loc[df_ticks_first.index[0],['bid','ask']]
      real_spread = abs(buy_price - sell_price)

      # if real_spread >= REAL_SPREAD_LIMIT_LIST[self.symbol]:
      if real_spread >= self.real_sprad_limit:
        logger.debug('Account.EvaluateRewrd:%s real_spread(%.3f) exceeds limit (%.3f).' %(utc_from,real_spread,self.real_sprad_limit))
        # 2026/8/9 試行回数の最後の1回でもリアルスプレッドが規定値より大きいの場合はアクションごとに振舞いを変える
        if(n == MAX_RETRY_NUM[self.period]-1):
          # no_action の時はそのままなので何もしない
          # long_entry,short_entryの時はエントリーできないので no_actionに変更
          if actn_idx == long_entry or actn_idx == short_entry:
            action_idx = no_action
          # close_and_long,close_and_shortの時はクローズだけするので position_close に変更
          elif actn_idx == close_and_long or actn_idx == close_and_short:
            action_idx = position_close

          # postion_closeのためにtickの値を格納する
          sell_close_price = buy_price
          buy_close_price = sell_price

        # tick取得時間を1分進める
        utc_from = utc_from + timedelta(minutes=1)
        utc_to = utc_from + timedelta(minutes=1)
      else:
        # sell_open_priceとbuy_open_priceを設定してForループを抜ける
        logger.debug('Account.EvaluateRewrd:real_spread(%.3f) is in limit (%.3f).' %(real_spread,self.real_sprad_limit))
        sell_open_price = sell_price
        buy_open_price = buy_price

        # 2026/8/9 postion_closeのためにtickの値を格納する
        sell_close_price = buy_price
        buy_close_price = sell_price
        break

    # acount(last_dt(1:00))の状態をaccount(2:00(test_dt))に一旦コピーする
    # logger.debug('Account.EvaluateRewrd:Last_datetime=%s, Account=\n%s' %(last_dt,self.df_account.loc[last_dt]))
    self.df_account.loc[test_dt] = self.df_account.loc[last_dt]

    if(actn_idx == no_action):
      # ただし、account(test_dt(2:00))は次の時間足account(3:00)ですぐに評価される
      logger.debug('Account.EvaluateRewrd:Action_index=%s, No Action.' %(actn_idx))
      # 何もしないので、close_plは0にする
      self.df_account.loc[test_dt, 'close_pl'] = 0.0
    elif(actn_idx == long_entry):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Long Entry.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = test_dt(2:00), pos_open_price = price_data(2:00).open, has_position = 1, float_pl = 0.0, close_pl = 0.0
      # pos_open_dataにはAskを格納する
      # longとshortを分割する
      # self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
      #   = [test_dt, df_price_data.loc[test_dt,'open'] + df_price_data.loc[test_dt,'spread']/(1000*self.magnification), self.pos_scale, 0.0, 0.0, 0.0]
      self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
         = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0, 0.0]
    elif(actn_idx == short_entry):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Short Entry.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = test_dt(2:00), pos_open_price = price_data(2:00).open, has_position = -1, float_pl = 0.0, close_pl = 0.0
      # pos_open_dataにはBidを格納する
      # longとshortを分割する
      self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
         = [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0, 0.0]
    elif(actn_idx == position_close):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Position Close.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = none, pos_open_price = 0.0, has_position = 0, float_pl = 0.0, close_pl = float_pl(1:00)
      # longとshortを分割する
      # self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
      #   = [None, 0.0, 0.0, 0.0 , 0.0, self.df_account.loc[last_dt,'float_pl']]
      # 2026/8/9 position_close の際は real_spread を考慮する
      if(self.df_account.loc[last_dt,'has_long'] > 0.0):
        # Long positionを持っている場合
        # buy_close_price は bid で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
          = [None, 0.0, 0.0, 0.0 , 0.0
             , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]

      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
          = [None, 0.0, 0.0, 0.0 , 0.0
             , (self.df_account.loc[last_dt,'pos_open_price'] - sell_close_price) * self.magnification]

    elif(actn_idx == close_and_long):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Close and Long.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = test_dt(2:00), pos_open_price = price_data(2:00).open, has_position = 1, float_pl = 0.0, close_pl = float_pl(1:00)
      # pos_open_dataにはAskを格納する
      # longとshortを分割する
      # 2026/8/9 position_close の際は real_spread を考慮する
      # self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
      #    = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0, self.df_account.loc[last_dt,'float_pl']]
      if(self.df_account.loc[last_dt,'has_long'] > 0.0):
        # Long positionを持っている場合
        # buy_close_price は bid で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0
              , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]

      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0
              , (self.df_account.loc[last_dt,'pos_open_price'] - sell_close_price) * self.magnification]
    elif(actn_idx == close_and_short):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Close and Short.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = test_dt(2:00), pos_open_price = price_data(2:00).open, has_position = 1, float_pl = 0.0, close_pl = float_pl(1:00)
      # pos_open_dataにはBidを格納する
      # longとshortを分割する
      # 2026/8/9 position_close の際は real_spread を考慮する
      # self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
      #    = [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0, self.df_account.loc[last_dt,'float_pl']]
      if(self.df_account.loc[last_dt,'has_long'] > 0.0):
        # Long positionを持っている場合
        # buy_close_price は bid で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0
              , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]

      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [test_dt, sell_open_price, 0.0, self.pos_scale, 0.0
              , (self.df_account.loc[last_dt,'pos_open_price'] - sell_close_price) * self.magnification]
    else:
      # 定義されていないので、エラー
      EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
                                         +'Account.EvaluateRewrd:Action is not defined.')
      sys.exit('Account.EvaluateRewrd:Action is not defined.')

  # TakeProfitを実行して利益を確定する。private method
  def __execTakeProfit(self):
    self.close_pl += self.tp_level
    self.pos_open_price = 0.0
    self.has_long = 0.0
    self.has_short = 0.0
    self.float_pl = 0.0
    return

  # LossCutを実行して損失を確定する。private method
  def __execLossCut(self):
    self.close_pl -= self.lc_level
    self.pos_open_price = 0.0
    self.has_long = 0.0
    self.has_short = 0.0
    self.float_pl = 0.0
    return

	# countdownを計算して、当該時間のAccount情報に追加する
  def CalcCountdown(self, cntdt):
    self.countdown_datetime = cntdt
    self.countdown = (self.countdown_datetime - self.start_period).total_seconds() / (self.end_trade - self.start_period).total_seconds()
    logger.debug('Account.CalcCountdown:Countown=%.4f' %(self.countdown))
    # 当該足をのカウントダウンを更新する
    self.df_account.loc[self.countdown_datetime,['countdown']] = (self.countdown)

  # DataFrame Accountに格納されている値を返す。カラムを指定した場合はそのカラムだけを返す
  def GetAccountInfo(self, idx_dt, cols=None):
    if cols is None:
      cols = self.df_account.columns

    return self.df_account.loc[idx_dt,cols]

  def DropAccount(self):
    self.df_account = self.df_account.drop(self.df_account.index)
    logger.debug('Account.DropAccount:df_account %s' %(self.df_account))

  def SetEnv(self,env):
    self.env = env

  def SetDeltaPeriod(self,dltprd):
    self.delta_period = dltprd


# # Package-Agnent
# - 取引のアクションを決めるパッケージ以下の機能を実装する
#  -  Environmentから加工された価格データを受け取りLSTMに渡す
#  - LSTMのアクションをEnvironmentへ渡す
#  - DQLを行う

# ## Dependency

# In[6]:


#@title Dependency
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import math


# ## Logger
# - Loggerの実行は、Environmentと切り離されたときに実行する

# In[7]:


#@title Logger Agent
import logging
import logging.config

# logging.config.fileConfig('./drive/My Drive/Colab Notebooks/02_DRLTradingSystem2020/Logging.ini')
# logging.config.fileConfig(LOGGING_INIFILE_PATH)
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger_agent = logging.getLogger('DRLAgent')
logger_agent.debug('Debug level massage.')
logger_agent.info('Info level massage.')
logger_agent.warning('Warning level massage.')
logger_agent.error('Error level massage.')
logger_agent.critical('Critical level massage.')


# ## クラス Agent

# In[8]:


#@title class Agent { output-height: 200 }
class Agent():

  def __init__(self,input_num,hidden_num,output_num,model_name,trade_mode):
    #引数の各素子数に応じたニューラルネットを作成する
    self.main_brain = Brain(input_num,output_num,hidden_num)

    # model_name は 'GBPJPY_D1_B_20240323165209' という形式のため
    # path = G:\マイドライブ\Colab Notebooks\00_Common\Model\Trade\Model_GBPJPY_D1_B_20240323165209.pth
    # にする。ファイルが見つからない場合は'FileNotFoundError'をキャッチする
    # トレードモードによってパスを分ける
    if trade_mode == ACCOUNT_TRADE_MODE_REAL:
      TRADE_MODEL_PATH = TRADE_MODEL_PATH_REAL
    else:
      TRADE_MODEL_PATH = TRADE_MODEL_PATH_DEMO

    FilePath = namedtuple('FilePath', ['account','agent','brain'])
    # Model_EURJPY_D1_T_20260309195912.pth
    FilePath.brain = TRADE_MODEL_PATH+'Model_'+model_name+'.pth'
    # Agent_EURJPY_D1_T_20260309195912.ini
    FilePath.agent = TRADE_MODEL_PATH+'Agent_'+model_name+'.ini'
    logger_agent.debug('Agent.__init()__:ModelFile loading.\n FilePath.brain=%s' %(FilePath.brain))

    #これまでのトレーニングパラメータを呼び出す
    self.main_brain.loadModel(FilePath.brain)

    self.device = self.main_brain.GetDevice()

    #Rewardを格納するデータフレームを作成する
    # 2026/2/14 DataFrameはログ用のデータを格納するためのもの
    # Tensor計算用に辞書を用意する
		# System2026用：リワードは使わなくて大丈夫なはず
    # self.df_reward = pd.DataFrame(
    #    columns=['symbol','period','reward','q_max','a_max','q_taken','a_taken','exp_s_a']
    # )

    # 学習用 Tensor を保持する辞書
    # この辞書の形はtensor_reward={set_datetime:{'reward':xx,'q_max':xx,'a_max':xx,'q_taken':xx,'a_taken':xx,'exp_s_a':xx}}
    # のように、set_datetimeをキーとして入れ子になっている辞書を取り出す感じ（のはず。Copilotの提案）
    # 
		# self.tensor_reward = {}

    # float_plに対する評価を0~1の倍で設定する
    logger.debug('Agent.__init__: INIFile loading. \n FilePath.agent=%s' %(FilePath.agent))
    agt_inifile = configparser.ConfigParser()
    agt_inifile.read(FilePath.agent)
    self.est_float_pl = float(agt_inifile.get('COMMOM', 'ESTIMATE_FLOAT_PL'))

    logger_agent.debug('Agent.__init()__: ESTIMATE_FLOAT_PL= %.3f' %(self.est_float_pl))

  def DecideAction(self, set_datetime, input_data, action_mask):
    # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
    # 計算用Tensorを格納する辞書とログ用のDataframeに分ける
    # input_dataを用いて、NNからすべてのアクションの行動評価関数Qを取得する
    self.main_state_action_values = self.main_brain(input_data)
    # self.target_state_action_values = self.target_brain(input_data)
    logger_agent.debug('Agent.DecideAction:main_state_action_values\n%s\n------'%(self.main_state_action_values))
    # logger_agent.debug('Agent.DecideAction:target_state_action_values\n%s\n------'%(self.target_state_action_values))

    # DDQNに対応させる。main_brain(main Q-network)とtarget_brain(target Q-network)からそれぞれQ値
    # Q_m(s_t,a),Q_t(S_t,a)を取得する
    # 現在の状態から取り得る行動と、状態関数が最大の行動を取得する
    # ε-greedy法により、次の行動が状態関数の最大とは限らない
    # DDQN対応。mani_brain(main Q-network)から得られたaction_indexと、target_brain(target Q-network)から得られたmax_action_indexを使用する
		# decideAction(self, action_mask, action_score, episode=-2, train_mode=True):
    (self.acition_index, _) = self.main_brain.decideAction(action_mask, self.main_state_action_values, episode=-2, train_mode=False)
    # (_, self.max_action_index) = self.target_brain.decideAction(action_mask, self.target_state_action_values, epi_num, self.train_mode)

		# 強化学習はしないので、ここはいらないはず
    '''
    if set_datetime not in self.tensor_reward:
        self.tensor_reward[set_datetime] = {}

    # q_max, a_max , q_taken, a_takenを登録する
    # ---- 学習用 Tensor を保存（計算グラフ保持）----
    self.tensor_reward[set_datetime].update({
        "q_max":   self.target_state_action_values[0][self.max_action_index].view(1),
        "q_taken": self.main_state_action_values[0][self.acition_index].view(1),
        "a_max":   self.max_action_index,
        "a_taken": self.acition_index,
    })

    # ---- ログ用 DataFrame には float を保存 ----
    self.df_reward.loc[set_datetime, 'q_max']   = self.tensor_reward[set_datetime]["q_max"].item()
    self.df_reward.loc[set_datetime, 'q_taken'] = self.tensor_reward[set_datetime]["q_taken"].item()
    self.df_reward.loc[set_datetime, 'a_max']   = self.max_action_index
    self.df_reward.loc[set_datetime, 'a_taken'] = self.acition_index
    '''
    return self.acition_index

  def GetMainHiddenCellState(self):
    return self.main_brain.getHiddenCellState()


# ## クラス Brain
# Agent内のニューラルネット部分をBrainクラスとして別途定義

# In[9]:


#@title class Brain { output-height: 200 }

class Brain(nn.Module):

  def __init__(self,input_num,output_num,hidden_num):

    self.input_num = input_num
    self.output_num = output_num
    self.hidden_num = hidden_num

    super(Brain, self).__init__()
    #引数の各素子数に応じたニューラルネットを作成する
    #モデルとしては入力層(input_num)→LSTM→LSTM出力層（=hidden_num)→出力層(output_num)
    self.lstm = nn.LSTM(input_num, hidden_num)
    self.lstm2action = nn.Linear(hidden_num, output_num)
    # DuelingNetwork対応。Value層(lstm2value)とAdvantage層(lstm2action)に分ける。
    self.lstm2value = nn.Linear(hidden_num, 1)

    # NN　ModelをGPUで実行する。可能な場合
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger_agent.info('Brain.__init__:torch.device=%s' %(self.device))

    self.lstm.to(self.device)
    self.lstm2action.to(self.device)
    self.lstm2value.to(self.device)

    # logger.debug('Brain.__init__:self.lstm.to(self.device)=%s, self.lstm2action.to(self.device)=%s' %(self.lstm.is_cuda(), self.lstm2action.is_cuda()))

    #hidden_stateとcell_stateを初期化する。hidden_cell_stateは(hidden_state,cell_state)のタプルにする
    # self.resetHiddenCellState()
    # 損失関数に渡すデータの中身を初期化する
    # self.clearLossData()

    #損失関数と最適化手法を定義する
    #損失関数は、（ひとまず）SmoothL1Loss(state_action_values(=output),expected_state_action_values(=target))を採用
    #optimiserは、optim.Adam(self.model.parameters(), lr=0.0001)
    self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
    self.criterion = nn.SmoothL1Loss()

    return
## For OnLocal
  def LoadModel(self, file_path):
    logger_agent.debug('Brain.__init()__:path %s' %(file_path))
    try:
      self.load_state_dict(torch.load(file_path,map_location=torch.device(self.device)))
    except (FileNotFoundError) as e:
      logger_agent.error(e)
      return 1
    else:
      logger_agent.debug('NN Model load successfully.')
      #hidden_stateとcell_stateを初期化する。hidden_cell_stateは(hidden_state,cell_state)のタプルにする
      self.resetHiddenCellState()
      return 0

  # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
  ## For OnLocal
  @torch.no_grad()
  def forward(self, input_data):

    # numpy → tensor
    x = torch.FloatTensor(input_data).to(self.device)

    # LSTM forward
    lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
        x.view(-1, 1, self.input_num),
        (self.hidden_state.to(self.device), self.cell_state.to(self.device))
    )

    # Dueling Network
    self.action_score = self.lstm2action(lstm_out.view(-1, self.hidden_num))
    logger_agent.debug('Brain.forward:self.action_score====\n%s' %(self.action_score))
    self.value_score  = self.lstm2value(lstm_out.view(-1, self.hidden_num)).expand(-1, self.action_score.size(1))
    logger_agent.debug('Brain.forward:self.value_score====\n%s' %(self.value_score))

    # Q(s,a)
    self.output = self.value_score + self.action_score - self.action_score.mean(1, keepdim=True).expand(-1, self.action_score.size(1))
    logger_agent.debug('Brain.forward:self.output====\n%s' %(self.output))

    # ← ログ用に保存しない（Agent が必要なら受け取って保存する）
    return self.output

  def evaluateLossFunction(self, output, target):
    self.loss = self.criterion(output, target)
    return

  def resetHiddenCellState(self):
    #それぞれのテンソルのサイズは(num_layers * num_directions, batch, hidden_size)となる。
    #初期化はrandn()で行う。
    # self.hidden_state = torch.randn(1, 1, self.hidden_num)
    # self.cell_state = torch.randn(1, 1, self.hidden_num)
    # 初期化はzeros()で行う。
    self.hidden_state = torch.zeros(1, 1, self.hidden_num)
    self.cell_state = torch.zeros(1, 1, self.hidden_num)
    return

  # 2026/2/14 Copiotの助けを借りて大幅リファクタリング
  def decideAction(self, action_mask, action_score, episode=-2, train_mode=True):
    #取得したaction_score[1,output_num]にその時に可能な行動をaction_mask[output_num]によりマスクし、
    #その中で最大の行動（インデックス）を返す
    # ε-greedy法を適用する
    masked = torch.where(
        torch.BoolTensor(action_mask).to(self.device),
        action_score,
        torch.full_like(action_score, float('-inf'))
    )
    self.masked_action_scores = masked
    logger_agent.debug('Brain.decideAction:masked_action_scores======\n%s\n=======' %(masked))

    max_value, max_index = masked.max(1)

    epsilon = 0.5 * (1 / (episode + 1))
    # 推論モードの時はε-greedy法は適用しない(ランダムで選ばない)
    if train_mode and epsilon > np.random.uniform(0, 1):
      logger_agent.debug('Brain.decideAction ==Action Random Select==')
      random_scores = torch.where(
          torch.BoolTensor(action_mask).to(self.device),
          torch.rand_like(action_score),
          torch.full_like(action_score, float('-inf'))
      )
      logger_agent.debug('Brain.decideAction:random_action_scores======\n%s\n=======' %(random_scores))
      action_value, action_index = random_scores.max(1)
    else:
        action_index = max_index
        action_value = max_value

    # ← ログ用に保存しない（Agent が必要なら受け取って保存する）
    logger_agent.debug('Brain.decideAction:max_action_index======\n%s\n======\naction_index=======\n%s\n=======' %(action_index,action_value))
    return (action_index, max_index)

  def updateNN(self):
    # ネットワークを更新します
    self.optimizer.zero_grad()  # 勾配をリセット
    self.loss.backward()  # バックプロパゲーションを計算
    self.optimizer.step()  # 結合パラメータを更新
    return

  def setHiddenCellState(self, hidden_cell_state):
    (self.hidden_state, self.cell_state) = hidden_cell_state
    return

  def getHiddenCellState(self):
    return (self.hidden_state, self.cell_state)

  def saveModel(self, PATH):
    # デフォルト状態(_use_new_zipfile_serialization=True)で、runtime errorとなったときは
    # _use_new_zipfile_serialization=Falseで再試行する
    try:
      torch.save(self.state_dict(), PATH)
    except RuntimeError as e:
      logger_agent.error('Brain.saveModel:%s' %(e))
      torch.save(self.state_dict(), PATH, _use_new_zipfile_serialization=False)
    logger_agent.info('Brain.saveModel: Model saved successfully.')
    return

## For OnLocal
# LoadModelを実装したため、こちらは使わない
  def loadModel(self, PATH):
    self.load_state_dict(torch.load(PATH,map_location=torch.device(self.device)))
    #hidden_stateとcell_stateを初期化する。hidden_cell_stateは(hidden_state,cell_state)のタプルにする
    self.resetHiddenCellState()
    return

  def copyNN(self, model_dict):
    self.load_state_dict(model_dict)
    return

  def GetDevice(self):
    return self.device

  def getOutput(self):
    # OutputとMaskOutputを返す
    return (self.action_score,self.masked_action_scores)


# # Package-Trader
# - トレードをする機能実装する
# - 空トレードも実装

# ## Logger
# - TradrerパッケージのLoggerインスタンスはlogger_traderとする

# In[10]:


#@title Logger Trainer
import logging
import logging.config

# logging.config.fileConfig('./drive/My Drive/Colab Notebooks/02_DRLTradingSystem2020/Logging.ini')
# logging.config.fileConfig(LOGGING_INIFILE_PATH)
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger_trader = logging.getLogger('DRLTrader')
logger_trader.debug('Debug level massage.')
logger_trader.info('Info level massage.')
logger_trader.warning('Warning level massage.')
logger_trader.error('Error level massage.')
logger_trader.critical('Critical level massage.')


# ## クラス Trader
# - System2026では、空トレードをするため、Brainを直接初期化せずにTraderクラスを初期化する。
# - 初期化をするときに空トレードをする

# In[11]:


#@title class Trainer
class Trader():

  # def __init__(self, epinum, sbl, tf, start_dt, tdur, gnum, anum, train_data, periods_dict, load_flg, filepth=None, prds=None, cp_frq=2, grd='X', mgc='yyyymmddHHMMSS',rsf=False):
  # 初期化に必要なパラメータだけを受け取るようにする
  def __init__(self,sbl,tf,tdur,prds,rsl,lcl,tpl,ps,ptw,ipn,opn,hdn,trdmd,mdlnm,tsl):
    logger_trader.debug('◆◆%s Initialize Trainer ◆◆')
    '''
    .setファイルから受け取れるパラメータ
    MODEL_NAME=EURJPY_D1_T_20260309195912
    LC_LEVEL=2.75
    TP_LEVEL=2.95
    LONG_PERIOD=167
    SHORT_PERIOD=59
    TICKVOL_PERIOD=5
    POS_SCALE=0.5
    MODEL_INPUT_NUM=76
    MODEL_HIDDEN_NUM=76
    MODEL_OUTPUT_NUM=6
    TRAIN_SYMBOL_LIST=['EURUSD', 'USDJPY', 'GBPUSD', 'EURJPY', 'EURGBP', 'GBPJPY']
    MAGIC=20260309195912
    TRADE_SYSTEM=DuelNet_2026
    REAL_SPREAD_LIMIT=1.0
    PRE_TRADE_WEEKS=2
    '''
    #self.epi_num = epinum
    self.symbol = sbl
    self.period = tf
    # self.delta_period = EnvironmentCommon.GetPeriodTimeDelta(self.period)
    # self.delta_periodは、Environmentを初期化する前には取得できなくなった
    self.delta_period = None
    self.test_duration = tdur
    # self.start_date_time(月曜日 0:00)と、self.end_period(トレード期間中の最後の時間。土曜日0:00の一つ前の足の開始時間）
    # self.end_trade(self.end_periodの時に参照する時間足。self.end_periodの1つ前の時間足）
    # self.start_date_time,self.end_period,self.end_tradeは、Trader.SetTradePeriod()で計算
    # self.SetTradePeriodは、self.delta_periodが取得できないと計算できない
    # self.SetTradePeriod(now_dt)
    #self.start_date_time = start_dt
    #self.gen_num = gnum
    #self.agent_num = anum
    #self.load_flg = load_flg
    #self.file_path = filepth
    self.periods = prds #[LongPeriod,ShortPeriod,TickvolPerod]
    #self.copy_frq = cp_frq
    #self.magic = mgc
    #self.real_spread_flg = rsf
    self.real_spread_limit = rsl
    # 2026/2/14 Copilotの助けにより大幅なリファクタリング
    #self.periods_dict = periods_dict
    #self.train_data_dict = { self.period: train_data }
    self.lc_level = lcl
    self.tp_level = tpl
    self.pos_scale = ps
    self.pre_trade_weeks = ptw
    # ipn,opn,hdn
    self.input_num = ipn
    self.output_num = opn
    self.hidden_num = hdn
    # trade_mode
    self.trade_mode = trdmd
    self.model_name = mdlnm
    self.trade_symbol_list = tsl
    self.env = None # envは、DRLTradeServer4で初期化したのちに、setEnv()にて格納する

    logger_trader.debug('Trainer.__init__:pre_trade_weeks=%s' %(self.pre_trade_weeks))

    # トレーニングの結果(P/Lの合計と標準偏差)を格納するDataFrame
    self.df_result = pd.DataFrame(columns=['tradenum','sum','mean','sd'])

    # Accountを初期化する
    # self.acnt = Account(self.symbol,self.period,self.start_date_time, self.test_duration, self.load_flg, self.file_path, self.periods,self.real_spread_flg)
    # def __init__(self,sbl,tf,rsl,lcl,tpl,ps,ptw)
    self.acnt = Account(self.symbol,self.period,self.real_spread_limit,
                        self.lc_level,self.tp_level,self.pos_scale,self.pre_trade_weeks)
    # System2026用：これは使わない?
    # self.acnt.SetTradePeriod(self.start_period, self.end_period, self.end_trade):

    # オブジェクトのトレード期間(start_period(0:00),end_period(23:00),end_trade(22:00))をself.SetTradePeriodで計算する
    # self.start_date_time, self.end_date_time = self.acnt.GetTradePeriod()
    # logger_trader.debug('start_period:%s, end_period:%s, end_trade:%s' %(self.start_period, self.end_period, self.end_trade))

    # Agentを初期化する
    #   def __init__(self,input_num,hidden_num,output_num,model_name,trade_mode):
    self.agnt = Agent(self.input_num,self.hidden_num,self.output_num,self.model_name,self.trade_mode)
    '''    
    # トレード期間(Mon 1:00-Fri 23:00)のprace_dataをまとめて取得する
    self.UpdateTradePeriod(self.start_date_time, self.test_duration)
    # 詳細のテスト結果(acnt.df_account)を格納する
    self.df_detail_testresult = pd.DataFrame()
    '''
  def SetTradePeriod(self,now_dt):
    # 初期化された時のトレード開始日時を計算する
    if(self.test_duration == 'D'):
      # 1epiの期間が1日の場合
      # 開始日時は、その日の0:00。ただし、土日の場合は、翌週の月曜とする
      self.start_period = (now_dt
                           + timedelta(days=(0 if now_dt.isoweekday() < 6 else 8 - now_dt.isoweekday()))).replace(hour=0,minute=0)
      # トレードの終了は翌日0:00の2つ前の時間足
      self.end_period = (self.start_period + timedelta(days=1)).replace(hour=0,minute=0)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period
    elif(self.test_duration == 'W'):
      # 1epiの期間が1週の場合
      # トレード開始はその週の月曜日の0:00(datetime.isoweekday()は、月曜日=1,日曜日=7)
      # 初期化時点が土日の場合は、翌月曜日、平日の場合はその週の月曜日
      q,r = divmod(now_dt.isoweekday(),6)
      self.start_period = (now_dt + timedelta(days=1*q+1-r)).replace(hour=0,minute=0,second=0,microsecond=0)
      # トレードの終了は、その週の土曜日の0:00の2つ前(*)の時間足
      # (*)金曜日の23:00(=self.end_period)の時点で、22:00(=self.end_trade)の時間足を参照するので、df_acount(22:00)のcountdownが1.0となるように計算する
      self.end_period = (self.start_period + timedelta(days=5)).replace(hour=0,minute=0,second=0,microsecond=0)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period
    # 1epiの期間が1年(?)の場合
    elif(self.test_duration == 'Y'):
      # 開始日はその日(self.ini_datetime)の属する年の第2週の月曜日0:00
      self.start_period = datetime.datetime.fromisocalendar(now_dt.year, 2, 1)
      # トレードの終了は、その年の51週目の土曜日0:00の2つ前の時間足
      self.end_period = datetime.datetime.fromisocalendar(now_dt.year, 51, 6)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period

  def ExecPreTrade(self, now_dt):
    # 現在時刻から、一旦正規のトレード期間を計算する
    self.SetTradePeriod(now_dt)
    # 「空トレード」を行う。空トレード期間の月曜リストを取得する
    from_date = self.start_period - timedelta(weeks=self.pre_trade_weeks)
    to_date = self.start_period - timedelta(weeks=1)
    every_monday = pd.date_range(from_date, to_date, freq='W-MON')

    # １週間ごとに「空トレード」をする。トレード処理はプライベートメソッドに書き出す
    # AccountDatFrameとRewardDataFrameは都度破棄する
    # NNの更新はしない
    for w in every_monday:
      # self._ExecTrade(w,train_mode,return_detail,pretrade_flg=True)
      self._ExecTrade(w,pretrade_flg=True)

  # def _ExecTrade(self,stdt,train_mode,return_detail,pretrade_flg):
  def _ExecTrade(self,stdt,pretrade_flg):
    if pretrade_flg:
      logger_trader.info('Trader._ExecTrade:◇◇PreTradeing Start on %s◇◇' %(stdt))
    else:
      logger_trader.info('Trader._ExecTrade:◆◆Trading Start on %s◆◆' %(stdt))

    # self.UpdateTradePeriod(stdt, self.test_duration)
    self.SetTradePeriod(stdt)
    lstdt = self._GetLastDateTime(stdt, self.delta_period)
    self.acnt.SetAccount(lstdt)
    # System2026用：リワードは使わないで大丈夫なはず
    # self.agnt.SetReward(lstdt, self.acnt.GetAccountInfo(lstdt,['float_pl','close_pl']))

    # 2026/2/14 Copilotの力を借りて大幅リファクタリング
		# System2026用：価格データはMetaTrader5サーバから直接取得して加工する
    # price_obj = self.periods_dict[self.period][self.symbol]

    # 2023/06/03 実際にテストをするのは、スタート期間のひとつ前の足から
		# System2026用：テスト期間（self.start_period(月曜0:00)~self.end_period(金曜0:00))の
		# date_randeを取得する	
    pd_freq = PERIOD_FREQ_DICT[self.period]
    dtrng = pd.date_range(self.start_period, self.end_period, freq=pd_freq)		
    for tstdt in dtrng:
      # ここで、時間の調整
      # 現在時刻(t)のひとつ前(t-1)の時系列データを扱うときのdataetime
      # tstdt(test_date_time) → t, lstdt(last_date_time) → t-1
      # tstdt = self._GetNextDateTime(lstdt, self.delta_period)
      lstdt = self._GetLastDateTime(tstdt, self.delta_period)
      logger_trader.debug('Trainer._ExecTrade:last_date=%s(weekday:%d)' %(lstdt,lstdt.isoweekday()))
      logger_trader.debug('Trainer._ExecTrade:◆◆Traing test_date%s (weekday:%d)◆◆' %(tstdt,tstdt.isoweekday()))

      # 1.直近足(t-1)の価格データを使ってTP,LCのチェックを行う
      #   def GetPriceData(self,sb,pd,trddt):
      df_prace_data = self.env.GetPriceData(self.symbol,self.period,lstdt)
      if df_prace_data.empty:
        logger_trader.warning('Trainer._ExecTrade:[Empty PriceData]Traing %s skipped.' %(lstdt))
        continue

      self.acnt.CheckTpLc(lstdt, df_prace_data)

      # 2.Account情報(t-1)の確定損益(close_pl(t-1))と評価損益(float_pl(t-1))をReward(t)として登録する
      # 例：close_pl(1:00),float_pl(1:00)→Reward(2:00)に格納
			# System2026用：リワードは使わないで大丈夫なはず
      # self.agnt.SetReward(tstdt, self.acnt.GetAccountInfo(lstdt,['float_pl','close_pl']))

      # 3. トレーニングを行い、その結果、t(2:00)におけるq_max, a_max, q_taken. a_takenを取得する
      #  3-1. t-1カウントダウンの計算
      #       23:00の時点で、22:00のカウントダウンを計算して、22:00のdf_accountに格納する(?)
      # 	def SetTradePeriod(self, stpd, edpd, edtd):
      self.acnt.SetTradePeriod(self.start_period,self.end_period,self.end_trade)
      self.acnt.CalcCountdown(lstdt)
      #  3-2. t-1(1:00)のAccount情報として取得する
      #       23:00の時点で、22:00.closeのdf_accountから('has_position','float_pl','countdown')を取得する
      #       22:00のcountdownで全てクローズとさせたい
      # self.df_acnt = self.acnt.GetAccountInfo(lstdt,['has_position','float_pl','countdown'])
      # longとshortを分割
      self.df_acnt = self.acnt.GetAccountInfo(lstdt,['has_long','has_short','float_pl','countdown'])
      #  3-3. Accountの状態('has_position','float_pl','countdown')から、取り得るアクションを限定する
      # self.action_mask = EnvironmentCommon.getAvailableAction(self.df_acnt.float_pl, self.df_acnt.has_position, self.df_acnt.countdown)
      self.action_mask = self.env.getAvailableAction(self.df_acnt.float_pl, self.df_acnt.has_long, self.df_acnt.has_short, self.df_acnt.countdown)
      #  3-4. 直近足(t-1)のトレードデータを取得する
			#  def GetTradeData(self,dt,timeframe,sbl,lprd=LONG_PERIOD,sprd=SHORT_PERIOD,tvprd=TICKVOL_PERIOD,tsl=None):
      df_trade = self.env.GetTradeData(lstdt,self.period,self.symbol,self.periods[0],self.periods[1],self.periods[2],self.trade_symbol_list)
      #  3-5. トレーニングデータにアカウント情報を結合する
      self.df_acnt_train = pd.concat([self.df_acnt,df_trade])
      logger_trader.debug('Trainer._ExecTrade:Size of df_acnt_train=%d' %(len(self.df_acnt_train)))
      #  3-6. トレーニングデータなどをもとに次の行動を決定する
      #       この時、q_max、a_max、q_taken、a_takenをdf_reward(t)(2:00)に格納する
      # self.action_index = self.agnt.DecideAction(tstdt,self.df_acnt_train.to_list(), self.action_mask, self.enum)
      self.action_index = self.agnt.DecideAction(tstdt,self.df_acnt_train.to_list(), self.action_mask)
      # self.action_index = self.agnt.DecideAction(tstdt,self.df_acnt_train.iloc[0].tolist(), self.action_mask)
      # 4.選択した行動から損益を評価する。Rewardへの格納は次の時間のループの最初(2.)に行う
      # self.acnt.EvaluateRewrd(self.action_index, tstdt, lstdt, price_obj.GetPriceData(tstdt))
      self.acnt.EvaluateRewrd(self.action_index, tstdt, lstdt, df_prace_data)
      # 5. 2:00のRewardと行動関数から教師データを作成する
			# System2026用：教師データも作成しなくて大丈夫なはず
      # self.agnt.CalcExpectStateActionValue(tstdt)
      # 6. (option)トレードの詳細をファイルに書き出す。return_detail=Trueの時のみ
			# System2026用：これも使わない
      # if return_detail:
        # self._WriteDetailFile(tstdt)

      # tstdt += self.delta_period
    # end of for loop
    '''
    logger_trader.debug('◆◆Final Traing on %s◆◆' %(tstdt))
		# Sat0:00の処理　★この処理はいらないかもしれない
    # この時は、取引をしないで直前の時間(23:00)のdf_accountのclose_plとfloat_plを
    # Evaluate Reward, Calc
    # 現時点(0:00)のRewardに格納する
    # 現在時刻(t)のひとつ前(t-1)の時系列データを扱うときのdataetime
    # test_date_time → t, last_date_time → t-1
    lstdt = self._GetLastDateTime(tstdt, self.delta_period)

    # 1.直近足(t-1)の価格データを使ってTP,LCのチェックを行う
    self.acnt.CheckTpLc(lstdt, self.periods_dict[self.period][self.symbol].GetPriceData(lstdt))

    # 2.Account情報(t-1)の確定損益(close_pl(t-1))と評価損益(float_pl(t-1))をReward(t)として登録する
    # 例：close_pl(23:00),float_pl(23:00)→Reward(0:00)に格納
    self.agnt.SetReward(tstdt, self.acnt.GetAccountInfo(lstdt,['float_pl','close_pl']))
    # Step3,4は実施しない。
    #  3-1. t-1カウントダウンの計算
    #       0:00の時点で、23:00のカウントダウンを計算して、23:00のdf_accountに格納する(?)
    self.acnt.CalcCountdown(lstdt)
    # 5. 0:00のRewardと行動関数から教師データを作成する
    #    q_maxは 0 で評価する
    self.agnt.CalcExpectStateActionValue(tstdt)
    '''
    # 「空トレード」の時は、隠れ層の状態を確認する
    if pretrade_flg:
      hdn,cel=self.agnt.GetMainHiddenCellState()
      logger_trader.debug('Trainer.TrainAgent:\n[HiddenState]%s \n[CellState]%s' %(hdn,cel)) 
      logger_trader.debug('Trainer.TrainAgent:◇◇PreTraning Finished on %s◇◇' %(tstdt))
      logger_trader.debug('Total P/L:%.3f' %(self.acnt.df_account.close_pl.sum()))
    else:
      # トレードの結果を評価する。
      # ここに入ったらエラーで落ちる(self.enumがないため)
      logger_trader.info('◆◆Training Finish.◆◆')
      logger_trader.info('Total P/L:%.3f' %(self.acnt.df_account.close_pl.sum()))
      logger_trader.debug('Total Avg:%.3f' %(self.acnt.df_account.close_pl.mean()))
      logger_trader.debug('Total SD:%.3f' %(self.acnt.df_account.close_pl.std()))
      # 結果をdf_resultに格納する
      # self.df_result.loc[self.enum] = [(self.acnt.df_account.close_pl != 0).sum(), self.acnt.df_account.close_pl.sum(), self.acnt.df_account.close_pl.mean(), self.acnt.df_account.close_pl.std()]
      # トレーニングデータ(q_taken(mon 1:00-fri 23:00))と教師データ(exp_s_a(mon 2:00- sat 0:00))を
      # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
      # 計算用のTensorとログ用のDataFrameを分離する
      # self.q_taken_tensor = torch.cat([s for s in self.agnt.df_reward.iloc[1:-1,self.agnt.df_reward.columns.get_loc('q_taken')]])
      # self.exp_s_a_tensor = torch.cat([s for s in self.agnt.df_reward.iloc[2:,self.agnt.df_reward.columns.get_loc('exp_s_a')]])
      ''' System2026用：NNモデルの学習は行わない
      # TensorDict から学習用 Tensor を取り出す
      tensor_list_q_taken = []
      tensor_list_exp_s_a = []

      # df_reward の index を使って時系列順に Tensor を取り出す
      reward_index = list(self.agnt.df_reward.index)

      for i in range(1, len(reward_index)-1):
          t = reward_index[i]
          rt = self.agnt.tensor_reward[t]
          tensor_list_q_taken.append(rt["q_taken"])

      for i in range(2, len(reward_index)):
          t = reward_index[i]
          rt = self.agnt.tensor_reward[t]
          tensor_list_exp_s_a.append(rt["exp_s_a"])

      # 連結（計算グラフ保持）
      self.q_taken_tensor = torch.cat(tensor_list_q_taken)
      self.exp_s_a_tensor = torch.cat(tensor_list_exp_s_a)

      if self.enum + 1 < self.epi_num:
        # トレーニングモードの時に損失関数に送りNNの更新をする。
        if train_mode:
          self.agnt.UpdateMainNN(self.q_taken_tensor.view(-1,1), self.exp_s_a_tensor.detach().view(-1,1))
          # copy_frq(=2)の回数ごとにTargetQ-NNをMainQ-NNと同じにする
          if((self.enum+1) % self.copy_frq == 0):
            logger_trader.debug('Trainer.TrainAgent:Copy MainNN to TargetNN Execute.')
            self.agnt.CopyMainNNToTargetNN()

      # バックテスト用にdf_accountを退避させる
      #if return_detail:
      #  self.df_detail_testresult = self.acnt.df_account.copy()

      # 隠れ層をリセットする
      self.agnt.ResetHiddenCellState()
      hdn,cel=self.agnt.GetMainHiddenCellState()
      logger_trader.debug('Trainer.TrainAgent:\n[HiddenState]%s \n[CellState]%s' %(hdn,cel)) 
      '''

    # df_account、df_rewardを空にする。
    self.acnt.DropAccount()
    # self.agnt.DropReward()
    return

  # 現在のDateTime(cur_dt)からMetaTrader5のDB上の一つ前の足の日付データを取得する
  def _GetLastDateTime(self,cur_dt, dlt_dt):
    tmp_last_dt = cur_dt - dlt_dt
    # タイムゾーンをUTCに設定する
    timezone = pytz.timezone("Etc/UTC")
    # TimeZoneをMataTraderサーバの稼働しているキプロス（ニコシア）に設定する
    # timezone = pytz.timezone("Asia/Nicosia")
    tmp_last_dt.replace(tzinfo=timezone)
    # MetaTrader5サーバーにアクセスして、現在足の１つ前の足のdatetimeを取得する
    # タイムゾーンは UTCとする。サーバから取得される時間はJSTに変化されているので、UTCにする
    if not mt5.initialize(MT5_PATH):
      logger_trader.error("Trader._GetLastDateTime:initialize() failed, error code =%s" %(mt5.last_error()))
      sys.exit()

    count=1
		# last_dtを「月曜日0:00の１つ前」の時間（例えばD1の場合は、日曜日0:00）とした場合、自動的に
		# データのある時間足(金曜日0:00)までさかのぼる
    rates = mt5.copy_rates_from(self.symbol,TIMEFRAME_DICT[self.period],tmp_last_dt,count)
		# rates=[time、open、high、low、close、tick_volume、spread、real_volume]列を持つNumPy配列
		# ただし、timeはローカルのタイムゾーン(JST)に変換されたエポック秒として返ってくるので、astimezoneでUTC(=>キプロス)に変換
    last_dt = datetime.fromtimestamp(rates[0][0]).astimezone(timezone)

    return last_dt

	# 初期化したEnvironmentを格納する
  def setEnv(self, env):
    self.env = env
    self.acnt.SetEnv(env)

  # タイムフレームに応じたdatetime.timedeltaを取得する
  def setPeriodTimeDelta(self):
    self.delta_period = self.env.GetPeriodTimeDelta(self.period)
    self.acnt.SetDeltaPeriod(self.delta_period)


# # Stab

# ## Logger
# - StabのLoggerインスタンスはlogger_rootとする

# In[12]:


#@title Logger Stab
import logging
import logging.config

# logging.config.fileConfig('./drive/My Drive/Colab Notebooks/02_DRLTradingSystem2020/Logging.ini')
# logging.config.fileConfig(LOGGING_INIFILE_PATH)
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger_root = logging.getLogger('DRLRoot')
logger_root.debug('Debug level massage.')
logger_root.info('Info level massage.')
logger_root.warning('Warning level massage.')
logger_root.error('Error level massage.')
logger_root.critical('Critical level massage.')


# ## BackTestDetail

# In[ ]:


if __name__ == '__main__':
  ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMO #@param ["ACCOUNT_TRADE_MODE_DEMO", "ACCOUNT_TRADE_MODE_CONTEST", "ACCOUNT_TRADE_MODE_REAL"] {type:"raw"}
  BACKTEST_START_DATE = '2024-08-05' #@param {type:"date"}
  BACKTEST_END_DATE = '2024-08-10' #@param {type:"date"}
  BACKTEST_DURATION = 'W' #@param ['D', 'W', 'Y']

  # ファイルパスを設定する
  EnvironmentCommon.setTradeModePath(ACCOUNT_TRADE_MODE)

  TEST_SYMBOL = "GBPAUD" # @param ["*", "EURUSD", "USDJPY", "GBPUSD", "EURJPY", "EURGBP", "GBPJPY", "GBPAUD", "GBPZD"]
  TEST_PERIOD = "*" #@param ["*","D1","H12","H8", "H6","H4", "H1", "M30",  "M1"]
  TEST_GRADE = "*" #@param ["*","G","S","B","[GS]","[SB]","[BG]"]
  TEST_MAGIC = "*" #@param {type:"string"}

