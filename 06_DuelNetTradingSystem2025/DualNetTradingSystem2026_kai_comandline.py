#!/usr/bin/env python
# coding: utf-8

# # DuelNetTradingSystem2026改Comandline
# DualNetTradingSystem2026.ipynbよりコピー
# Copilotの助けを借りてパフォーマンスチューニング
# ## バッチファイル呼び出し用への修正
# 1. 余分なセルを削除する
# 2. セルアウトプットを削除する
# 3. 以前の.ipynbファイルに埋め込まれているfor comandlineの修正をする
# ## .pyファイルへの変換
# pthon仮想環境がアクティベートされているコマンドライン上で  
# `jupyter nbconvert --to python app.ipynb`
# を実行  
# ## What's new (backlog)
# - 最重要：tick → bar のマッピングを前計算して、pandas.query を完全に排除する

# # Package-Environment
# - 価格や損益の情報を提供するパッケージ。以下の機能を実装する
#  - 価格CSVファイルを結合して1つのpandas dataframeにする

# ## Dependency

# In[20]:


#@title Dependency
import pandas as pd
import numpy as np
import datetime
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
# 2026/6/6 SQLAlchemyより高速なConnectorXに変更
#from sqlalchemy import create_engine, text
import connectorx as cx


# ## クラス変数
# - SYMBOL_5DIGITS：小数点以下が5桁の通貨ペア(EURUSD等)の配列
# - SYMBOL_4DIGITS：小数点以下が4桁の通貨ペア(USDZAR等)の配列
# - SYMBOL_3DIGITS：小数点以下が3桁の通貨ペア(USDJPY等)の配列
# - DIGIT_MAGNIFICATION：通貨ペアごとの倍率辞書(5桁：100倍、4桁10倍、3桁1倍)

# In[21]:


# @title Class Valiables

EXEC_ENV = "LOCAL" #@param ["COLABO", "LOCAL"]

#
inifile = configparser.ConfigParser()

# .iniファイルのエンコード
ENC = None
if EXEC_ENV == 'COLABO':
  inifile.read('./drive/MyDrive/Colab Notebooks/06_DuelNetTradingSystem2025/settings.ini',ENC)
else:
  ENC = 'UTF-8'
  inifile.read(r'D:\ColabNotebooks\06_DuelNetTradingSystem2025\settings.ini',ENC)


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
# LINE_MESSAGING_API_TOKEN = inifile.get('COMMOM', 'LINE_MESSAGING_API_TOKEN')
# 2026/6/17 Line Message APIだと1か月に200メッセージしか送れないためntfy(ntfy.sh)に切り替える
URL_TOPIC_NAME = inifile.get('COMMOM', 'URL_TOPIC_NAME')

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
# 2026/6/6 SQLAlchemyの代わりにConnectorXを使う
# ENGINE=None
CONN_URL=None


# ## Logger
# ロギングを設定する。ログレベルは以下
# 1. CRITICAL
# 1. ERROR
# 1. WARNING
# 1. INFO
# 1. DEBUG
# 
# Environment package では、ログ空間を"DRL.Environment"とする

# In[22]:


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


# ## クラスEnvironmentCommon
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

# In[23]:


#@title class EnvironmentCommon
class EnvironmentCommon:
  @classmethod
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
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                  +'EnvironmentCommon.GetDigitMagnification:Symbol is not defined.')
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      sys.exit('Symbol is not defined.')

  @classmethod
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
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                   +'EnvironmentCommon.SetLimitStopLevel:Period is not Implemented.')
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      sys.exit('EnvironmentCommon.SetLimitStopLevel:Period is not Implemented.')

    losscut_level =  (1/23)*period_value + (22/23)
    logger.debug('## Losscut Level: %.2f' %(losscut_level))
    return losscut_level

  @classmethod
  def GetPeriodTimeDelta(cls,period):
    if(period == 'M1'):
      delta_period = datetime.timedelta(minutes=1)
    elif(period == 'M5'):
      delta_period = datetime.timedelta(minutes=5)
    elif(period == 'M15'):
      delta_period = datetime.timedelta(minutes=15)
    elif(period == 'M30'):
      delta_period = datetime.timedelta(minutes=30)
    elif(period == 'H1'):
      delta_period = datetime.timedelta(hours=1)
    elif(period == 'H4'):
      delta_period = datetime.timedelta(hours=4)
    elif(period == 'H6'):
      delta_period = datetime.timedelta(hours=6)
    elif(period == 'H8'):
      delta_period = datetime.timedelta(hours=8)
    elif(period == 'H12'):
      delta_period = datetime.timedelta(hours=12)
    elif(period == 'D1'):
      delta_period = datetime.timedelta(days=1)
    else:
      logger.warning('■■Period Delta is not Implemented')
      # delta_period = datetime.timedelta(hours=1)
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                   +'EnvironmentCommon.GetPeriodTimeDelta:Period Delta is not Implemented.')
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      sys.exit('EnvironmentCommon.GetPeriodTimeDelta:Period Delta is not Implemented.')

    return delta_period

  # longとshortで分ける
  @classmethod
  # def getAvailableAction(self,pl_, pos_, cd_):
  def getAvailableAction(self,pl_, lg_, st_, cd_):
    action_mask = np.empty(6, dtype='bool')
    # logger.debug('EnvironmentCommon.getAvailableAction:plofit_loss=%.3f position=%d countwdown=%.3f' %(pl_,pos_,cd_))
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
        # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
        #                           +'EnvironmentCommon.getAvailableAction:pattern 4 error %s' %(action_mask))
        ## ForComandline
        # 標準出力で999を返す
        print(999)
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
        # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
        #                           +'EnvironmentCommon.getAvailableAction:pattern 8 error %s' %(action_mask))
        ## ForComandline
        # 標準出力で999を返す
        print(999)
        sys.exit('EnvironmentCommon.getAvailableAction:pattern 8 error ' ,str(action_mask))

    return action_mask

  @classmethod
  def periodRandomSelect(cls):
    # 各Periodの値をランダムに選択する
    prime_number = [5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97\
                    ,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191]
    period_numbers = random.sample(prime_number,2)
    long_period = max(period_numbers)
    short_period = min(period_numbers)
    tickvol_period = random.choice(prime_number)

    return (long_period,short_period,tickvol_period)

  @classmethod
  def setTradeModePath(cls, trdmd):
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
  '''
  @classmethod
  def send_line_notify(cls,notification_message):
    headers = {"Content_Type": "application/json","Authorization": "Bearer " + LINE_MESSAGING_API_TOKEN}
    requests.post("https://api.line.me/v2/bot/message/broadcast",headers=headers,json={"messages": [{"type": "text","text": notification_message}]}).json()
  '''
  @classmethod
  def send_ntfy_message(cls,ttl,msg):
    # 通知を送信
    response = requests.post(URL_TOPIC_NAME, data=msg.encode('utf-8'), headers={"Title": ttl.encode('utf-8')})

    # 通知が成功したか確認
    if response.status_code == 200:
        logger.debug('EnvironmentCommon.send_ntfy_message:Send Message Success. Title:%s, Message:%s' %(ttl,msg))
    else:
        logger.debug('EnvironmentCommon.send_ntfy_message:Send Message Faled. Title:%s. Message:%s' %(ttl,msg))

  @classmethod
  def getRealSpreadLimit(cls,sbl,flg=False):
      if flg:
          return REAL_SPREAD_LIMIT_LIST[sbl]
      else:
          return 1.0

  # 2026/6/6 SQLAlchemyの代わりにConnectorXを使う
  @classmethod
  def connectDB(csl):
      # global ENGINE
      # ENGINE = create_engine(RDBMS+"://"+USER+":"+PASSWORD+"@"+HOST+"/"+DBNAME)

      # CONN_URLでは、DB接続を確立するわけではない
      global CONN_URL
      CONN_URL = RDBMS+"://"+USER+":"+PASSWORD+"@"+HOST+"/"+DBNAME

  @classmethod
  def compareEvalValue(cls,sbl,tf,eval_value):
      # 新しく取得したeval_valueと比較して、新しいほうの値が良ければ、既存のファイルをvoidファイルに移動する
      # 戻り値：true→新規作成　false→変更なし
      # 同じ通貨ペアと時間足のファイルAgentファイルがあるかを確認する
      # フォルダにある同じ通貨ペア、時間足のファイルを取得する
      # 
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


# ## クラス PriceData
# 価格データを取り扱うクラス
# ### コンストラクタ [PriceData()](#init)
# - オブジェクト化する通貨ペアと時間足を受け取り、対象となるCSVファイルをPandas Dataframeに変換する
#  - 引数：
#    - sbl=通貨ペア('EURUSD'など)
#    - prd=時間足('M1','H1'など)
#  - 戻り値：
#  - 入力：
#  - 出力：
# 
# ### オブジェクト関数 [ConvertPriceDataFileToDataFrame()](#method1)
# - Google Driveにある価格データをPandasDataframe(price_list)に通貨ペア、時間足ごとに変換する。csvファイルが例えば年単位で複数ある場合は結合する
# - DataFrameの内容はMT5から取得したデータのままとする
# - DataFrame(price_list)の構成は以下の通り
#   - Index:time(datetime)
#   - open
#   - high
#   - low
#   - close
#   - tick_volume
#   - spread
#   - real_volume
# 
# ### オブジェクト関数[AddStaticInfo()](#method2)
# - ConvertPriceDataFileToDataFrame()で作成したDataFeame(static_data)に、SMA(単純移動平均)とStdev(標準偏差)、z-score等を追加する
# - open,high,low,closeは通貨ペアの小数点以下の桁数によって100倍(5桁(UERUSD等))、10倍(4桁(USDZAR等))、1倍(3桁(USDJPY等))する
# - DataFrame(static_data)の構成は以下の通り
#   - Index:time
#   - tick_volume
#     - ~~tick_volume/100,000する~~
#     - TANH(LOG10(AVERAGE()))とする
#   - Close-Open:price_list.close-price_list.open
#   - High-Low:price_list.high-price_list.low
#   - STD_Short：短期標準偏差.close
#   - STD_Long：長期標準偏差.cloe
#   - z_score_short：短期Z-Score.close
#   - z_score_long：長期Z-Score.close
#   - sma_close_short-long：SMA(Short.close)-SMA(Long.close)
#   - std_short-long：STD(Short.close)-STD(Long.close)
#   - z_score_short-long：Z-Score(Short.close)-Z-Score(Long.close)
#   - sma_close-open_short：SMA(Short.close)-SMA(Short.open)
#   - sma_close-open_long：SMA(Long.close)-SMA(Long.open)
# 
#   - ※SMALong:price_list.closeの長期単純移動平均
#   - ※SMAShort:price_list.closeの短期単純移動平均
#   - ※カラム名には通貨ペア名(USDJPY等)つける
# 
# ### オブジェクト関数[GetPriceData()](#method3)
# - 指定した日時の時間足データを取得する
#  - 引数
#    - 時間(datetime)
#  - 戻り値
#    - 時間足データ(pandas.DataFrame)
# 

# In[24]:


#@title class PriceData
class PriceData:

  def __init__(self, sbl, prd):
    self.symbol = sbl
    self.period = prd
    return

  def ConvertPriceDataFileToDataFrame(self):

    # ファイル名は"EURUSD_H1_202301020000_202312312359.csv"
    filenames=glob.glob('%s%s_%s_*.csv'%(PRICEDATA_PATH,self.symbol,self.period))

    # ファイルリストを1つずつ取り出して、detaframeに加工する
    list_ = []
    for file in filenames:
      # csv(tsv)ファイルをpd.dataframeに変換
      # df = pd.read_table(file)
      df = pd.read_csv(file)
      # logger.debug(df)

      # <DATE>と<TIME>を結合し(間にスペースを挿入)、datetime型に変換
      # 新たな列<DATETIME>に結合結果を追加する
      # df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S')
      df['time'] = pd.to_datetime(df['time'])

      #<DATETIME>列をインデックスにした後に、<DATE>列と<TIME>列を削除する
      #InedexがDateTimeIndexとなり、時間でのスライスが可能
      # df = df.set_index('<DATETIME>')
      df = df.set_index('time')
      list_.append(df)

    self.price_list = pd.concat(list_)

    # データの加工
    # Indexを昇順に並べる(念のため)
    self.price_list.sort_index(inplace=True)
    # 重複している行を削除
    logger.debug(('PriceData.ConvertPriceDataFileToDataFrame:Duplicated Rows:',self.price_list[self.price_list.duplicated()]))
    self.price_list.drop_duplicates(inplace=True)

  #@markdown ##AddSMAStdev() <a name = "method2"></a>
  def AddStaticInfo(self, periods=None):
    self.static_data = pd.DataFrame(index=self.price_list.index)
    # self.price_listの各列を追加していく
    # open,high,low,closeは通貨ペアによって桁数調整をする
    self.static_data[['open','high','low','close']]\
    = self.price_list[['open','high','low','close']]*EnvironmentCommon.GetDigitMagnification(self.symbol)

    # LONG_PERIOD,SHORT_PERIOD,TICKVOL_PERIODはインスタンス毎に変更する
    if periods != None:
      self.long_period = periods[0]
      self.short_period = periods[1]
      self.tickvol_period = periods[2]
    else:
      self.long_period = LONG_PERIOD
      self.short_period = SHORT_PERIOD
      self.tickvol_period = TICKVOL_PERIOD
    logger.info('PriceData.AddStaticInfo:self.long_period=%d, self.short_period=%d, self.tickvol_period=%d' %(self.long_period,self.short_period,self.tickvol_period))

    # TickVolは10万分の1にする
    # 2023/7/9 glocab変数を変更しない
    self.static_data['tick_volume'] = np.tanh(np.log10(self.price_list['tick_volume']/self.price_list['tick_volume'].rolling(self.tickvol_period).mean()))
    # 統計情報の追加するものは上記の説明参照
    self.static_data['close-open'] = self.static_data['close'] - self.static_data['open']
    self.static_data['high-low'] = self.static_data['high'] - self.static_data['low']
    self.static_data['sma_close_short'] = self.static_data['close'].rolling(self.short_period).mean()
    self.static_data['sma_close_long'] = self.static_data['close'].rolling(self.long_period).mean()
    self.static_data['sma_open_short'] = self.static_data['open'].rolling(self.short_period).mean()
    self.static_data['sma_open_long'] = self.static_data['open'].rolling(self.long_period).mean()
    self.static_data['std_short'] = self.static_data['close'].rolling(self.short_period).std()
    self.static_data['std_long'] = self.static_data['close'].rolling(self.long_period).std()
    self.static_data['z_score_short'] = (self.static_data['close'] - self.static_data['sma_close_short']) / self.static_data['std_short']
    self.static_data['z_score_long'] = (self.static_data['close'] - self.static_data['sma_close_long']) / self.static_data['std_long']
    self.static_data['sma_close_short-long'] = self.static_data['sma_close_short'] - self.static_data['sma_close_long']
    self.static_data['std_short-long'] = self.static_data['std_short'] - self.static_data['std_long']
    self.static_data['z_score_short-long'] = self.static_data['z_score_short'] - self.static_data['z_score_long']
    self.static_data['sma_close-open_short'] = self.static_data['sma_close_short'] - self.static_data['sma_open_short']
    self.static_data['sma_close-open_long'] = self.static_data['sma_close_long'] - self.static_data['sma_open_long']

    # NNに投入しないカラム(open,high,low,close)は削除する
    self.static_data.drop(columns=['open','high','low','close','sma_close_short','sma_close_long','sma_open_short','sma_open_long']
                          , inplace=True)

    # カラム名に一括で通貨ペアを付加する
    self.static_data.rename(columns=lambda s: self.symbol + '_' + s, inplace = True)
    # logger.debug('PriceData.AddStaticInfo:%s' %(self.static_data))

  def GetPriceData(self,dt):
    return self.price_list[self.price_list.index == dt]


# ## クラスTrainData
# テストデータを扱うクラス
# ### コンストラクタTrainData()
# - 複数通貨ペアのPriceDataを受け取り、結合してTrainDataを作成する
# - TradeAgentへ渡す訓練データとして、現在のポジションの情報を格納するカラムを追加する
# - DataFrame(train_data_list)の構造は以下の通り
#   - static_data:各通貨ペア分結合
#   - current_pl:現在の損益
#   - long_position_flg:ロングを保有していれば"1"
#   - short_position_flg:ショートを保有していれば"1"
#   (cuttrent_pl、各position_flgは"0"で初期化)
#  - 引数：
#    - [PriceDataList]：オブジェクトPriceDataのリスト
#  - 入力：
#    - PositionOpenPrice:現在保有ポジションのエントリー価格(初期値0.0)
#    - LongFlg:保有ポジションがLongならば1(初期値0)
#    - ShortFlg:保有ポジションがShortならば1(初期値0)
#    - Reward:報酬(初期値0.0)
# 
# メモ：GetRewardとResetRewardを作る

# In[25]:


#@title class TrainData
class TrainData:

  #@markdown ## Constractor <a name = "init1"></a>
  def __init__(self, pdlst, idx=None):
    # static_dataのデータを結合して1つのデータフレームにする
    # axis=1で、横方向に結合(カラムを追加する)
    # join='outer'を指定することで、全ての行が残る
    self.train_data = pd.concat([p.static_data for p in pdlst], axis=1, join='outer')
    # さらに取引通貨ペアのインデックスと合わせる
    if idx is not None:
      self.train_data = self.train_data.reindex(index=idx)

    # 保有ポジションの情報を格納するカラムを追加する
    # self.train_data['current_pl'] = 0
    # self.train_data['long_pos_flg'] = 0
    # self.train_data['short_pos_flg'] = 0
    return

  #  指定したindexのトレーニングデータを返す
  def GetTrainData(self, indx_timedate):
    return self.train_data.loc[indx_timedate]

  # 指定したindexの期間のトレーニングデータを返す
  def GetTrainDataPeriod(self, start_timedate, end_timedate):
    return self.train_data.loc[start_timedate:end_timedate]

  def GetTrainDataColLen(self):
    return len(self.train_data.columns)


# ## クラス Account
# - 訓練するTradeAgentのポジション情報と報酬を管理する
# ### コンストラクタ  Account()
#  - pandas.DataFrame:Accountを初期化する。構造は以下
#    - index(Datetime):時間足のオープン時刻
#    - ~~reward:報酬。損益が確定した段階で報酬に反映させる。1,000pips=1とする。報酬が確定するときは①ポジションをクローズしたとき②t/pやl/cにかかったとき~~
#    →Rewardは、Agentで保持する
#    - position_open_price：現在保有しているポジションのエントリー価格
#    - has_long：現在保有しているポジションがLongの場合はTrue
#    - has_short：現在保有しているポジションがShortの場合はTrue
#    - float_pl：現在保有しているポジションの含み損益
#    - close_pl：この時間足で確定した損益。報酬と基本的には同じ
#  - 引数(Symbol,TimeFrame)を格納する
#  - TimeFrameに応じたLimitStopLevelを決定する。private関数setLimitStopLevel()を呼びだす
#  - Symbolに対応するマージンを取得する
# 
# ### オブジェクト関数　CheckTpLc()
# - n:00(ex 1:00)に、(n-1):00(ex0:00)足のデータを用いて、Tp/Lcに引っかかていないかをチェックする。
#  - 引数
#    - last_datetime:Datetime 直近(1つ前)足のOpentime(n-1:00)
#    - last_price_data:DataFrame 直前足のPriceData
#    - now_datetime:Datetime 現在時刻のOpenTime(n:00)
# 
#  - 入力
#    - Account((n-1):00) 直近のAccount情報
# 
#  - 処理
#    1. 直近のAccount情報を取得する
#    1. Account.position_open_priceとLastCloseの差から現在時刻(n:00)のprofit_lossを計算する
#      - has_long && !has_short → profit_loss = LastClose - position_open_price
#      - !has_long && has_short → profit_loss = position_open_price - LastClose
#      - !has_long && !has_short → profit_loss = 0
#      - has_long && has_short → sys.exit()
#   この時、profit_lossからmarginを引く
#    1. Tp/Lcに引っかかっていないかチェックする
# 
# ### オブジェクト関数 ExecuteAction()
# - ニューラルネットの判断(Action)に基づきアクションを実行し、Account情報を更新する
# - 実行するアクションは
#  1. a0:None
#  1. a1:LongEntry
#  1. a2:ShortEntry
#  1. a3:PositionClose
#  1. a4:PositionClose&LongEntry
#  1. a5:PositionClose&ShortEntry
# 
# ### オブジェクト関数 CalcCountdown()
# - 1 episodeの最後に向かってカウントダウン(実際は0から1に向かってカウントアップ)をする。
#  - 1epi=1日の時は、当日の0:00が最初(=0)、翌日0:00の2つ前の足が最後(=1)
#  - 1epi=1週の時は、月曜日の0:00が最初(=0)、土曜日の0:00の2つ前の足が最後(=1)
#  - 1epi=1年の時は、当年のISO第1週目の月曜日の0:00が最初(=0)、翌年のISO第1週の2週前の土曜日の0:00の2つ前の足が最後(=1)

# In[26]:


#@title class Account
class Account:

  def __init__(self,sbl,tf,inidt,trndrt,load_flg=False,file_name=None,periods=None,real_spread_flg=False):
    # accountを格納する。pandas dataframe
    self.df_account = pd.DataFrame(columns=['symbol','period','pos_open_datetime','pos_open_price','has_long','has_short','float_pl','close_pl','countdown'])
    self.symbol = sbl
    self.period = tf
    self.ini_datetime = inidt
    if load_flg and (file_name.account != None) :
      logger.debug('Account.__init__: INIFile loading. \n %s' %(file_name.account))
      #Load FlagがTrueでファイルパスが指定されている場合は、そのファイルを読み込む
      acc_inifile = configparser.ConfigParser()
      acc_inifile.read(file_name.account)
      self.tp_level = float(acc_inifile.get('COMMOM', 'TAKEPROFIT_LEVEL'))
      self.lc_level = float(acc_inifile.get('COMMOM', 'LOSSCUT_LEVEL'))
      # 保有ポジションのモデルに入力する際の倍率(0.1～1.0倍 0.1刻み)
      self.pos_scale = float(acc_inifile.get('COMMOM', 'POS_SCALE'))
      self.real_sprad_limit = float(acc_inifile.get('COMMOM', 'REAL_SPREAD_LIMIT'))
      # 2025/3/15 「空テスト」を行う期間(週数)
      self.pre_trade_weeks = int(acc_inifile.get('COMMOM', 'PRE_TRADE_WEEKS'))
    else:
      logger.debug('Account.__init__: LC/TP Level is set.')
      [self.tp_level] = random.choices([random.randrange(TP_LC_MIN_LEVEL,TP_LC_MAX_LEVEL + 1, TP_LC_STEP),0],\
                                   [math.floor((math.floor(TP_LC_MAX_LEVEL-TP_LC_MIN_LEVEL)+1)/TP_LC_STEP),1])
      [self.lc_level] = random.choices([random.randrange(TP_LC_MIN_LEVEL,TP_LC_MAX_LEVEL + 1, TP_LC_STEP),0],\
                                   [math.floor((math.floor(TP_LC_MAX_LEVEL-TP_LC_MIN_LEVEL)+1)/TP_LC_STEP),1])
      self.tp_level /= 1000
      self.lc_level /= 1000
       # 保有ポジションのモデルに入力する際の倍率(0.1～1.0倍 0.1刻み)
      self.pos_scale = random.randrange(1, 11, 1)/10
      # real_spread_limitを取得する
      self.real_sprad_limit = EnvironmentCommon.getRealSpreadLimit(sbl,real_spread_flg)
      logger.debug("Account.__init__:RealSpreadLimit:%.3f" %(self.real_sprad_limit))
      # 2025/3/15 「空テスト」を行う期間(週数)
      # 2026/2/26 12週間はCPU負荷を考慮して削除
      self.pre_trade_weeks = random.choice([0,1,2,4])
      # ★★ForTest 1週間で固定
      # self.pre_trade_weeks = 1
      logger.debug("Account.__init__:pre_trade_weeks:%d" %(self.pre_trade_weeks))

    logger.debug('Account.__init__:tp_level=%.3f, lc_level=%.3f, pos_scale=%.1f' %(self.tp_level, self.lc_level ,self.pos_scale))

    self.margin = MARGIN_DICT[self.symbol]
    self.magnification = EnvironmentCommon.GetDigitMagnification(self.symbol)

    self.pos_open_price = 0.0
    # LongとShortを別々に入力する
    self.has_long = 0.0
    self.has_short = 0.0
    # self.has_position = 0.0
    self.float_pl = 0.0
    self.close_pl = 0.0
    self.train_duration = trndrt
    # 時間足の長さに応じたTimeDeltaを取得する
    self.delta_period = EnvironmentCommon.GetPeriodTimeDelta(self.period)
    self.countdown = 0.0

    self.SetTradePeriod(self.ini_datetime, self.train_duration)

    # INIファイルに記録するため、LONG,SHORT,TICKVOL_PERIODの値をオブジェクトに格納する
    if periods != None:
      self.long_period = periods[0]
      self.short_period = periods[1]
      self.tickvol_period = periods[2]
    else:
      self.long_period = LONG_PERIOD
      self.short_period = SHORT_PERIOD
      self.tickvol_period = TICKVOL_PERIOD

    self.ticks_frame = None

    # 2026/3/15 Copilotによるパフォーマンスチューニング
    # bar_to_tick = {
    #     bar_datetime: (tick_start_idx, tick_end_idx)
    # }
    self.bar_to_tick = {}
    self.bar_to_tick_minute = {}

  def SetTradePeriod(self, stdt, trdl):
    self.ini_datetime = stdt
    self.train_duration = trdl
    # 初期化するときにトレードの期間(start,end)と、カウントダウンの最後を計算する
    if(self.train_duration == 'D'):
      # 1epiの期間が1日の場合
      # 開始日時は、その日の0:00。ただし、土日の場合は、翌週の月曜とする
      self.start_period = (self.ini_datetime
                           + datetime.timedelta(days=(0 if self.ini_datetime.isoweekday() < 6 else 8 - self.ini_datetime.isoweekday()))).replace(hour=0,minute=0)
      # トレードの終了は翌日0:00の2つ前の時間足
      self.end_period = (self.start_period
                          + datetime.timedelta(days=1)).replace(hour=0,minute=0)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period
    elif(self.train_duration == 'W'):
      # 1epiの期間が1週の場合
      # トレード開始はその週の月曜日の0:00(datetime.isoweekday()は、月曜日=1,日曜日=7)
      # 日曜日は次の週とするため剰余を使う
      self.start_period = (self.ini_datetime
                          + datetime.timedelta(days=1 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0,minute=0,second=0,microsecond=0)
      # トレードの終了は、その週の土曜日の0:00の2つ前(*)の時間足
      # (*)金曜日の23:00(=self.end_period)の時点で、22:00(=self.end_trade)の時間足を参照するので、df_acount(22:00)のcountdownが1.0となるように計算する
      self.end_period = (self.ini_datetime
                          + datetime.timedelta(days=6 - (self.ini_datetime.isoweekday() % 7))).replace(hour=0,minute=0,second=0,microsecond=0)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period
    # 1epiの期間が1年(?)の場合
    elif(self.train_duration == 'Y'):
      # 開始日はその日(self.ini_datetime)の属する年の第2週の月曜日0:00
      self.start_period = datetime.datetime.fromisocalendar(self.ini_datetime.year, 2, 1)
      # トレードの終了は、その年の51週目の土曜日0:00の2つ前の時間足
      self.end_period = datetime.datetime.fromisocalendar(self.ini_datetime.year, 51, 6)
      self.end_period -= self.delta_period
      self.end_trade = self.end_period - self.delta_period

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

      # self.df_account.loc[set_datetime] = [symbol, period, pos_open_datetime, pos_open_price,
      #                                    has_position, float_pl, close_pl, countdown]
      self.df_account.loc[set_datetime] = [symbol, period, pos_open_datetime, pos_open_price,
                                         has_long, has_short, float_pl, close_pl, countdown]
    # logger.debug('Account.SetAccount:df_account(%s)\n%s' %(set_datetime, self.df_account.loc[set_datetime]))

  def CheckTpLc(self, lstdt, lstprc):
    self.last_close_price = 0.0

    self.utc_from = lstdt
    self.utc_to = lstdt + self.delta_period
    # 直近足のすべてのtickを取得する
    # 2025/02/11 MT5ではなく、DBから取得するようにする
    # 2025/2/22 DBからではなく、あらかじめDBから取得したDataFrameから直近の時間足だけ取得する
    # 2026/3/15 Copilotによるパフォーマンスチューニング　pandsのqueryは致命的に遅い(らしい)ので、使わない
    # _ticks_period = self.ticks_frame.query('@self.utc_from<=index<@self.utc_to')
    start_idx, end_idx = self.bar_to_tick[self.utc_from]

    # if len(_ticks_period) == 0:
    if start_idx==end_idx:
      # tickが取得できなかった場合は、その期間は取引がなかったため、TpLcの評価はしない
      logger.warning('Account.CheckTpLc:Due to No ticks TpLc Skipped. From %s To %s' 
                     %(self.utc_from.strftime('%Y-%m-%d %H:%M'),self.utc_to.strftime('%Y-%m-%d %H:%M')))
      return

    _ticks_period = self.ticks_frame.iloc[start_idx:end_idx]   
    logger.debug('Account.CheckTpLc:From %s To %s Count of _ticks_period %d' 
                 %(self.utc_from.strftime('%Y-%m-%d %H:%M'),self.utc_to.strftime('%Y-%m-%d %H:%M'),len(_ticks_period)))



    # 直近足のopen(ask/bid)、close(ask/bid)を取得する
    # 2026/3/15 Copilotによるパフォーマンスチューニング
    # self.last_open_ask, self.last_open_bid=_ticks_period.iloc[0,[_ticks_period.columns.get_loc('ask'),_ticks_period.columns.get_loc('bid')]]
    # self.last_close_ask, self.last_close_bid=_ticks_period.iloc[-1,[_ticks_period.columns.get_loc('ask'),_ticks_period.columns.get_loc('bid')]]
    self.last_open_ask  = _ticks_period.iloc[0]['ask']
    self.last_open_bid  = _ticks_period.iloc[0]['bid']
    self.last_close_ask = _ticks_period.iloc[-1]['ask']
    self.last_close_bid = _ticks_period.iloc[-1]['bid']


    # 2024/12/9 pos_open_priceに対するTP/LC priceを格納する
    self.takeprofit_price = 0.0
    self.losscut_price = 0.0
    # 直近のAccount情報を取得する
    # 休場などにより、前日のaccountデータが存在しない場合は、データ存在するaccountデータをコピーする
    if self.utc_from not in self.df_account.index:
      # データの入っている直近のaccountデータの日付を取得する
      last_dt = self.df_account.iloc[-1].name
      # last_dtからutc_fromまでのdate_rangeを取得して、平日については値の入っているaccountデータをコピーする
      for bd in pd.date_range(start=last_dt, end=self.utc_from,freq=PERIOD_FREQ_DICT[self.period]):
        if bd.weekday() < 5:
          self.df_account.loc[bd] = self.df_account.loc[last_dt]

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

        # TP\LCの評価
        # self.ticks_frameから最初にTC/LCの水準を超えたtickを抽出する
        # 2026/3/15 Copilotによるパフォーマンスチューニング
        # PandasDataframeのqueryは使わない
        '''
        self.ticks_tplc = _ticks_period.query('bid <= %f | bid >= %f' %(self.losscut_price,self.takeprofit_price))
        self.ticks_tplc.sort_index(inplace=True)
        logger.debug('Account.CheckTpLc:Count of rows:%s %d' %(self.utc_from,len(self.ticks_tplc)))
        if len(self.ticks_tplc) > 0:
          self.tplc_bid = self.ticks_tplc.iloc[0,self.ticks_tplc.columns.get_loc('bid')]
          if self.tplc_bid <= self.losscut_price:
            # LCにかかった時には、直近のAccountに利益を追加して保有ポジションを解除する
            logger.debug('Account.CheckTpLc:Loss cut was executed.(10)')
            self.__execLossCut()
          elif self.tplc_bid >= self.takeprofit_price:
            # TPにかかった時には、直近のAccountに利益を追加して保有ポジションを解除する
            logger.debug('Account.CheckTpLc:Take profit was executed.(10)')
            self.__execTakeProfit()
          else:
            # どちらにも引っかからない場合は、何もしない。ここには来ないはず
            logger.debug('Account.CheckTpLc:Neither Take profit nor Loss cut was executed.(10)')
            pass
        '''
        # ---- pandas.query を使わず numpy で高速判定 ----
        # 1角時間足分のtick dataframe(_ticks_period)からbidの値をnumpyに抜き出す
        bids = _ticks_period['bid'].to_numpy()
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
        # self.ticks_frameから最初にTC/LCの水準を超えたtickを抽出する
        # self.ticks_tplc = self.ticks_frame.query('ask <= %f | ask >= %f' %(self.takeprofit_price,self.losscut_price))
        # 2026/3/15 Copilotによるパフォーマンスチューニング
        # PandasDataframeのqueryは使わない
        '''
        self.ticks_tplc = _ticks_period.query('ask <= %f | ask >= %f' %(self.takeprofit_price,self.losscut_price))
        self.ticks_tplc.sort_index(inplace=True)
        logger.debug('Account.CheckTpLc:Count of rows:%s %d' %(self.utc_from,len(self.ticks_tplc)))
        if len(self.ticks_tplc) > 0:
          self.tplc_ask = self.ticks_tplc.iloc[0,self.ticks_tplc.columns.get_loc('ask')]

          if self.tplc_ask >= self.losscut_price:
            # LCにかかった時には、直近のAccountに利益を追加して保有ポジションを解除する
            logger.debug('Account.CheckTpLc:Loss cut was executed.(20)')
            self.__execLossCut()
          elif self.tplc_ask <= self.takeprofit_price:
            # TPにかかった時には、直近のAccountに利益を追加して保有ポジションを解除する
            logger.debug('Account.CheckTpLc:Take profit was executed.(20)')
            self.__execTakeProfit()
          else:
            # どちらにも引っかからない場合は、何もしない。ここには来ないはず
            logger.debug('Account.CheckTpLc:Neither Take profit nor Loss cut was executed.(20)')
            pass
        '''
        asks = _ticks_period['ask'].to_numpy()
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
        # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
        #                                   +'Account.CheckTpLc:Position opened but no flags.')
        ## ForComandline
        # 標準出力で999を返す
        print(999)
        sys.exit("Account.CheckTpLc:Position opened but no flags.")

      logger.debug("Account.CheckTpLc:%s takeprofit_price=%.3f, losscut_price=%.3f, last_close_price=%.3f, float_pl=%.3f" \
                   %(self.utc_to, self.takeprofit_price, self.losscut_price, self.last_close_price, self.float_pl))
      # logger.info("Account.CheckTpLc:max_profit=%.3f, max_loss=%.3f, float_pl=%.3f" %(self.max_profit, self.max_loss,self.float_pl))

    # 直近足を更新する
    self.df_account.loc[self.utc_from,['pos_open_price','has_long','has_short','float_pl','close_pl']]\
    = (self.pos_open_price, self.has_long, self.has_short, self.float_pl, self.close_pl)
    # logger.debug('Account.CheckTpLc:df_account(%s)\n%s\n-----'%(self.utc_from,self.df_account.loc[self.utc_from]))
    return

  # countdownを計算して、当該時間のAccount情報に追加する
  def CalcCountdown(self, cntdt):
    self.countdown_datetime = cntdt
    self.countdown = (self.countdown_datetime - self.start_period).total_seconds() / (self.end_trade - self.start_period).total_seconds()
    logger.debug('Account.CalcCountdown:Countown=%.4f' %(self.countdown))
    # 当該足をのカウントダウンを更新する
    self.df_account.loc[self.countdown_datetime,['countdown']] = (self.countdown)
    return

  # DataFrame Accountに格納されている値を返す。カラムを指定した場合はそのカラムだけを返す
  def GetAccountInfo(self, idx_dt, cols=None):
    if cols is None:
      cols = self.df_account.columns

    return self.df_account.loc[idx_dt,cols]

  def GetAccountColLen(self):
    return len(self.df_account.columns)

  # このAccountオブジェクトのトレード開始、終了datetimeを返す
  def GetTradePeriod(self):
    return (self.start_period, self.end_period)

  def EvaluateRewrd(self, actn_idx, test_dt, last_dt, df_price_data):

    no_action = 0
    long_entry = 1
    short_entry = 2
    position_close = 3
    close_and_long = 4
    close_and_short = 5

    utc_from = test_dt
    utc_to = utc_from + datetime.timedelta(minutes=1)
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
      # PandasDataframeのqueryを使わない
      # _ticks_first = self.ticks_frame.query('@utc_from<=index<@utc_to')
      # self.bar_to_tick_minuteにキーが存在していない場合は、1分進める
      if utc_from in self.bar_to_tick_minute:
        start_idx, end_idx = self.bar_to_tick_minute[utc_from]
      else:
        logger.warning('Account.EvaluateRewrd:No data in self.bar_to_tick_minute. From %s' 
                       %(utc_from.strftime('%Y-%m-%d %H:%M')))
        utc_from = utc_from + datetime.timedelta(minutes=1)
        utc_to = utc_from + datetime.timedelta(minutes=1)

        continue

      # if len(_ticks_first) == 0:
      if start_idx==end_idx:
        # tickが取得できなかった場合は、１分進める
        logger.warning('Account.EvaluateRewrd:No ticks. From %s To %s' 
                       %(utc_from.strftime('%Y-%m-%d %H:%M'),utc_to.strftime('%Y-%m-%d %H:%M')))

        # 2026/8/18 試行回数の最後の1回でもtickが取得できない場合はアクションごとに振舞いを変える
        if(n == MAX_RETRY_NUM[self.period]-1):
          # no_action position_closeの時はそのままなので何もしない
          # long_entry,short_entryの時はエントリーできないので no_actionに変更
          if actn_idx == long_entry or actn_idx == short_entry:
            action_idx = no_action
          # close_and_long,close_and_shortの時はクローズだけするので position_close に変更
          elif actn_idx == close_and_long or actn_idx == close_and_short:
            action_idx = position_close

        utc_from = utc_from + datetime.timedelta(minutes=1)
        utc_to = utc_from + datetime.timedelta(minutes=1)

        continue

      _ticks_first = self.ticks_frame.iloc[start_idx:end_idx]
      logger.debug('Account.EvaluateRewrd:From %s To %s Count of ticks_first %d' 
                   %(utc_from.strftime('%Y-%m-%d %H:%M'),utc_to.strftime('%Y-%m-%d %H:%M'),len(_ticks_first)))



      [sell_price,buy_price]=_ticks_first.loc[_ticks_first.index[0],['bid','ask']]
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
        utc_from = utc_from + datetime.timedelta(minutes=1)
        utc_to = utc_from + datetime.timedelta(minutes=1)
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
      # 2026/8/19 bug fix: buy_open_price が0.0の時はエントリーしない（できない）
      # ここに来るときは、ポジションも損益も0のはずなのでdf_accountは空にする
      if(buy_open_price == 0.0):
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
      else:
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [test_dt, buy_open_price, self.pos_scale, 0.0, 0.0, 0.0]
    elif(actn_idx == short_entry):
      logger.debug('Account.EvaluateRewrd:Action_index=%s, Short Entry.' %(actn_idx))
      # account(2:00)に、次の値を格納する
      # pos_open_datetime = test_dt(2:00), pos_open_price = price_data(2:00).open, has_position = -1, float_pl = 0.0, close_pl = 0.0
      # pos_open_dataにはBidを格納する
      # longとshortを分割する
      # 2026/8/19 bug fix: sell_open_price が0.0の時はエントリーしない（できない）
      # ここに来るときは、ポジションも損益も0のはずなのでdf_accountは空にする
      if(sell_open_price == 0.0):
        self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
           = [None, 0.0, 0.0, 0.0, 0.0, 0.0]
      else:
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
        # 2026/08/18 bug fix:buy_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        if(buy_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
            = [None, 0.0, 0.0, 0.0 , 0.0, self.df_account.loc[last_dt,'float_pl']] 
        else:
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
            = [None, 0.0, 0.0, 0.0 , 0.0
               , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]
      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        # 2026/08/18 bug fix:sell_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        if(sell_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
            = [None, 0.0, 0.0, 0.0 , 0.0, self.df_account.loc[last_dt,'float_pl']]
        else:
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
        # 2026/08/18 bug fix:buy_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        # 2026/08/19 bug fix:closeとentryの処理を分離する
        if(buy_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0, self.df_account.loc[last_dt,'float_pl']]
        else:
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0
                , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]
        # 2026/8/19 bug fix: buy_open_price が0.0の時はエントリーしない（できない）
        # ただし、クローズした損益は残す
        if(buy_open_price > 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']]\
             = [test_dt, buy_open_price, self.pos_scale]
      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        # 2026/08/18 bug fix:sell_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        # 2026/08/19 bug fix:closeとentryの処理を分離する
        if(sell_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0, self.df_account.loc[last_dt,'float_pl']]
        else:
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0
                , (self.df_account.loc[last_dt,'pos_open_price'] - sell_close_price) * self.magnification]
        # 2026/8/19 bug fix: buy_open_price が0.0の時はエントリーしない（できない）
        # ただし、クローズした損益は残す
        if(buy_open_price > 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long']]\
             = [test_dt, buy_open_price, self.pos_scale]
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
        # 2026/08/18 bug fix:buy_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        # 2026/08/19 bug fix:closeとentryの処理を分離する
        if(buy_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0, self.df_account.loc[last_dt,'float_pl']]
        else:
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0
                , (buy_close_price - self.df_account.loc[last_dt,'pos_open_price']) * self.magnification]
        # 2026/8/19 bug fix: buy_open_price が0.0の時はエントリーしない（できない）
        # ただし、クローズした損益は残す
        if(sell_open_price > 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']]\
             = [test_dt, sell_open_price, self.pos_scale]
      if(self.df_account.loc[last_dt,'has_short'] > 0.0):
        # Short positionを持っている場合
        # sell_close_price は ask で評価する
        # 2026/08/18 bug fix:sell_close_price==0.0の時（tickが取得できなかった時）、便宜上float_plでクローズさせる
        # 2026/08/19 bug fix:closeとentryの処理を分離する
        if(sell_close_price == 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0, self.df_account.loc[last_dt,'float_pl']]
        else:
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_long','has_short', 'float_pl', 'close_pl']]\
             = [None, 0.0, 0.0, 0.0, 0.0
                , (self.df_account.loc[last_dt,'pos_open_price'] - sell_close_price) * self.magnification]
        # 2026/8/19 bug fix: buy_open_price が0.0の時はエントリーしない（できない）
        # ただし、クローズした損益は残す
        if(sell_open_price > 0.0):
          self.df_account.loc[test_dt, ['pos_open_datetime', 'pos_open_price', 'has_short']]\
             = [test_dt, sell_open_price, self.pos_scale]
    else:
      # 定義されていないので、エラー
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                    +'Account.EvaluateRewrd:Action is not defined.')
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      sys.exit('Account.EvaluateRewrd:Action is not defined.')

    return

  def DropAccount(self):
    self.df_account = self.df_account.drop(self.df_account.index)
    logger.debug('Account.DropAccount:df_account %s' %(self.df_account))

  def SaveIniFile(self, filepath):
    with open(filepath , mode='w') as f:
      f.write('[COMMOM]\n')
      f.write('LOSSCUT_LEVEL=' + str(self.lc_level) + '\n')
      f.write('TAKEPROFIT_LEVEL=' + str(self.tp_level) + '\n')
      f.write('LONG_PERIOD='+str(self.long_period)+'\n')
      f.write('SHORT_PERIOD='+str(self.short_period)+'\n')
      f.write('TICKVOL_PERIOD='+str(self.tickvol_period)+'\n')
      f.write('POS_SCALE='+str(self.pos_scale)+'\n')
      f.write('TRAIN_SYMBOL_LIST='+str(TRAIN_SYMBOL_LIST)+'\n')
      f.write('REAL_SPREAD_LIMIT='+str(self.real_sprad_limit)+'\n')
      f.write('PRE_TRADE_WEEKS='+str(self.pre_trade_weeks)+'\n')
    return

  def SaveSetFile(self, mdlname):
    with open(mdlname.set_file_path , mode='w') as f:
      f.write('MODEL_NAME='+mdlname.model_name+'\n')
      f.write('LC_LEVEL=' + str(self.lc_level) + '\n')
      f.write('TP_LEVEL=' + str(self.tp_level) + '\n')
      f.write('LONG_PERIOD='+str(self.long_period)+'\n')
      f.write('SHORT_PERIOD='+str(self.short_period)+'\n')
      f.write('TICKVOL_PERIOD='+str(self.tickvol_period)+'\n')
      f.write('POS_SCALE='+str(self.pos_scale)+'\n')
      f.write('MODEL_INPUT_NUM='+str(MODEL_INPUT_NUM)+'\n')
      f.write('MODEL_HIDDEN_NUM='+str(MODEL_HIDDEN_NUM)+'\n')
      f.write('MODEL_OUTPUT_NUM='+str(MODEL_OUTPUT_NUM)+'\n')
      f.write('TRAIN_SYMBOL_LIST='+str(TRAIN_SYMBOL_LIST)+'\n')
      f.write('MAGIC='+mdlname.magic+'\n')
      f.write('TRADE_SYSTEM='+TRADE_SYSTEM+'\n')
      f.write('REAL_SPREAD_LIMIT='+str(self.real_sprad_limit)+'\n')
      f.write('PRE_TRADE_WEEKS='+str(self.pre_trade_weeks)+'\n')
    return

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

  # トレード期間のtick dataをDBから取得してDataFrameに格納する
  # 2026/6/6 SQLAlchemyの代わりにConnctorXを使う
  def SetTickDataPeriod(self, start_date, end_date):
    tick_copy_flg = False

    schema_tbl = str.lower(SCHEMA+'.pricedata_'+self.symbol+'_tick')
    # SELECT * は使わない
    sql="SELECT bid, ask, time_msc FROM "+schema_tbl+" WHERE time_msc>='"+start_date.strftime('%Y-%m-%d %H:%M') \
        +"' AND time_msc<'"+end_date.strftime('%Y-%m-%d %H:%M')+"' ORDER BY time_msc ASC"

    for i in range(10):
      '''
      with ENGINE.connect() as conn:
        schema_tbl = str.lower(SCHEMA+'.pricedata_'+self.symbol+'_tick')
        sql="SELECT * FROM "+schema_tbl+" WHERE time_msc>='"+start_date.strftime('%Y-%m-%d %H:%M') \
          +"' AND time_msc<'"+end_date.strftime('%Y-%m-%d %H:%M')+"' ORDER BY time_msc ASC"
        self.ticks_frame = pd.read_sql(sql=text(sql), con=conn)
      '''
      self.ticks_frame = cx.read_sql(query=sql, conn=CONN_URL)

      logger.debug('Account.SetTickDataPeriod:From %s To %s Count of ticks_frame %d' 
                   %(start_date.strftime('%Y-%m-%d %H:%M'),end_date.strftime('%Y-%m-%d %H:%M'),len(self.ticks_frame)))

      if len(self.ticks_frame) > 0:
        tick_copy_flg = True
        break
      else:
        continue
    if not tick_copy_flg:
      # コピーに失敗したらエラー
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                   +'Account.SetTickDataPeriod:Copy Tick Range Failed.\n'
      #                                   +'From %s To %s Count of ticks_frame %d'
      #                                   %(start_date.strftime('%Y-%m-%d %H:%M'),end_date.strftime('%Y-%m-%d %H:%M'),len(self.ticks_frame)))
      # sys.exit("Account.SetTickDataPeriod:Copy Tick Range Failed.")
      # 2026/8/21 コピーに失敗した場合はシステムエラーとせずにFalseを返す
      logger.warning("Account.SetTickDataPeriod:Copy Tick Range Failed.")
      return(False)

    # 秒での時間をdatetime形式に変換する
    # DBにはtime_mscはTimestamp型で格納されている。DBからデータ取得時にソートされている
    # self.ticks_frame['time']=pd.to_datetime(self.ticks_frame['time'], unit='s')
    # self.ticks_frame['time_msc']=pd.to_datetime(self.ticks_frame['time_msc'], unit='ms')
    self.ticks_frame = self.ticks_frame.set_index('time_msc')
    # self.ticks_frame.sort_index(inplace=True)
    return(True)

  def DropTickDataPeriod(self):
    if self.ticks_frame is not None:
        self.ticks_frame.drop(self.ticks_frame.index, inplace=True)
    return

  def GetPreTradeWeeks(self):
    return self.pre_trade_weeks


# # Package-Agnent
# - 取引のアクションを決めるパッケージ以下の機能を実装する
#  -  Environmentから加工された価格データを受け取りLSTMに渡す
#  - LSTMのアクションをEnvironmentへ渡す
#  - DQLを行う

# ## Dependency

# In[27]:


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

# In[28]:


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
# ・LSTMのニューラルネットを構築する<br>
# ・与えられた状態から行動を決定する<br>
# ・次の行動から得られた報酬とその時の状態からその次の行動関数を算出し、誤差を補正する<br>
# 
# ###input_data(状態S[seq,8])
# s0 = 現在の損益(pips/1000) 例えば101.000USDJPY - 100.000USDJPY=1000pips<br>
# s1 = LongPosition (0=none, 1=entry)<br>
# s2 = ShortPosition (0=none, 1=entry)<br>
# s3 = Weekday(0/6 = Monday, 6/6 = Sunday)<br>
# s4 = Countdown：毎週月曜日の0:00足を1、金曜日の22:00足を0となるようにする。金曜日の23:00足はマイナスの値となる。<br>
# s5 = Close-Open<br>
# s6 = High-Low<br>
# s7 = Tickvol(1/100000)<br>
# ※今後、複数symbolをインプットとする場合は、s5-s7のセットを追加する。
# 
# ###output_data(行動A[seq,6])
# a0:何もしない<br>
# a1:LongEntry<br>
# a2:ShortEntry<br>
# a3:PositionClose<br>
# a4:PositionClose&LongEntry<br>
# a5:PositionClose&ShortEntry<br>
# ※両建てはしない。ドテンを想定する。

# In[29]:


#@title class Agent { output-height: 200 }
# hidden_num = 8 #@param {type:"integer"}

class Agent():

  def __init__(self,sbl,prd,input_num,output_num,ini_datetime,LoadModel=False, FilePath=None, train_mode=True):
    self.hidden_num = input_num
    self.symbol = sbl
    self.period = prd
    self.train_mode = train_mode

    #引数の各素子数に応じたニューラルネットを作成する
    # DDQNに対応する
    self.main_brain = Brain(input_num,output_num,self.hidden_num)
    self.target_brain = Brain(input_num,output_num,self.hidden_num)
    #これまでのトレーニングパラメータを呼び出す
    if(LoadModel and FilePath.brain != None):
      self.main_brain.loadModel(FilePath.brain)

    self.target_brain.copyNN(self.main_brain.state_dict())

    self.device = self.main_brain.GetDevice()

    #Rewardを格納するデータフレームを作成する
    # 2026/2/14 DataFrameはログ用のデータを格納するためのもの
    # Tensor計算用に辞書を用意する
    self.df_reward = pd.DataFrame(
        columns=['symbol','period','reward','q_max','a_max','q_taken','a_taken','exp_s_a']
    )
    # 初期行は不要。必要になったときに行を追加する方が自然。

    # 学習用 Tensor を保持する辞書
    # この辞書の形はtensor_reward={set_datetime:{'reward':xx,'q_max':xx,'a_max':xx,'q_taken':xx,'a_taken':xx,'exp_s_a':xx}}
    # のように、set_datetimeをキーとして入れ子になっている辞書を取り出す感じ（のはず。Copilotの提案）
    self.tensor_reward = {}

    # float_plに対する評価を0~1の倍でRandomに設定する
    if LoadModel and (FilePath.agent != None) :
      logger.debug('Agent.__init__: INIFile loading. \n %s' %(FilePath.agent))
      #Load FlagがTrueでファイルパスが指定されている場合は、そのファイルを読み込む
      agt_inifile = configparser.ConfigParser()
      agt_inifile.read(FilePath.agent)
      self.est_float_pl = float(agt_inifile.get('COMMOM', 'ESTIMATE_FLOAT_PL'))
    else:
      logger_agent.debug('Agent.__init__: ESTIMATE_FLOAT_PL is set.')
      self.est_float_pl = random.random()

    logger_agent.debug('Agent.__init()__: ESTIMATE_FLOAT_PL= %.3f' %(self.est_float_pl))

  def SetReward(self, set_datetime, df_float_close_pl):
    logger_agent.debug('Agent.SetReward:float_pl=%.3f, close_pl=%.3f, est_float_pl=%.3f' \
                 %(df_float_close_pl.float_pl, df_float_close_pl.close_pl, df_float_close_pl.float_pl * self.est_float_pl))

    # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
    # Tensor学習用の辞書とログ用のDetaFrameにそれぞれ格納する
    # reward を Tensor として計算
    prev_reward = 0.0
    if set_datetime in self.df_reward.index:
        # prev_reward = self.df_reward.loc[set_datetime, 'reward']
        prev_reward = self.tensor_reward[set_datetime]["reward"]

    reward_tensor = torch.tensor(prev_reward, device=self.device) + \
        (df_float_close_pl.float_pl * self.est_float_pl + df_float_close_pl.close_pl)

    # Tensor を保存（計算グラフ保持）
    if set_datetime not in self.tensor_reward:
        self.tensor_reward[set_datetime] = {}
    # reward の shape 統一
    reward_tensor = reward_tensor.view(1)
    self.tensor_reward[set_datetime]["reward"] = reward_tensor

    # ログ用
    self.df_reward.loc[set_datetime, 'symbol'] = self.symbol
    self.df_reward.loc[set_datetime, 'period'] = self.period
    self.df_reward.loc[set_datetime, 'reward'] = reward_tensor.item()

    return

  def CalcExpectStateActionValue(self, set_datetime):
    # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
    # Tensorを計算する辞書とログ用のDataframeを分ける
    # 教師データとなる R + γmax(Q(s_t+1,a))を計算してTensor辞書に格納する
    # rt = self.tensor_reward[set_datetime]
    rt = self.tensor_reward.setdefault(set_datetime, {})

    # exp_s_a を Tensor として計算（計算グラフ保持）
    # q_maxが存在しない場合はq_maxをtensor([0.0])にする
    # 併せてq_maxにfloat（スカラー）が入っている場合はTensor([value]) に変換
    # rt.setdefault("q_max", 0.0)
    if "q_max" not in rt:
        rt["q_max"] = torch.tensor([0.0], device=self.device)
    elif isinstance(rt["q_max"], float):
        rt["q_max"] = torch.tensor([rt["q_max"]], device=self.device)

    # reward も shape=[1] に統一
    if "reward" not in rt:
        rt["reward"] = torch.tensor([0.0], device=self.device)
    elif isinstance(rt["reward"], float):
        rt["reward"] = torch.tensor([rt["reward"]], device=self.device)

    # exp_s_a を Tensor として計算
    exp_s_a_tensor = rt["reward"] + GAMMA * rt["q_max"]
    # shape=[1] を保証
    exp_s_a_tensor = exp_s_a_tensor.view(1)
    # 保存
    rt["exp_s_a"] = exp_s_a_tensor

    # TensorDict に保存← 再代入は不要（rt は参照）だが、明示的に書いても OK
    self.tensor_reward[set_datetime] = rt
    # DataFrame には float を保存
    self.df_reward.loc[set_datetime, 'exp_s_a'] = exp_s_a_tensor.item()

    logger_agent.debug('Agent.CalcExpectStatuActionValue:exp_s_a=%.4f' % exp_s_a_tensor.item())

  def DecideAction(self, set_datetime, input_data, action_mask, epi_num):
    # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
    # 計算用Tensorを格納する辞書とログ用のDataframeに分ける
    # input_dataを用いて、NNからすべてのアクションの行動評価関数Qを取得する
    self.main_state_action_values = self.main_brain(input_data)
    self.target_state_action_values = self.target_brain(input_data)
    logger_agent.debug('Agent.DecideAction:main_state_action_values\n%s\n------'%(self.main_state_action_values))
    logger_agent.debug('Agent.DecideAction:target_state_action_values\n%s\n------'%(self.target_state_action_values))

    # DDQNに対応させる。main_brain(main Q-network)とtarget_brain(target Q-network)からそれぞれQ値
    # Q_m(s_t,a),Q_t(S_t,a)を取得する
    # 現在の状態から取り得る行動と、状態関数が最大の行動を取得する
    # ε-greedy法により、次の行動が状態関数の最大とは限らない
    # DDQN対応。mani_brain(main Q-network)から得られたaction_indexと、target_brain(target Q-network)から得られたmax_action_indexを使用する
    (self.acition_index, _) = self.main_brain.decideAction(action_mask, self.main_state_action_values, epi_num, self.train_mode)
    (_, self.max_action_index) = self.target_brain.decideAction(action_mask, self.target_state_action_values, epi_num, self.train_mode)

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

    return self.acition_index

  def UpdateMainNN(self, output, target):
    self.main_brain.evaluateLossFunction(output,target)
    self.main_brain.updateNN()

  def CopyMainNNToTargetNN(self):
    self.target_brain.copyNN(self.main_brain.state_dict())

  def ResetHiddenCellState(self):
    # Main Q-NNのHiddenCellStateをリセット
    (hdn, cel) = self.main_brain.getHiddenCellState()
    logger_agent.debug('Agent.ResetMainHiddenCellState:Before [HiddenState]%s, [CellState]%s' %(hdn, cel))
    self.main_brain.resetHiddenCellState()
    (hdn, cel) = self.main_brain.getHiddenCellState()
    logger_agent.debug('Agent.ResetMainHiddenCellState:After [HiddenState]%s, [CellState]%s' %(hdn, cel))
    # Target Q-NNのHiddenCellStateをリセット
    (hdn, cel) = self.target_brain.getHiddenCellState()
    logger_agent.debug('Agent.ResetTargetHiddenCellState:Before [HiddenState]%s, [CellState]%s' %(hdn, cel))
    self.target_brain.resetHiddenCellState()
    (hdn, cel) = self.target_brain.getHiddenCellState()
    logger_agent.debug('Agent.ResetTargetHiddenCellState:After [HiddenState]%s, [CellState]%s' %(hdn, cel))

  def DropReward(self):
    self.df_reward = self.df_reward.drop(self.df_reward.index)
    # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
    self.tensor_reward = {}   # ← TensorDict もクリア

    logger_agent.debug('Agent.DropReward:df_reward %s' %(self.df_reward))

  def SaveAgentModel(self, AgentFilePath, BrainFilePath,eval_value):
      with open(AgentFilePath , mode='w') as f:
        f.write('[COMMOM]\n')
        f.write('ESTIMATE_FLOAT_PL = ' + str(self.est_float_pl) + '\n')
        f.write('EVAL_VALUE = ' + str(eval_value) + '\n')

      self.main_brain.saveModel(BrainFilePath)

  def SetModelMode(self, train_mode):
    self.train_mode = train_mode
    self.main_brain.train(train_mode)
    self.target_brain.train(train_mode)

  def GetMainHiddenCellState(self):
    return self.main_brain.getHiddenCellState()

  def GetTargetHiddenCellState(self):
    return self.target_brain.getHiddenCellState()

  def GetOutput(self):
    return self.main_brain.getOutput()


# ## クラス Brain
# Agent内のニューラルネット部分をBrainクラスとして別途定義

# In[30]:


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
    self.resetHiddenCellState()
    # 損失関数に渡すデータの中身を初期化する
    # self.clearLossData()

    #損失関数と最適化手法を定義する
    #損失関数は、（ひとまず）SmoothL1Loss(state_action_values(=output),expected_state_action_values(=target))を採用
    #optimiserは、optim.Adam(self.model.parameters(), lr=0.0001)
    self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
    self.criterion = nn.SmoothL1Loss()

    # .setファイルに書き込む素子数をグローバル変数に代入する
    global MODEL_INPUT_NUM, MODEL_OUTPUT_NUM, MODEL_HIDDEN_NUM
    MODEL_INPUT_NUM = self.input_num
    MODEL_OUTPUT_NUM = self.output_num
    MODEL_HIDDEN_NUM = self.hidden_num
    logger_agent.debug('Brain.__init__:MODEL_INPUT_NUM=%s, MODEL_OUTPUT_NUM=%s, MODEL_HIDDEN_NUM=%s' %(MODEL_INPUT_NUM, MODEL_OUTPUT_NUM, MODEL_HIDDEN_NUM))

    return

  # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
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


# # Package-Trainer
# - NNをトレーニングするための機能を実装する

# ## Logger
# - TrainerパッケージのLoggerインスタンスはlogger_trainerとする

# In[31]:


#@title Logger Trainer
import logging
import logging.config

# logging.config.fileConfig('./drive/My Drive/Colab Notebooks/02_DRLTradingSystem2020/Logging.ini')
# logging.config.fileConfig(LOGGING_INIFILE_PATH)
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger_trainer = logging.getLogger('DRLTrainer')
logger_trainer.debug('Debug level massage.')
logger_trainer.info('Info level massage.')
logger_trainer.warning('Warning level massage.')
logger_trainer.error('Error level massage.')
logger_trainer.critical('Critical level massage.')


# ## クラス Trainer

# In[39]:


#@title class Trainer
class Trainer():

  def __init__(self, epinum, sbl, tf, start_dt, tdur, gnum, anum, train_data, periods_dict, load_flg, filepth=None, prds=None, cp_frq=2, grd='X', mgc='yyyymmddHHMMSS',rsf=False):

    self.epi_num = epinum
    self.symbol = sbl
    self.period = tf
    self.start_date_time = start_dt
    self.test_duration = tdur
    self.gen_num = gnum
    self.agent_num = anum
    self.load_flg = load_flg
    self.file_path = filepth
    self.periods = prds
    self.copy_frq = cp_frq
    self.magic = mgc
    self.real_spread_flg = rsf
    # 2026/2/14 Copilotの助けにより大幅なリファクタリング
    self.periods_dict = periods_dict
    self.train_data_dict = { self.period: train_data }

    # Trainerの初期化
    logger_trainer.debug('◆◆%s Initialize Trainer ◆◆' %(self.start_date_time))
    # トレーニングの結果(P/Lの合計と標準偏差)を格納するDataFrame
    self.df_result = pd.DataFrame(columns=['tradenum','sum','mean','sd'])

    # Accountを初期化する
    self.acnt = Account(self.symbol,self.period,self.start_date_time, self.test_duration, self.load_flg, self.file_path, self.periods,self.real_spread_flg)
    # 2025/3/15 「空トレード」を行う期間(週数)は、agentから取得する
    self.pre_trade_weeks = self.acnt.GetPreTradeWeeks()
    logger_trainer.debug('Trainer.__init__:pre_trade_weeks=%s' %(self.pre_trade_weeks))
    # オブジェクトのトレード期間(start_date_time(0:00),end_date_time(23:00))を取得する
    self.start_date_time, self.end_date_time = self.acnt.GetTradePeriod()
    logger_trainer.debug('start_date_time:%s, end_date_time:%s' %(self.start_date_time, self.end_date_time))

    # Agentを初期化する
    # この時、input_numとしてAccount('has_long','has_short','float_pl','countdown')とTrainDataの長さを渡す必要がある
    self.delta_time = EnvironmentCommon.GetPeriodTimeDelta(self.period)
    self.input_num = train_data.GetTrainDataColLen() + 4
    self.output_num = 6
    self.agnt = Agent(self.symbol,self.period,self.input_num,self.output_num,self.start_date_time,self.load_flg,self.file_path)

    # トレード期間(Mon 1:00-Fri 23:00)のprace_dataをまとめて取得する
    # 2026/8/21 tickデータがないなどで UpdateTradePeriod が失敗した場合は処理を中断する
    if(not self.UpdateTradePeriod(self.start_date_time, self.test_duration)):
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      logger_trainer.error('Trainer.__init()__:Some problem has occred in UpdateTradePeriod.')
      sys.exit('Training halted.') 

    # 詳細のテスト結果(acnt.df_account)を格納する
    self.df_detail_testresult = pd.DataFrame()

    # 詳細のテストの情報(input,hidden,cellstate,output,maskoutput)を書き出すファイル名
    self.now_date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    self.input_filename = TRAIN_DETAIL_INPUT_FILE_NAME+'_'+self.symbol+'_'+self.period+'_'+self.magic+'_'+self.now_date_time+'.csv'
    self.hidden_filename = TRAIN_DETAIL_HIDDEN_FILE_NAME+'_'+self.symbol+'_'+self.period+'_'+self.magic+'_'+self.now_date_time+'.csv'
    self.cellstate_filename = TRAIN_DETAIL_CELLSTATE_FILE_NAME+'_'+self.symbol+'_'+self.period+'_'+self.magic+'_'+self.now_date_time+'.csv'
    self.output_filename = TRAIN_DETAIL_OUTPUT_FILE_NAME+'_'+self.symbol+'_'+self.period+'_'+self.magic+'_'+self.now_date_time+'.csv'
    self.maskoutput_filename = TRAIN_DETAIL_MASKOUTPUT_FILE_NAME+'_'+self.symbol+'_'+self.period+'_'+self.magic+'_'+self.now_date_time+'.csv'

  # 2025/3/16 NNのhidden_cell stateをリセットしないフラグを追加する→やはりなし
  def TrainAgent(self, epinum=-1, train_mode=True, return_detail=False):
    # epinumが-1より大きい場合は、引数を代入する
    if epinum > -1:
      self.epi_num = epinum
    # モデルのモード(トレーニング/推論)の設定
    self.agnt.SetModelMode(train_mode)
    # トレーニングの開始
    logger_trainer.info('Trainer.TrainAgent:◆◆TrainAgent Start from %s to %s◆◆' %(self.start_date_time,self.end_date_time))
    # Backtestの場合は、1周だけ
    for self.enum in range(self.epi_num):
      # 「空トレード」を行う。空トレード期間の月曜リストを取得する
      from_date = self.start_date_time - datetime.timedelta(weeks=self.pre_trade_weeks)
      to_date = self.start_date_time - datetime.timedelta(weeks=1)
      every_monday = pd.date_range(from_date, to_date, freq='W-MON')

      # １週間ごとに「空トレード」をする。トレード処理はプライベートメソッドに書き出す
      # AccountDatFrameとRewardDataFrameは都度破棄する
      # NNの更新はしない
      for w in every_monday:
        not self._ExecTrade(w,train_mode,return_detail,pretrade_flg=True)
      # トレード・トレーニングを実施(１週間分)
      self._ExecTrade(self.start_date_time,train_mode,return_detail,pretrade_flg=False)

    # すべてのトレーニングが終了したら結果を返す
    if return_detail:
      return self.df_result, self.df_detail_testresult
    else:
      return self.df_result

  def _ExecTrade(self,stdt,train_mode,return_detail,pretrade_flg):
    if pretrade_flg:
      logger_trainer.info('Trainer._ExecTrade:◇◇PreTraning Epsorde %d Start on %s◇◇' %(self.enum+1,stdt))
    else:
      logger_trainer.info('Trainer._ExecTrade:◆◆Traning Epsorde %d Start on %s◆◆' %(self.enum+1,stdt))
    # 2026/8/21 tickデータ不備などによりテストデータがセットできないときは処理を中断する（トレーニング中止）
    if(not self.UpdateTradePeriod(stdt, self.test_duration)):
      # トレーニングを終了させる
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      logger_trainer.error('Trainer._ExecTrade:While UpdateTradePeriod, some problem has occured.')
      sys.exit('Training halted.') 
    lstdt = self._GetLastDateTime(stdt, self.delta_time)
    tstdt = stdt
    self.acnt.SetAccount(lstdt)
    self.agnt.SetReward(lstdt, self.acnt.GetAccountInfo(lstdt,['float_pl','close_pl']))

    # 2026/2/14 Copilotの力を借りて大幅リファクタリング
    price_obj = self.periods_dict[self.period][self.symbol]

    # 2023/06/03 実際にテストをするのは、スタート期間のひとつ前の足から
    for lstdt, self.df_train in self.df_trade_data_period.iterrows():
      # ここで、時間の調整
      # 現在時刻(t)のひとつ前(t-1)の時系列データを扱うときのdataetime
      # tstdt(test_date_time) → t, lstdt(last_date_time) → t-1
      # tstdt = self._GetNextDateTime(lstdt, self.delta_time)

      logger_trainer.debug('Trainer.TrainAgent:last_date=%s(weekday:%d)' %(lstdt,lstdt.isoweekday()))
      logger_trainer.debug('Trainer.TrainAgent:◆◆Traing test_date%s (weekday:%d)◆◆' %(tstdt,tstdt.isoweekday()))

      # 1.直近足(t-1)の価格データを使ってTP,LCのチェックを行う
      if price_obj.GetPriceData(lstdt).empty:
        logger_trainer.warning('Trainer.TrainAgent:[Empty PriceData]Traing %s skipped.' %(lstdt))
        continue

      self.acnt.CheckTpLc(lstdt, price_obj.GetPriceData(lstdt))

      # 2.Account情報(t-1)の確定損益(close_pl(t-1))と評価損益(float_pl(t-1))をReward(t)として登録する
      # 例：close_pl(1:00),float_pl(1:00)→Reward(2:00)に格納
      self.agnt.SetReward(tstdt, self.acnt.GetAccountInfo(lstdt,['float_pl','close_pl']))

      # 3. トレーニングを行い、その結果、t(2:00)におけるq_max, a_max, q_taken. a_takenを取得する
      #  3-1. t-1カウントダウンの計算
      #       23:00の時点で、22:00のカウントダウンを計算して、22:00のdf_accountに格納する(?)
      self.acnt.CalcCountdown(lstdt)
      #  3-2. t-1(1:00)のAccount情報として取得する
      #       23:00の時点で、22:00.closeのdf_accountから('has_position','float_pl','countdown')を取得する
      #       22:00のcountdownで全てクローズとさせたい
      # self.df_acnt = self.acnt.GetAccountInfo(lstdt,['has_position','float_pl','countdown'])
      # longとshortを分割
      self.df_acnt = self.acnt.GetAccountInfo(lstdt,['has_long','has_short','float_pl','countdown'])
      #  3-3. Accountの状態('has_position','float_pl','countdown')から、取り得るアクションを限定する
      # self.action_mask = EnvironmentCommon.getAvailableAction(self.df_acnt.float_pl, self.df_acnt.has_position, self.df_acnt.countdown)
      self.action_mask = EnvironmentCommon.getAvailableAction(self.df_acnt.float_pl, self.df_acnt.has_long, self.df_acnt.has_short, self.df_acnt.countdown)
      #  3-4. 直近足(t-1)のトレーニングデータを取得する
      # df_train = train.GetTrainData(last_date_time)
      #  3-5. トレーニングデータにアカウント情報を結合する
      self.df_acnt_train = pd.concat([self.df_acnt,self.df_train])
      #  3-6. トレーニングデータなどをもとに次の行動を決定する
      #       この時、q_max、a_max、q_taken、a_takenをdf_reward(t)(2:00)に格納する
      self.action_index = self.agnt.DecideAction(tstdt,self.df_acnt_train.to_list(), self.action_mask, self.enum)
      # 4.選択した行動から損益を評価する。Rewardへの格納は次の時間のループの最初(2.)に行う
      self.acnt.EvaluateRewrd(self.action_index, tstdt, lstdt, price_obj.GetPriceData(tstdt))
      # 5. 2:00のRewardと行動関数から教師データを作成する
      self.agnt.CalcExpectStateActionValue(tstdt)
      # 6. (option)トレードの詳細をファイルに書き出す。return_detail=Trueの時のみ
      if return_detail:
        self._WriteDetailFile(tstdt)

      tstdt += self.delta_time
    # end of for loop
    # Sat0:00の処理　★この処理はいらないかもしれない
    # この時は、取引をしないで直前の時間(23:00)のdf_accountのclose_plとfloat_plを
    # Evaluate Reward, Calc
    # 現時点(0:00)のRewardに格納する
    logger_trainer.debug('◆◆Final Traing on %s◆◆' %(tstdt))
    # 現在時刻(t)のひとつ前(t-1)の時系列データを扱うときのdataetime
    # test_date_time → t, last_date_time → t-1
    lstdt = self._GetLastDateTime(tstdt, self.delta_time)

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

    # 「空トレード」の時は、隠れ層の状態を確認する
    if pretrade_flg:
      hdn,cel=self.agnt.GetMainHiddenCellState()
      logger_trainer.debug('Trainer.TrainAgent:\n[HiddenState]%s \n[CellState]%s' %(hdn,cel)) 
      logger_trainer.debug('Trainer.TrainAgent:◇◇PreTraning Epsorde %d Finished on %s◇◇' %(self.enum+1,tstdt))
      logger_trainer.debug('Total P/L:%.3f' %(self.acnt.df_account.close_pl.sum()))
    else:
      # トレードの結果を評価する。
      logger_trainer.info('◆◆Training Episorde %d Finish.◆◆' %(self.enum+1))
      logger_trainer.info('Total P/L:%.3f' %(self.acnt.df_account.close_pl.sum()))
      logger_trainer.debug('Total Avg:%.3f' %(self.acnt.df_account.close_pl.mean()))
      logger_trainer.debug('Total SD:%.3f' %(self.acnt.df_account.close_pl.std()))
      # 結果をdf_resultに格納する
      self.df_result.loc[self.enum] = [(self.acnt.df_account.close_pl != 0).sum(), self.acnt.df_account.close_pl.sum(), self.acnt.df_account.close_pl.mean(), self.acnt.df_account.close_pl.std()]
      # トレーニングデータ(q_taken(mon 1:00-fri 23:00))と教師データ(exp_s_a(mon 2:00- sat 0:00))を
      # 2026/2/14 Copilotの助けを借りて大幅リファクタリング
      # 計算用のTensorとログ用のDataFrameを分離する
      # self.q_taken_tensor = torch.cat([s for s in self.agnt.df_reward.iloc[1:-1,self.agnt.df_reward.columns.get_loc('q_taken')]])
      # self.exp_s_a_tensor = torch.cat([s for s in self.agnt.df_reward.iloc[2:,self.agnt.df_reward.columns.get_loc('exp_s_a')]])

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
            logger_trainer.debug('Trainer.TrainAgent:Copy MainNN to TargetNN Execute.')
            self.agnt.CopyMainNNToTargetNN()

      # バックテスト用にdf_accountを退避させる
      if return_detail:
        self.df_detail_testresult = self.acnt.df_account.copy()
      # 隠れ層をリセットする
      self.agnt.ResetHiddenCellState()
      hdn,cel=self.agnt.GetMainHiddenCellState()
      logger_trainer.debug('Trainer.TrainAgent:\n[HiddenState]%s \n[CellState]%s' %(hdn,cel)) 

    # df_account、df_rewardを空にする。
    self.acnt.DropAccount()
    self.agnt.DropReward()
    return

  def _WriteDetailFile(self,test_dt):
    # input情報
    with open(TRAIN_RESULT_PATH+self.input_filename, mode='a',newline='') as f:
      writer = csv.writer(f)
      data = [test_dt,*self.df_acnt_train.to_list()]
      writer.writerow(data)
    # hidden情報、cell情報
    hdnst, clst = self.agnt.GetMainHiddenCellState()
    with open(TRAIN_RESULT_PATH+self.hidden_filename, mode='a',newline='') as f:
      writer = csv.writer(f)
      data = [test_dt]
      lst = hdnst.tolist()
      for l in lst:
        for m in l:
          data.extend(m)
      writer.writerow(data)
    with open(TRAIN_RESULT_PATH+self.cellstate_filename, mode='a',newline='') as f:
      writer = csv.writer(f)
      data = [test_dt]
      lst = clst()
      for l in lst:
        for m in l:
          data.extend(m)
      writer.writerow(data)
    # output,maskoutput情報
    outpt, mskoutpt = self.agnt.GetOutput()
    with open(TRAIN_RESULT_PATH+self.output_filename, mode='a',newline='') as f:
      writer = csv.writer(f)
      data = [test_dt]
      lst = outpt.tolist()
      for l in lst:
        data.extend(l)
      writer.writerow(data)
    with open(TRAIN_RESULT_PATH+self.maskoutput_filename, mode='a',newline='') as f:
      writer = csv.writer(f)
      data = [test_dt]
      lst = mskoutpt.tolist()
      for l in lst:
        data.extend(l)
      writer.writerow(data)

  def ExecBacktest(self, start_year, end_year, test_duration):
    # バックテストの結果を格納するデータフレーム
    # 一行にテスト期間(例えば１週間)毎に結果を集計する
    self.df_backtest_result = pd.DataFrame(columns=['tradenum','sum','mean','sd','cum_sum'])
    self.df_backtest_result_detail = pd.DataFrame(columns=['tradenum','sum','mean','sd','cum_sum'])

    self.acnt.DropAccount()
    self.agnt.DropReward()
    self._DropResult()

    # テスト期間が週間の場合、１週間ごとに結果を集計する
    # 各年の2週目～51週目までをテストする。
    logger_trainer.info('Trainer.ExecBacktest:◆◇◆Start Backtesting.◆◇◆')
    if(test_duration == 'W'):
      for y in range(start_year,end_year+1):
        # 2026/8/21テスト期間をその年のiso 2週目からiso 最終週(52or53)の１週前とする
        # 12/28は必ずiso weekの最終週に含まれることを利用する
        test_end_week = datetime.datetime(y, 12, 28).isocalendar().week
        for w in range(2,test_end_week): # range()は test_end_week そのものは含まれないため、自然と１週前になる
          # 初めの時間足の初期値を作る
          self.start_date_time = datetime.datetime.fromisocalendar(y, w, 1)
          # 2026/8/21 tickが存在しないなど、テスト対象期間のデータがセットできないときはその週のテストをスキップする 
          if(not self.UpdateTradePeriod(self.start_date_time, test_duration)):
            continue
          # 推論モードでバックテストを実施する
          self.TrainAgent(1,train_mode=False)
          logger_trainer.debug('Trainer.ExecBacktest:self.df_result:%s' %(self.df_result))
          # 1週間ごとに結果を記録する
          self.df_backtest_result.loc[self.start_date_time] = self.df_result.iloc[-1]
          self.df_backtest_result.loc[self.start_date_time,'cum_sum'] = self.df_backtest_result['sum'].sum()
          self._DropResult()
    elif(test_duration == 'Y'):
      # 未実装
      pass
    elif(test_duration == 'D'):
      # 未実装
      pass
    else:
      # 未実装
      pass
    logger_trainer.info('Trainer.ExecBacktest:◇◆◇Finish Backtesting.◇◆◇')
    return self.df_backtest_result

  def ExecBacktestDetail(self, start_day, end_day, test_duration):
    # バックテストの結果を格納するデータフレーム
    # 一行にテスト期間(例えば１週間)毎に結果を集計する
    self.df_backtest_result = pd.DataFrame(columns=['tradenum','sum','mean','sd','cum_sum'])
    self.df_backtest_detail_result = pd.DataFrame(columns=['pos_open_datetime','pos_open_price','has_long','has_short','close_pl'])

    self.acnt.DropAccount()
    self.agnt.DropReward()
    self._DropResult()

    # start_dayとend_dayから、毎週月曜日のリストを作成する
    every_monday = pd.date_range(start_day, end_day, freq='W-MON')
    # テスト期間が週間の場合、１週間ごとに結果を集計する
    # 各年の2週目～51週目までをテストする。
    logger_trainer.info('Trainer.ExecBacktest:◆◇◆Start DetailBacktesting. From:%s, To:%s◆◇◆' %(start_day,end_day))
    if(test_duration == 'W'):
      for w in every_monday:
        #ISOWEEKが1週目、52週目以降の時はテストしない
        if w.isocalendar().week == 1 or w.isocalendar().week >= 52:
          logger_trainer.info('Trainer.ExecBacktestDetail: %d week test skiped.' %(w.isocalendar().week))
          continue

        # 初めの時間足の初期値を作る
        # self.UpdateTradePeriod(w, test_duration)
        self.acnt.SetTradePeriod(w, test_duration)
        # オブジェクトのトレード期間(start_date_time(0:00),end_date_time(23:00))を取得する
        self.start_date_time, self.end_date_time = self.acnt.GetTradePeriod()
        # 推論モードでバックテストを実施する
        self.df_result, self.df_detail_testresult = self.TrainAgent(1,train_mode=False,return_detail=True)
        logger_trainer.debug('Trainer.ExecBacktest:self.df_result:%s' %(self.df_result))
        # 1週間ごとに結果を記録する
        self.df_backtest_result.loc[self.start_date_time] = self.df_result.iloc[-1]
        self.df_backtest_result.loc[self.start_date_time,'cum_sum'] = self.df_backtest_result['sum'].sum()
        self._DropResult()
        self.df_backtest_detail_result = pd.concat([self.df_backtest_detail_result,self.df_detail_testresult],join='inner')
    elif(test_duration == 'Y'):
      # 未実装
      pass
    elif(test_duration == 'D'):
      # 未実装
      pass
    else:
      # 未実装
      pass
    logger_trainer.info('Trainer.ExecBacktest:◇◆◇Finish Backtesting.◇◆◇')
    return self.df_backtest_result,self.df_backtest_detail_result

  def UpdateTradePeriod(self, stdt, tstdl):
    self.acnt.SetTradePeriod(stdt, tstdl)
    # オブジェクトのトレード期間(start_date_time(0:00),end_date_time(23:00))を取得する
    tmp_start_date_time, tmp_end_date_time = self.acnt.GetTradePeriod()
    logger_trainer.debug('Trainer.UpdateTradePeriod:tmp_start_date_time:%s' %(tmp_start_date_time))
    logger_trainer.debug('Trainer.UpdateTradePeriod:tmp_end_date_time:%s' %(tmp_end_date_time))
    # 2026/2/14 self.end_date_timeが更新されないのでここでする
    self.end_date_time = tmp_end_date_time
    # トレード期間(Mon 0:00-Fri 23:00)のprace_dataをまとめて取得する
    # self.df_trade_data_period = train_data_dict[self.period].GetTrainDataPeriod(self.start_date_time, self.end_date_time)
    # 2023/06/03 取得するトレードデータは、トレード期間のひとつ前の足から終了のひとつ前まで
    previous_start_datetime = self._GetLastDateTime(tmp_start_date_time, self.delta_time)
    previous_end_date_time = self._GetLastDateTime(tmp_end_date_time, self.delta_time)
    logger_trainer.debug('Trainer.UpdateTradePeriod:previous_start_datetime:%s' %(previous_start_datetime))
    logger_trainer.debug('Trainer.UpdateTradePeriod:previous_end_date_time:%s' %(previous_end_date_time))
    self.df_trade_data_period = self.train_data_dict[self.period].GetTrainDataPeriod(previous_start_datetime, previous_end_date_time)

    # 2025/2/22 accountにトレード期間中(開始の１つ前と終了のひとつ後)のtick dataをdataFrameで渡す
    self.acnt.DropTickDataPeriod()
    # 2026/8/21 tickデータ不備によりtickデータがセットできない場合は、処理を中断する（呼び出し元でテストをスキップさせる）
    if(not self.acnt.SetTickDataPeriod(previous_start_datetime,tmp_end_date_time+self.delta_time)):
      logger_trainer.warning('Trainer.UpdateTradePeriod:No tick data in SetTickDataPeriod()')
      return(False)

    # 2026/3/15 Copilotによるパフォーマンスチューニング
    ticks = self.acnt.ticks_frame
    tick_index = ticks.index.to_numpy()

    # ---- ここで bar → tick のマッピングを前計算する ----
    # CheckTpLcで使う時間足のはじめと終わりのtick行番号(index)を格納する
    self.bar_to_tick = {}

    bar_dt = previous_start_datetime

    while bar_dt <= tmp_end_date_time:
        next_bar_dt = bar_dt + self.delta_time

        # numpy の二分探索で高速に index を取得
        start_idx = np.searchsorted(tick_index, np.datetime64(bar_dt), side='left')
        end_idx   = np.searchsorted(tick_index, np.datetime64(next_bar_dt), side='left')

        self.bar_to_tick[bar_dt] = (start_idx, end_idx)

        bar_dt = next_bar_dt

    # Accountにbar_to_tickを渡す
    self.acnt.bar_to_tick = self.bar_to_tick

    # Evluaterewardで使う１分ごとのはじめと終わりの行番号(index)を格納する
    self.bar_to_tick_minute = {}
    bar_dt = previous_start_datetime

    # 最後の足の00分に全く取引がない（あるいはreal spread超過）の時に1分後のtickを見に行くと
    # 値が存在しないためエラーとなることを回避する
    # while bar_dt <= tmp_end_date_time:
    while bar_dt <= tmp_end_date_time+self.delta_time:
        next_bar_dt = bar_dt + datetime.timedelta(minutes=1)

        # numpy の二分探索で高速に index を取得
        start_idx = np.searchsorted(tick_index, np.datetime64(bar_dt), side='left')
        end_idx   = np.searchsorted(tick_index, np.datetime64(next_bar_dt), side='left')

        self.bar_to_tick_minute[bar_dt] = (start_idx, end_idx)

        bar_dt = next_bar_dt
    # Accountにbar_to_tickを渡す
    self.acnt.bar_to_tick_minute = self.bar_to_tick_minute
    return(True)

  def SaveParameters(self,Parameters,Modelname=None):
    self.acnt.SaveIniFile(Parameters.account)
    self.agnt.SaveAgentModel(Parameters.agent,Parameters.brain,Parameters.eval_value)
    if Modelname != None:
      self.acnt.SaveSetFile(Modelname)

  # 現在のDateTime(cur_dt)からDeltaTime(dlt_dt)分前のDateTimeを返す
  # ただし、dlt_dt分戻した曜日が土日の場合は金曜日の同時刻までさかのぼる
  def _GetLastDateTime(self,cur_dt, dlt_dt):
    last_dt = cur_dt - dlt_dt
    # 2023/9/9 曜日に関係なく、PriceDataとTrainDataが揃っている時間までさかのぼる
    # if last_dt.isoweekday() >= 6:
    #   last_dt -= (last_dt.isoweekday() % 5)*datetime.timedelta(days=1)
    # LastDateTimeに対応するPriceDataかTrainDataがなければもう1つ時間を戻す
    while self.periods_dict[self.period][self.symbol].GetPriceData(last_dt).empty or\
      self.train_data_dict[self.period].GetTrainDataPeriod(last_dt, last_dt).empty:
      logger_trainer.debug('Trainer._GetLastDateTime:PriceData or TrainData in %s is not exist.' %(last_dt))
      last_dt -= dlt_dt
    return last_dt

  # 現在のDateTime(cur_dt)からDeltaTime(dlt_dt)分"後"のDateTimeを返す
  # ただし、dlt_dt分進めた曜日が土日の場合は月曜日の同時刻まで進める
  def _GetNextDateTime(self,cur_dt, dlt_dt):
    next_dt = cur_dt + dlt_dt
    # 2023/9/9 曜日に関係なく、PriceDataとTrainDataが揃っている時間まで進める
    # if next_dt.isoweekday() >= 6:
    #   next_dt += (3 - (next_dt.isoweekday() % 5))*datetime.timedelta(days=1)
    # NextDateTimeに対応するPriceDataかTrainDataがなければもう1つ時間を進める
    while self.periods_dict[self.period][self.symbol].GetPriceData(next_dt).empty or\
      self.train_data_dict[self.period].GetTrainDataPeriod(next_dt, next_dt).empty:
      logger_trainer.warning('Trainer._GetNextDateTime:PriceData or TrainData in %s is not exist.' %(next_dt))
      next_dt += dlt_dt
    return next_dt

  def _DropResult(self):
    # df_resultの表を削除する
    self.df_result = self.df_result.drop(self.df_result.index)
    logger_trainer.debug('Train._DropResult:df_result %s' %(self.df_result))


# ## クラス TrainDataMaker
# - PraiceData, StaticData, TrainDataの作成をコントロールする

# In[34]:


#@title class TrainDataMaker
class TrainDataMaker():

  def __init__(self,period_random_select=False,prmlst=None):
    # PriceData, StaticDataを取り扱うPriceDataオブジェクトを格納する辞書、時間足と通貨ペアの２重構造
    # prices_dict = {'EURUSD': pricedata(eurusd), 'USDJPY': pricedata(usdjpy),...}
    # periods_dict = {'D1': prices_dict(d1), 'H4': prices_dict(h4), ...}
    self.periods_dict = {}
    self.prices_dict = {}
    # train_dataを時間足毎に格納する辞書。train_dataはすべての通貨ペアが結合されているものか、取引対象通貨ペアのみのいずれか(２重構造ではない)
    # train_data_dict = {'D1': traindata(all_symbols/trade_symbol), 'H4': traindata(all_symbols/trade_symbol), ...}
    self.train_data_dict = {}

    # train_dataの窓関数のピリオドを決定する。
    # トレーニング再開の場合は、設定ファイル(account.ini)に記録されているので読み込む
    # 設定ファイルが存在しない(或いは設定ファイルにピリオドの記載がない)場合は、ランダムに選ぶかデフォルト値を使う
    self.train_prm_list = prmlst
    self.periods = [LONG_PERIOD,SHORT_PERIOD,TICKVOL_PERIOD]

    if self.train_prm_list != None and self.train_prm_list.account != None:
      logger_trainer.debug('TrainDataMaker.__init__: INIFile loading. \n %s' %(self.train_prm_list.account))
      #Load FlagがTrueでファイルパスが指定されている場合は、そのファイルを読み込む
      acc_inifile = configparser.ConfigParser()
      acc_inifile.read(self.train_prm_list.account)
      try:
        self.periods[0] = int(acc_inifile.get('COMMOM', 'LONG_PERIOD'))
        self.periods[1] = int(acc_inifile.get('COMMOM', 'SHORT_PERIOD'))
        self.periods[2] = int(acc_inifile.get('COMMOM', 'TICKVOL_PERIOD'))
      except configparser.NoOptionError:
        # Account.iniに設定がない場合は、デフォルトの値を使う
        logger_trainer.warning('TrainDataMaker.__init__:No Such Option Found.')
        pass
    elif period_random_select:
      # ランダムで設定する
      self.periods = EnvironmentCommon.periodRandomSelect()
    else:
      # デフォルト値を使う
      pass

  def MakePriceData(self,train_period_list,train_symbol_list):
    # 時間足、通貨ペア別の価格データをcsvファイルから取り出して、複数年分を１つに結合する

    for prd in train_period_list:
      for sbl in train_symbol_list:
        logger_trainer.info('TrainDataMaker.MakePriceData:Making %s, %s pricedata.' %(sbl,prd))
        prices = PriceData(sbl,prd)
        prices.ConvertPriceDataFileToDataFrame()
        if(self.periods_dict.setdefault(prd) is not None):
          self.prices_dict = self.periods_dict[prd]
        self.prices_dict[sbl] = prices
        self.periods_dict[prd] = self.prices_dict.copy()
        self.prices_dict.clear()
        logger_trainer.info('TrainDataMaker.MakePriceData:Finish %s, %s pricedata.' %(sbl,prd))

  def MakeTrainData(self, train_period, train_symbol, input_all_symbols_flg=True):
    # 対象となる時間足のTrainDataを作成する
    # まず、PriceDataからStaticDataを作成する
    self.prices_dict = self.periods_dict[train_period]
    for symbol, prices in self.prices_dict.items():
      prices.AddStaticInfo(self.periods)
      self.prices_dict[symbol]

    # TrainDataを作成する
    if input_all_symbols_flg:
      logger_trainer.info('TrainDataMaker.MakeTrainData:Making traindata. All %s pricedata combined.' %(train_period))
      self.train_data_dict[train_period] = TrainData(list(self.periods_dict[train_period].values()), self.periods_dict[train_period][train_symbol].price_list.index)
      logger_trainer.info('TrainDataMaker.MakeTrainData:Finish traindata.')
    else:
      logger_trainer.info('TrainDataMaker.MakeTrainData:Making %s, %s traindata.' %(train_symbol,train_period))
      self.train_data_dict[train_period] = TrainData([self.periods_dict[train_period][train_symbol]])
      logger_trainer.info('TrainDataMaker.MakeTrainData:Finish traindata.')

  def GetPeriodsDict(self):
    return self.periods_dict

  def GetPricesDict(self, train_period):
    return self.periods_dict[train_period]

  def SetPeriodsDict(self, pddct):
    # ポインタ渡し
    self.periods_dict = pddct

  def SetPricesDict(self, train_period, pcdct):
    # コピー
    self.periods_dict[train_period] = pcdct.copy()

  def GetTrainDataDict(self):
    return self.train_data_dict

  def GetPeriods(self):
    return self.periods


# # Package-Portfolio
# トレード候補となるNN Modelの過去の成績から効率的フロンティアを作成し、ポートフォリオを計算する。

# ## Dependency
# このパッケージを実行する前にPyPortfolioOptをインストールする

# In[35]:


from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import matplotlib.pyplot as plt
## for comandline 以下をコメントアウトする
# %matplotlib inline
import MetaTrader5 as mt5


# ## Logger Portfolio

# In[36]:


#@title Logger Portfolio
import logging
import logging.config

# logging.config.fileConfig('./drive/My Drive/Colab Notebooks/02_DRLTradingSystem2020/Logging.ini')
# logging.config.fileConfig(LOGGING_INIFILE_PATH)
logconfigfile = configparser.ConfigParser()
logconfigfile.read(LOGGING_INIFILE_PATH,ENC)
logging.config.fileConfig(logconfigfile)
logger_portfolio = logging.getLogger('DRLPortfolio')
logger_portfolio.debug('Debug level massage.')
logger_portfolio.info('Info level massage.')
logger_portfolio.warning('Warning level massage.')
logger_portfolio.error('Error level massage.')
logger_portfolio.critical('Critical level massage.')


# ## クラス PortfolioConstructer

# In[37]:


#@title PortfolioConstructer
class PortfolioConstructer:

  def __init__(self,target_ex):
    self.target_excepted_value = target_ex
    pass

  def ConstructPortfolio(self,df_models):
    # df_modelの内容をプロットする？

    # モデルの平均、分散・共分散行列を計算する
    self.np_models = df_models.to_numpy()
    self.np_cov = np.cov(self.np_models,rowvar=False,bias=True)
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:CovMatrix=======\n%s' %(self.np_cov))
    self.np_mean = np.mean(self.np_models,axis=0)
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:Means=======\n%s'%(self.np_mean))
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:Min mean=%.4f'%(self.np_mean.min()))
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:Max mean=%.4f'%(self.np_mean.max()))

    self.S = pd.DataFrame(self.np_cov,columns=df_models.columns)
    self.mu = pd.Series(self.np_mean,index=df_models.columns)
    self.ef = EfficientFrontier(self.mu, self.S)
    self.w_min_dict = self.ef.min_volatility() # 戻り値は辞書型
    self.w_min_np = pd.Series(self.w_min_dict).values
    self.pf_mu_min = np.dot(self.mu,self.w_min_np.T)
    self.v_min = np.sqrt(np.dot(np.dot(self.w_min_np,self.np_cov),self.w_min_np.T)) # 標準偏差の計算
    logger_portfolio.info('EfficientFrontier.ConstructPortfolio:min_volatility pf_mu_min=%.4f, v_min=%.4f' %(self.pf_mu_min,self.v_min))

    # ポートフォリオを構成する各nnの期待値がtarget_expected_valeより小さい場合は、ポートフォリオ内の最大の期待値(np_mean.max())とする
    if self.np_mean.max() < self.target_excepted_value:
      self.ef = EfficientFrontier(self.mu, self.S)
      self.w_dict = ef.efficient_return(target_return=self.np_mean.max())
      self.w_np = pd.Series(self.w_dict).values
      self.pf_mu = np.dot(self.mu,self.w_np.T) # ポートフォリオの期待値
      self.v = np.sqrt(np.dot(np.dot(self.w_np,self.np_cov),self.w_np.T)) # 分散の計算
    # リスク最小の期待値がtarget_expected_valeより小さい場合は期待値が1.0のポートフォリオを採用する
    elif self.pf_mu_min < self.target_excepted_value:
      self.ef = EfficientFrontier(self.mu, self.S)
      self.w_dict = self.ef.efficient_return(target_return=self.target_excepted_value)
      self.w_np = pd.Series(self.w_dict).values
      self.pf_mu = np.dot(self.mu,self.w_np.T) # ポートフォリオの期待値
      self.v = np.sqrt(np.dot(np.dot(self.w_np,self.np_cov),self.w_np.T)) # 分散の計算
    # リスク最小の期待値がtarget_expected_vale以上の場合はリスク最小におけるポートフォリオを採用する
    else:
      self.w_dict = self.w_min_dict
      self.w_np = self.w_min_np
      self.pf_mu = self.pf_mu_min # ポートフォリオの期待値
      self.v = self.v_min

    self.w_dict['mu'] = self.pf_mu
    self.w_dict['sigma'] = self.v
    logger_portfolio.info('EfficientFrontier.ConstructPortfolio:pf_mu=%.4f, v=%.4f' %(self.pf_mu,self.v))
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:Portfolio=====\n%s' %(self.w_dict))

    return self.w_dict

  def SavePortfolioFile(self,m_dict):
    # NN モデルのポートフォリオをModelPortfolio.iniファイルに保存する
    self.p_ini = configparser.ConfigParser()
    self.p_ini['LOCAL'] = m_dict

    with open (TRADE_MODEL_PATH+MODEL_PORTFOLIO_FILE_NAME,'w') as fp:
      self.p_ini.write(fp)
    return

  def DrawEfficientFrontier(self,df_models):
    # モデルの平均、分散・共分散行列を計算する
    self.np_models = df_models.to_numpy()
    self.np_cov = np.cov(self.np_models,rowvar=False,bias=True)
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:CovMatrix=======\n%s' %(self.np_cov))
    self.np_mean = np.mean(self.np_models,axis=0)
    logger_portfolio.debug('EfficientFrontier.ConstructPortfolio:Means=======\n%s'%(self.np_mean))
    self.S = pd.DataFrame(self.np_cov,columns=df_models.columns)
    self.mu = pd.Series(self.np_mean,index=df_models.columns)
    self.ef = EfficientFrontier(self.mu, self.S)

    self.trets = [i/100 for i in range(round(self.np_mean.min()*100),round(self.np_mean.max()*100))]
    self.tvols = []
    for tr in self.trets:
      w = self.ef.efficient_return(target_return=tr)
      w = pd.Series(w).values
      v = np.sqrt(np.dot(np.dot(w,np.array(self.S)),w.T)) # 分散の計算
      self.tvols += [v]

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(self.tvols, self.trets, marker='x')
    ax.set_xlim([0.0, max(self.tvols)*1.2])
    ax.set_ylim([0.0, self.np_mean.max()*1.2])
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected return')
    ax.grid(True)
    plt.show()

    def CalcOrderLots(self,trdmd,mname=None):
        # この関数は、ローカル環境でのみ機能する
        if EXEC_ENV == 'COLABO':
            # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
            #                                    +'EfficientFrontier.CalcOrderLots:This function will not work in COLABO.')
            sys.exit("EfficientFrontier.CalcOrderLots:This function will not work in COLABO.")

        # MT5クライアントに接続して、口座情報(残高)を取得する
        # トレードモード(MT5より借用)
        if trdmd == ACCOUNT_TRADE_MODE_DEMO:
          MT5_PATH = inifile.get(EXEC_ENV, 'mt5_demo_path')
          growth_rate = float(inifile.get(EXEC_ENV, 'GROWTH_RATE_DEMO'))
        elif trdmd == ACCOUNT_TRADE_MODE_REAL:
          MT5_PATH = inifile.get(EXEC_ENV, 'mt5_real_path')
          growth_rate = float(inifile.get(EXEC_ENV, 'GROWTH_RATE_REAL'))
        else:
          # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
          #                                      +'EfficientFrontier.CalcOrderLots:Account Trade Mode is not defined.')
          sys.exit("EfficientFrontier.CalcOrderLots:Account Trade Mode is not defined.")

        portfolio.read(TRADE_MODEL_PATH+MODEL_PORTFOLIO_FILE_NAME,'UTF-8')
        logger_portfolio.debug("EfficientFrontier.CalcOrderLots:MT5_PATH=%s" %(MT5_PATH))

        # 取引ロットを計算する
        if not mt5.initialize(MT5_PATH):
            # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
            #                                    +'EfficientFrontier.CalcOrderLots:initialize() failed, error code = %s' %(str(mt5.last_error())))
            sys.exit("EfficientFrontier.CalcOrderLots:initialize() failed, error code =", mt5.last_error())

        account_info = mt5.account_info()
        # mt5.shutdown()

        logger_portfolio.debug("EfficientFrontier.CalcOrderLots:account_info=%s" %(str(account_info)))
        equity = account_info.equity
        logger_portfolio.info("EfficientFrontier.CalcOrderLots:account_info.equity=%s" %(str(equity)))

        # ロット計算に必要な情報を取得する
        if mname is None:
          investment_rate = 1.0
        else:
          investment_rate = float(portfolio.get(EXEC_ENV,mname))
        expected_value = float(portfolio.get(EXEC_ENV,'mu'))
        # logger_trader.debug("Environment:%s growth_rate=%s, investment_rate=%s" %(mname,str(growth_rate),str(investment_rate)))
        logger_portfolio.debug("EfficientFrontier.CalcOrderLots:%s growth_rate=%s, investment_rate=%s, expected_value=%s" %(mname,str(growth_rate),str(investment_rate),str(expected_value)))
        # order_lots = (equity/100000)*growth_rate*investment_rate
        order_lots = (equity/100000)*(growth_rate/expected_value)*investment_rate
        # OrderLotsを少数第3位を四捨五入する
        order_lots = round(order_lots,2)
        logger_portfolio.debug("EfficientFrontier.CalcOrderLots:order_lots=%s" %(str(order_lots)))

        return order_lots


# # Stab
# - Google Colab から実行するためのスタブ。Mainを実行する前に、GoogleDriveをマウントする。

# ## Logger
# - StabのLoggerインスタンスはlogger_rootとする

# In[38]:


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


# ## Make Train Data and Train Model
# テストモデル毎にLong/Short/Tickvol Periodを変更するため、都度TrainDataを作成する

# In[ ]:


if __name__ == '__main__':
  ## For Comandline
  # コマンドラインからの実行に対応させる
  # 引数は 0:PythonCode 1:trade_mode 2:symbol 3:period 4:maxtrainnum 5:trainnum
  args = sys.argv
  if len(args) != 6:
    logger_root.error('Stab Make_Train_Data_and_Train_Model: Usage \n [trade_mode(REAL/DEMO)] [symbol] [period] [max_train_num] [train_num]')
    # 標準出力で999を返す
    print(999)
    sys.exit('Stab Make_Train_Data_and_Train_Model: Usage \n [trade_mode(REAL/DEMO)] [symbol] [period] [max_train_num] [train_num]')

  ## For Comandline
  # trade_modeの判定。引数は"REAL" "DEMO"の想定
  if args[1] == 'REAL':
    ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_REAL
  elif args[1] == 'DEMO':
    ACCOUNT_TRADE_MODE = ACCOUNT_TRADE_MODE_DEMO
  else:
    logger_root.error('Stab Make_Train_Data_and_Train_Model: train_mode can select from REAL or DEMO.')
    # 標準出力で999を返す
    print(999)
    sys.exit('Stab Make_Train_Data_and_Train_Model: train_mode can select from REAL or DEMO.')

  TRAIN_SYMBOL_LIST = ['EURUSD','USDJPY','GBPUSD','EURJPY','EURGBP','GBPJPY']
  ## For Comandline
  TRADE_SYMBOL = args[2]
  TRADE_PERIOD = args[3]
  # 20240713 DETAIL_CHECK_PERIDは読み込まない
  # DETAIL_CHECK_PERID = 'M1'
  # TRAIN_PERIOD_LIST = [TRADE_PERIOD,DETAIL_CHECK_PERID]
  TRAIN_PERIOD_LIST = [TRADE_PERIOD]
  INPUT_ALL_SYMBOLS_FLG = True

  delta_time = EnvironmentCommon.GetPeriodTimeDelta(TRADE_PERIOD)
  # ファイルパスを設定する
  EnvironmentCommon.setTradeModePath(ACCOUNT_TRADE_MODE)

  ## For Comandline
  # TRAIN_MODEL_PATHからTrainNumList.csvを読み込んでpd.DataFrameを作成する
  # TrainNumList.csvが存在しない場合は、すべてが0のDataFrameを作成する
  trainnum_file_path = TRAIN_MODEL_PATH + 'TrainNumList.csv'
  if os.path.isfile(trainnum_file_path):
    logger_root.debug('Stab Make_Train_Data_and_Train_Model: %s read.' %(trainnum_file_path))
    dfTrainNumList = pd.read_csv(trainnum_file_path, index_col=0)
  else:
    logger_root.debug('Stab Make_Train_Data_and_Train_Model: New dfTrainNumList create.')
    cols = ['D1','H12','H8','H6','H4','H1','M30']
    rows = ['EURUSD','USDJPY','GBPUSD','EURJPY','EURGBP','GBPJPY']
    dfTrainNumList = pd.DataFrame(np.zeros((len(rows),len(cols)),dtype='int16'),columns=cols,index=rows)

  logger_root.debug('Stab Make_Train_Data_and_Train_Model: dfTrainNumList. \n %s' %(dfTrainNumList))

  ## For Comandline
  # SymbolとPeriodに対応したトレーニング完了数を取得し、すでにmaxtrainnum以上トレーニングをしていたら終了させる
  already_train_num = dfTrainNumList.at[TRADE_SYMBOL,TRADE_PERIOD]
  logger_root.debug('Stab Make_Train_Data_and_Train_Model: already_train_num=%d' %(already_train_num))

  if already_train_num >= int(args[4]):
    logger_root.error('Stab Make_Train_Data_and_Train_Model: already_train_num is exceeded max_train_num.')
    # トレーニング回数を標準出力で返す
    print(int(already_train_num))
    sys.exit('Stab Make_Train_Data_and_Train_Model: already_train_num is exceeded max_train_num.')

  LOAD_FLAG = True
  FILE_PATH = None
  # LONG,SHORT,TICKVO+_PERIODをランダムに選択するか
  PERIOD_RANDOM_SELECT = True #@param {type:"boolean"}

  # 1つの世代で作成するTrainAgentの数(default:5)
  TRAIN_AGENT_NUM = 5 #@param {type:"number"}
  # 次の世代に行けるTrainAgentの数(上位)(default:3)
  AGENT_TOGO_NEXT_NUM = 3 #@param {type:"number"}
  # 訓練を最大何世代行うか。最大世代数までに条件にあうTrainAgent(聖杯！）が見つかれば訓練中断(default:5)

  ## For Comandline
  MAX_GEN_NUM = int(already_train_num) + int(args[5])

  # 1つのエージェントに対して繰り返す訓練の数
  ## EPISODE_NUMは辞書で持って、TRADE_PERIODに対応して取り出すか。(default:200)
  EPISODE_NUM = 201 #@param {type:"integer"}
  # テスト期間は1episodeの長さ。1日(D)、1週間(W)、1年(Y)を選択する
  TEST_DURATION = 'W' #@param ['D', 'W', 'Y']
  # BackTestを行う期間(default:2020-2023)
  BACKTEST_START_YEAR = 2021 #@param {type:"slider", min:2018, max:2030, step:1}
  BACKTEST_END_YEAR = 2024 #@param {type:"slider", min:2018, max:2030, step:1}
  BACKTEST_DURATION = 'W' #@param ['D', 'W', 'Y']

  # For DDQN Target Q-NNとMain Q-NNを同期する頻度(何episode毎か)(defoult:3)
  COPY_FREQ_TARGET_Q_NN = 3 # @param {type:"integer"}
  # エントリーする際のスプレッドを制限するか(しない場合は一律1,000pips)
  REAL_SPREAD_FLG = False

  # 世代別TradeAgent事のBacktestの結果とObjectを格納するDataFrame
  # result:Backtestの損益の合計、trade_avg:1週間当たりの平均損益、std:1週間当たりの損益の標準偏差
  # cdp_0:pips=0の時の累積確率密度関数(cumulative distribution probability function)の値、agt_obj:TradeAgentのObject
  df_gen_rank = pd.DataFrame(columns=['gen','agt_num','result','trade_avg','std','cdp_0','eval_value','agt_obj'])
  # ParameterFileは、AccountのINIファイルとAgentのINIファイル、BrainのPTHファイルがある
  # それぞれのファイルパスをランクごとにNamedTupleに格納する
  # TrainParametersList = [None] * TRAIN_AGENT_NUM
  # 2026/01/22 NNModelの評価値(eval_value)を格納できるようにする
  TrainParameters = namedtuple('TrainParameters', ['account','agent','brain','eval_value'])
  TrainParametersList=[TrainParameters(None,None,None,None)]*TRAIN_AGENT_NUM
  # Trainerを格納するリスト
  AgentList = [None] * TRAIN_AGENT_NUM
  # periods_dictを格納するリスト
  PriceDataDicts = [None] * TRAIN_AGENT_NUM
  # train_data_dictを格納するリスト
  TrainDataDicts = [None] * TRAIN_AGENT_NUM
  # ModelNameとMAGICを格納するNamedTaple
  ModelName = namedtuple('ModelName',['model_name','magic','set_file_path'])

  # 2025/02/11 MT5との接続をやめて、DB(postgresql)と接続する
  # クラス関数で、engineを取得する
  EnvironmentCommon.connectDB()
  #logger.debug("DB Conneced.")
  # 2026/6/6 SQLALchemyの代わりにConnectorXを使う
  logger.debug("CONN_URL Created.")

  # PriceDataはすべての世代、エージェントで共通して使用するため1つだけ作成する
  # 個々のTrainDataを作成する際に、ポインタを渡す
  price_data_common = TrainDataMaker()

  ## For Comandline
  try:
    price_data_common.MakePriceData(TRAIN_PERIOD_LIST,TRAIN_SYMBOL_LIST)
    # 20240713 DetailCheckはテスト対象の通貨ペア分だけ作る
    # 2025/2/15 DetailCheckデータは作らない
    # price_data_common.MakePriceData([DETAIL_CHECK_PERID],[TRADE_SYMBOL])
  except MemoryError as e:
    # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\nGen '+str(gnum+1)+' MemoryError occured. Training has halted.')
    logger_root.error('Stab.MakeTrainDataAndTrainModel:MemoryError occurred.')
    # トレーニング回数を標準出力で返す
    print(int(already_train_num))
    sys.exit('Stab.MakeTrainDataAndTrainModel:MemoryError occurred.')

  for gnum in range(MAX_GEN_NUM):
    # 訓練する世代のNN Paramファイルが存在していれば、それを取得し
    # 当該世代の訓練をスキップする
    # ParameterFileは、AccountのINIファイルとAgentのINIファイル、BrainのPTHファイルがある
    # 各世代において、それぞれのファイルの数が同じかを確認し、同じであればRank毎にNamedTupleを生成する
    tmp_account_params=glob.glob('%s%s_%s_Account_Gen%s_Rank*.ini'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
    logger_root.debug('tmp_account_params:%s' %(tmp_account_params))
    tmp_agent_params=glob.glob('%s%s_%s_Agent_Gen%s_Rank*.ini'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
    logger_root.debug('tmp_agent_params:%s' %(tmp_agent_params))
    tmp_brain_params=glob.glob('%s%s_%s_Brain_Gen%s_Rank*.pth'%(TMP_TRAIN_MODEL_PATH,TRADE_SYMBOL,TRADE_PERIOD,gnum+1))
    logger_root.debug('tmp_brain_params:%s' %(tmp_brain_params))

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
        logger_root.debug('TrainParameters:%s' %(str(TrainParametersList[rnk])))
        logger_root.info('TrainModel:Generation %s Parameters Loaded.' %(str(gnum+1)))

      # 次の世代へ
      ## For Comandline
      # TrainNumListにこれまでに完了したトレーニング数を記録する
      dfTrainNumList.at[TRADE_SYMBOL,TRADE_PERIOD] = gnum+1
      continue

    # 訓練する時間(開始日)をランダムに設定する。
    # どの訓練期間であっても、年は共通(2018～2022)
    # 2024/12/21 MT5サーバのtickが取得できないため、2020～2023に変更
    YEAR = random.randint(2021,2024)
    if(TEST_DURATION == 'Y'):
      # 訓練期間が年単位の場合は、第2週目の月曜日(weekday=1)から
      WEEKNUM = 2
      WEEKDAY = 1
    elif(TEST_DURATION == 'W'):
      # 訓練期間が週単位の場合は、2～51週(ランダム)の月曜日(weekday=1)から
      WEEKNUM = random.randint(2,51)
      WEEKDAY = 1
    elif(TEST_DURATION == 'D'):
      # 訓練期間が日単位の場合は、2～51週(ランダム)の月曜日(weekday=1)～金曜日(weekday=5)(ランダム)
      WEEKNUM = random.randint(2,51)
      WEEKDAY = random.randint(1,5)
    else:
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                    +'Stub:This period is not implemented.')
      ## ForComandline
      # 標準出力で999を返す
      print(999)
      sys.exit('This period is not implemented.')

    # fromisocalendar()は、python3.8以降でないと実装されていない
    # 月曜日(weekday=1)と最終週の日曜日(weekday=7)
    test_date_time = datetime.datetime.fromisocalendar(YEAR,WEEKNUM,WEEKDAY)
    logger_root.info('Stab.MakeTrainDataAndTrainModel:◆◆Generation %d Start Training.' %(gnum+1))

    for p, prmlst in enumerate(TrainParametersList):
      logger_root.debug('Stab.MakeTrainDataAndTrainModel: Gen %d, Agent %d, Start Training.' %(gnum+1, p+1))
      # AgentListの中身がNoneの場合に新たにTrainerAgentを作成する
      # その時、TrainParameterにファイルパスが格納されている場合はファイルをロードする
      # TrainParameterの値がNoneの場合は、新規作成
      if AgentList[p] == None:
        # Agentが無い場合、Agentに渡すためのPriceDataとTrainDataを作成する。
        # PriceDataは全体で共通のものをポイント参照
        # TrainDataは都度作成する
        price_train_data = TrainDataMaker(PERIOD_RANDOM_SELECT, prmlst)
        price_train_data.SetPeriodsDict(price_data_common.GetPeriodsDict())
        price_train_data.MakeTrainData(TRADE_PERIOD, TRADE_SYMBOL, INPUT_ALL_SYMBOLS_FLG)
        train_data_dict = price_train_data.GetTrainDataDict()
        periods_dict = price_train_data.GetPeriodsDict()
        periods = price_train_data.GetPeriods()

        PriceDataDicts[p] = periods_dict
        TrainDataDicts[p] = train_data_dict
        #__init__(self, epinum, sbl, tf, start_dt, tdur,
        #  gnum, anum, train_data, periods_dict, load_flg, filepth=None, prds=None, cp_frq=2, grd='X', mgc='yyyymmddHHMMSS',rsf=False)
        AgentList[p] = Trainer(EPISODE_NUM, TRADE_SYMBOL, TRADE_PERIOD, test_date_time, TEST_DURATION
                        , gnum+1, p+1, train_data_dict[TRADE_PERIOD], periods_dict, LOAD_FLAG, filepth=prmlst, prds=periods, cp_frq=COPY_FREQ_TARGET_Q_NN, rsf=REAL_SPREAD_FLG)

      else:
        # AgentList[p]にオブジェクトがある場合(前世代からの生き残り)でも、訓練期間(test_date_time)を更新する
        # 2026/8/21 トレーニング期間中のtickが取得できないなど、トレーニングができないときはシステムエラーとする
        if(not AgentList[p].UpdateTradePeriod(test_date_time,TEST_DURATION)):
          # トレーニングを終了させる
          ## ForComandline
          # 標準出力で999を返す
          print(999)
          logger_root.error('Stab.MakeTrainDataAndTrainModel:While UpdateTradePeriod, some problem has occured.')
          sys.exit('Training halted.')         
        # PriceDataとTrainDataも取得する
        periods_dict = PriceDataDicts[p]
        train_data_dict = TrainDataDicts[p]

      # 実際のトレーニング
      df_result = AgentList[p].TrainAgent(epinum=EPISODE_NUM, train_mode=True)
      # 2026/8/21 トレーニング期間中のtickが取得できないなど、トレーニングができないときは Noneを取得し、システムエラーとする
      if(df_result is None):
        # トレーニングを終了させる
        ## ForComandline
        # 標準出力で999を返す
        print(999)
        logger_root.error('Stab.MakeTrainDataAndTrainModel:While Training, some problem has occured.')
        sys.exit('Training halted.')         

      logger_root.info('■□■TrainModel:%s_%s Gen %d, Agent %d, Train Finish.■□■\n %s' %(TRADE_SYMBOL, TRADE_PERIOD, gnum+1, p+1,df_result.tail()))

      # ★★ForTest
      # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD
      #                                   +'Stub:Test finished.')
      # sys.exit()

      df_result.to_csv(TMP_TRAIN_RESULT_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Gen'+str(gnum+1)+'_Agent'+str(p+1)+'_TrainResult.csv')

      df_backtest_result = AgentList[p].ExecBacktest(BACKTEST_START_YEAR,BACKTEST_END_YEAR,BACKTEST_DURATION)
      # トレーニングの結果をdf_gen_rankに登録する
      # ['gen','agt_num','result','trade_avg','std','cdp_0','evae','agt_obj']

      # 2023/9/22 eval_valueの評価方法を変更。trade_avgが負の場合は、pips0における累積分布確率関数の左側を掛ける
      result = df_backtest_result['sum'].sum()
      trade_avg = df_backtest_result['sum'].mean()
      std = df_backtest_result['sum'].std()
      cdp_0 = stats.norm(loc=trade_avg,scale=std).cdf(0)
      eval_value = trade_avg*cdp_0 if trade_avg<0 else trade_avg*(1-cdp_0)
      df_gen_rank.loc['%s%s'%(gnum+1, p+1)]=[gnum+1,p+1, result, trade_avg, std, cdp_0, eval_value, AgentList[p]]

      logger.info('TrainModel:Gen %d, Agent %d, BackTest Finish.\n %s' %(gnum+1, p+1,df_gen_rank.loc['%s%s'%(gnum+1, p+1)]))
      df_backtest_result.to_csv(TMP_TRAIN_RESULT_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Gen'+str(gnum+1)+'_Agent'+str(p+1)+'_BacktestResult.csv')
      # Exit Criteriaを満たしているかチェック
      # 1.年間平均週間獲得pips1,000(=1.000)以上かつ獲得pips0における累積分布確率関数が5%以下
      # if(df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['trade_avg'] >1 and df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['cdp_0'] < 0.05):
      # 1.年間平均週間獲得pips(千単位)と、[1-獲得pips0における累積分布確率関数]の積(eval_value)が0.95より大きい
      # 2026/2/14 
      # 2026/6/13 Exit Criteriaチェックをリファクタリング
      model_file_path = None
      result_file_path = None
      model_grade = None
      line_msg = None
      now_date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

      # eval_valueが既存のModelより良ければ入れ替え
      # そうでない場合でもexit criteriaを満たしていればVOIDに保存
      if EnvironmentCommon.compareEvalValue(TRADE_SYMBOL,TRADE_PERIOD,eval_value):
        model_file_path = TRAIN_MODEL_PATH
        result_file_path = TRAIN_RESULT_PATH
      else:
        model_file_path = TRAIN_MODEL_PATH_VOID
        result_file_path = TRAIN_RESULT_PATH_VOID

      if(df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['eval_value'] >0.95):
        # 【聖杯の発見！】
        logger_root.info('TrainModel:%s_%s ★☆★☆★Grail has got!★☆★☆★' %(TRADE_SYMBOL, TRADE_PERIOD))
        model_grade = 'G'
        line_msg = '★☆★☆★Grail has got!★☆★☆★'
        # 内部関数呼び出し
        SaveParametersAndModelAndSendLine()
      # agentの訓練が完了の都度Modelが採用できるかチェックする
      # 採用条件：年間平均週間獲得pips500(=0.500)以上かつ獲得pips0における累積分布確率関数が40%以下
      # if(df_cur_gen_rank.iloc[0]['trade_avg'] >0.5 and df_cur_gen_rank.iloc[0]['cdp_0'] < 0.4):
      # 採用条件：年間平均週間獲得pips(千pips)と[1-獲得pips0における累積分布確率関数]の積が0.3より大きい
      # 2023/5/4 採用条件の変更0.3→0.6(μ=1.0かつ１σ1.0以内)
      # 2023/6/14 採用条件の変更0.6→0.3(μ=0.5かつ１σ1.0以内)
      elif(df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['eval_value'] >0.3):
        # モデルとして採用する
        logger_root.info('TrainModel:%s_%s ★☆★Silver Model has got.★☆★' %(TRADE_SYMBOL,TRADE_PERIOD))
        model_grade = 'S'
        line_msg = '★☆★Silver Model has got.★☆★'
        # 内部関数呼び出し
        SaveParametersAndModelAndSendLine()
      # 評価値が0.15より大きいモデルについても採用する
      elif(df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['eval_value'] >0.15):
        # モデルとして採用する
        logger_root.info('TrainModel:%s_%s ★★Blonze Model has got.★★' %(TRADE_SYMBOL,TRADE_PERIOD))
        model_grade = 'B'
        line_msg = '★★Blonze Model has got.★★'
        # 内部関数呼び出し
        SaveParametersAndModelAndSendLine()
      # 評価値が0.08より大きいモデルについても採用する
      elif(df_gen_rank.loc['%s%s'%(gnum+1, p+1)]['eval_value'] >0.08):
        # モデルとして採用する
        logger_root.info('TrainModel:%s_%s ★Test Model has got.★' %(TRADE_SYMBOL,TRADE_PERIOD))
        model_grade = 'T'
        line_msg = '★Test Model has got.★'
        # 内部関数呼び出し
        SaveParametersAndModelAndSendLine()
      else:
        logger_root.debug('TrainModel:Continue training.')

      # LINEへメッセージをNNモデルやiniファイルを保存したり、送ったり、CSVを保存する処理を内部関数化する
      def SaveParametersAndModelAndSendLine():
        # iniファイルとNNモデルを保存する
        ModelParameters = TrainParameters(account=model_file_path+'Account_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.ini',
                                          agent=model_file_path+'Agent_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.ini',
                                          brain=model_file_path+'Model_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.pth',
                                          eval_value=eval_value)
        ModelNNName = ModelName(model_name=TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time,
                                magic=now_date_time,
                                set_file_path=model_file_path+'EA_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.set')
        AgentList[p].SaveParameters(ModelParameters,ModelNNName)
        # CSVファイルを保存する
        df_gen_rank.loc['%s%s'%(gnum+1, p+1)].to_csv(result_file_path+'Result_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.csv')
        df_result.to_csv(result_file_path+'TrainResult_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.csv')
        df_backtest_result.to_csv(result_file_path+'BacktestResult_'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'.csv')
        # LINEにメッセージを送る
        # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'
        #                                   +TRADE_SYMBOL+'_'+TRADE_PERIOD+'_T_'+now_date_time+'\n'+line_msg+'\n'
        #                                   +str(df_gen_rank.loc['%s%s'%(gnum+1, p+1),['agt_num','result','eval_value']]))
        EnvironmentCommon.send_ntfy_message(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
                                            TRADE_SYSTEM+'\n'
                                           +TRADE_SYMBOL+'_'+TRADE_PERIOD+'_'+model_grade+'_'+now_date_time+'\n'+line_msg+'\n'
                                           +str(df_gen_rank.loc['%s%s'%(gnum+1, p+1),['agt_num','result','eval_value']]))

    # Rankの並べ替え、世代別に評価値(eval_value)の大きい順に並べ替える。その後、最新の世代のランキングだけ取り出す
    df_gen_rank.sort_values(['gen','eval_value'],ascending=[True,False],inplace=True)
    df_cur_gen_rank = df_gen_rank.query('gen == %d' %(gnum+1))
    # df_cur_gen_rank.reset_index(inplace=True, drop=True)
    logger_root.info('TrainModel:Generation %d train finished.\n%s' %(gnum+1,df_cur_gen_rank))
    df_cur_gen_rank.to_csv(TMP_TRAIN_RESULT_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Gen'+str(gnum+1)+'_Rank.csv')

    # AgentToGoの数だけ生き残り、生き残り分のACCOUNT.INI、AGRNT.INI、MODEL.PTHファイルをMODEL/TMPディレクトリに保存
    # 足りない分はNoneで見たす
    # 2023/7/16 メモリを開放する
    # AgentList.clear()
    del AgentList
    AgentList = [None] * TRAIN_AGENT_NUM
    # TrainParametersList.clear()
    del TrainParametersList
    TrainParametersList=[TrainParameters(None,None,None,None)]*TRAIN_AGENT_NUM

    # 次の世代にperiods_dictを格納するリスト
    NextPriceDataDicts = [None] * TRAIN_AGENT_NUM
    # 次の世代にtrain_data_dictを格納するリスト
    NextTrainDataDicts = [None] * TRAIN_AGENT_NUM

    for rnk in range(AGENT_TOGO_NEXT_NUM):
      AgentList[rnk] = df_cur_gen_rank.iloc[rnk]['agt_obj']
      # ACCOUNT.INI、AGRNT.INI、MODEL.PTHファイルをMODEL/TMPディレクトリに保存
      TrainParametersList[rnk] = TrainParameters(account=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Account_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.ini',
                                                 agent=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Agent_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.ini',
                                                 brain=TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'_Brain_Gen'+str(gnum+1)+'_Rank'+str(rnk+1)+'.pth',
                                                 eval_value=eval_value)
      AgentList[rnk].SaveParameters(TrainParametersList[rnk],None)
      NextPriceDataDicts[rnk] = PriceDataDicts[int(df_cur_gen_rank.iloc[rnk]['agt_num'])-1]
      NextTrainDataDicts[rnk] = TrainDataDicts[int(df_cur_gen_rank.iloc[rnk]['agt_num'])-1]

    # 2023/7/15 メモリを解放する
    #PriceDataDicts.clear()
    del PriceDataDicts
    PriceDataDicts = NextPriceDataDicts.copy()
    #NextPriceDataDicts.clear()
    del NextPriceDataDicts

    # TrainDataDicts.clear()
    del TrainDataDicts
    TrainDataDicts = NextTrainDataDicts.copy()
    # NextTrainDataDicts.clear()
    del NextTrainDataDicts
    gc.collect()

    ## For Comandline
    # TrainNumListにこれまでに完了したトレーニング数を記録してcsvファイルに書き出す
    dfTrainNumList.at[TRADE_SYMBOL,TRADE_PERIOD] = gnum+1
    dfTrainNumList.to_csv(trainnum_file_path)

    #EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]
    #                                   +'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\nGen '+str(gnum+1)
    #                                   +' Train has finished.\n'+str(df_cur_gen_rank.iloc[0,[1,2,6]]))
    EnvironmentCommon.send_ntfy_message(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
                                       TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\nGen '+str(gnum+1)
                                       +' Train has finished.\n'+str(df_cur_gen_rank.iloc[0,[1,2,6]]))

  # 所定の世代数のトレーニングが終了
  # 1-2. MODEL/TMP以下のファイルを削除
  '''
  テスト用にTMPファイルを残しておく
  for p in glob.glob(TMP_TRAIN_MODEL_PATH+TRADE_SYMBOL+'_'+TRADE_PERIOD+'*'):
    if os.path.isfile(p):
        os.remove(p)
  logger_root.info('TrainModel:TMP File Deleted.')
  '''

  # 2025/2/15 MT5との接続はしない。
  # MT5との接続を終了する
  # mt5.shutdown()
  logger_root.debug('MT5 ShutDown Completed.')

# logger_root.info('TrainModel:%s_%s ▼▼No Model has found.▼▼' %(TRADE_SYMBOL,TRADE_PERIOD))
  # EnvironmentCommon.send_line_notify(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE]+'\n'+TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\n▼▼Train has finished.▼▼')
  EnvironmentCommon.send_ntfy_message(ACCOUNT_TRADE_MODE_STR[ACCOUNT_TRADE_MODE],
                                      TRADE_SYSTEM+'\n'+TRADE_SYMBOL+'_'+TRADE_PERIOD+'\n▼▼Train has finished.▼▼')
  logger_root.info('TrainModel:%s_%s Train has finished.' %(TRADE_SYMBOL,TRADE_PERIOD))
  ## For Comandline
  # これまでのトレーニング回数をcsvファイルに書き出す
  dfTrainNumList.to_csv(trainnum_file_path)
  # トレーニング回数を標準出力で返す
  print(int(dfTrainNumList.at[TRADE_SYMBOL,TRADE_PERIOD]))
  sys.exit(0)

