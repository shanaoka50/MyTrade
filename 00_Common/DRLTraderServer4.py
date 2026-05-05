#!/usr/bin/env python
# coding: utf-8
'''
DuelingNetにも対応させる。
MT5から送られてくるリクエストのTRADE_SYSTEMの種類により
初期化するModel,Environmentを変える

System2026(DuelNet_2026)にも対応させる
System2026はBrainを初期化するのではなく、Traderを初期化する
'''
from flask import Flask,request,abort
from waitress import serve

import torch
import DRLTraderOnLocal as dr # DQN2023用
import DuelNetTraderOnLocalKai as dn # DuelNet用
import DualNetTradingSystem2026_kaiOnLocal as ds # DuelNet_2026用
import configparser
import logging
import logging.config
from datetime import datetime,timezone,timedelta
import sys
import csv
from collections import namedtuple
import pandas as pd
import time

# NNのobjectを格納する辞書
# nndict = {'EURUSD_H1_20230327130606':nnobj_eurusd_h1,'USDJPY_H1_20230329014443':nnobj_usdjpy_h1...}
# というイメージ
# サーバ実行中は同じクラス変数として保持する
nndict = {}
envdict ={}
# トレードの詳細情報を格納する辞書
trddtldict = {}

# 詳細のテストの情報(input,hidden,cellstate,output,maskoutput)を書き出すファイル名
input_filename = ''
hidden_filename = ''
cellstate_filename = ''
output_filename = ''
maskoutput_filename = ''

app = Flask(__name__)

# Loggerの設定
logconfigfile = configparser.ConfigParser()
logconfigfile.read(r'D:\ColabNotebooks\00_Common\Logging.ini','UTF-8')
logging.config.fileConfig(logconfigfile)
logger_trader_server = logging.getLogger('DRLTraderServer')
logger_trader_server.debug('Debug level massage.')
logger_trader_server.info('Info level massage.')
logger_trader_server.warning('Warning level massage.')
logger_trader_server.error('Error level massage.')
logger_trader_server.critical('Critical level massage.')

#チャートにEAを投入した際にinit()にて呼び出される想定
#SymbolとPeriodに対応したニューラルネットワークを初期化して、
#Brainクラスを格納する
@app.route('/init_nn', methods=['POST'])
def init_nn():

    # Model NNを初期化する。
    # 同時にこのモデルのTp/LCレベルを返して、EAに格納させる
    symbol = request.form['Symbol']
    period = request.form['Period'][7:] # PERIOD_H1
    model_name = request.form['ModelName']
    trade_system = request.form['TradeSystem']
    account_trade_mode = int(request.form['AccountTradeMode'])
    logger_trader_server.info('[InitNN]:Symbol=%s, Period=%s, TradeSystem=%s, ModelName=%s.' %(symbol,period,trade_system,model_name))

    if trade_system == 'DQN2023':
        brain = dr.Brain()
        result_cd = brain.LoadModel(model_name,account_trade_mode)
    elif trade_system == 'DuelNet':
        brain = dn.Brain()
        result_cd = brain.LoadModel(model_name,account_trade_mode)
    elif trade_system == 'DuelNet_Kai':
        model_input_num = int(request.form['ModelInputNum'])
        model_hidden_num = int(request.form['ModelHiddenNum'])
        model_output_num = int(request.form['ModelOutputNum'])
        logger_trader_server.info('[InitNN]:ModelInputNum=%s, ModelHiddenNum=%s, ModelOutputNum=%s.' %(model_input_num,model_hidden_num,model_output_num))
        brain = dn.Brain(model_input_num,model_hidden_num,)
        result_cd = brain.LoadModel(model_name,account_trade_mode)
    elif trade_system == 'DuelNet_2026':
        model_input_num = int(request.form['ModelInputNum'])
        model_hidden_num = int(request.form['ModelHiddenNum'])
        model_output_num = int(request.form['ModelOutputNum'])

        long_period = int(request.form['LongPeriod'])
        short_period = int(request.form['ShortPeriod'])
        tickol_period = int(request.form['TickvolPeriod'])

        real_spread_limit = float(request.form['RealSpreadLimit'])
        lc_level = float(request.form['LcLevel'])
        tp_level = float(request.form['TpLevel'])
        pos_scale = float(request.form['PosScale'])
        pre_trade_weeks = int(request.form['PreTradeWeeks'])

        train_symbol_list = eval(request.form['TrainSymbolList'])

        logger_trader_server.info('[InitNN]:ModelInputNum=%s, ModelHiddenNum=%s, ModelOutputNum=%s.' %(model_input_num,model_hidden_num,model_output_num))

        # def __init__(self,sbl,tf,now_dt,tdur,prds,rsl,lcl,tpl,ps,ptw,ipn,opn,hdn,trdmd,mdlnm):
        # テスト期間は週で固定 datetime.now(timezone.utc)は使わない
        trader = ds.Trader(symbol,period,'W',[long_period,short_period,tickol_period],real_spread_limit,
                          lc_level,tp_level,pos_scale,pre_trade_weeks,model_input_num,model_output_num,model_hidden_num,account_trade_mode,
                          model_name,train_symbol_list)
        '''
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
        
    logger_trader_server.debug('[InitNN]:ResultCD %d' %(result_cd))

    if result_cd == 1:
        logger_trader_server.warning('[InitNN]:Model %s Initialize Failed.'%(model_name))
        return str(1)
    
    # MT5の接続が切れないようにEnvironmentもオブジェクト化する
    try:
      if trade_system == 'DQN2023':
        env = dr.Environment(account_trade_mode)
      elif trade_system == 'DuelNet':
        env = dn.Environment(account_trade_mode)
      elif trade_system == 'DuelNet_Kai':
        env = dn.Environment(account_trade_mode)
      elif trade_system == 'DuelNet_2026':
        env = ds.Environment(account_trade_mode)
        trader.setEnv(env)
        trader.setPeriodTimeDelta()
    except SystemExit as e:
      logger_trader_server.error('[init_nn]:SystemExit occured.')
      logger_trader_server.error(e)
      abort(500)

    # breinのHiddenStateとCellStateを確認する
    hidden_state, cell_state = brain.GetHiddenCellState()
    logger_trader_server.debug('[InitNN]:HiddenState %s \n CellState %s' %(hidden_state,cell_state))
    #作成したBrainObjectをdicに格納する。
    nndict[model_name] = brain
    envdict[model_name] = env

    # DuelNet_2026の時は、空トレードをする
    if trade_system == 'DuelNet_2026':
        trader.ExecPreTrade(datetime.now(timezone.utc))

    # 詳細のテストの情報(input,hidden,cellstate,output,maskoutput)を書き出すファイル名
    TradeDetailes = namedtuple('TradeDetailes', ['input_filename','hidden_filename','cellstate_filename','output_filename','maskoutput_filename'])
    now_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    trddtldict[model_name] = TradeDetailes(input_filename = dr.TRADE_DETAIL_INPUT_FILE_NAME+'_'+model_name+'_'+now_date_time+'.csv',
                                           hidden_filename = dr.TRADE_DETAIL_HIDDEN_FILE_NAME+'_'+model_name+'_'+now_date_time+'.csv',
                                           cellstate_filename = dr.TRADE_DETAIL_CELLSTATE_FILE_NAME+'_'+model_name+'_'+now_date_time+'.csv',
                                           output_filename = dr.TRADE_DETAIL_OUTPUT_FILE_NAME+'_'+model_name+'_'+now_date_time+'.csv',
                                           maskoutput_filename = dr.TRADE_DETAIL_MASKOUTPUT_FILE_NAME+'_'+model_name+'_'+now_date_time+'.csv')

    logger_trader_server.info('[InitNN]: Model %s Initialized.' %(model_name))
    logger_trader_server.debug('[InitNN]:input_filename=%s, hidden_filename=%s, cellstate_filename=%s, output_filename=%s, maskoutput_filename=%s' %(trddtldict[model_name].input_filename,trddtldict[model_name].hidden_filename,trddtldict[model_name].cellstate_filename,trddtldict[model_name].output_filename,trddtldict[model_name].maskoutput_filename))
    return str(0)

# 時間足が更新された時の価格データを受け取り、アクションを返す
@app.route('/get_action', methods=['POST'])
def get_action():
    '''
    Input
    0_Symbol
    1_Period
    2_ProfitLoss：現在の損益(pips/1000→USDJPY1円相当=1,000pips→ProfitLoss=1.000)。
    [DQN2023]
    3_OrderType：ポジション(OP_BUY=1,OP_SELL=-1,NO_POSITION=0)
    [DuelNet]
    3_1_HasLong:Longポジション(保有=1,保有せず=0)
    3_2_HasShort:Shortポジション(保有=1,保有せず=0)
    4_DateTime：YYYY.MM.DD HH:MM:SS形式
    5_ModelName
    6_LongPeriod(長期移動平均の期間）
    7_ShortPeriod(短期移動平均の期間）
    8_TickVolPeriod(ティックボリュームの移動平均期間）
    9_AccountTradeMode:口座のモード(デモ、リアル）
        0:ACCOUNT_TRADE_MODE_DEMO,1:ACCOUNT_TRADE_MODE_CONTEST,2:ACCOUNT_TRADE_MODE_REAL
    Return 戻り値はカンマで区切る
    0_Action
    1_Lots
    
    ActionIndex
    a0:何もしない  
    a1:LongEntry  
    a2:ShortEntry  
    a3:PositionClose  
    a4:PositionClose&LongEntry  
    a5:PositionClose&ShortEntry
    '''

    # POSTされたFromには、上記の情報がdic型に格納されている
    logger_trader_server.debug('[GetAction]:request.form %s' %(request.form))
    symbol = request.form['Symbol']
    period = request.form['Period']
    profit_loss = float(request.form['ProfitLoss'])
    date_time = request.form['DateTime']
    model_name = request.form['ModelName']
    long_period = int(request.form['LongPeriod'])
    short_period = int(request.form['ShortPeriod'])
    tickvol_period = int(request.form['TickVolPeriod'])
    trade_system = request.form['TradeSystem']
    account_trade_mode = int(request.form['AccountTradeMode'])

    train_symbol_list=None
    if trade_system == 'DQN2023':
      order_type = float(request.form['OrderType'])
    elif trade_system == 'DuelNet':
      has_long = float(request.form['HasLong'])
      has_short = float(request.form['HasShort'])
    elif trade_system == 'DuelNet_Kai':
      has_long = float(request.form['HasLong'])
      has_short = float(request.form['HasShort'])
      # fortest
      train_symbol_list = eval(request.form['TrainSymbolList'])
      logger_trader_server.debug('[InitNN]:TrainSymbolList=%s.' %(train_symbol_list))

    # TradeSystemにより、order_typeを分けて取得するように実装する
    # DateTime(String)をpythonのDateTimeに変換する
    dt_format = '%Y.%m.%d %H:%M:%S'
    logger_trader_server.debug('[GetAction]:date_time %s' %(date_time))
    dt = datetime.strptime(date_time,dt_format)
    dt = dt.replace(tzinfo=timezone.utc)
    logger_trader_server.debug('[GetAction]:dt %s' %(dt.tzinfo))

    # MT5 clientから送信されてくる時間軸(Period)の文字列はPERIOD_XXであるため
    # PERIOD_を削除する
    timeframe = period[7:]
    
    # nndictからmodel_nnと、envdictからEnvironmentを取得する
    brain = nndict[model_name]
    env = envdict[model_name]
    
    # 1.COUNTDOWNを計算する(sblとtfはスレッドが正しく処理されていることを確認するためのもの)
    countdown, sbl, tf = env.CalcCountdown(dt,timeframe,symbol)
    logger_trader_server.debug('[GetAction]:%s,%s,%s countdown:%.4f' %(model_name,sbl,tf,countdown))
    # 2. Accountの状態('has_position','countdown')から、取り得るアクションを限定する
    if trade_system == 'DQN2023':
      action_mask, sbl, tf = env.GetAvailableAction(order_type, countdown, timeframe, symbol)
    elif trade_system == 'DuelNet':
      action_mask, sbl, tf = env.GetAvailableAction(has_long, has_short, countdown, timeframe, symbol)
    elif trade_system == 'DuelNet_Kai':
      action_mask, sbl, tf = env.GetAvailableAction(has_long, has_short, countdown, timeframe, symbol)
    logger_trader_server.debug('[GetAction]:%s,%s,%s action_mask:%s' %(model_name,sbl,tf,action_mask))
    # 3.Trade情報を作成する
    try:
      # 加工したtrade_dataにNaNが発生していないかを確認する
      # NaNがあった場合は、5回再取得する
      for i in range(5):
        if trade_system == 'DQN2023':
          trade_data, sbl, tf = env.GetTradeData(dt,timeframe,symbol,account_trade_mode,long_period,short_period,tickvol_period)
        elif trade_system == 'DuelNet':
          trade_data, sbl, tf = env.GetTradeData(dt,timeframe,symbol,account_trade_mode,long_period,short_period,tickvol_period,train_symbol_list)
        elif trade_system == 'DuelNet_Kai':
          trade_data, sbl, tf = env.GetTradeData(dt,timeframe,symbol,account_trade_mode,long_period,short_period,tickvol_period,train_symbol_list)
        logger_trader_server.debug('[GetAction]:%s,%s,%s trade_data:%s' %(model_name,sbl,tf,trade_data))
        logger_trader_server.debug('[GetAction]:%s,%s,%s :pd.isna(trade_data).any(): %s' %(model_name,sbl,tf,pd.isna(trade_data).any()))
        if pd.isna(trade_data).any():
          logger_trader_server.debug('[GetAction]:%s,%s,%s :pd.isna(trade_data):NG %d times.' %(model_name,sbl,tf,i+1))
          # 5回再試行してもNGの場合はサーバエラーを返す
          if i >=4:
            logger_trader_server.error('[GetAction]:%s,%s,%s :env.GetTradeData() failed. trade_data:%s'%(model_name,sbl,tf,trade_data))
            abort(500)
          else:
            # 1分待つ 2024/8/17 wait→sleepに変更
            # time.wait(60)
            time.sleep(60)
        else:
          #NaNがなかった場合は、forループを抜ける
          logger_trader_server.debug('[GetAction]:%s,%s,%s :pd.isna(trade_data):OK.' %(model_name,sbl,tf))
          break
    except SystemExit as e:
      logger_trader_server.error('[GetAction]:SystemExit occured.')
      logger_trader_server.error(e)
      abort(500)
    # 4.Account情報とTrade情報を結合する
    if trade_system == 'DQN2023':
      input_data = [order_type,profit_loss,countdown,*trade_data]
    elif trade_system == 'DuelNet':
      input_data = [has_long,has_short,profit_loss,countdown,*trade_data]
    elif trade_system == 'DuelNet_Kai':
      input_data = [has_long,has_short,profit_loss,countdown,*trade_data]
    # 5.Model NN にて各行動の評価値を取得する
    with torch.no_grad():
        state_action_values = brain(input_data)
    logger_trader_server.debug('[GetAction]:%s state_action_values:%s' %(model_name,state_action_values))
    # 6.ActinoMaskを考慮して、次の行動を決定する
    acition_index, masked_action_scores = brain.DecideAction(action_mask, state_action_values)
    logger_trader_server.debug('[GetAction]:%s,%s,%s acition_index:%s' %(model_name,sbl,tf,acition_index[0].item()))
    
    # 7.現在の口座残高とモデル別ポートフォリオから購入ロットを計算する
    if acition_index in [0,3]:
        # Action が「何もしない」「クローズ」の場合はロットの計算はしない
        order_lots = 0
    else:
        # Action が「LongEntry」「ShortEntry」「PositionClose&LongEntry」「PositionClose&ShortEntry」の場合は購入ロットの計算をする
        try:
          order_lots, sbl, tf = env.CalcOrderLots(model_name,timeframe,symbol,account_trade_mode)
        except SystemExit as e:
          logger_trader_server.error('[get_action]:SystemExit occured.')
          logger_trader_server.error(e)
          abort(500)
    # 8. brainは、cell state が変更されているため、辞書に再度格納する
    # ポインタ参照のため、この処理はいらない？
    nndict[model_name] = brain
    envdict[model_name] = env
    
    #logger_trader_server.debug('[GetAction]:HiddenState %s \n CellState %s' %(hidden_state,cell_state))
    
    #logger_trader_server.debug('[GetAction]:%s,%s,%s order_lots:%s' %(model_name,sbl,tf,order_lots))
    
    #9. (option)トレードの詳細をファイルに書き出す。
    # トレードモードによってパスを分ける
    if account_trade_mode == dr.ACCOUNT_TRADE_MODE_REAL:
        #TRADE_MODEL_PATH = dr.TRADE_MODEL_REAL_PATH
        TRADE_RESULT_PATH = dr.TRADE_RESULT_PATH_REAL
    else:
        #TRADE_MODEL_PATH = dr.TRADE_MODEL_DEMO_PATH
        TRADE_RESULT_PATH = dr.TRADE_RESULT_PATH_DEMO
        
    trddtls = trddtldict[model_name]
    logger_trader_server.debug('[GetAction]:model_name=%s' %(model_name))
    # input情報
    logger_trader_server.debug('[GetAction]:input_filename=%s' %(TRADE_RESULT_PATH+trddtls.input_filename))
    with open(TRADE_RESULT_PATH+trddtls.input_filename, mode='a',newline='') as f:
        writer = csv.writer(f)
        data = [dt,*input_data]
        writer.writerow(data)
    # breinのHiddenStateとCellStateを確認する
    hidden_state, cell_state = brain.GetHiddenCellState()
    logger_trader_server.debug('[GetAction]:hidden_filename=%s' %(TRADE_RESULT_PATH+trddtls.hidden_filename))
    with open(TRADE_RESULT_PATH+trddtls.hidden_filename, mode='a',newline='') as f:
        writer = csv.writer(f)
        data = [dt]
        lst = hidden_state.tolist()
        for l in lst:
            for m in l:
                data.extend(m)
        writer.writerow(data)
    logger_trader_server.debug('[GetAction]:cellstate_filename=%s' %(TRADE_RESULT_PATH+trddtls.cellstate_filename))
    with open(TRADE_RESULT_PATH+trddtls.cellstate_filename, mode='a',newline='') as f:
        writer = csv.writer(f)
        data = [dt]
        lst = cell_state.tolist()
        for l in lst:
            for m in l:
                data.extend(m)
        writer.writerow(data)
    # output,maskoutput情報
    logger_trader_server.debug('[GetAction]:output_filename=%s' %(TRADE_RESULT_PATH+trddtls.output_filename))
    with open(TRADE_RESULT_PATH+trddtls.output_filename, mode='a',newline='') as f:
        writer = csv.writer(f)
        data = [dt]
        lst = state_action_values.tolist()
        for l in lst:
            data.extend(l)
        writer.writerow(data)
    logger_trader_server.debug('[GetAction]:maskoutput_filename=%s' %(TRADE_RESULT_PATH+trddtls.maskoutput_filename))
    with open(TRADE_RESULT_PATH+trddtls.maskoutput_filename, mode='a',newline='') as f:
        writer = csv.writer(f)
        data = [dt]
        lst = masked_action_scores.tolist()
        for l in lst:
            data.extend(l)
        writer.writerow(data)
    
    res = str(acition_index[0].item())+','+str(order_lots)
    return res

@app.route('/check_models')
def check_models():
    res = ''
    for k,v in nndict.items():
        print('Key:%s, ID:%s' %(k,id(v)))
        hidden_state, cell_state = v.GetHiddenCellState()
        res += '<h2>Key:%s, ID:%s</h2><br>HiddenState:%s<br>CellState:%s<br>' %(k,id(v),hidden_state,cell_state)     
    return res

@app.route('/')
def hello_world():
    print('hello_world:Hello Waitress!')
    return "<h1>Hello Waitress!</h1>"

@app.route('/index')
def hello_index():
    print('hello_indrx:Hello Index!')
    return "<h1>Hello Index!</h1>"
    
@app.route('/post_test', methods=['POST']) #Methodを明示する必要あり
def post_test():
  open = '0'
  high = '0'
  low = '0'
  close ='0'
  
  if request.method == 'POST':
    open = request.form['Open']
    close = request.form['Close']
    high = request.form['High']
    low = request.form['Low']
    
  print('[Post]:Open=%s, High=%s, Low=%s, Close=%s' %(open,high,low,close))
  return '[Post]:Open=%s, High=%s, Low=%s, Close=%s' %(open,high,low,close)

@app.route('/get_test', methods=['GET']) #Methodを明示する必要はない
def get_test():
  open = '0'
  high = '0'
  low = '0'
  close ='0'
  
  if request.method == 'GET':
    open = request.args.get('Open')
    close = request.args.get('Close')
    high = request.args.get('High')
    low = request.args.get('Low')
    
  print('[Get]:Open=%s, High=%s, Low=%s, Close=%s' %(open,high,low,close))
  return '[Get]:Open=%s, High=%s, Low=%s, Close=%s' %(open,high,low,close)

@app.route('/error_test', methods=['POST']) #Methodを明示する必要あり
def error_test():
  # 疑似的にエラーを発生させる
  logger_trader_server.debug('[ErrorTest]:start')
  # envdictからEnvironmentを取得する
  model_name = request.form['ModelName']
  env = envdict[model_name]
  try:
    env.GenerateSysExit()
  except SystemExit as e:
    logger_trader_server.error('[ErrorTest]:SystemExit occured.')
    logger_trader_server.error(e)
    abort(500)
    return '6,0'
  logger_trader_server.debug('[ErrorTest]:end')
  return '6,0'

if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=80)
    # serve(app, host='0.0.0.0', port=80)
    # serve(app, host='localhost', port=8088)
    # ホスト名とポート番号をpythonのコマンドライン引数から取得するようにする
    args = sys.argv
    if (args[1] == None) or (args[2] == None):
        logger_trader_server.error('Usage: python DRLTraderServer.py [hostname] [portNo.] [threads]')
    else:
        logger_trader_server.info('Server running on Host:%s, Port:%s, Threads:%s' %(args[1],args[2],args[3]))
    
    serve(app, host=args[1], port=int(args[2]), threads=int(args[3]))

