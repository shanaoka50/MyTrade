#!/usr/bin/env python
# coding: utf-8
'''
DRLTraderServer_v2026.py
TradingCore2027基盤連携・Waitress本番配信用サーバー
'''
from flask import Flask, request, abort
from waitress import serve

import pandas as pd
import torch
import configparser
import logging
import logging.config
from datetime import datetime, timezone
import sys
import csv
from collections import namedtuple
import time
import ast
import gc

# 共通基盤モジュールと推論用OnLocalモジュール
from TradingCore2027 import (
    NO_ACTION, POSITION_CLOSE, CoreEnvironment
)
import DualNetTradingSystem2026_kaiOnLocal3 as ds

nndict = {}
envdict = {}
trddtldict = {}

TradeDetails = namedtuple('TradeDetails', [
    'input_filename', 'hidden_filename', 'cellstate_filename',
    'output_filename', 'maskoutput_filename'
])

app = Flask(__name__)

logconfigfile = configparser.ConfigParser()
logconfigfile.read(r'D:\ColabNotebooks\00_Common\Logging.ini', 'UTF-8')
logging.config.fileConfig(logconfigfile)
logger = logging.getLogger('DRLTraderServer')


# =====================================================================
# System別ハンドラ定義 (将来の System2027 拡張用)
# =====================================================================
def init_duelnet_2026(form, symbol, period, model_name, account_trade_mode):
    model_input_num = int(form['ModelInputNum'])
    model_hidden_num = int(form['ModelHiddenNum'])
    model_output_num = int(form['ModelOutputNum'])
    long_period = int(form['LongPeriod'])
    short_period = int(form['ShortPeriod'])
    tickvol_period = int(form['TickvolPeriod'])
    real_spread_limit = float(form['RealSpreadLimit'])
    lc_level = float(form['LCLevel'])
    tp_level = float(form['TPLevel'])
    pos_scale = float(form['PosScale'])
    pre_trade_weeks = int(form['PreTradeWeeks'])
    train_symbol_list = ast.literal_eval(form['TrainSymbolList'])

    trader = ds.Trader(
        symbol, period, 'W', [long_period, short_period, tickvol_period],
        real_spread_limit, lc_level, tp_level, pos_scale, pre_trade_weeks,
        model_input_num, model_output_num, model_hidden_num,
        account_trade_mode, model_name, train_symbol_list
    )

    env = ds.Environment(account_trade_mode)
    trader.setEnv(env)
    trader.setPeriodTimeDelta()
    trader.ExecPreTrade(datetime.now(timezone.utc))

    return trader, env


def build_input_duelnet_2026(form, profit_loss, countdown, trade_data):
    has_long = float(form['HasLong'])
    has_short = float(form['HasShort'])
    return [has_long, has_short, profit_loss, countdown, *trade_data], has_long, has_short

# =====================================================================
# System2027 ハンドラ定義
# =====================================================================
def init_system_2027(form, symbol, period, model_name, account_trade_mode):
    """System2027 専用初期化 (マルチタイムフレーム・単一通貨モデル)"""
    # System2027 は時間足によらず通貨ペアごとにモデルを解決
    # 必要パラメータを取り出し、Trader2027 を初期化
    model_input_num = int(form['ModelInputNum'])
    model_hidden_num = int(form['ModelHiddenNum'])
    model_output_num = int(form['ModelOutputNum'])
    real_spread_limit = float(form['RealSpreadLimit'])
    lc_level = float(form['LCLevel'])
    tp_level = float(form['TPLevel'])
    pos_scale = float(form['PosScale'])
    pre_trade_weeks = int(form['PreTradeWeeks'])
    train_symbol_list = ast.literal_eval(form['TrainSymbolList'])

    # 将来作成する Trader2027 オブジェクト
    trader = ds.Trader(
        symbol, period, 'W', [167, 59, 5],
        real_spread_limit, lc_level, tp_level, pos_scale, pre_trade_weeks,
        model_input_num, model_output_num, model_hidden_num,
        account_trade_mode, model_name, train_symbol_list
    )

    env = ds.Environment(account_trade_mode)
    trader.setEnv(env)
    trader.setPeriodTimeDelta()
    trader.ExecPreTrade(datetime.now(timezone.utc))
    return trader, env

def build_input_system_2027(form, profit_loss, countdown, trade_data):
    """System2027 専用入力生成"""
    has_long = float(form['HasLong'])
    has_short = float(form['HasShort'])
    # 必要に応じて時間足固有のフラグやマルチタイムフレーム特徴量を結合
    return [has_long, has_short, profit_loss, countdown, *trade_data], has_long, has_short

SYSTEM_HANDLERS = {
    'DuelNet_2026': {
        'init': init_duelnet_2026,
        'build_input': build_input_duelnet_2026
    }
    # 将来の追加用スロット:
    'System2027': {
        'init': init_system_2027,
        'build_input': build_input_system_2027
    }
}


# =====================================================================
# エンドポイント
# =====================================================================
@app.route('/init_nn', methods=['POST'])
def init_nn():
    symbol = request.form['Symbol']
    period = request.form['Period'][7:]  # "PERIOD_H1" -> "H1"
    model_name = request.form['ModelName']
    trade_system = request.form['TradeSystem']
    account_trade_mode = int(request.form['AccountTradeMode'])

    logger.info(f'[InitNN]: Symbol={symbol}, Period={period}, TradeSystem={trade_system}, ModelName={model_name}')

    handler = SYSTEM_HANDLERS.get(trade_system)
    if not handler:
        logger.error(f'[InitNN]: Unsupported TradeSystem: {trade_system}')
        abort(400)

    try:
        trader, env = handler['init'](request.form, symbol, period, model_name, account_trade_mode)
        nndict[model_name] = trader
        envdict[model_name] = env
    except Exception as e:
        logger.error(f'[InitNN]: Initialization failed for model {model_name}: {e}')
        abort(500)

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    trddtldict[model_name] = TradeDetails(
        input_filename=f"TrainDetailInput_{model_name}_{now_str}.csv",
        hidden_filename=f"TrainDetailHidden_{model_name}_{now_str}.csv",
        cellstate_filename=f"TrainDetailCellState_{model_name}_{now_str}.csv",
        output_filename=f"TrainDetailOutput_{model_name}_{now_str}.csv",
        maskoutput_filename=f"TrainDetailMaskOutput_{model_name}_{now_str}.csv"
    )

    logger.info(f'[InitNN]: Model {model_name} Initialized successfully.')
    return "0"


@app.route('/get_action', methods=['POST'])
def get_action():
    symbol = request.form['Symbol']
    period = request.form['Period']
    timeframe = period[7:]
    profit_loss = float(request.form['ProfitLoss'])
    date_time = request.form['DateTime']
    model_name = request.form['ModelName']
    long_period = int(request.form['LongPeriod'])
    short_period = int(request.form['ShortPeriod'])
    tickvol_period = int(request.form['TickVolPeriod'])
    trade_system = request.form['TradeSystem']
    account_trade_mode = int(request.form['AccountTradeMode'])

    handler = SYSTEM_HANDLERS.get(trade_system)
    if not handler:
        logger.error(f'[GetAction]: Unsupported TradeSystem: {trade_system}')
        abort(400)

    dt = datetime.strptime(date_time, '%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc)

    trader = nndict.get(model_name)
    env = envdict.get(model_name)
    if not trader or not env:
        logger.error(f'[GetAction]: Model/Env not found for {model_name}')
        abort(500)

    countdown, _, _ = env.CalcCountdown(dt, timeframe, symbol)

    has_long = float(request.form['HasLong'])
    has_short = float(request.form['HasShort'])
    train_symbol_list = ast.literal_eval(request.form['TrainSymbolList'])

    # 共通コアのアクションマスクを使用
    action_mask = CoreEnvironment.getAvailableAction(profit_loss, has_long, has_short, countdown)

    try:
        for i in range(5):
            trade_data = env.GetTradeData(dt, timeframe, symbol, long_period, short_period, tickvol_period, train_symbol_list)
            if pd.isna(trade_data).any():
                if i >= 4:
                    logger.error('[GetAction]: GetTradeData failed (NaN detected).')
                    abort(500)
                time.sleep(1)
            else:
                break
    except Exception as e:
        logger.error(f'[GetAction]: Error in GetTradeData: {e}')
        abort(500)

    input_data, _, _ = handler['build_input'](request.form, profit_loss, countdown, trade_data)

    with torch.no_grad():
        action_index = trader.agnt.DecideAction(dt, input_data, action_mask)
        state_action_values = trader.agnt.main_state_action_values
        masked_action_scores = trader.agnt.main_brain.masked_action_scores

    action_val = action_index[0].item()

    # 行動定数によるロット判定
    if action_val in [NO_ACTION, POSITION_CLOSE]:
        order_lots = 0.0
    else:
        try:
            order_lots, _, _ = env.CalcOrderLots(model_name, timeframe, symbol, account_trade_mode)
        except Exception as e:
            logger.error(f'[GetAction]: Error in CalcOrderLots: {e}')
            abort(500)

    trade_result_path = ds.TRADE_RESULT_PATH_REAL if account_trade_mode == ds.ACCOUNT_TRADE_MODE_REAL else ds.TRADE_RESULT_PATH_DEMO
    trddtls = trddtldict.get(model_name)

    if trddtls:
        try:
            with open(trade_result_path + trddtls.input_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([dt, *input_data])

            hidden_state, cell_state = trader.agnt.GetMainHiddenCellState()
            with open(trade_result_path + trddtls.hidden_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([dt, *[v for r in hidden_state.tolist() for sub in r for v in sub]])
            with open(trade_result_path + trddtls.cellstate_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([dt, *[v for r in cell_state.tolist() for sub in r for v in sub]])

            with open(trade_result_path + trddtls.output_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([dt, *[v for r in state_action_values.tolist() for v in r]])
            with open(trade_result_path + trddtls.maskoutput_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([dt, *[v for r in masked_action_scores.tolist() for v in r]])
        except Exception as e:
            logger.warning(f'[GetAction]: CSV logging failed: {e}')

    return f"{action_val},{order_lots}"


@app.route('/deinit_nn', methods=['POST'])
def deinit_nn():
    model_name = request.form.get('ModelName')
    if not model_name:
        abort(400)

    logger.info(f'[DeinitNN]: Deinitializing model {model_name}...')
    trader = nndict.pop(model_name, None)
    env = envdict.pop(model_name, None)
    details = trddtldict.pop(model_name, None)

    del trader, env, details
    gc.collect()

    logger.info(f'[DeinitNN]: Model {model_name} removed from memory.')
    return "0"


@app.route('/check_models')
def check_models():
    res = ''
    for k, v in nndict.items():
        hidden_state, cell_state = v.agnt.GetMainHiddenCellState()
        res += f'<h2>Key: {k}, ID: {id(v)}</h2><br>HiddenState: {hidden_state}<br>CellState: {cell_state}<br>'
    return res


@app.route('/')
def health_check():
    return "<h1>Waitress is Running (TradingCore2027 Integrated)</h1>"


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        logger.error('Usage: python DRLTraderServer.py [hostname] [portNo.] [threads]')
        sys.exit(1)

    logger.info(f'Server running on Host:{args[1]}, Port:{args[2]}, Threads:{args[3]}')
    serve(app, host=args[1], port=int(args[2]), threads=int(args[3]))