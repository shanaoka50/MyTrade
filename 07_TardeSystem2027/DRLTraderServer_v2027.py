#!/usr/bin/env python
# coding: utf-8
'''
DRLTraderServer (System2026専用・将来のSystem2027拡張対応版)
'''
from flask import Flask, request, abort
from waitress import serve

import pandas as pd
import torch
import DualNetTradingSystem2026_kaiOnLocal3 as ds  # DuelNet_2026用
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

# グローバルオブジェクト保持
nndict = {}
envdict = {}
trddtldict = {}

TradeDetails = namedtuple('TradeDetails', [
    'input_filename', 'hidden_filename', 'cellstate_filename',
    'output_filename', 'maskoutput_filename'
])

app = Flask(__name__)

# Logger設定
logconfigfile = configparser.ConfigParser()
logconfigfile.read(r'D:\ColabNotebooks\00_Common\Logging.ini', 'UTF-8')
logging.config.fileConfig(logconfigfile)
logger = logging.getLogger('DRLTraderServer')


# =====================================================================
# System別ハンドラ定義 (将来の System2027 追加時はここに関数を追加)
# =====================================================================

def init_duelnet_2026(form, symbol, period, model_name, account_trade_mode):
    """DuelNet_2026 専用の初期化処理"""
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

    # 空トレードの実行
    trader.ExecPreTrade(datetime.now(timezone.utc))

    return trader, env


def build_input_duelnet_2026(form, profit_loss, countdown, trade_data):
    """DuelNet_2026 専用の入力データ作成"""
    has_long = float(form['HasLong'])
    has_short = float(form['HasShort'])
    input_data = [has_long, has_short, profit_loss, countdown, *trade_data]
    return input_data, has_long, has_short


# システム名とハンドラのマッピング辞書
SYSTEM_HANDLERS = {
    'DuelNet_2026': {
        'init': init_duelnet_2026,
        'build_input': build_input_duelnet_2026
    }
    # 将来の追加例:
    # 'System2027': {
    #     'init': init_system_2027,
    #     'build_input': build_input_system_2027
    # }
}


# =====================================================================
# API エンドポイント
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
        logger.error(f'[InitNN]: Initialization failed for model {model_name}')
        logger.error(e)
        abort(500)

    # ログ出力用ファイル名の生成 (ds/dr からパスを参照)
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

    # 日時変換 (UTC)
    dt = datetime.strptime(date_time, '%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc)

    trader = nndict.get(model_name)
    env = envdict.get(model_name)
    if not trader or not env:
        logger.error(f'[GetAction]: Model or Env not found for {model_name}')
        abort(500)

    # 1. カウントダウン計算
    countdown, _, _ = env.CalcCountdown(dt, timeframe, symbol)

    # 2. アクションマスクの作成
    has_long = float(request.form['HasLong'])
    has_short = float(request.form['HasShort'])
    train_symbol_list = ast.literal_eval(request.form['TrainSymbolList'])

    action_mask = env.getAvailableAction(profit_loss, has_long, has_short, countdown)
    logger.debug(f'[GetAction]: {model_name},{symbol},{timeframe} action_mask: {action_mask}')

    # 3. Trade情報 (OHLCV加工データ) 取得
    try:
        for i in range(5):
            trade_data = env.GetTradeData(dt, timeframe, symbol, long_period, short_period, tickvol_period, train_symbol_list)
            if pd.isna(trade_data).any():
                if i >= 4:
                    logger.error(f'[GetAction]: GetTradeData failed (NaN detected).')
                    abort(500)
                time.sleep(1)
            else:
                break
    except Exception as e:
        logger.error(f'[GetAction]: Error in GetTradeData: {e}')
        abort(500)

    # 4. 入力データの結合
    input_data, _, _ = handler['build_input'](request.form, profit_loss, countdown, trade_data)

    # 5. 推論の実行
    with torch.no_grad():
        action_index = trader.agnt.DecideAction(dt, input_data, action_mask)
        state_action_values = trader.agnt.main_state_action_values
        masked_action_scores = trader.agnt.main_brain.masked_action_scores

    action_val = action_index[0].item()
    logger.debug(f'[GetAction]: {model_name},{symbol},{timeframe} action_index: {action_val}')

    # 6. ロット計算
    if action_val in [0, 3]:  # NO_ACTION または POSITION_CLOSE
        order_lots = 0.0
    else:
        try:
            order_lots, _, _ = env.CalcOrderLots(model_name, timeframe, symbol, account_trade_mode)
        except Exception as e:
            logger.error(f'[GetAction]: Error in CalcOrderLots: {e}')
            abort(500)

    # 7. (オプション) 詳細ログ出力
    trade_result_path = ds.TRADE_RESULT_PATH_REAL if account_trade_mode == ds.ACCOUNT_TRADE_MODE_REAL else ds.TRADE_RESULT_PATH_DEMO
    trddtls = trddtldict[model_name]

    try:
        # Inputログ
        with open(trade_result_path + trddtls.input_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([dt, *input_data])

        # Hidden / CellStateログ
        hidden_state, cell_state = trader.agnt.GetMainHiddenCellState()
        with open(trade_result_path + trddtls.hidden_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([dt, *[val for row in hidden_state.tolist() for sub in row for val in sub]])
        with open(trade_result_path + trddtls.cellstate_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([dt, *[val for row in cell_state.tolist() for sub in row for val in sub]])

        # Output / MaskedOutputログ
        with open(trade_result_path + trddtls.output_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([dt, *[val for row in state_action_values.tolist() for val in row]])
        with open(trade_result_path + trddtls.maskoutput_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([dt, *[val for row in masked_action_scores.tolist() for val in row]])
    except Exception as e:
        logger.warning(f'[GetAction]: Failed to write trade detail CSVs: {e}')

    return f"{action_val},{order_lots}"



@app.route('/deinit_nn', methods=['POST'])
def deinit_nn():
    model_name = request.form.get('ModelName')
    if not model_name:
        abort(400)

    logger.info(f'[DeinitNN]: Deinitializing model {model_name}...')

    # 辞書から安全に削除 (存在しない場合でもエラーにしない)
    trader = nndict.pop(model_name, None)
    env = envdict.pop(model_name, None)
    details = trddtldict.pop(model_name, None)

    # オブジェクト参照を切断
    del trader
    del env
    del details

    # 明示的なメモリ解放
    gc.collect()

    logger.info(f'[DeinitNN]: Model {model_name} successfully removed from memory.')
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
    return "<h1>Waitress is Running (DuelNet_2026 Ready)</h1>"


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        logger.error('Usage: python DRLTraderServer.py [hostname] [portNo.] [threads]')
        sys.exit(1)

    logger.info(f'Server running on Host:{args[1]}, Port:{args[2]}, Threads:{args[3]}')
    serve(app, host=args[1], port=int(args[2]), threads=int(args[3]))