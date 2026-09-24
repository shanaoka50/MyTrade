#!/usr/bin/env python
# coding: utf-8
'''
TradingCore2027.py
トレーニング環境と本番推論環境で共通して使用する基底クラス・定数定義
System2027: マルチタイムフレーム + トレーリングストップ(Action 6)対応版
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import configparser
import logging

logger_core = logging.getLogger('DRLLogging')

# 行動コードの定義 (7行動)
NO_ACTION = 0
LONG_ENTRY = 1
SHORT_ENTRY = 2
POSITION_CLOSE = 3
CLOSE_AND_LONG = 4
CLOSE_AND_SHORT = 5
TRAILING_STOP = 6  # ★新規追加: ロスカット引き上げ/引き下げ

GAMMA = 0.99


# =====================================================================
# 1. ニューラルネットワーク定義 (Brain)
# =====================================================================
class Brain(nn.Module):
    def __init__(self, input_num, output_num, hidden_num):
        super(Brain, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        self.hidden_num = hidden_num

        # LSTM + Dueling Network 構造
        self.lstm = nn.LSTM(input_num, hidden_num)
        self.lstm2action = nn.Linear(hidden_num, output_num)  # Advantage層
        self.lstm2value = nn.Linear(hidden_num, 1)            # Value層

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.hidden_state = None
        self.cell_state = None
        self.resetHiddenCellState()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, input_data):
        x = torch.FloatTensor(input_data).to(self.device)
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x.view(-1, 1, self.input_num),
            (self.hidden_state.to(self.device), self.cell_state.to(self.device))
        )

        hidden_flat = lstm_out.view(-1, self.hidden_num)
        self.action_score = self.lstm2action(hidden_flat)
        self.value_score = self.lstm2value(hidden_flat).expand(-1, self.action_score.size(1))

        # Dueling 合成: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        output = self.value_score + self.action_score - self.action_score.mean(1, keepdim=True).expand(-1, self.action_score.size(1))
        return output

    def resetHiddenCellState(self):
        self.hidden_state = torch.zeros(1, 1, self.hidden_num, device=self.device)
        self.cell_state = torch.zeros(1, 1, self.hidden_num, device=self.device)

    def decideAction(self, action_mask, action_score, episode=-2, train_mode=True):
        masked = torch.where(
            torch.BoolTensor(action_mask).to(self.device),
            action_score,
            torch.full_like(action_score, float('-inf'))
        )

        max_value, max_index = masked.max(1)

        epsilon = 0.5 * (1 / (episode + 1)) if episode >= 0 else 0.0
        if train_mode and epsilon > np.random.uniform(0, 1):
            random_scores = torch.where(
                torch.BoolTensor(action_mask).to(self.device),
                torch.rand_like(action_score),
                torch.full_like(action_score, float('-inf'))
            )
            _, action_index = random_scores.max(1)
        else:
            action_index = max_index

        return action_index, max_index

    def evaluateLossFunction(self, output, target):
        return self.criterion(output, target)

    def updateNN(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def getHiddenCellState(self):
        return self.hidden_state, self.cell_state

    def setHiddenCellState(self, state_tuple):
        self.hidden_state, self.cell_state = state_tuple

    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.resetHiddenCellState()

    def copyNN(self, state_dict):
        self.load_state_dict(state_dict)


# =====================================================================
# 2. 共通ロジック (アクションマスク・時間軸変換)
# =====================================================================
class CoreEnvironment:
    PERIOD_TIMEDELTA = {
        'M1': timedelta(minutes=1),
        'M5': timedelta(minutes=5),
        'M15': timedelta(minutes=15),
        'M30': timedelta(minutes=30),
        'H1': timedelta(hours=1),
        'H4': timedelta(hours=4),
        'H6': timedelta(hours=6),
        'H8': timedelta(hours=8),
        'H12': timedelta(hours=12),
        'D1': timedelta(days=1),
        'W1': timedelta(weeks=1),
    }

    @classmethod
    def GetPeriodTimeDelta(cls, period):
        delta = cls.PERIOD_TIMEDELTA.get(period)
        if delta is None:
            raise RuntimeError(f"Period Delta is not defined for {period}")
        return delta

    @classmethod
    def getAvailableAction(cls, profit_loss, has_long, has_short, countdown):
        """
        ポジション状態・含み損益・カウントダウンに基づくアクションマスク判定 (7行動)
        [a0: None, a1: Long, a2: Short, a3: Close, a4: Close&Long, a5: Close&Short, a6: TrailingStop]
        """
        if countdown < 1.0:
            # 評価損益がプラスの時のみ TrailingStop (a6) を選択可能
            can_trail = (profit_loss > 0.0)

            if has_long > 0.0 and has_short == 0.0:
                return [True, False, False, True, True, True, can_trail]
            elif has_short > 0.0 and has_long == 0.0:
                return [True, False, False, True, True, True, can_trail]
            elif has_long == 0.0 and has_short == 0.0:
                return [True, True, True, False, False, False, False]
            else:
                # 異常両建て状態
                return [False, False, False, True, False, False, False]
        else:
            # 取引時間外 (週末クローズなど)
            if has_long > 0.0 or has_short > 0.0:
                return [False, False, False, True, False, False, False]
            else:
                return [True, False, False, False, False, False, False]


# =====================================================================
# 3. 共通アカウント基底クラス (BaseAccount)
# =====================================================================
class BaseAccount:
    def __init__(self, symbol, period, real_spread_limit, lc_level, tp_level, pos_scale, pre_trade_weeks, margin=0.02):
        self.symbol = symbol
        self.period = period
        self.real_spread_limit = real_spread_limit
        self.lc_level = lc_level
        self.tp_level = tp_level
        self.pos_scale = pos_scale
        self.pre_trade_weeks = pre_trade_weeks
        self.margin = margin

        self.df_account = pd.DataFrame(
            columns=['symbol', 'period', 'pos_open_datetime', 'pos_open_price', 'has_long', 'has_short', 'float_pl', 'close_pl', 'countdown']
        )
        self.delta_period = CoreEnvironment.GetPeriodTimeDelta(period)

        self.pos_open_price = 0.0
        self.has_long = 0.0
        self.has_short = 0.0
        self.float_pl = 0.0
        self.close_pl = 0.0
        self.losscut_price = 0.0
        self.takeprofit_price = 0.0

    def SetAccount(self, dt, pos_open_datetime=None, pos_open_price=0.0, has_long=0.0, has_short=0.0, float_pl=0.0, close_pl=0.0, countdown=0.0):
        if dt not in self.df_account.index:
            self.df_account.loc[dt] = [
                self.symbol, self.period, pos_open_datetime, pos_open_price,
                has_long, has_short, float_pl, close_pl, countdown
            ]

    def DropAccount(self):
        self.df_account = self.df_account.drop(self.df_account.index)


# =====================================================================
# 4. System2027 専用拡張ネットワーク (Brain2027)
# =====================================================================
class Brain2027(Brain):
    def __init__(self, input_num, output_num, hidden_num):
        super(Brain2027, self).__init__(input_num, output_num, hidden_num)
        self.input_layer_norm = nn.LayerNorm(input_num)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_data):
        x = torch.FloatTensor(input_data).to(self.device)
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x.view(-1, 1, self.input_num),
            (self.hidden_state.to(self.device), self.cell_state.to(self.device))
        )

        hidden_flat = lstm_out.view(-1, self.hidden_num)
        self.action_score = self.lstm2action(hidden_flat)
        self.value_score = self.lstm2value(hidden_flat).expand(-1, self.action_score.size(1))

        output = self.value_score + self.action_score - self.action_score.mean(1, keepdim=True).expand(-1, self.action_score.size(1))
        return output


# =====================================================================
# 5. マルチタイムフレーム特徴量アライメント (MTFAligner)
# =====================================================================
class MTFAligner:
    @staticmethod
    def AlignFeatures(base_df, higher_dfs):
        combined = base_df.copy()
        for h_df in higher_dfs:
            combined = pd.merge_asof(
                combined,
                h_df,
                left_index=True,
                right_index=True,
                direction='backward'
            )
        combined.bfill(inplace=True)
        combined.ffill(inplace=True)
        return combined