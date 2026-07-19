#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ГИБРИД v4.0-LITE + V2.3 — ТАКТИЧЕСКИЙ СЛОЙ АКЦИЙ (ГИБРИДНЫЙ ЗАГРУЗЧИК)
================================================================================
Версия: гибридная (локальные CSV + API Мосбиржи как fallback)
Приоритет: локальные CSV → API (если CSV не хватает или устарел)
Источник: https://github.com/ML-Finteh/Portfel
"""

import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# 0. КОНФИГУРАЦИЯ ПУТЕЙ
# ============================================================
DATA_DIR = os.environ.get('PORTFEL_DATA_DIR', '/mnt/agents/output/Portfel')

# ============================================================
# 1. ЗАГРУЗЧИК ДАННЫХ (локальные CSV + API Мосбиржи)
# ============================================================

class DataLoader:
    """
    Гибридный загрузчик:
      1. Загружает данные из локальных CSV (приоритет)
      2. Если CSV не хватает / устарел / отсутствует — запрашивает API Мосбиржи
      3. Объединяет данные, устраняя дубликаты
    """

    BASE_URL = 'https://iss.moex.com/iss'

    CSV_MAP = {
        'IMOEX': {
            'file': 'IMOEX_index 2026.07.17.csv',
            'date_col': 'date',
            'close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
        },
        'RGBITR': {
            'file': 'RGBITR_history.csv',
            'date_col': 'TRADEDATE',
            'close': 'CLOSE',
            'open': 'OPEN',
            'high': 'HIGH',
            'low': 'LOW',
        },
        'RUCBITR': {
            'file': 'RUCBTRNS_history.csv',
            'date_col': 'TRADEDATE',
            'close': 'CLOSE',
            'open': 'OPEN',
            'high': 'HIGH',
            'low': 'LOW',
        },
        'GLDRUB_TOM': {
            'file': 'GLDRUB_TOM_history_2013_2026.csv',
            'date_col': 'date',
            'close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
        },
        'CNYRUB_TOM': {
            'file': 'CNYRUB_TOM_history.csv',
            'date_col': 'TRADEDATE',
            'close': 'CLOSE',
            'open': 'OPEN',
            'high': 'HIGH',
            'low': 'LOW',
        },
    }

    # Параметры API для каждого тикера
    API_MAP = {
        'IMOEX':    {'engine': 'stock',  'market': 'index'},
        'RGBITR':   {'engine': 'stock',  'market': 'index'},
        'RUCBITR':  {'engine': 'stock',  'market': 'index'},
        'GLDRUB_TOM': {'engine': 'currency', 'market': 'selt'},
        'CNYRUB_TOM': {'engine': 'currency', 'market': 'selt'},
    }

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self._real_data = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HybridBacktest/2.0',
            'Accept': 'application/json'
        })

    def _load_csv(self, ticker):
        """Загрузка из локального CSV."""
        if ticker not in self.CSV_MAP:
            return pd.DataFrame()

        cfg = self.CSV_MAP[ticker]
        path = os.path.join(self.data_dir, cfg['file'])

        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)
        df[cfg['date_col']] = pd.to_datetime(df[cfg['date_col']])

        for std_col in ['close', 'open', 'high', 'low']:
            src = cfg.get(std_col)
            if src and src in df.columns:
                df[std_col] = pd.to_numeric(df[src], errors='coerce')

        # Исправление: нулевые close (дни без торгов) → NaN → ffill
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                zero_mask = df[col] == 0
                if zero_mask.sum() > 0:
                    df.loc[zero_mask, col] = np.nan
                df[col] = df[col].ffill()

        df = df.sort_values(cfg['date_col']).set_index(cfg['date_col'])
        df.index.name = 'TRADEDATE'
        return df

    def _fetch_moex(self, ticker, start='2006-01-01', end=None):
        """Загрузка с API Мосбиржи (ISS)."""
        if end is None:
            end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        if ticker not in self.API_MAP:
            print(f'[WARN] Нет API-параметров для {ticker}')
            return pd.DataFrame()

        api_cfg = self.API_MAP[ticker]
        all_rows = []
        start_idx = 0
        url = f"{self.BASE_URL}/history/engines/{api_cfg['engine']}/markets/{api_cfg['market']}/securities/{ticker}.json"

        try:
            while True:
                params = {'from': start, 'till': end, 'start': start_idx}
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code != 200:
                    print(f'[API ERR] {ticker}: HTTP {resp.status_code}')
                    break

                data = resp.json()
                rows = data['history']['data']
                if not rows:
                    break

                all_rows.extend(rows)
                start_idx += len(rows)
                if len(rows) < 100:
                    break

            if all_rows:
                columns = data['history']['columns']
                df = pd.DataFrame(all_rows, columns=columns)
                df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
                for col in ['CLOSE', 'OPEN', 'HIGH', 'LOW']:
                    if col in df.columns:
                        df[col.lower()] = pd.to_numeric(df[col], errors='coerce')
                df = df.sort_values('TRADEDATE').set_index('TRADEDATE')
                print(f'[API OK] {ticker}: загружено {len(df)} строк с API')
                return df
        except Exception as e:
            print(f'[API ERR] {ticker}: {e}')

        return pd.DataFrame()

    def load(self, ticker, force_api=False, min_date=None):
        """
        Гибридная загрузка:
          - force_api=True: игнорировать CSV, грузить только с API
          - min_date: если CSV заканчивается раньше этой даты — догрузить с API
        """
        df_csv = pd.DataFrame()
        df_api = pd.DataFrame()

        # Шаг 1: пробуем CSV (если не force_api)
        if not force_api:
            df_csv = self._load_csv(ticker)
            if not df_csv.empty:
                print(f'[CSV OK] {ticker}: {len(df_csv)} строк, {df_csv.index.min().date()} — {df_csv.index.max().date()}')

        # Шаг 2: проверяем, нужен ли API
        need_api = force_api or df_csv.empty
        if not need_api and min_date is not None:
            if df_csv.index.max() < pd.to_datetime(min_date):
                need_api = True
                print(f'[INFO] {ticker}: CSV устарел ({df_csv.index.max().date()} < {min_date}), догружаем с API...')

        # Шаг 3: загружаем с API если нужно
        if need_api:
            start_api = '2000-01-01'
            if not df_csv.empty:
                # Догружаем только недостающий период
                start_api = (df_csv.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')

            df_api = self._fetch_moex(ticker, start=start_api)

        # Шаг 4: объединяем
        if not df_csv.empty and not df_api.empty:
            # Убираем дубликаты по индексу, приоритет у API (свежие данные)
            df_combined = pd.concat([df_csv, df_api])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined = df_combined.sort_index()
            print(f'[MERGE] {ticker}: объединено {len(df_combined)} строк (CSV + API)')
        elif not df_api.empty:
            df_combined = df_api
        elif not df_csv.empty:
            df_combined = df_csv
        else:
            print(f'[ERR] {ticker}: данные не получены ни из CSV, ни из API')
            return pd.DataFrame()

        self._real_data[ticker] = df_combined.copy()
        return df_combined


# ============================================================
# 2. ДВИЖОК V2.3 — ТАКТИЧЕСКАЯ МОДЕЛЬ АКЦИЙ
# ============================================================

class V23Engine:
    """
    Полная реализация V2.3 для управления equity exposure.
    Возвращает ежедневный сигнал: 0 (OUT) или 1 (IN)
    """

    def __init__(self, df_imoex, 
                 ema_fast=15, ema_slow=50,
                 adx_weak=20, adx_strong=40,
                 emergency_stop_atr=1.5,
                 partial_exit_atr=4.0,
                 base_risk=0.05, max_risk=0.10,
                 entry_mode='limit_ema'):

        self.df = df_imoex.copy()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_weak = adx_weak
        self.adx_strong = adx_strong
        self.emergency_stop = emergency_stop_atr
        self.partial_exit_atr = partial_exit_atr
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.entry_mode = entry_mode

        self._calculate_indicators()
        self._generate_signals()

    def _calculate_indicators(self):
        df = self.df

        # EMA
        df['EMA_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(span=14, adjust=False).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.clip(lower=0)
        minus_dm = minus_dm.clip(lower=0)

        plus_dm = plus_dm.copy()
        minus_dm = minus_dm.copy()
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0

        atr_ema = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_ema.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_ema.replace(0, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(span=14, adjust=False).mean()

        # Режим тренда
        df['TrendRegime'] = 0
        df.loc[df['ADX'] > self.adx_weak, 'TrendRegime'] = 1
        df.loc[df['ADX'] > self.adx_strong, 'TrendRegime'] = 2

    def _generate_signals(self):
        df = self.df.copy()

        entry_cond = (df['close'] > df['EMA_fast']) & (df['close'] > df['EMA_slow'])
        exit_cond = df['close'] < df['EMA_slow']

        signal = np.zeros(len(df), dtype=int)
        in_position = False

        for i in range(len(df)):
            if not in_position:
                if entry_cond.iloc[i]:
                    if df['TrendRegime'].iloc[i] == 0:
                        signal[i] = 0
                    else:
                        signal[i] = 1
                        in_position = True
            else:
                if exit_cond.iloc[i]:
                    signal[i] = 0
                    in_position = False
                else:
                    signal[i] = 1

        df['V23_Signal'] = signal

        if self.entry_mode == 'limit_ema':
            next_open = df['open'].shift(-1)
            ema_fast_val = df['EMA_fast']
            df['V23_EntryPrice'] = np.where(next_open <= ema_fast_val, next_open, ema_fast_val)
        else:
            df['V23_EntryPrice'] = df['open'].shift(-1)

        df.loc[df.index[-1], 'V23_EntryPrice'] = df.loc[df.index[-1], 'close']
        df['V23_ExitPrice'] = df['open'].shift(-1)
        df.loc[df.index[-1], 'V23_ExitPrice'] = df.loc[df.index[-1], 'close']

        self.df = df

    def get_daily_signals(self):
        return self.df['V23_Signal']

    def get_signal_df(self):
        return self.df[['close', 'open', 'high', 'low', 'EMA_fast', 'EMA_slow', 
                        'ADX', 'TrendRegime', 'V23_Signal', 'V23_EntryPrice', 'ATR']]


# ============================================================
# 3. ГИБРИДНЫЙ БЭКТЕСТ: v4.0-LITE + V2.3
# ============================================================

class HybridBacktest:

    CASH_RATES_BY_YEAR = {
        2006: 11.25, 2007: 10.25, 2008: 10.75, 2009: 10.00, 2010: 8.00,
        2011: 8.08,  2012: 8.25,  2013: 5.50,  2014: 9.08,  2015: 11.00,
        2016: 10.50, 2017: 8.25,  2018: 7.42,  2019: 7.00,  2020: 5.08,
        2021: 6.42,  2022: 11.08, 2023: 10.58, 2024: 17.50, 2025: 17.33,
        2026: 15.08
    }

    BASE_TARGET = {
        'high_rate': {
            'gold': 0.20, 'cny_bonds': 0.25, 'money_market': 0.05,
            'floaters': 0.25, 'ofz_pk': 0.05, 'stocks': 0.15, 'real_estate': 0.05,
        },
        'transition': {
            'gold': 0.25, 'cny_bonds': 0.20, 'money_market': 0.05,
            'floaters': 0.15, 'ofz_pk': 0.10, 'stocks': 0.15, 'real_estate': 0.10,
        },
        'low_rate': {
            'gold': 0.30, 'cny_bonds': 0.15, 'money_market': 0.05,
            'floaters': 0.05, 'ofz_pk': 0.15, 'stocks': 0.15, 'real_estate': 0.15,
        }
    }

    def __init__(self, data_loader, v23_engine):
        self.data = data_loader
        self.v23 = v23_engine
        self.daily_data = None
        self.results = {}

    def _get_phase(self, rate):
        if rate > 12:
            return 'high_rate'
        elif rate > 8:
            return 'transition'
        else:
            return 'low_rate'

    def _get_enhanced_weights(self, key_rate, imoex_drop, v23_signal,
                              rate_cut_fast=False, usd_above_85=False):
        phase = self._get_phase(key_rate)
        weights = self.BASE_TARGET[phase].copy()
        base_stocks = weights['stocks']

        if v23_signal == 0:
            weights['stocks'] = 0.0
            if key_rate > 15:
                weights['money_market'] += base_stocks * 0.7
                weights['floaters'] += base_stocks * 0.3
            elif key_rate > 12:
                weights['floaters'] += base_stocks * 0.5
                weights['gold'] += base_stocks * 0.3
                weights['money_market'] += base_stocks * 0.2
            else:
                weights['ofz_pk'] += base_stocks * 0.4
                weights['gold'] += base_stocks * 0.4
                weights['money_market'] += base_stocks * 0.2
        else:
            if self.v23.df.loc[self.v23.df.index[-1], 'TrendRegime'] == 2:
                extra = 0.05
                weights['stocks'] = min(base_stocks + extra, 0.25)
                weights['money_market'] = max(weights['money_market'] - extra, 0)

        if key_rate > 15:
            extra_mm = min((key_rate - 15) / 200, 0.10)
            weights['money_market'] += extra_mm
            total_f = weights['floaters'] + weights['cny_bonds']
            if total_f > 0:
                weights['floaters'] -= extra_mm * weights['floaters'] / total_f
                weights['cny_bonds'] -= extra_mm * weights['cny_bonds'] / total_f

        if rate_cut_fast:
            shift = 0.05
            weights['floaters'] -= shift
            weights['gold'] += shift * 0.6
            weights['ofz_pk'] += shift * 0.4

        if usd_above_85:
            target_cny = 0.30
            current_cny = weights['cny_bonds']
            if current_cny < target_cny:
                delta = min(target_cny - current_cny, weights['floaters'])
                weights['cny_bonds'] += delta
                weights['floaters'] -= delta

        if imoex_drop >= 15:
            extra_stocks = min(0.10, weights['money_market'])
            weights['stocks'] += extra_stocks
            weights['money_market'] -= extra_stocks

        total = sum(weights.values())
        if total > 0:
            weights = {k: max(0.0, v / total) for k, v in weights.items()}
        total2 = sum(weights.values())
        if total2 > 0 and abs(total2 - 1.0) > 1e-9:
            weights = {k: v / total2 for k, v in weights.items()}

        return weights

    def run(self):
        imoex = self.data._real_data.get('IMOEX')
        rgbitr = self.data._real_data.get('RGBITR')
        rucbitr = self.data._real_data.get('RUCBITR')

        if imoex is None or rgbitr is None or rucbitr is None:
            raise ValueError("Отсутствуют обязательные данные: IMOEX, RGBITR, RUCBITR")

        df = imoex[['close']].rename(columns={'close': 'imoex'}).copy()

        for ticker, col_name in [('RGBITR', 'rgbitr'), ('RUCBITR', 'rucbitr'),
                                  ('GLDRUB_TOM', 'gold'), ('CNYRUB_TOM', 'cny')]:
            data = self.data._real_data.get(ticker)
            if data is not None:
                df = df.join(data[['close']].rename(columns={'close': col_name}), how='outer')

        df = df.sort_index().ffill().bfill()

        # Ключевая ставка
        kr_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        kr_df = pd.DataFrame({'date': kr_dates})
        kr_df['key_rate'] = np.nan

        RATE_PERIODS = [
            ('2006-01-01', '2006-06-25', 12.00), ('2006-06-26', '2006-10-22', 11.50),
            ('2006-10-23', '2007-01-28', 11.00), ('2007-01-29', '2007-04-29', 10.50),
            ('2007-04-30', '2007-06-18', 10.25), ('2007-06-19', '2007-07-30', 10.00),
            ('2008-02-04', '2008-04-28', 10.25), ('2008-04-29', '2008-06-09', 10.50),
            ('2008-06-10', '2008-07-11', 10.75), ('2008-07-12', '2008-11-11', 11.00),
            ('2008-11-12', '2008-12-11', 12.00), ('2008-12-12', '2009-04-23', 13.00),
            ('2009-04-24', '2009-05-13', 12.50), ('2009-05-14', '2009-06-04', 12.00),
            ('2009-06-05', '2009-07-12', 11.50), ('2009-07-13', '2009-08-09', 10.75),
            ('2009-08-10', '2009-09-13', 10.50), ('2009-09-14', '2009-09-29', 10.00),
            ('2009-09-30', '2009-10-29', 9.75),  ('2009-10-30', '2009-11-24', 9.50),
            ('2009-11-25', '2009-12-27', 9.00),  ('2009-12-28', '2010-02-23', 8.75),
            ('2010-02-24', '2010-03-28', 8.50),  ('2010-03-29', '2010-04-29', 8.00),
            ('2010-04-30', '2010-05-30', 7.75),  ('2010-05-31', '2010-06-08', 7.50),
            ('2010-06-09', '2011-02-27', 7.75),  ('2011-02-28', '2011-04-24', 8.00),
            ('2011-04-25', '2011-05-02', 8.25),  ('2011-05-03', '2011-12-25', 8.25),
            ('2011-12-26', '2012-09-13', 8.00),  ('2012-09-14', '2013-03-31', 8.25),
            ('2013-04-01', '2013-09-12', 8.25),  ('2013-09-13', '2014-03-02', 5.50),
            ('2014-03-03', '2014-03-27', 5.50),  ('2014-03-28', '2014-04-27', 7.00),
            ('2014-04-28', '2014-07-27', 7.50),  ('2014-07-28', '2014-10-30', 8.00),
            ('2014-10-31', '2014-12-10', 9.50),  ('2014-12-11', '2014-12-15', 10.50),
            ('2014-12-16', '2015-01-29', 17.00), ('2015-01-30', '2015-03-02', 15.00),
            ('2015-03-03', '2015-05-04', 14.00), ('2015-05-05', '2015-06-14', 12.50),
            ('2015-06-15', '2016-08-09', 11.00), ('2016-08-10', '2017-03-26', 10.50),
            ('2017-03-27', '2017-09-17', 9.75),  ('2017-09-18', '2017-12-17', 8.50),
            ('2017-12-18', '2018-02-11', 7.75),  ('2018-02-12', '2018-03-25', 7.50),
            ('2018-03-26', '2018-09-13', 7.25),  ('2018-09-14', '2018-12-16', 7.50),
            ('2018-12-17', '2019-06-16', 7.75),  ('2019-06-17', '2019-07-28', 7.50),
            ('2019-07-29', '2019-09-08', 7.25),  ('2019-09-09', '2020-02-09', 7.00),
            ('2020-02-10', '2020-04-26', 6.00),  ('2020-04-27', '2020-06-18', 5.50),
            ('2020-06-19', '2020-07-26', 4.50),  ('2020-07-27', '2021-03-21', 4.25),
            ('2021-03-22', '2021-04-22', 4.50),  ('2021-04-23', '2021-06-10', 5.00),
            ('2021-06-11', '2021-07-25', 5.50),  ('2021-07-26', '2021-09-12', 6.50),
            ('2021-09-13', '2022-02-27', 7.50),  ('2022-02-28', '2022-04-10', 9.50),
            ('2022-04-11', '2022-05-03', 11.00), ('2022-05-04', '2022-05-25', 14.00),
            ('2022-05-26', '2022-07-21', 11.00), ('2022-07-22', '2022-09-18', 8.00),
            ('2022-09-19', '2023-07-23', 7.50),  ('2023-07-24', '2023-08-14', 8.50),
            ('2023-08-15', '2023-12-17', 12.00), ('2023-12-18', '2024-07-28', 16.00),
            ('2024-07-29', '2024-09-15', 18.00), ('2024-09-16', '2025-07-27', 21.00),
            ('2025-07-28', '2026-07-06', 15.25),
        ]

        for s, e, rate in RATE_PERIODS:
            mask = (kr_df['date'] >= pd.to_datetime(s)) & (kr_df['date'] <= pd.to_datetime(e))
            kr_df.loc[mask, 'key_rate'] = rate

        kr_df = kr_df.set_index('date')
        kr_df['key_rate'] = kr_df['key_rate'].ffill().bfill()
        df = df.join(kr_df, how='left')

        # Макро-индикаторы
        df['imoex_peak'] = df['imoex'].rolling(window=252, min_periods=50).max()
        df['imoex_drop'] = (df['imoex_peak'] - df['imoex']) / df['imoex_peak'] * 100
        df['imoex_drop'] = df['imoex_drop'].fillna(0)

        df['key_rate_lag63'] = df['key_rate'].shift(63)
        df['rate_cut_fast'] = (df['key_rate_lag63'] - df['key_rate']) >= 2.0

        # Доходности активов
        for col in ['imoex', 'rgbitr', 'rucbitr', 'gold', 'cny']:
            if col in df.columns:
                df[f'{col}_ret'] = df[col].pct_change()

        if 'gold_ret' not in df.columns:
            df['gold_ret'] = df['rgbitr_ret'] * 0.3
        if 'cny_ret' not in df.columns:
            df['cny_ret'] = df['rgbitr_ret'] * 0.5

        df['money_ret'] = (df['key_rate'] - 0.5) / 100 / 252
        df['ofz_pk_ret'] = df['rgbitr_ret'] + 0.5 / 100 / 252
        df['floaters_ret'] = df['rucbitr_ret'] if 'rucbitr_ret' in df.columns else df['rgbitr_ret']
        df['real_estate_ret'] = df['imoex_ret'] * 0.8

        # Защита от inf/NaN в доходностях
        ret_cols = [c for c in df.columns if c.endswith('_ret')]
        for col in ret_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)

        # ИНТЕГРАЦИЯ V2.3
        v23_signals = self.v23.get_daily_signals()
        v23_df = v23_signals.to_frame(name='v23_signal')
        df = df.join(v23_df, how='left')
        df['v23_signal'] = df['v23_signal'].ffill().fillna(0).astype(int)

        weights_records = []
        for idx, row in df.iterrows():
            w = self._get_enhanced_weights(
                key_rate=row['key_rate'],
                imoex_drop=row['imoex_drop'],
                v23_signal=row['v23_signal'],
                rate_cut_fast=row.get('rate_cut_fast', False),
                usd_above_85=False
            )
            weights_records.append(w)

        weights_df = pd.DataFrame(weights_records, index=df.index)

        portfolio_ret = pd.Series(0.0, index=df.index)

        asset_map = {
            'stocks': 'imoex_ret',
            'ofz_pk': 'ofz_pk_ret',
            'floaters': 'floaters_ret',
            'gold': 'gold_ret',
            'real_estate': 'real_estate_ret',
            'cny_bonds': 'cny_ret',
            'money_market': 'money_ret',
        }

        for asset, ret_col in asset_map.items():
            if ret_col in df.columns and asset in weights_df.columns:
                portfolio_ret += weights_df[asset].fillna(0) * df[ret_col].fillna(0)

        portfolio_ret = portfolio_ret.replace([np.inf, -np.inf], 0)

        df['portfolio_ret'] = portfolio_ret
        df['portfolio_cum'] = (1 + portfolio_ret).cumprod()

        total_return = df['portfolio_cum'].iloc[-1] - 1
        n_years = len(df) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        cum_max = df['portfolio_cum'].cummax()
        drawdown = (df['portfolio_cum'] - cum_max) / cum_max
        max_dd = drawdown.min()

        ann_vol = portfolio_ret.std() * np.sqrt(252)
        sharpe = (cagr - 0.05) / ann_vol if ann_vol > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        v23_in_market = df['v23_signal'].sum() / len(df) * 100
        v23_trades = (df['v23_signal'].diff().abs().sum()) / 2

        self.results = {
            'cagr': cagr,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'ann_volatility': ann_vol,
            'sharpe': sharpe,
            'calmar': calmar,
            'n_years': n_years,
            'start_date': df.index.min().date(),
            'end_date': df.index.max().date(),
            'n_days': len(df),
            'v23_time_in_market': v23_in_market,
            'v23_trades_approx': v23_trades,
        }

        self.daily_data = df
        self.weights_df = weights_df

        return df, weights_df

    def print_report(self):
        r = self.results
        print('\n' + '╔' + '═' * 78 + '╗')
        print('║' + ' ГИБРИД v4.0-LITE + V2.3 — РЕЗУЛЬТАТЫ БЭКТЕСТА'.center(78) + '║')
        print('╠' + '═' * 78 + '╣')
        print(f"║  Период: {r['start_date']} — {r['end_date']} ({r['n_years']:.1f} лет)".ljust(78) + '║')
        print('╠' + '═' * 78 + '╣')
        print(f"║  📈 CAGR:                    {r['cagr']*100:>8.2f}%".ljust(78) + '║')
        print(f"║  📊 Общая доходность:        {r['total_return']*100:>8.2f}%".ljust(78) + '║')
        print(f"║  📉 Макс. просадка:          {r['max_drawdown']*100:>8.2f}%".ljust(78) + '║')
        print(f"║  📊 Волатильность:           {r['ann_volatility']*100:>8.2f}%".ljust(78) + '║')
        print(f"║  🎯 Sharpe:                  {r['sharpe']:>8.3f}".ljust(78) + '║')
        print(f"║  🎯 Calmar:                  {r['calmar']:>8.3f}".ljust(78) + '║')
        print('╠' + '═' * 78 + '╣')
        print(f"║  ⚡ V2.3 статистика:".ljust(78) + '║')
        print(f"║     • Время в рынке:         {r['v23_time_in_market']:>6.1f}%".ljust(78) + '║')
        print(f"║     • Примерное число сделок: {r['v23_trades_approx']:>6.0f}".ljust(78) + '║')
        print('╚' + '═' * 78 + '╝')


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print('=' * 80)
    print('ГИБРИД v4.0-LITE + V2.3 — Гибридный загрузчик (CSV + API Мосбиржи)')
    print('=' * 80)

    loader = DataLoader()

    print('\n--- Загрузка обязательных данных ---')
    loader.load('IMOEX')
    loader.load('RGBITR')
    loader.load('RUCBITR')

    print('\n--- Загрузка опциональных данных ---')
    loader.load('GLDRUB_TOM')
    loader.load('CNYRUB_TOM')

    if 'IMOEX' not in loader._real_data:
        print('КРИТИЧЕСКАЯ ОШИБКА: Нет данных IMOEX')
        return

    print('\n--- Инициализация V2.3 Engine ---')
    v23 = V23Engine(loader._real_data['IMOEX'])
    print(f'V2.3 готов: {len(v23.df)} дней')
    print(f'Сигналов IN: {v23.df["V23_Signal"].sum()} дней')
    print(f'Сигналов OUT: {(v23.df["V23_Signal"] == 0).sum()} дней')

    print('\n--- Запуск гибридного бэктеста ---')
    engine = HybridBacktest(loader, v23)
    df, weights = engine.run()
    engine.print_report()

    out_dir = os.environ.get('PORTFEL_OUTPUT_DIR', '/mnt/agents/output')
    weights_path = os.path.join(out_dir, 'hybrid_weights.csv')
    equity_path = os.path.join(out_dir, 'hybrid_equity.csv')

    weights.to_csv(weights_path)
    df[['portfolio_cum', 'v23_signal']].to_csv(equity_path)
    print(f'\n[OK] Результаты сохранены: {weights_path}, {equity_path}')

    report_path = os.path.join(out_dir, 'hybrid_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(engine.results, f, indent=2, default=str)
    print(f'[OK] JSON-отчёт: {report_path}')

if __name__ == '__main__':
    main()
