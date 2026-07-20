#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГИБРИД v5.3.0-ASYNC — Тактическая модель управления портфелем российских активов
ИСПРАВЛЕННАЯ ВЕРСИЯ: автоопределение файлов по префиксу + автоопределение колонок

Версия: 5.3.0-prefix
Дата: 2026-07-20

Исправления:
  • CSV_MAP использует префиксы — файлы определяются по начальному слову
  • Автоопределение колонок date/value/high/low для разных форматов CSV
  • Корректная обработка M2 (фильтрация по индикатору)
  • Graceful degradation при недоступности API
  • Поддержка CNYRUB из репозитория

Зависимости:
  pip install pandas numpy httpx matplotlib

Автор: ML-Finteh Team (оригинал), adapted
Лицензия: MIT
"""

import os
import sys
import math
import logging
import time
import hashlib
import json
import asyncio
import warnings
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

# Async HTTP
import httpx

# Matplotlib
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# КОНФИГУРАЦИЯ (Dataclass)
# ============================================================================

@dataclass
class ModelConfig:
    """Централизованная конфигурация модели."""

    data_dir: str = field(default_factory=lambda: os.environ.get(
        'PORTFEL_DATA_DIR', os.path.dirname(os.path.abspath(__file__))))
    output_dir: str = field(default_factory=lambda: os.environ.get(
        'PORTFEL_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')))
    cache_dir: str = field(default_factory=lambda: os.environ.get(
        'PORTFEL_CACHE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')))

    api_timeout: int = 30
    api_max_retries: int = 3
    api_retry_delay: float = 1.5
    api_concurrency: int = 8
    cache_ttl_hours: int = 6
    max_pagination_pages: int = 50

    freshness_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'daily_prices': 3,
        'key_rate': 7,
        'monthly_macro': 60,
        'fx_rates': 3,
    })

    rebalance_threshold: float = 0.03
    transaction_cost: float = 0.0005
    slippage: float = 0.0002
    tax_rate: float = 0.13
    risk_free_rate_source: str = 'key_rate'
    max_leverage: float = 1.0

    adx_period: int = 14
    ema_fast: int = 15
    ema_slow: int = 50
    hl_range_pct: float = 0.02

    max_gap_days: int = 5
    outlier_threshold: float = 0.20

    cvar_alpha: float = 0.05
    target_volatility: Optional[float] = None

    subperiods_enabled: bool = False
    subperiods_train_years: float = 2.0
    subperiods_test_years: float = 1.0
    subperiods_step_years: float = 1.0

    # Fallback: если API недоступен, работаем только с CSV
    offline_mode: bool = False

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# КЭШИРОВАНИЕ API
# ============================================================================

class APICache:
    """Кэширование HTTP-ответов."""

    def __init__(self, cache_dir: str, ttl_hours: int):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)

    def _cache_key(self, url: str, params: Optional[Dict]) -> str:
        key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cache"

    def get(self, url: str, params: Optional[Dict]) -> Optional[bytes]:
        key = self._cache_key(url, params)
        path = self._cache_path(key)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            return None
        try:
            return path.read_bytes()
        except Exception:
            return None

    def save(self, url: str, params: Optional[Dict], content: bytes):
        key = self._cache_key(url, params)
        path = self._cache_path(key)
        try:
            path.write_bytes(content)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")


# ============================================================================
# АВТООБНАРУЖЕНИЕ ФАЙЛОВ ПО ПРЕФИКСУ
# ============================================================================

def resolve_csv_file(prefix: str, data_dir: str) -> Optional[str]:
    """
    Находит CSV-файл в data_dir, имя которого начинается с prefix.
    Возвращает полный путь или None.
    """
    if not os.path.isdir(data_dir):
        return None
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.csv') and fname.startswith(prefix):
            return os.path.join(data_dir, fname)
    return None


# ============================================================================
# КАРТЫ ДАННЫХ (ПРЕФИКСЫ — ищутся по начальному слову)
# ============================================================================

CSV_MAP = {
    'IMOEX': {
        'prefix': 'IMOEX_index',       # найдёт IMOEX_index 2026.07.17.csv, IMOEX_index.csv и т.д.
        'date_col': 'date',
        'value_col': 'close',
        'high_col': 'high',
        'low_col': 'low',
    },
    'OFZ': {
        'prefix': 'RGBITR',            # найдёт RGBITR_history.csv, RGBI_history.csv и т.д.
        'date_col': 'TRADEDATE',
        'value_col': 'CLOSE',
        'high_col': 'HIGH',
        'low_col': 'LOW',
    },
    'GOLD': {
        'prefix': 'GLDRUB',            # найдёт GLDRUB_TOM_history_2013_2026.csv и т.д.
        'date_col': 'date',
        'value_col': 'close',
        'high_col': 'high',
        'low_col': 'low',
    },
    'CNY_BONDS': {
        'prefix': 'RUCBTRNS',          # найдёт RUCBTRNS_history.csv и т.д.
        'date_col': 'TRADEDATE',
        'value_col': 'CLOSE',
        'high_col': 'HIGH',
        'low_col': 'LOW',
    },
    'USD_RUB': {
        'prefix': 'USD000UTSTOM',      # найдёт USD000UTSTOM.csv и т.д.
        'date_col': 'Date',
        'value_col': 'Close_USD_RUB',
        'high_col': 'High_USD_RUB',
        'low_col': 'Low_USD_RUB',
    },
    'CNY_RUB': {
        'prefix': 'CNYRUB',            # найдёт CNYRUB_TOM_history.csv и т.д.
        'date_col': 'TRADEDATE',
        'value_col': 'CLOSE',
        'high_col': 'HIGH',
        'low_col': 'LOW',
    },
    'MREDC': {
        'prefix': 'MREDC',             # найдёт MREDC_historical_2016-2026.csv и т.д.
        'date_col': 'Date',
        'value_col': 'Close',
        'high_col': 'High',
        'low_col': 'Low',
    },
    'M2': {
        'prefix': 'emiss_37697_m2',    # найдёт emiss_37697_m2_money_supply.csv и т.д.
        'date_col': 'date',
        'value_col': 'value',
        'indicator_filter': 'Денежная масса в национальном определении (денежный агрегат М2)',
    },
}

MOEX_API_ENDPOINTS = {
    'IMOEX': {
        'url_template': 'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/{ticker}.csv',
        'ticker': 'IMOEX', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'OFZ': {
        'url_template': 'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/{ticker}.csv',
        'ticker': 'RGBITR', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'FLOATER': {
        'url_template': 'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/{ticker}.csv',
        'ticker': 'MOEXFLTR', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'GOLD': {
        'url_template': 'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/{ticker}.csv',
        'ticker': 'GLDRUB', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'USD_RUB': {
        'url_template': 'https://iss.moex.com/iss/history/engines/currency/markets/supt/boards/CETS/securities/{ticker}.csv',
        'ticker': 'USD000UTSTOM', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'CNY_RUB': {
        'url_template': 'https://iss.moex.com/iss/history/engines/currency/markets/supt/boards/CETS/securities/{ticker}.csv',
        'ticker': 'CNYRUB_TOM', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
    'MONEY_MARKET': {
        'url_template': 'https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQTF/securities/{ticker}.csv',
        'ticker': 'LQDT', 'sep': ',', 'date_col': 'TRADEDATE', 'value_col': 'CLOSE',
        'high_col': 'HIGH', 'low_col': 'LOW'
    },
}

CRITICAL_DATASETS = {'IMOEX'}
IMPORTANT_DATASETS = {'OFZ', 'GOLD', 'USD_RUB', 'key_rate'}
PRICE_ASSETS = {'IMOEX', 'OFZ', 'GOLD', 'CNY_BONDS', 'USD_RUB', 'CNY_RUB', 'MREDC'}
MACRO_DATASETS = {'M2', 'MREDC'}


# ============================================================================
# ИСТОРИЯ КЛЮЧЕВОЙ СТАВКИ ЦБ РФ (FALLBACK)
# ============================================================================

RATE_PERIODS = [
    ('2016-01-01', '2016-06-14', 11.00), ('2016-06-14', '2016-09-16', 10.50),
    ('2016-09-16', '2017-03-24', 10.00), ('2017-03-24', '2017-04-28', 9.75),
    ('2017-04-28', '2017-06-16', 9.25), ('2017-06-16', '2017-09-15', 9.00),
    ('2017-09-15', '2017-10-27', 8.50), ('2017-10-27', '2017-12-15', 8.25),
    ('2017-12-15', '2018-02-09', 7.75), ('2018-02-09', '2018-03-23', 7.50),
    ('2018-03-23', '2018-09-14', 7.25), ('2018-09-14', '2018-12-14', 7.50),
    ('2018-12-14', '2019-06-14', 7.75), ('2019-06-14', '2019-07-26', 7.50),
    ('2019-07-26', '2019-09-06', 7.25), ('2019-09-06', '2019-10-25', 7.00),
    ('2019-10-25', '2019-12-13', 6.50), ('2019-12-13', '2020-02-07', 6.25),
    ('2020-02-07', '2020-04-24', 6.00), ('2020-04-24', '2020-06-19', 5.50),
    ('2020-06-19', '2020-07-24', 4.50), ('2020-07-24', '2021-03-19', 4.25),
    ('2021-03-19', '2021-04-23', 4.50), ('2021-04-23', '2021-06-11', 5.00),
    ('2021-06-11', '2021-07-23', 5.50), ('2021-07-23', '2021-09-10', 6.50),
    ('2021-09-10', '2021-10-22', 6.75), ('2021-10-22', '2021-12-17', 7.50),
    ('2021-12-17', '2022-02-28', 8.50), ('2022-02-28', '2022-04-08', 20.00),
    ('2022-04-08', '2022-04-29', 17.00), ('2022-04-29', '2022-05-27', 14.00),
    ('2022-05-27', '2022-06-10', 11.00), ('2022-06-10', '2022-07-22', 9.50),
    ('2022-07-22', '2023-07-21', 7.50), ('2023-07-21', '2023-08-15', 8.50),
    ('2023-08-15', '2023-09-15', 12.00), ('2023-09-15', '2023-10-27', 13.00),
    ('2023-10-27', '2023-12-15', 15.00), ('2023-12-15', '2024-10-25', 16.00),
    ('2024-10-25', '2025-07-28', 21.00), ('2025-07-28', '2026-07-06', 18.00),
]


# ============================================================================
# АСИНХРОННЫЙ HTTP-ЗАГРУЗЧИК
# ============================================================================

async def async_http_get(
    client: httpx.AsyncClient,
    url: str,
    params: Dict = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.5,
    cache: Optional[APICache] = None
) -> Optional[bytes]:
    """Асинхронный HTTP GET с автоматическими повторами и кэшированием."""

    if cache:
        cached = cache.get(url, params)
        if cached is not None:
            return cached

    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                content = response.content
                if cache:
                    cache.save(url, params, content)
                return content
            logger.warning(f"HTTP {response.status_code} для {url} (попытка {attempt + 1})")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning(f"Ошибка сети для {url}: {e} (попытка {attempt + 1})")
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))
    return None


# ============================================================================
# ЗАГРУЗКА CSV (автоопределение колонок + автообнаружение файла по префиксу)
# ============================================================================

def _detect_columns(df: pd.DataFrame, asset_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Автоопределяет колонки даты, close, high, low из DataFrame."""
    cols = [c.strip() for c in df.columns]
    cols_lower = [c.lower() for c in cols]
    cols_upper = [c.upper() for c in cols]

    # Date column
    date_col = None
    for candidates in [['date'], ['tradedate'], ['дата']]:
        for c in candidates:
            if c in cols_lower:
                date_col = cols[cols_lower.index(c)]
                break
        if date_col:
            break
    if not date_col:
        for c in cols:
            if 'date' in c.lower() or 'дата' in c.lower():
                date_col = c
                break

    # Value/close column
    value_col = None
    for candidates in [['close'], ['close_usd_rub'], ['value'], ['цена'], ['закрытие']]:
        for c in candidates:
            if c in cols_lower:
                value_col = cols[cols_lower.index(c)]
                break
        if value_col:
            break
    if not value_col:
        for c in cols:
            if c.lower() in ('close', 'value', 'price', 'закрытие'):
                value_col = c
                break

    # High column
    high_col = None
    for candidates in [['high'], ['high_usd_rub'], ['макс']]:
        for c in candidates:
            if c in cols_lower:
                high_col = cols[cols_lower.index(c)]
                break
        if high_col:
            break
    if not high_col:
        for c in cols:
            if c.lower() in ('high', 'max', 'максимум'):
                high_col = c
                break

    # Low column
    low_col = None
    for candidates in [['low'], ['low_usd_rub'], ['мин']]:
        for c in candidates:
            if c in cols_lower:
                low_col = cols[cols_lower.index(c)]
                break
        if low_col:
            break
    if not low_col:
        for c in cols:
            if c.lower() in ('low', 'min', 'минимум'):
                low_col = c
                break

    return date_col, value_col, high_col, low_col


def load_csv_strict(filepath: str, asset_name: str = '') -> Tuple[pd.Series, pd.Series, pd.Series, List[str]]:
    """Загружает CSV, удаляя некорректные строки. Возвращает (close, high, low, errors)."""
    errors = []
    empty = (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), errors)

    if not filepath or not os.path.exists(filepath):
        errors.append(f"Файл не найден: {filepath}")
        return empty

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        errors.append(f"Ошибка чтения {filepath}: {str(e)}")
        return empty

    if df.empty:
        errors.append(f"Файл пустой: {filepath}")
        return empty

    # Попробуем взять колонки из CSV_MAP если asset_name известен
    date_col = value_col = high_col = low_col = None
    if asset_name and asset_name in CSV_MAP:
        cfg = CSV_MAP[asset_name]
        date_col = cfg.get('date_col')
        value_col = cfg.get('value_col')
        high_col = cfg.get('high_col')
        low_col = cfg.get('low_col')

    # Проверим, что колонки существуют
    cols = list(df.columns)
    if date_col and date_col not in cols:
        date_col = None
    if value_col and value_col not in cols:
        value_col = None
    if high_col and high_col not in cols:
        high_col = None
    if low_col and low_col not in cols:
        low_col = None

    # Автоопределение если нужно
    if not date_col or not value_col:
        d, v, h, l = _detect_columns(df, asset_name)
        if not date_col:
            date_col = d
        if not value_col:
            value_col = v
        if not high_col:
            high_col = h
        if not low_col:
            low_col = l

    if not date_col or not value_col:
        errors.append(f"Не найдены колонки даты/значения в {os.path.basename(filepath)}. Колонки: {cols}")
        return empty

    # Специальная обработка M2: фильтрация по индикатору
    if asset_name == 'M2' and 'indicator_filter' in CSV_MAP.get('M2', {}):
        indicator_filter = CSV_MAP['M2']['indicator_filter']
        if 'indicator' in df.columns:
            df = df[df['indicator'] == indicator_filter].copy()
            if df.empty:
                errors.append(f"M2: не найден индикатор '{indicator_filter}'")
                return empty

    dates = pd.to_datetime(df[date_col], errors='coerce')
    values = pd.to_numeric(df[value_col], errors='coerce')

    valid = dates.notna() & values.notna()
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        errors.append(f"{os.path.basename(filepath)}: удалено {n_invalid} строк с некорректными данными")

    close_series = pd.Series(
        data=values[valid].values,
        index=dates[valid],
        name=f"{asset_name}_close"
    ).sort_index()

    # High/Low
    high_series = pd.Series(index=close_series.index, dtype=float, name=f"{asset_name}_high")
    low_series = pd.Series(index=close_series.index, dtype=float, name=f"{asset_name}_low")

    if high_col and high_col in df.columns:
        h = pd.to_numeric(df[high_col], errors='coerce')
        high_series = pd.Series(data=h[valid].values, index=dates[valid], name=f"{asset_name}_high").sort_index()
    if low_col and low_col in df.columns:
        l = pd.to_numeric(df[low_col], errors='coerce')
        low_series = pd.Series(data=l[valid].values, index=dates[valid], name=f"{asset_name}_low").sort_index()

    if close_series.index.duplicated().any():
        close_series = close_series[~close_series.index.duplicated(keep='first')]
        high_series = high_series[~high_series.index.duplicated(keep='first')]
        low_series = low_series[~low_series.index.duplicated(keep='first')]

    return close_series, high_series, low_series, errors


# ============================================================================
# ВАЛИДАЦИЯ ДАННЫХ
# ============================================================================

def validate_series(series: pd.Series, name: str,
                   max_gap_days: int = 5,
                   outlier_threshold: float = 0.20) -> List[str]:
    """Проверяет данные на пропуски и выбросы."""
    warnings_list = []
    if series.empty:
        warnings_list.append(f"{name}: пустой ряд")
        return warnings_list

    diffs = series.index.to_series().diff().dt.days.dropna()
    big_gaps = diffs[diffs > max_gap_days]
    if not big_gaps.empty:
        warnings_list.append(f"{name}: обнаружены пропуски > {max_gap_days} дней "
                            f"({len(big_gaps)} случаев, макс. {big_gaps.max()} дней)")

    if len(series) >= 2:
        returns = series.pct_change().dropna()
        outliers = returns[returns.abs() > outlier_threshold]
        if not outliers.empty:
            warnings_list.append(f"{name}: обнаружены выбросы "
                                f"({len(outliers)} дней с изменением > {outlier_threshold:.0%})")

    return warnings_list


# ============================================================================
# АСИНХРОННАЯ ЗАГРУЗКА ИЗ MOEX ISS API
# ============================================================================

async def async_fetch_moex_history(
    client: httpx.AsyncClient,
    asset_name: str,
    from_date: str,
    till_date: str = None,
    cache: Optional[APICache] = None,
    config: Optional[ModelConfig] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, List[str]]:
    """
    Загружает историю с Мосбиржи с пагинацией.
    Возвращает: (close_series, high_series, low_series, errors)
    """
    errors = []
    cfg = MOEX_API_ENDPOINTS.get(asset_name)
    if not cfg:
        errors.append(f"Нет endpoint для {asset_name}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), errors

    if till_date is None:
        till_date = datetime.now().strftime('%Y-%m-%d')

    url = cfg['url_template'].format(ticker=cfg['ticker'])
    all_close, all_high, all_low = [], [], []
    current_from = from_date
    page = 0
    max_pages = config.max_pagination_pages if config else 50

    while page < max_pages:
        params = {'from': current_from, 'till': till_date, 'iss.meta': 'off'}
        content = await async_http_get(client, url, params=params, cache=cache)

        if content is None:
            errors.append(f"{asset_name}: не удалось получить данные от MOEX за {current_from}—{till_date}")
            break

        try:
            # MOEX отдаёт в cp1251
            text = content.decode('cp1251', errors='ignore')
            df = pd.read_csv(StringIO(text), sep=cfg['sep'])
        except Exception as e:
            errors.append(f"{asset_name}: ошибка парсинга CSV: {e}")
            break

        if df.empty:
            break

        # Fallback на разделитель ';'
        if cfg['date_col'] not in df.columns or cfg['value_col'] not in df.columns:
            try:
                df = pd.read_csv(StringIO(text), sep=';')
            except Exception:
                errors.append(f"{asset_name}: неожиданный формат. Колонки: {list(df.columns)}")
                break

        if cfg['date_col'] not in df.columns or cfg['value_col'] not in df.columns:
            errors.append(f"{asset_name}: в ответе API нет нужных колонок: {list(df.columns)}")
            break

        dates = pd.to_datetime(df[cfg['date_col']], errors='coerce')
        close_vals = pd.to_numeric(df[cfg['value_col']], errors='coerce')

        # Пытаемся загрузить реальные High/Low
        high_vals = close_vals.copy()
        low_vals = close_vals.copy()
        if cfg.get('high_col') in df.columns and cfg.get('low_col') in df.columns:
            h = pd.to_numeric(df[cfg['high_col']], errors='coerce')
            l = pd.to_numeric(df[cfg['low_col']], errors='coerce')
            high_vals = h.combine_first(close_vals)
            low_vals = l.combine_first(close_vals)

        valid = dates.notna() & close_vals.notna()
        batch_close = pd.Series(data=close_vals[valid].values, index=dates[valid], name=asset_name).sort_index()
        batch_high = pd.Series(data=high_vals[valid].values, index=dates[valid], name=f"{asset_name}_high").sort_index()
        batch_low = pd.Series(data=low_vals[valid].values, index=dates[valid], name=f"{asset_name}_low").sort_index()

        if batch_close.empty:
            break

        from_dt = pd.to_datetime(from_date)
        valid_batch = batch_close[batch_close.index >= from_dt]
        if valid_batch.empty:
            errors.append(f"{asset_name}: API вернул данные только до {from_date}")
            break

        all_close.append(batch_close)
        all_high.append(batch_high)
        all_low.append(batch_low)

        last_in_batch = batch_close.index.max()
        if last_in_batch >= pd.to_datetime(till_date) - pd.Timedelta(days=1):
            break

        next_from = (last_in_batch + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if next_from > till_date or next_from <= current_from:
            break

        current_from = next_from
        page += 1

    if not all_close:
        if not errors:
            errors.append(f"{asset_name}: API не вернул данных за период {from_date}—{till_date}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), errors

    close_s = pd.concat(all_close).sort_index()
    high_s = pd.concat(all_high).sort_index()
    low_s = pd.concat(all_low).sort_index()
    close_s = close_s[~close_s.index.duplicated(keep='first')]
    high_s = high_s[~high_s.index.duplicated(keep='first')]
    low_s = low_s[~low_s.index.duplicated(keep='first')]
    return close_s, high_s, low_s, errors


# ============================================================================
# АСИНХРОННАЯ ЗАГРУЗКА КЛЮЧЕВОЙ СТАВКИ С ЦБ РФ
# ============================================================================

async def async_fetch_cbr_key_rate_html(
    client: httpx.AsyncClient,
    cache: Optional[APICache] = None
) -> Tuple[pd.Series, List[str]]:
    """Парсит HTML страницу ключевой ставки с сайта ЦБ РФ."""
    errors = []
    url = "https://www.cbr.ru/hd/base/keyrate/"
    content = await async_http_get(client, url, cache=cache)

    if content is None:
        errors.append("HTML страница ключевой ставки недоступна")
        return pd.Series(dtype=float), errors

    try:
        import re
        text = content.decode('utf-8')
        pattern = r'(\d{2}\.\d{2}\.\d{4}).*?(\d+[\.,]\d+)\s*%'
        matches = re.findall(pattern, text)

        if not matches:
            errors.append("Не удалось извлечь данные со страницы ЦБ РФ")
            return pd.Series(dtype=float), errors

        dates, rates = [], []
        for date_str, rate_str in matches:
            try:
                dt = pd.to_datetime(date_str, format='%d.%m.%Y')
                rate = float(rate_str.replace(',', '.'))
                if 0.25 <= rate <= 50.0 and dt.year >= 2010 and dt <= pd.Timestamp.now():
                    dates.append(dt)
                    rates.append(rate)
            except Exception:
                continue

        if not dates:
            errors.append("Не удалось распарсить даты/ставки со страницы ЦБ РФ")
            return pd.Series(dtype=float), errors

        df = pd.DataFrame({'date': dates, 'rate': rates}).sort_values('date')
        df = df.drop_duplicates(subset='date', keep='last')

        result_dates, result_rates = [], []
        for i in range(len(df)):
            start = df.iloc[i]['date']
            rate = df.iloc[i]['rate']
            end = df.iloc[i + 1]['date'] if i + 1 < len(df) else pd.Timestamp.now()
            period_dates = pd.bdate_range(start, end, freq='B')
            result_dates.extend(period_dates)
            result_rates.extend([rate] * len(period_dates))

        series = pd.Series(data=result_rates, index=result_dates, name='key_rate')
        series = series[~series.index.duplicated(keep='first')].sort_index()
        logger.info(f"✓ Ключевая ставка получена с ЦБ РФ: {len(df)} периодов, до {df['date'].max().date()}")
        return series, errors

    except Exception as e:
        errors.append(f"Ошибка парсинга HTML ЦБ РФ: {str(e)}")
        return pd.Series(dtype=float), errors


# ============================================================================
# АСИНХРОННАЯ ЗАГРУЗКА M2 С ЦБ РФ
# ============================================================================

async def async_fetch_m2_from_cbr(
    client: httpx.AsyncClient,
    cache: Optional[APICache] = None
) -> Tuple[pd.Series, List[str]]:
    """Загружает агрегат M2 с сайта ЦБ РФ."""
    errors = []
    url = "https://www.cbr.ru/hd/base/d.aspx"
    params = {'PrtId': 'm2'}
    content = await async_http_get(client, url, params=params, cache=cache)

    if content is None:
        errors.append("Не удалось получить M2 с ЦБ РФ")
        return pd.Series(dtype=float), errors

    try:
        import re
        text = content.decode('utf-8')
        pattern = r'(\d{2}\.\d{2}\.\d{4}).*?(\d+[\.,]?\d*)\s*(?:млрд|млн)'
        matches = re.findall(pattern, text)

        if not matches:
            errors.append("Не удалось извлечь M2 со страницы ЦБ РФ")
            return pd.Series(dtype=float), errors

        dates, values = [], []
        for date_str, val_str in matches:
            try:
                dt = pd.to_datetime(date_str, format='%d.%m.%Y')
                val = float(val_str.replace(',', '.'))
                dates.append(dt)
                values.append(val)
            except Exception:
                continue

        series = pd.Series(data=values, index=dates, name='m2').sort_index()
        series = series[~series.index.duplicated(keep='first')]
        logger.info(f"✓ M2 получен с ЦБ РФ: {len(series)} точек")
        return series, errors

    except Exception as e:
        errors.append(f"Ошибка парсинга M2 с ЦБ РФ: {str(e)}")
        return pd.Series(dtype=float), errors


# ============================================================================
# ГИБРИДНЫЕ ЗАГРУЗЧИКИ С АВТОДОГРУЗКОЙ
# ============================================================================

async def async_load_asset_with_autofetch(
    client: httpx.AsyncClient,
    asset_name: str,
    config: ModelConfig,
    start_date: str = '2016-01-01',
    cache: Optional[APICache] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict]:
    """
    Гибридный загрузчик: CSV (по префиксу) → проверка актуальности → догрузка из API.
    Возвращает: (close, high, low, metadata)
    """
    metadata = {
        'source': None,
        'csv_status': 'not_loaded',
        'fetched_from_date': None,
        'fetched_points': 0,
        'errors': [],
        'warnings': [],
        'last_date': None
    }

    csv_close = pd.Series(dtype=float)
    csv_high = pd.Series(dtype=float)
    csv_low = pd.Series(dtype=float)

    if asset_name in CSV_MAP:
        prefix = CSV_MAP[asset_name].get('prefix', asset_name)
        csv_path = resolve_csv_file(prefix, config.data_dir)
        if csv_path:
            csv_close, csv_high, csv_low, csv_errors = load_csv_strict(csv_path, asset_name)
            if csv_errors:
                metadata['errors'].extend([f"[CSV] {e}" for e in csv_errors])
            if not csv_close.empty:
                metadata['csv_status'] = 'loaded'
                metadata['last_date'] = csv_close.index.max()
                logger.info(f"  CSV {asset_name}: {len(csv_close)} записей, до {csv_close.index.max().date()} (файл: {os.path.basename(csv_path)})")
        else:
            metadata['csv_status'] = 'missing'
            metadata['errors'].append(f"[CSV] Не найден файл с префиксом '{prefix}' в {config.data_dir}")
            logger.warning(f"  CSV {asset_name}: не найден файл с префиксом '{prefix}'")

    # Если offline_mode — не лезем в API
    if config.offline_mode:
        if not csv_close.empty:
            metadata['source'] = 'CSV'
            if csv_high.isna().all() or csv_low.isna().all():
                csv_high = csv_close * (1 + config.hl_range_pct)
                csv_low = csv_close * (1 - config.hl_range_pct)
            return csv_close, csv_high, csv_low, metadata
        else:
            metadata['source'] = 'FAILED'
            metadata['errors'].append(f"{asset_name}: нет CSV и offline_mode включён")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), metadata

    needs_fetch = False
    fetch_from = start_date

    if csv_close.empty:
        needs_fetch = True
        metadata['csv_status'] = 'missing'
        logger.warning(f"  CSV {asset_name} отсутствует — полная загрузка из API")
    else:
        days_old = (pd.Timestamp.now().normalize() - csv_close.index.max()).days
        threshold = config.freshness_thresholds.get('daily_prices', 3)
        if days_old > threshold:
            needs_fetch = True
            fetch_from = (csv_close.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            metadata['csv_status'] = 'stale'
            metadata['warnings'].append(
                f"{asset_name}: CSV устарел на {days_old} дней (последняя дата: {csv_close.index.max().date()})"
            )

    api_close = pd.Series(dtype=float)
    api_high = pd.Series(dtype=float)
    api_low = pd.Series(dtype=float)

    if needs_fetch and asset_name in MOEX_API_ENDPOINTS:
        logger.info(f"  → Загрузка {asset_name} из MOEX API с {fetch_from}...")
        api_close, api_high, api_low, api_errors = await async_fetch_moex_history(
            client, asset_name, fetch_from, cache=cache, config=config
        )
        if api_errors:
            metadata['errors'].extend([f"[API] {e}" for e in api_errors])
        if not api_close.empty:
            metadata['fetched_from_date'] = fetch_from
            metadata['fetched_points'] = len(api_close)
            logger.info(f"  ✓ Догружено из API: {len(api_close)} записей")
        else:
            metadata['warnings'].append(f"{asset_name}: API не вернул данных")

    # Объединение
    if not csv_close.empty and not api_close.empty:
        new_mask = ~api_close.index.isin(csv_close.index)
        new_records = api_close[new_mask]
        if not new_records.empty:
            combined_close = pd.concat([csv_close, api_close[new_mask]]).sort_index()
            combined_high = pd.concat([csv_high, api_high[new_mask]]).sort_index() if not api_high.empty else csv_high
            combined_low = pd.concat([csv_low, api_low[new_mask]]).sort_index() if not api_low.empty else csv_low
            metadata['fetched_points'] = len(new_records)
            metadata['source'] = 'CSV+API'
        else:
            combined_close = csv_close
            combined_high = csv_high
            combined_low = csv_low
            metadata['warnings'].append(f"{asset_name}: API вернул только дубликаты CSV")
            metadata['source'] = 'CSV'
    elif not csv_close.empty:
        combined_close = csv_close
        combined_high = csv_high
        combined_low = csv_low
        metadata['source'] = 'CSV'
    elif not api_close.empty:
        combined_close = api_close
        combined_high = api_high
        combined_low = api_low
        metadata['source'] = 'API'
    else:
        metadata['source'] = 'FAILED'
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), metadata

    metadata['last_date'] = combined_close.index.max()

    # Fallback High/Low из Close ±2%, если реальные не загружены
    if combined_high.isna().all() or combined_low.isna().all():
        combined_high = combined_close * (1 + config.hl_range_pct)
        combined_low = combined_close * (1 - config.hl_range_pct)

    return combined_close, combined_high, combined_low, metadata


async def async_load_key_rate_hybrid(
    client: httpx.AsyncClient,
    config: ModelConfig,
    cache: Optional[APICache] = None
) -> Tuple[pd.Series, Dict]:
    """Загружает ключевую ставку: встроенная таблица + догрузка с ЦБ РФ."""
    metadata = {
        'source': 'RATE_PERIODS',
        'errors': [],
        'warnings': [],
        'last_date': None,
        'fetched_from_web': False
    }

    dates, rates = [], []
    for start_str, end_str, rate in RATE_PERIODS:
        try:
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            period_dates = pd.bdate_range(start, end, freq='B')
            dates.extend(period_dates)
            rates.extend([rate] * len(period_dates))
        except Exception as e:
            metadata['errors'].append(f"Ошибка периода {start_str}—{end_str}: {e}")

    base_series = pd.Series(data=rates, index=dates, name='key_rate')
    base_series = base_series[~base_series.index.duplicated(keep='first')].sort_index()

    if base_series.empty:
        metadata['errors'].append("Встроенная таблица RATE_PERIODS не создала Series")
        return pd.Series(dtype=float), metadata

    last_builtin_date = base_series.index.max()
    metadata['last_date'] = last_builtin_date
    logger.info(f"  Ключевая ставка (встроенная): до {last_builtin_date.date()} ({base_series.iloc[-1]:.2f}%)")

    days_old = (pd.Timestamp.now().normalize() - last_builtin_date).days
    threshold = config.freshness_thresholds.get('key_rate', 7)

    if days_old <= threshold or config.offline_mode:
        if config.offline_mode:
            logger.info(f"  Ключевая ставка (offline): до {last_builtin_date.date()}")
        else:
            logger.info(f"  Ключевая ставка актуальна (задержка {days_old} дней)")
        return base_series, metadata

    metadata['warnings'].append(f"Встроенная таблица устарела на {days_old} дней — попытка догрузки с ЦБ РФ")
    logger.info(f"  → Дозагрузка ключевой ставки с ЦБ РФ...")

    web_series, web_errors = await async_fetch_cbr_key_rate_html(client, cache=cache)
    if web_errors:
        metadata['errors'].extend([f"[CBR] {e}" for e in web_errors])

    if not web_series.empty:
        new_records = web_series[web_series.index > last_builtin_date]
        if not new_records.empty:
            combined = pd.concat([base_series, new_records]).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            metadata['source'] = 'RATE_PERIODS+CBR'
            metadata['fetched_from_web'] = True
            metadata['last_date'] = combined.index.max()
            logger.info(f"  ✓ Догружено с ЦБ РФ: {len(new_records)} записей до {new_records.index.max().date()}")
            return combined, metadata
        else:
            metadata['warnings'].append("ЦБ РФ вернул данные, но все они устарели")

    metadata['warnings'].append("Догрузка с ЦБ РФ не удалась — используется встроенная таблица")
    return base_series, metadata


async def async_load_macro_with_autofetch(
    client: httpx.AsyncClient,
    name: str,
    config: ModelConfig,
    cache: Optional[APICache] = None
) -> Tuple[pd.Series, Dict]:
    """Загрузка макро-данных (M2, MREDC) с автодогрузкой."""
    metadata = {'source': None, 'errors': [], 'warnings': [], 'last_date': None}

    csv_series = pd.Series(dtype=float)
    if name in CSV_MAP:
        prefix = CSV_MAP[name].get('prefix', name)
        csv_path = resolve_csv_file(prefix, config.data_dir)
        if csv_path:
            csv_close, _, _, csv_errors = load_csv_strict(csv_path, name)
            if csv_errors:
                metadata['errors'].extend([f"[CSV] {e}" for e in csv_errors])
            if not csv_close.empty:
                csv_series = csv_close
                metadata['last_date'] = csv_series.index.max()
                metadata['source'] = 'CSV'
        else:
            metadata['errors'].append(f"[CSV] Не найден файл с префиксом '{prefix}'")

    threshold = config.freshness_thresholds.get('monthly_macro', 60)
    if name == 'M2' and not config.offline_mode and (csv_series.empty or
                         (pd.Timestamp.now() - csv_series.index.max()).days > threshold):
        logger.info(f"  → Дозагрузка M2 с ЦБ РФ...")
        web_series, web_errors = await async_fetch_m2_from_cbr(client, cache=cache)
        if web_errors:
            metadata['errors'].extend([f"[CBR] {e}" for e in web_errors])
        if not web_series.empty:
            if csv_series.empty:
                csv_series = web_series
                metadata['source'] = 'CBR_API'
            else:
                new_records = web_series[~web_series.index.isin(csv_series.index)]
                if not new_records.empty:
                    csv_series = pd.concat([csv_series, new_records]).sort_index()
                    metadata['source'] = 'CSV+CBR'
                    logger.info(f"  ✓ M2 дополнен: +{len(new_records)} точек с ЦБ РФ")

    if csv_series.empty:
        metadata['source'] = 'FAILED'
    else:
        metadata['last_date'] = csv_series.index.max()

    return csv_series, metadata


# ============================================================================
# ПАРАЛЛЕЛЬНАЯ ЗАГРУЗКА ВСЕХ АКТИВОВ
# ============================================================================

async def async_load_all_assets(
    config: ModelConfig,
    asset_names: List[str],
    start_date: str = '2016-01-01',
    cache: Optional[APICache] = None
) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, Dict]]:
    """Параллельно загружает все активы через семафор ограничения конкурентности."""
    semaphore = asyncio.Semaphore(config.api_concurrency)

    async def _fetch_one(client: httpx.AsyncClient, name: str):
        async with semaphore:
            return name, await async_load_asset_with_autofetch(
                client, name, config, start_date, cache
            )

    limits = httpx.Limits(max_connections=config.api_concurrency * 2, max_keepalive_connections=4)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(config.api_timeout)) as client:
        tasks = [_fetch_one(client, name) for name in asset_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Критическая ошибка загрузки актива: {res}")
            continue
        name, (close, high, low, meta) = res
        output[name] = (close, high, low, meta)
    return output


# ============================================================================
# ПРОВЕРКА АКТУАЛЬНОСТИ ДАННЫХ
# ============================================================================

def check_freshness(series: pd.Series, max_lag_days: int, name: str) -> Tuple[bool, str]:
    """Проверяет актуальность данных."""
    if series.empty:
        return False, f"{name}: данные отсутствуют"
    last_date = series.index.max()
    today = pd.Timestamp.now().normalize()
    lag_days = (today - last_date).days
    if lag_days > max_lag_days:
        return False, f"{name}: устарели на {lag_days} дней (последнее значение от {last_date.date()})"
    return True, f"{name}: актуальны (последнее значение от {last_date.date()})"


# ============================================================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    """Экспоненциальная скользящая средняя."""
    return series.ewm(span=span, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Расчёт ADX (Average Directional Index)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(25)


def v23_engine(close: pd.Series, high: pd.Series, low: pd.Series,
               config: ModelConfig) -> pd.DataFrame:
    """V2.3 движок: EMA 15/50 + ADX для тактического управления."""
    df = pd.DataFrame(index=close.index)
    df['close'] = close
    df['ema_fast'] = ema(close, config.ema_fast)
    df['ema_slow'] = ema(close, config.ema_slow)

    df['high'] = high
    df['low'] = low

    df['adx'] = compute_adx(df['high'], df['low'], df['close'], config.adx_period)

    # Сигналы
    df['signal_raw'] = np.where(df['ema_fast'] > df['ema_slow'], 1.0, 0.0)
    # ADX-фильтр: торгуем только при ADX > 20 (есть тренд)
    df['signal'] = np.where(df['adx'] > 20, df['signal_raw'], 0.0)

    # Позиция: 1 = в рынке, 0 = вне рынка
    df['position'] = df['signal'].shift(1).fillna(0.0)

    return df


# ============================================================================
# СИМУЛЯЦИЯ ПОРТФЕЛЯ
# ============================================================================

def simulate_portfolio(
    prices: Dict[str, pd.Series],
    signals: Dict[str, pd.Series],
    config: ModelConfig,
    initial_capital: float = 1_000_000.0,
    target_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Симулирует тактический портфель.
    target_weights: базовые веса активов (сумма = 1.0).
    signals: 1.0 = в рынке, 0.0 = вне рынка (денежный рынок).
    """
    if target_weights is None:
        n = len(prices)
        target_weights = {k: 1.0 / n for k in prices}

    # Общий индекс дат
    all_dates = pd.DatetimeIndex(sorted(set().union(*[p.index for p in prices.values()])))
    df = pd.DataFrame(index=all_dates)
    df['cash'] = 0.0
    df['portfolio_value'] = initial_capital

    # Дневные доходности активов
    returns = {}
    for name, price in prices.items():
        returns[name] = price.pct_change().reindex(all_dates, fill_value=0.0)

    # Сигналы на каждую дату
    sig_aligned = {}
    for name, sig in signals.items():
        sig_aligned[name] = sig.reindex(all_dates, fill_value=0.0)

    current_weights = {k: 0.0 for k in prices}
    portfolio_value = initial_capital
    prev_weights = target_weights.copy()

    records = []
    for date in all_dates:
        day_returns = {}
        day_signals = {}
        for name in prices:
            day_returns[name] = returns[name].loc[date] if date in returns[name].index else 0.0
            day_signals[name] = sig_aligned[name].loc[date] if date in sig_aligned[name].index else 0.0

        # Тактические веса: базовый вес * сигнал (0 или 1)
        tactical_weights = {}
        total_active = 0.0
        for name in prices:
            w = target_weights.get(name, 0.0) * day_signals[name]
            tactical_weights[name] = w
            total_active += w

        # Остаток в денежный рынок
        cash_weight = max(0.0, 1.0 - total_active)
        if 'MONEY_MARKET' in prices and 'MONEY_MARKET' in target_weights:
            tactical_weights['MONEY_MARKET'] = cash_weight + tactical_weights.get('MONEY_MARKET', 0.0)
        else:
            pass

        # Нормализация
        total = sum(tactical_weights.values())
        if total > 0:
            tactical_weights = {k: v / total for k, v in tactical_weights.items()}

        # Проверка порога ребалансировки
        rebalance = False
        for name in tactical_weights:
            if abs(tactical_weights[name] - prev_weights.get(name, 0.0)) > config.rebalance_threshold:
                rebalance = True
                break

        turnover = 0.0
        cost = 0.0
        if rebalance:
            turnover = sum(abs(tactical_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in set(tactical_weights) | set(prev_weights)) / 2.0
            cost = turnover * (config.transaction_cost + config.slippage)
            portfolio_value *= (1 - cost)
            prev_weights = tactical_weights.copy()

        # Дневная доходность портфеля
        port_return = 0.0
        for name, w in tactical_weights.items():
            port_return += w * day_returns.get(name, 0.0)

        portfolio_value *= (1 + port_return)

        records.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'port_return': port_return,
            'turnover': turnover if rebalance else 0.0,
            'cost': cost if rebalance else 0.0,
        })

    result = pd.DataFrame(records).set_index('date')
    return result


# ============================================================================
# МЕТРИКИ
# ============================================================================

def calculate_metrics(portfolio: pd.DataFrame, rf_annual: float = 0.18,
                      tax_rate: float = 0.13) -> Dict[str, float]:
    """Расширенные метрики эффективности."""
    returns = portfolio['port_return'].dropna()
    if returns.empty or len(returns) < 2:
        return {}

    # Годовая доходность (по 252 дням)
    n_days = len(returns)
    total_return = portfolio['portfolio_value'].iloc[-1] / portfolio['portfolio_value'].iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1

    # Волатильность
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_return - rf_annual) / downside if downside > 0 else 0.0

    # Jensen's Alpha
    jensen_alpha = ann_return - rf_annual

    # Максимальная просадка
    cummax = portfolio['portfolio_value'].cummax()
    drawdown = (portfolio['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # CVaR (95%)
    cvar = returns[returns <= returns.quantile(0.05)].mean() * np.sqrt(252)

    # Post-tax доходность
    post_tax_return = ann_return * (1 - tax_rate)

    return {
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'jensens_alpha': jensen_alpha,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'cvar_95_annual': cvar,
        'post_tax_return': post_tax_return,
        'total_return': total_return,
        'n_days': n_days,
    }


# ============================================================================
# КОРРЕЛЯЦИОННАЯ МАТРИЦА И РИСК
# ============================================================================

def compute_correlation_matrix(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Рассчитывает корреляционную матрицу доходностей активов."""
    df_returns = pd.DataFrame({k: v.pct_change() for k, v in prices.items()})
    return df_returns.corr()


def compute_cvar_by_asset(prices: Dict[str, pd.Series], alpha: float = 0.05) -> pd.Series:
    """CVaR по каждому активу."""
    cvars = {}
    for name, price in prices.items():
        rets = price.pct_change().dropna()
        cvars[name] = rets[rets <= rets.quantile(alpha)].mean()
    return pd.Series(cvars)


# ============================================================================
# ОТЧЁТ И ВИЗУАЛИЗАЦИЯ
# ============================================================================

def generate_report(
    portfolio: pd.DataFrame,
    metrics: Dict[str, float],
    corr_matrix: pd.DataFrame,
    cvar_assets: pd.Series,
    config: ModelConfig,
    data_status: Dict[str, Dict]
) -> str:
    """Генерирует текстовый отчёт."""
    lines = []
    lines.append("=" * 70)
    lines.append("  ГИБРИД v5.3.0-ASYNC — ОТЧЁТ О ТАКТИЧЕСКОМ УПРАВЛЕНИИ")
    lines.append("=" * 70)
    lines.append(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  1. СТАТУС ДАННЫХ")
    lines.append("─" * 70)
    for name, meta in data_status.items():
        status = meta.get('source', 'UNKNOWN')
        last = meta.get('last_date')
        last_str = last.strftime('%Y-%m-%d') if hasattr(last, 'strftime') else str(last)
        lines.append(f"  {name:12s} | источник: {status:12s} | последняя дата: {last_str}")
        for w in meta.get('warnings', []):
            lines.append(f"    ⚠ {w}")
        for e in meta.get('errors', []):
            lines.append(f"    ✗ {e}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  2. МЕТРИКИ ПОРТФЕЛЯ")
    lines.append("─" * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:25s}: {v:>12.4f}")
        else:
            lines.append(f"  {k:25s}: {v}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  3. КОРРЕЛЯЦИОННАЯ МАТРИЦА (доходности)")
    lines.append("─" * 70)
    lines.append(corr_matrix.round(3).to_string())
    lines.append("")

    lines.append("─" * 70)
    lines.append("  4. CVaR ПО АКТИВАМ (95%, дневная)")
    lines.append("─" * 70)
    for name, val in cvar_assets.items():
        lines.append(f"  {name:12s}: {val:>10.4%}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  5. ПРИМЕЧАНИЯ")
    lines.append("─" * 70)
    lines.append(f"  • Транзакционные издержки: {config.transaction_cost:.4%}")
    lines.append(f"  • Slippage: {config.slippage:.4%}")
    lines.append(f"  • Налоговая ставка (НДФЛ): {config.tax_rate:.0%}")
    lines.append(f"  • Порог ребалансировки: {config.rebalance_threshold:.2%}")
    lines.append(f"  • ADX-период: {config.adx_period}, EMA: {config.ema_fast}/{config.ema_slow}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def plot_results(portfolio: pd.DataFrame, output_dir: str):
    """Строит графики портфеля."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib недоступен — графики не построены")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Стоимость портфеля
    ax = axes[0]
    ax.plot(portfolio.index, portfolio['portfolio_value'], color='#1f77b4', linewidth=1.2)
    ax.set_title('Стоимость портфеля', fontsize=12, fontweight='bold')
    ax.set_ylabel('RUB')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # 2. Дневная доходность
    ax = axes[1]
    rets = portfolio['port_return'] * 100
    ax.bar(portfolio.index, rets, color=np.where(rets >= 0, '#2ca02c', '#d62728'), width=1.0, alpha=0.7)
    ax.set_title('Дневная доходность, %', fontsize=12, fontweight='bold')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)

    # 3. Просадка
    ax = axes[2]
    cummax = portfolio['portfolio_value'].cummax()
    dd = (portfolio['portfolio_value'] - cummax) / cummax * 100
    ax.fill_between(portfolio.index, dd, 0, color='#d62728', alpha=0.3)
    ax.plot(portfolio.index, dd, color='#d62728', linewidth=0.8)
    ax.set_title('Просадка, %', fontsize=12, fontweight='bold')
    ax.set_ylabel('%')
    ax.set_xlabel('Дата')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'portfolio_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"График сохранён: {path}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    t0 = time.time()
    config = ModelConfig()
    cache = APICache(config.cache_dir, config.cache_ttl_hours)
    start_date = '2016-01-01'

    logger.info("=" * 60)
    logger.info("ГИБРИД v5.3.0-ASYNC: запуск загрузки данных")
    logger.info("=" * 60)

    # 1. Параллельная загрузка ценовых активов
    price_assets = list(PRICE_ASSETS)
    logger.info(f"Загрузка {len(price_assets)} активов (параллельно, concurrency={config.api_concurrency})...")
    asset_data = await async_load_all_assets(config, price_assets, start_date, cache)

    prices = {}
    highs = {}
    lows = {}
    data_status = {}
    for name, (close, high, low, meta) in asset_data.items():
        prices[name] = close
        highs[name] = high
        lows[name] = low
        data_status[name] = meta

    # 2. Ключевая ставка
    logger.info("Загрузка ключевой ставки...")
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(config.api_timeout)) as client:
        key_rate, kr_meta = await async_load_key_rate_hybrid(client, config, cache)
    data_status['key_rate'] = kr_meta

    # 3. Макро
    logger.info("Загрузка макро-данных...")
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(config.api_timeout)) as client:
        m2_series, m2_meta = await async_load_macro_with_autofetch(client, 'M2', config, cache)
        mredc_series, mredc_meta = await async_load_macro_with_autofetch(client, 'MREDC', config, cache)
    data_status['M2'] = m2_meta
    data_status['MREDC'] = mredc_meta

    # 4. Проверка критических данных
    logger.info("Валидация данных...")
    critical_ok = True
    for name in CRITICAL_DATASETS:
        if name not in prices or prices[name].empty:
            logger.error(f"КРИТИЧЕСКИЙ АКТИВ {name} НЕ ЗАГРУЖЕН")
            critical_ok = False
    if not critical_ok:
        logger.error("Прерывание: отсутствуют критические данные")
        return 1

    # 5. Расчёт сигналов V2.3
    logger.info("Расчёт технических индикаторов V2.3...")
    signals = {}
    for name in prices:
        if name in highs and name in lows:
            engine_df = v23_engine(prices[name], highs[name], lows[name], config)
        else:
            h = prices[name] * (1 + config.hl_range_pct)
            l = prices[name] * (1 - config.hl_range_pct)
            engine_df = v23_engine(prices[name], h, l, config)
        signals[name] = engine_df['position']

    # 6. Симуляция портфеля
    logger.info("Симуляция портфеля...")
    # Базовые веса: равновзвешенные по доступным активам
    n_active = len([k for k in prices if k not in {'MREDC', 'M2'}])
    weight_per_asset = 1.0 / n_active if n_active > 0 else 0.0
    base_weights = {}
    for k in prices:
        if k not in {'MREDC', 'M2'}:
            base_weights[k] = weight_per_asset

    portfolio = simulate_portfolio(prices, signals, config, target_weights=base_weights)

    # 7. Метрики
    rf = key_rate.iloc[-1] / 100.0 if not key_rate.empty else 0.18
    metrics = calculate_metrics(portfolio, rf_annual=rf, tax_rate=config.tax_rate)

    # 8. Корреляции и CVaR
    corr_matrix = compute_correlation_matrix(prices)
    cvar_assets = compute_cvar_by_asset(prices, alpha=config.cvar_alpha)

    # 9. Отчёт
    report = generate_report(portfolio, metrics, corr_matrix, cvar_assets, config, data_status)
    report_path = os.path.join(config.output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Отчёт сохранён: {report_path}")
    print("\n" + report)

    # 10. Графики
    plot_results(portfolio, config.output_dir)

    elapsed = time.time() - t0
    logger.info(f"Выполнено за {elapsed:.2f} секунд")
    return 0


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        sys.exit(130)
