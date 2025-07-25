import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast
import math
import json
import yaml
import imageio
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_absolute_percentage_error,
)
from datetime import datetime
from pathlib import Path


# ======================= dataframe processing =======================
def load_tables(root_dir, file_types=None):
    """
    Recursively load table-form files (CSV, Excel) from a directory.
    
    Args:
        root_dir (str or Path): Root directory to search.
        file_types (list of str): File extensions to include (e.g., ['csv','xlsx','xls']).
                                  Defaults to ['csv','xlsx','xls'].
                                  
    Returns:
        list of pd.DataFrame: List of dataframes loaded.
    """
    if file_types is None:
        file_types = ['csv', 'xlsx', 'xls']
    root = Path(root_dir)
    dfs = []
    
    for ext in file_types:
        for file in root.rglob(f'*.{ext}'):
            try:
                df = (pd.read_csv(file) if ext == 'csv' 
                      else pd.read_excel(file))
                dfs.append(df)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    
    return dfs

def merge_dfs(dfs):
    """
    Union-merge a list of DataFrames on their shared columns,
    preserving column order from the first DataFrame.
    
    Args:
        dfs (list of pd.DataFrame): DataFrames to merge.
        
    Returns:
        pd.DataFrame: Single DataFrame containing all rows, in the order
                      of shared columns as they appear in the first DataFrame.
    """
    if not dfs:
        return pd.DataFrame()
    
    # Find common columns
    shared_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        shared_cols &= set(df.columns)
    # Preserve order from the first DataFrame
    ordered_cols = [col for col in dfs[0].columns if col in shared_cols]
    
    # Concatenate slices in that order
    return pd.concat([df[ordered_cols] for df in dfs], ignore_index=True)

def clip_df(df: pd.DataFrame, lower: float = 0.0, upper: float = 1.0) -> pd.DataFrame:
    return df.clip(lower=lower, upper=upper)

def pick_from_list_str(df: pd.DataFrame, cols: list, index: int) -> pd.DataFrame:
    for col in cols:
        parts = df[col].str.strip('[]').str.split(',', expand=True)
        idx = index if index >= 0 else parts.shape[1] + index
        df[col] = parts.iloc[:, idx].str.strip()
    return df

def convert_strings_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any column values that look like numbers into floats.
    Non‑numeric strings are left unchanged.
    """
    def try_numeric(col: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(col, errors="raise")
        except (ValueError, TypeError):
            return col 
    return df.apply(try_numeric)
    
def normalize_min_max(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Scale specified columns to the [0, 1] range using min–max normalization,
    modifying the DataFrame in place and returning it.
    """
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def log_scale_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Return a copy of df where each column in cols is mapped to its natural log.
    """
    for col in cols:
        df[col] = np.log(df[col])
    return df

def add_valid_flag(df: pd.DataFrame, cols: list, pct: float) -> pd.DataFrame:
    """
    Flags rows as valid (1) if all specified columns are within the [pct, 1–pct] quantiles,
    otherwise flags them as invalid (0), and stores the result in 'valid'.
    """
    # start with all ones (all valid)
    flags = pd.Series(1, index=df.index)
    # zero out any row where any specified column is an outlier
    for c in cols:
        lo = df[c].quantile(pct)
        hi = df[c].quantile(1 - pct)
        mask = (df[c] < lo) | (df[c] > hi)
        flags[mask] = 0
    # attach and return
    df['valid'] = flags
    return df

def compute_bounds(dfs):
    """
    dfs: list of pandas.DataFrame with identical columns
    returns: (min_bounds, max_bounds) as two dicts
    """
    # build a DataFrame of per-DF mins/maxs, then collapse
    mins = pd.concat([df.min() for df in dfs], axis=1).min(axis=1).to_dict()
    maxs = pd.concat([df.max() for df in dfs], axis=1).max(axis=1).to_dict()
    return mins, maxs

def merge_dicts(*dicts) -> dict:
    """
    Merge any number of dicts; later ones override earlier keys.
    """
    merged = {}
    for d in dicts:
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
        merged.update(d)
    return merged

# ======================= plot functions =======================

# APS-style color palette (Physical Review Letters)
APS_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion/red
    "#CC79A7",  # reddish purple
    "#999999",  # medium gray
]

# Preselect color groups
LEFT_COLORS = [APS_COLORS[1], APS_COLORS[2], APS_COLORS[4]]      # sky blue, bluish green, blue
RIGHT_COLORS = [APS_COLORS[0], APS_COLORS[5], APS_COLORS[6]]     # orange, vermillion/red, reddish purple

# Lookup table for metric direction: 'min' means lower is better, 'max' means higher is better
METRIC_DIRECTION = {
    'rmse': 'min',
    'mse': 'min',
    'mae': 'min',
    'log_loss': 'min',
    'cross_entropy': 'min',
    'accuracy': 'max',
    'r2': 'max',
    'precision': 'max',
    'recall': 'max',
    'f1': 'max',
    'auc': 'max',
    'roc_auc': 'max'
}


def plot_history(history):
    h = history.history
    plt.plot(h['loss'],    label='train loss')
    plt.plot(h['val_loss'], label='val loss')
    if 'accuracy' in h:
        plt.plot(h['accuracy'],     label='train acc')
        plt.plot(h['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def process_df(df: pd.DataFrame, inputs, outputs, save_to, funcs):
    """
    df       : full DataFrame
    inputs   : list of feature column names (kept in order, rows untouched)
    outputs  : list of base output names, e.g. ['mean_x','sigma_y',...]
    
    Assumes df has columns true_<output> and pred_<output> for each entry in outputs.
    """
    X = df[inputs]
    res = []
    for name in outputs:
        tcol = f"true_{name}"
        pcol = f"pred_{name}"
        if tcol in df.columns and pcol in df.columns:
            y_true = df[tcol]
            y_pred = df[pcol]
            temp = (y_true, y_pred, name)
            # TODO: call your subroutine here, e.g.
            for func in funcs:
                func(*temp)
            dic = compute_regression_metrics(*temp)
            res.append(dic)
        else:
            raise KeyError(f"Columns {tcol} and/or {pcol} not found in DataFrame")
    save_to_csv(res, save_to)

def plot_true_vs_pred(y_true, y_pred, name, rng):
    # label
    if name in ('emittance_x','emittance_y'):
        var = 'x' if name=='emittance_x' else 'y'
        label_tex = rf'$\varepsilon_{var}\;(\mathrm{{mm}}\cdot\mathrm{{mrad}})$'
    else:
        label_tex = name

    # constant 0–1 range
    x = np.linspace(0, 1, 200)
    blue   = '#1f77b4'
    orange = '#ff7f0e'

    # square figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)

    # ±rng/2 band
    ax.fill_between(x, x - rng/2, x + rng/2,
                    color=blue, alpha=0.3,
                    label='Simulation variability')
    ax.plot(x, x - rng/2, '-', color=blue, linewidth=0.5)
    ax.plot(x, x + rng/2, '-', color=blue, linewidth=0.5)

    # perfect‑prediction line
    ax.plot([0, 1], [0, 1], '--', color=blue, linewidth=1,
            label=r'$y = x$')

    # data
    ax.scatter(y_true, y_pred,
               c=orange, alpha=0.8, s=5,
               label='Data samples')

    # labels
    ax.set_xlabel(f'True {label_tex}')
    ax.set_ylabel(f'Predicted {label_tex}')

    # fixed axes and square aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')

    # tick formatting (hide zero on y)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: '' if v == 0 else f'{v:g}')
    )
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: f'{v:g}')
    )

    # ax.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true, y_pred, name):
    # compute percent error (since normalized to 1)
    errors = (y_pred - y_true) * 100
    plt.figure()
    plt.hist(errors, bins=100, alpha=0.7)
    plt.axvline(0, linestyle='--')  # zero‑error line
    plt.xlabel(f"Error (%) for {name}")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution: {name}")
    plt.tight_layout()
    plt.show()

def plot_distributions(df, bins=10,
                       min_bounds=None, max_bounds=None,
                       title=None):
    """
    Create a grid of histograms for each DataFrame column,
    fixing the x‑axis to given bounds if provided,
    and optionally adding a big title above all subplots.

    Parameters:
    - df: pandas.DataFrame with numeric columns
    - bins: number of histogram bins
    - min_bounds: dict mapping column name -> min x‑value
    - max_bounds: dict mapping column name -> max x‑value
    - title: optional str, a super‐title for the entire figure

    Returns:
    - fig: matplotlib.figure.Figure
    """
    min_bounds = min_bounds or {}
    max_bounds = max_bounds or {}
    n = df.shape[1]

    # determine grid size (square-ish)
    n_cols = int(math.ceil(math.sqrt(n)))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    if title:
        fig.suptitle(title, fontsize=16)
        # make room for suptitle
        plt.subplots_adjust(top=0.92)

    for idx, col in enumerate(df.columns):
        data = df[col].dropna()
        color = APS_COLORS[idx % len(APS_COLORS)]
        lo = min_bounds.get(col, data.min())
        hi = max_bounds.get(col, data.max())

        axes[idx].hist(data, bins=bins, range=(lo, hi), color=color)
        axes[idx].set_xlim(lo, hi)
        axes[idx].set_title(f"Distribution of '{col}'")
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Count")

    # hide any unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig

def plot_key_lines(data_list, y_keys, x_key, margin_frac=0.05):
    """
    Plot one or more y‑series against an x‑series from a list of dicts,
    using dual y-axes based on whether lower (min) or higher (max) is better.

    Adds a fractional margin around each y-axis range for clarity.

    Parameters
    ----------
    data_list : list of dict
        Your raw data, e.g. [{'time':0, 'a':1, 'b':2}, …]
    y_keys : list of str
        Keys to plot on the y‑axis (multiple lines)
    x_key : str
        Key to plot on the x‑axis
    margin_frac : float
        Fractional margin to add above and below each axis range (default 0.05)
    """
    # sort by x for a clean line
    sorted_data = sorted(data_list, key=lambda d: d[x_key])
    x = [d[x_key] for d in sorted_data]

    # assign metrics to axes
    left_keys, right_keys = [], []
    for key in y_keys:
        direction = METRIC_DIRECTION.get(key.lower(), 'min')
        if direction == 'max':
            right_keys.append(key)
        else:
            left_keys.append(key)

    # gather values
    left_values = [[d[k] for d in sorted_data] for k in left_keys] or None
    right_values = [[d[k] for d in sorted_data] for k in right_keys] or None

    # compute ranges with margins
    def compute_limits(values):
        vmin = min(min(vals) for vals in values)
        vmax = max(max(vals) for vals in values)
        vrange = vmax - vmin
        return vmin - margin_frac * vrange, vmax + margin_frac * vrange

    left_limits = compute_limits(left_values) if left_values else None
    right_limits = compute_limits(right_values) if right_values else None

    # plot setup
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx() if right_limits else None

    markers = ['o','s','^','d','P','X','v','<','>','*']

    # draw left (minimize) metrics in blue spectrum
    for i, key in enumerate(left_keys):
        y = [d[key] for d in sorted_data]
        color = LEFT_COLORS[i % len(LEFT_COLORS)]
        ax_left.plot(x, y,
                     marker=markers[i % len(markers)],
                     label=key,
                     color=color)
    if left_limits:
        ax_left.set_ylim(*left_limits)

    # draw right (maximize) metrics in red/orange spectrum
    if ax_right:
        for j, key in enumerate(right_keys):
            y = [d[key] for d in sorted_data]
            color = RIGHT_COLORS[j % len(RIGHT_COLORS)]
            ax_right.plot(x, y,
                          marker=markers[(j + len(left_keys)) % len(markers)],
                          label=key,
                          color=color)
        ax_right.set_ylim(*right_limits)

    # labels
    ax_left.set_xlabel(x_key)
    ax_left.set_ylabel('Metric value\n(lower better)')
    if ax_right:
        ax_right.set_ylabel('Metric value\n(higher better)')

    # legend combined
    hL, lL = ax_left.get_legend_handles_labels()
    if ax_right:
        hR, lR = ax_right.get_legend_handles_labels()
        ax_left.legend(hL + hR, lL + lR, loc='best')
    else:
        ax_left.legend(hL, lL, loc='best')

    plt.tight_layout()
    plt.show()


# ======================= evaluation metrics =======================

def compute_regression_metrics(y_true, y_pred, name=None):
    """
    Compute common regression metrics between true and predicted values,
    returning a dict under the given `name` key.
    """
    return {name: metrics(y_true, y_pred)}

def metrics(y_true, y_pred):
    """
    Compute common regression metrics between true and predicted values,
    returning a dict under the given `name` key.
    """
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'ExplainedVar': explained_variance_score(y_true, y_pred),
            'MaxError': max_error(y_true, y_pred),
            'MedianAE': median_absolute_error(y_true, y_pred),
            'MAPE': mape,
            'Accuracy': max(0.0, 1.0 - mape)  # clip so it never goes below 0,
        }

# ======================= file I/O operations =======================

def load_config(path):
    """
    Read a YAML config file from `path` and return it as a dict.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)  # parses into Python dict
        
def save_to_csv(data, path):
    """
    Save `data` (a list of dicts) to a CSV file with a timestamp in its name.
    """
    # build DataFrame: each item is {feature: {metric: value}}
    records = {}
    for item in data:
        for feature, metrics in item.items():
            records[feature] = metrics
    df = pd.DataFrame.from_dict(records, orient='index')
    # ensure .csv extension and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = path[:-4] if path.lower().endswith('.csv') else path
    filename = f"{base}_{timestamp}.csv"
    # save
    df.to_csv(filename, index_label='feature')

def figures_to_gif(fig_list, path, fps=1):
    """
    Convert a list of matplotlib Figure objects into an animated GIF.
    The output filename will include a timestamp.

    Parameters:
    - fig_list: list of matplotlib.figure.Figure
    - path: str, base file path (with or without .gif extension)
    - fps: int, frames per second for the GIF

    Returns:
    - filename: the actual filename saved (with timestamp and .gif)
    """
    # Ensure .gif extension and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = path[:-4] if path.lower().endswith('.gif') else path
    filename = f"{base}_{timestamp}.gif"

    frames = []
    for fig in fig_list:
        # Render the figure to a RGB array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape((h, w, 3))
        frames.append(img)

    # Write out as an animated GIF
    imageio.mimsave(filename, frames, fps=fps)
    return filename
