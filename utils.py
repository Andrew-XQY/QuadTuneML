import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ======================= dataframe processing =======================

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
    
def normalize_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame where each column is scaled to the [0, 1] range
    using min-max normalization.
    """
    return (df - df.min()) / (df.max() - df.min())

def log_scale_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Return a copy of df where each column in cols is mapped to its natural log.
    """
    for col in cols:
        df[col] = np.log(df[col])
    return df

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
    
# ======================= plot functions =======================

# APS-style color palette (Physical Review Letters)
aps_colors = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion/red
    "#CC79A7",  # reddish purple
    "#999999",  # medium gray
]

def process_df(df: pd.DataFrame, inputs, outputs, save_to):
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
            plot_true_vs_pred(*temp)
            plot_error_distribution(*temp)
            dic = compute_regression_metrics(*temp)
            res.append(dic)
        else:
            raise KeyError(f"Columns {tcol} and/or {pcol} not found in DataFrame")
    save_to_csv(res, save_to)

def plot_true_vs_pred(y_true, y_pred, name):
    # compute axis limits
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6, s=10)
    plt.plot([mn, mx], [mn, mx], linestyle='--')  # 45° line
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs. Predicted: {name}")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.gca().set_aspect('equal', 'box')
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
        color = aps_colors[idx % len(aps_colors)]
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


# ======================= evaluation metrics =======================

def compute_regression_metrics(y_true, y_pred, name):
    """
    Compute common regression metrics between true and predicted values,
    returning a dict under the given `name` key.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = max(0.0, 1.0 - mape)  # clip so it never goes below 0
    return {
        name: {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2': r2_score(y_true, y_pred),
            'ExplainedVar': explained_variance_score(y_true, y_pred),
            'MaxError': max_error(y_true, y_pred),
            'MedianAE': median_absolute_error(y_true, y_pred),
            'MAPE': mape,
            'Accuracy': accuracy,
        }
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
