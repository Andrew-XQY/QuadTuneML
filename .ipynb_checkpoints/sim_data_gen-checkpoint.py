import itertools
import random
import os
import glob
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from cpymad.madx import Madx
import pymadx

###############################################################################
# CONFIGURATION (all parameters here for easy future config-driven use)
###############################################################################
CONFIG = {
    'home_path': '/Users/andrewxu/Documents/Projects/MADX GuI Trails/madx/acc-models-tls-main/elena_extraction/lne02/line',
    'lne02_str_path': '/Users/andrewxu/Documents/Projects/MADX GuI Trails/madx/acc-models-tls-main/elena_extraction/lne02/line/lne_repo/lne02/lne02_k.str',
    'trackone_dir': '/Users/andrewxu/Documents/Projects/MADX GuI Trails/madx/acc-models-tls-main/elena_extraction/lne02/line',
    'quad_keys': [
        'klne.zqmd.0208',
        'klne.zqmf.0209',
        'klne.zqmd.0214',
        'klne.zqmf.0215',
    ],
    'quad_range': [-100, 100],
    'quad_step': 20,
    'particles_per_sim': 6000,
    'beam_params': {
        'gemx': 2e-6/6,
        'betax': 4.2206719082,
        'alfx': 2.6283720873,
        'gemy': 4e-6/6,
        'betay': 5.3866337131,
        'alfy': 5.1980977991e-01,
    },
    'output_csv': 'data/batch_results.csv',
    'flush_every': 10,
}
os.makedirs(os.path.join(CONFIG['home_path'], 'data'), exist_ok=True)

###############################################################################
# QUEUE CONSTRUCTION (grid scan)
###############################################################################

def build_quad_grid_queue(config):
    keys = config['quad_keys']
    vals = np.arange(config['quad_range'][0], config['quad_range'][1]+config['quad_step'], config['quad_step'])
    queue = [dict(zip(keys, combo)) for combo in itertools.product(*(vals for _ in keys))]
    # Add index to each dict
    for idx, d in enumerate(queue):
        d['index'] = idx
    return queue

def build_quad_random_queue(config, n, m):
    """
    Generate n sets of random quadrupole strengths (for the 4 quads),
    each set repeated m times, for a total of n*m queue elements.
    """
    keys = config['quad_keys']
    low, high = config['quad_range']
    queue = []
    idx = 0
    for n_idx in range(n):
        quad_strengths = {k: random.uniform(low, high) for k in keys}
        for _ in range(m):
            entry = quad_strengths.copy()
            entry['index'] = n_idx
            queue.append(entry)
            idx += 1
    return queue


# Set working directory from CONFIG
os.chdir(CONFIG['home_path'])

# ─────────────────────────────────────────────────────────────────────────────
# 1) PARSERS & CALCULATORS
# ─────────────────────────────────────────────────────────────────────────────

def parse_trackone_to_df(filepath: str) -> pd.DataFrame:
    """
    Parse a MAD-X TRACKONE file into a pandas DataFrame with a 'segment' column.
    """
    rows, seg_ids, columns = [], [], []
    segment = 0

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('@') or line.startswith('$'):
                continue
            if line.startswith('*'):
                columns = line[1:].split()
                continue
            if line.startswith('#segment'):
                segment += 1
                continue

            parts = line.split()
            if columns and len(parts) == len(columns):
                values = [float(x) for x in parts]
                rows.append(values)
                seg_ids.append(segment)

    df = pd.DataFrame(rows, columns=columns)
    df['segment'] = seg_ids
    return df


def calculate_beam_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean_x, mean_y, sigma_x, sigma_y, emittance_x, emittance_y, transmission
    for each segment in the TRACKONE DataFrame.
    """
    def rms_emittance(x, px):
        x2 = np.mean(x*x)
        px2 = np.mean(px*px)
        xp =  np.mean(x*px)
        return np.sqrt(x2*px2 - xp*xp)

    groups = df.groupby('segment')
    N0 = groups.size().iloc[0]

    records = []
    for seg, g in groups:
        x, y = g['X'].to_numpy(), g['Y'].to_numpy()
        px, py = g['PX'].to_numpy(), g['PY'].to_numpy()
        N = len(x)

        # centroids
        mx, my = x.mean(), y.mean()
        # rms sizes
        sx = np.sqrt(np.mean((x - mx)**2))
        sy = np.sqrt(np.mean((y - my)**2))
        # emittances (unchanged)
        ex = rms_emittance(x, px)
        ey = rms_emittance(y, py)
        # transmission
        T  = N / N0

        records.append({
            'segment':       seg,
            'mean_x':        mx,
            'mean_y':        my,
            'sigma_x':       sx,
            'sigma_y':       sy,
            'emittance_x':   ex,
            'emittance_y':   ey,
            'transmission':  T,
        })

    out = pd.DataFrame(records).set_index('segment')
    return out


def beam_df_to_dict(beam_df: pd.DataFrame) -> dict:
    """
    Flatten an 8×N beam‐parameter DataFrame into a dict of lists.
    """
    return beam_df.to_dict(orient="list")


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

def save_metadata_txt(metadata: dict, filepath: str):
    """
    Save a flat dict of metadata to a human-readable TXT file,
    one key:value per line.
    """
    # make sure the directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2) SIMULATION DRIVER
# ─────────────────────────────────────────────────────────────────────────────

# Output/input file locations to match CONFIG
TRACKONE_DIR = CONFIG['trackone_dir']
OUTPUT_CSV   = CONFIG['output_csv']
FLUSH_EVERY  = CONFIG['flush_every']


def run_single(gconf: dict) -> dict:
    """
    Run one MAD-X job with quadrupoles set by gconf.
    Returns the merged dict of beam params + gconf.
    """


    # 1) Write quadrupole strengths to .str file (as in GUI code)
    with open(CONFIG['lne02_str_path'], 'w') as file:
        file.write(f"! LNE02\n")
        for k in CONFIG['quad_keys']:
            file.write(f"{k} = {gconf[k]};\n")

    # 2) start MAD-X quietly
    madx = Madx(stdout=False)

    # 3) generate beam with pymadx GaussGenerator
    bp = CONFIG['beam_params']
    G = pymadx.Ptc.GaussGenerator(
        gemx=bp['gemx'], betax=bp['betax'], alfx=bp['alfx'],
        gemy=bp['gemy'], betay=bp['betay'], alfy=bp['alfy'],
        sigmat=1e-12, sigmapt=1e-12
    )
    G.Generate(nToGenerate=CONFIG['particles_per_sim'], fileName='inrays.madx')

    # 4) call your MAD-X sequence
    madx.call(file='general_lne02.madx')

    # 5) pick up TRACKONE and parse
    trackone_file = os.path.join(CONFIG['trackone_dir'], 'trackone')
    df = parse_trackone_to_df(trackone_file)

    # 6) beam parameters
    beam_df = calculate_beam_parameters(df)
    beam_dict = beam_df_to_dict(beam_df)

    # 7) merge and return
    return merge_dicts(beam_dict, gconf)





###############################################################################
# MAIN LOOP: minimal, modular, queue-driven
###############################################################################
if __name__ == '__main__':
    import time
    start_time = time.time()
    # queue = build_quad_grid_queue(CONFIG)
    queue = build_quad_random_queue(CONFIG, n=5, m=50)
    all_dicts = []
    for idx, quad_conf in enumerate(tqdm(queue, desc="Quad grid scan"), 1):
        merged = run_single(quad_conf)
        all_dicts.append(merged)
        if idx % CONFIG['flush_every'] == 0:
            pd.DataFrame(all_dicts).to_csv(CONFIG['output_csv'], index=False)
    pd.DataFrame(all_dicts).to_csv(CONFIG['output_csv'], index=False)
    print(f"\nAll done! Results in {CONFIG['output_csv']}")

    # Save metadata
    end_time = time.time()
    total_runs = len(all_dicts)
    total_time = end_time - start_time
    avg_time = total_time / total_runs if total_runs else 0
    metadata = {
        'total_simulations': total_runs,
        'total_time_seconds': total_time,
        'avg_time_per_simulation_seconds': avg_time,
        'total_time_hours': total_time / 3600,
        'avg_time_per_simulation_ms': avg_time * 1000,
        'timestamp_start': start_time,
        'timestamp_end': end_time
    }
    META_JSON = CONFIG['output_csv'].replace('.csv', '.json')
    with open(META_JSON, 'w') as f:
        json.dump(metadata, f, indent=2)
