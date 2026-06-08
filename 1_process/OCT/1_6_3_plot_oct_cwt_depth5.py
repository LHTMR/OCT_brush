# -*- coding: utf-8 -*-
"""
OCT CWT – 深さ5種類を比較する3行×5列図（Amplitude / Phase / CWT, dBスケール）
- ベース：OCT CWT quicklook (3段図) のレイアウト思想
- 深さ5本（offset 5種類）を 1枚の 3行×5列図で比較
- CWT:
    - Morlet (PyWavelets)
    - linear power |coeff|^2 -> dB 変換
    - dB reference は「1試行の全offset・全time・全freq」の中央値（共通基準）
    - カラースケール vmin/vmax も1試行の5offsetで共通
- 深さ処理:
    - USE_TRITAP = True で tri-tap (i-1, i, i+1) 平均
    - USE_TRITAP = False で raw（単一深さ）
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from matplotlib.gridspec import GridSpec

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools

# ============================================================
# Settings
# ============================================================

USE_TRITAP = True   # True: tri-tap average, False: raw depth

data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 26472 #10000
npyname = "phase_change_data.npy"

# CWT parameters
FMIN = 20.0
FMAX = 500.0
VOICES_PER_OCT = 32
CMOR_B, CMOR_C = 2.5, 1.0

# Depth bounds for safety
DEP_MIN = 1
DEP_MAX = 1022

WIN_SAMPLES = 3000  # number of time samples to visualize

# CWT dB display range (per trial, across 5 depths)
# If None, use data-driven min/max (global over 5 offsets)
CWT_VMIN_DB = None   # e.g. -40.0
CWT_VMAX_DB = None   # e.g.  20.0

SAVE_FIGURES = False   # True: save PNG per trial
SHOW_FIGURES = True    # True: show figure window

# ============================================================
# Depth processing: raw / tri-tap
# ============================================================

def build_trace_raw(phase_data, dep_indices, time_indices, offset):
    """Use depth index + offset (clipped) directly."""
    dep_core = np.clip(dep_indices + offset, DEP_MIN, DEP_MAX)
    t_idx = np.clip(time_indices, 0, phase_data.shape[2] - 1)
    return phase_data[0, dep_core, t_idx].astype(float)

def build_trace_tritap(phase_data, dep_indices, time_indices, offset):
    """Tri-tap average over depths i-1, i, i+1 with weights [0.25, 0.5, 0.25]."""
    out = []
    K = np.array([0.25, 0.5, 0.25], dtype=float)

    for base, t_idx in zip(dep_indices + offset, time_indices):
        i = int(np.clip(base, DEP_MIN, DEP_MAX))
        i0 = max(DEP_MIN, i - 1)
        i1 = i
        i2 = min(DEP_MAX, i + 1)

        vals = np.array([
            phase_data[0, i0, t_idx],
            phase_data[0, i1, t_idx],
            phase_data[0, i2, t_idx]
        ], dtype=float)

        out.append(np.sum(vals * K))

    return np.array(out, dtype=float)

# ============================================================
# CWT (Morlet, PyWavelets)
# ============================================================

def cwt_morlet_pywt(x, fs):
    """Morlet CWT -> linear power |coeff|^2 (time x freq)."""
    n_freqs = int(np.log2(FMAX / FMIN) * VOICES_PER_OCT)
    n_freqs = max(n_freqs, 8)
    freqs = np.geomspace(FMIN, FMAX, n_freqs)

    wavelet = f"cmor{CMOR_B}-{CMOR_C}"
    fc = pywt.central_frequency(wavelet)
    scales = (fc * fs) / freqs

    coeffs, _ = pywt.cwt(x, scales=scales, wavelet=wavelet, sampling_period=1.0 / fs)
    power = (np.abs(coeffs) ** 2).T  # shape: [time x freq]

    return freqs, power

# ============================================================
# split dataframe (original logic)
# ============================================================

def split_dataframe(df):
    consecutive_zeros = 0
    start_index = -1
    end_index = -1

    for i in range(len(df)):
        if df.iloc[i, 0] == 0:
            consecutive_zeros += 1
            if consecutive_zeros == 10 and start_index == -1:
                start_index = i - 9
        else:
            if consecutive_zeros >= 10 and end_index == -1:
                end_index = i
            consecutive_zeros = 0

    before_df = pd.DataFrame()
    after_df = pd.DataFrame()

    if start_index != -1:
        before_index = max(0, start_index - 5001)
        before_df = df.iloc[before_index:start_index - 1]

    if end_index != -1:
        after_rows = []
        for i in range(end_index + 1, min(end_index + 5001, len(df))):
            if df.iloc[i, 0] == 0:
                break
            after_rows.append(df.iloc[i])
        after_df = pd.DataFrame(after_rows)

    return [("before_brushing", before_df), ("after_brushing", after_df)]

# ============================================================
# 3行×5列 図の描画
# ============================================================

def plot_grid_3x5(amp_imgs, phase_list, freqs, power_db_list,
                  offsets, save_path, vmin_db, vmax_db):
    """
    3 rows x 5 cols:
        row 0: Amplitude image (depth x time)
        row 1: Phase trace (time)
        row 2: CWT power in dB (time x freq)
    x-axis: samples (0..WIN_SAMPLES-1), shared within each column.
    y-axis labels: only left column.
    """

    n_depths = len(offsets)
    assert n_depths == 5, "This function assumes exactly 5 depths (offsets)."

    fig = plt.figure(figsize=(22, 9))
    gs = GridSpec(3, 5, figure=fig, wspace=0.05, hspace=0.08)

    # --- Axes containers for sharex ---
    ax_amp_row = []
    ax_phase_row = []
    ax_cwt_row = []

    # X extents (samples)
    n_samples = phase_list[0].size
    x_start = 0
    x_end = n_samples - 1

    # ---- Row 0: Amplitude images ----
    for j in range(5):
        if j == 0:
            ax0 = fig.add_subplot(gs[0, j])
        else:
            # sharex within column: Amplitude row is reference
            ax0 = fig.add_subplot(gs[0, j], sharex=ax_amp_row[0])

        img = amp_imgs[j]
        depth_max = img.shape[0] - 1
        depth_min = 0
        extent0 = [x_start, x_end, depth_max, depth_min]

        ax0.imshow(img, aspect='auto', cmap='gray', origin='upper', extent=extent0)

        if j == 0:
            ax0.set_ylabel("Depth (px)")
        else:
            ax0.set_ylabel("")

        ax0.set_title(f"offset {offsets[j]}")
        ax0.set_xticks([])  # hide x ticks on row 0
        ax0.set_yticks([])

        ax_amp_row.append(ax0)

    # ---- Row 1: Phase (rad), shared x with row 0 ----
    for j in range(5):
        ax1 = fig.add_subplot(gs[1, j], sharex=ax_amp_row[0])
        ax1.plot(np.arange(n_samples), phase_list[j], lw=1.0)

        ax1.set_ylim(-np.pi, np.pi)

        if j == 0:
            ax1.set_ylabel("Phase (rad)")
        else:
            ax1.set_ylabel("")

        ax1.set_xticks([])   # hide x ticks on row 1
        ax1.set_yticks([])

        ax_phase_row.append(ax1)

    # ---- Row 2: CWT (dB), shared x with row 0 ----
    for j in range(5):
        ax2 = fig.add_subplot(gs[2, j], sharex=ax_amp_row[0])

        power_db = power_db_list[j]
        extent2 = [x_start, x_end, freqs[0], freqs[-1]]

        im = ax2.imshow(
            power_db.T,
            aspect='auto',
            origin='lower',
            extent=extent2,
            vmin=vmin_db,
            vmax=vmax_db
        )

        if j == 0:
            ax2.set_ylabel("Freq (Hz)")
        else:
            ax2.set_ylabel("")

        # x-axis: samples, 1000-step ticks
        ticks = np.arange(0, n_samples, 1000)
        ax2.set_xticks(ticks)
        ax2.set_xlabel("Samples")

        ax_cwt_row.append(ax2)

    # カラーバーは CWT の右端だけに付ける
    cbar = fig.colorbar(im, ax=ax_cwt_row, location='right', fraction=0.02, pad=0.02)
    cbar.set_label("CWT Power (dB)")

    # Tight layout
    fig.tight_layout()

    # Save / show
    if SAVE_FIGURES:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():

    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")

    input_names, input_abs, _ = path_tools.get_folders_with_file(
        db_path_input,
        target_file,
        automatic=set_path_automatic,
        select_multiple=False,
        verbose=True
    )

    if not input_abs:
        print("❌ No folders found")
        return

    for idx, cond in enumerate(input_names):

        print(f"\n===== Condition: {cond} =====")

        # ---- load phase / csv ----
        phase_path = os.path.join(input_abs[idx], npyname)
        csv_path = os.path.join(input_abs[idx], target_file)

        try:
            phase_data = np.load(phase_path)
        except Exception as e:
            print(f"⚠ Failed to load phase data: {phase_path} | {e}")
            continue

        try:
            csv_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠ Failed to read CSV: {csv_path} | {e}")
            continue

        # ---- split into before / after ----
        dfs = split_dataframe(csv_data)

        # we use only after_brushing
        label_after, df_after = dfs[1]
        if df_after.empty:
            print("⚠ after_brushing segment is empty, skip.")
            continue

        dep_indices = df_after.iloc[:, 0].astype(int).values
        time_indices = df_after.index.values

        # ---- offsets by cover (bare / tegaderm) ----
        cond_lower = cond.lower()
        if "bare" in cond_lower:
            offsets = [2, 20, 40, 160, 320]
        else:
            offsets = [22, 40, 60, 180, 340]

        if len(offsets) != 5:
            print("⚠ This script assumes exactly 5 offsets, but got:", offsets)
            continue

        # ---- Amplitude image (fallback: abs of phase_data) ----
        # phase_data shape: [1, depth, time]
        amp_full = np.abs(phase_data[0, :, :])  # depth x time

        # --------------------------------------------------------
        # Step 1: unify scales (dB reference & vmax) across 5 depths
        # --------------------------------------------------------
        trace_store = []
        power_store = []
        amp_store = []

        all_power_values = []

        # choose time window
        t0 = int(time_indices[0])
        time_win = np.arange(t0, t0 + WIN_SAMPLES)
        # NOTE: we will use index 0..WIN_SAMPLES-1 in plotting

        for offset in offsets:

            # --- build trace (raw or tri-tap) ---
            if USE_TRITAP:
                trace_sparse = build_trace_tritap(phase_data, dep_indices, time_indices, offset)
            else:
                trace_sparse = build_trace_raw(phase_data, dep_indices, time_indices, offset)

            # --- interpolate onto uniform time window (WIN_SAMPLES) ---
            trace = np.full(WIN_SAMPLES, np.nan, dtype=float)
            cols = np.searchsorted(time_win, time_indices)
            valid = (cols >= 0) & (cols < WIN_SAMPLES)
            trace[cols[valid]] = trace_sparse[: valid.sum()]

            valid_idx = np.where(~np.isnan(trace))[0]
            if valid_idx.size < 2:
                print(f"⚠ trace too sparse: {cond} | off{offset}")
                continue

            trace = np.interp(
                np.arange(WIN_SAMPLES),
                valid_idx,
                trace[valid_idx]
            )

            # --- CWT (linear power) ---
            freqs, power_lin = cwt_morlet_pywt(trace, sampling_rate)

            # --- collect for global reference & vmax ---
            power_store.append(power_lin)
            trace_store.append(trace)
            amp_store.append(amp_full[:, time_win])  # depth x WIN_SAMPLES

            all_power_values.append(power_lin.ravel())

        if not power_store:
            print("⚠ No valid traces for this condition, skip.")
            continue

        # ---- build global reference for dB (median of linear power) ----
        all_power_values = np.concatenate(all_power_values)
        finite_mask = np.isfinite(all_power_values) & (all_power_values > 0)
        if not np.any(finite_mask):
            print("⚠ No positive finite power values, skip.")
            continue

        ref_power = np.nanmedian(all_power_values[finite_mask])
        eps = np.finfo(float).tiny

        # ---- convert each power to dB and find global dB min/max ----
        power_db_list = []
        global_min_db = np.inf
        global_max_db = -np.inf

        for p_lin in power_store:
            p_db = 10.0 * np.log10((p_lin + eps) / ref_power)
            power_db_list.append(p_db)

            local_min = np.nanmin(p_db)
            local_max = np.nanmax(p_db)

            if np.isfinite(local_min):
                global_min_db = min(global_min_db, local_min)
            if np.isfinite(local_max):
                global_max_db = max(global_max_db, local_max)

        # decide final vmin/vmax in dB
        if CWT_VMIN_DB is not None:
            vmin_db = CWT_VMIN_DB
        else:
            vmin_db = global_min_db

        if CWT_VMAX_DB is not None:
            vmax_db = CWT_VMAX_DB
        else:
            vmax_db = global_max_db

        # --------------------------------------------------------
        # Step 3: plot 3x5 grid for this condition
        # --------------------------------------------------------
        out_png = Path(input_abs[idx]) / "depth5_compare_grid_dB.png"

        # sanity: we expect 5 traces/amps/powers
        if len(amp_store) != 5 or len(trace_store) != 5 or len(power_db_list) != 5:
            print("⚠ Expected 5 valid offsets but got "
                  f"amp:{len(amp_store)}, trace:{len(trace_store)}, power:{len(power_db_list)}")
            continue

        plot_grid_3x5(
            amp_imgs=amp_store,
            phase_list=trace_store,
            freqs=freqs,
            power_db_list=power_db_list,
            offsets=offsets,
            save_path=str(out_png),
            vmin_db=vmin_db,
            vmax_db=vmax_db
        )

        if SAVE_FIGURES:
            print(f"✅ Saved: {out_png}")
        else:
            print(f"ℹ️ Figure not saved (SAVE_FIGURES = False): {out_png}")

    print("\n🎉 Done. 3x5 depth comparison figures (dB) processed for all conditions.")


if __name__ == "__main__":
    main()
