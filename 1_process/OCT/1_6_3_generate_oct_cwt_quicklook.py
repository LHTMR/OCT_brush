# -*- coding: utf-8 -*-
"""
OCT CWT quicklook (図出力のみ) — 元の入出力/選択ロジックは維持
- フォルダ選択やCSV/NPY読み込みはオリジナルのまま
- トラックしたピクセル（深さ）に沿ってトレースを抽出
- CWT（Morlet, PyWavelets）でスカログラム + リッジ
- 2 Hz 幅の平均バンドパワー曲線
- PNG を <サンプルフォルダ>/figs_cwt/ に保存

依存：
    numpy, pandas, matplotlib, pywt, scipy (既存ライブラリ群)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Keep your project-specific path setup as-is ----
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # (未使用でも互換のため残置)
from library_python.sensors.OCT.OCTMorph import OCTMorph                        # (未使用でも互換のため残置)

# === Settings (元のまま／必要最小限のみ明示) ===
data_external_hdd = False
set_path_automatic = False
dataset = "OCT_BRUSH"
target_file = "skin_displacement_estimation_corrected.csv"
sampling_rate = 10000  # Fs (Hz) - 既存ロジックに合わせる
npyname = "phase_change_data.npy"

# 解析パラメータ（CWT）
FMIN = 20.0
FMAX = 1000.0
VOICES_PER_OCT = 32
CMOR_B, CMOR_C = 2.5, 1.0  # PyWaveletsの complex Morlet 'cmorB-C'

# 深さのtri-tap（境界安全）
DEP_KERNEL = np.array([0.25, 0.50, 0.25], dtype=float)
DEP_MIN = 1
DEP_MAX = 1022  # 0..1023 の想定で tri-tap を安全化するための内側範囲

# =====================================================================
# 解析ヘルパー（トラッキング沿いトレース → CWT → 図保存）
# =====================================================================

def safe_triplet_indices(i):
    """(i-1, i, i+1) を [DEP_MIN, DEP_MAX] に収めて返す"""
    i = int(i)
    a = max(DEP_MIN, i - 1)
    b = min(DEP_MAX, i)
    c = min(DEP_MAX, i + 1)
    return a, b, c

def build_trace(phase_data, dep_indices, time_indices, offset):
    """
    既存ロジックに合わせたトレース抽出：
    - 深さ：dep_indices + offset を境界クリップ
    - tri-tap（i-1, i, i+1）× 0.25/0.5/0.25（境界では重み再正規化）
    - 時間：df_part.index の (t_idx-1) を用いる既存の合わせ方
    """
    if phase_data is None or len(dep_indices) == 0 or len(time_indices) == 0:
        return np.array([])

    n = min(len(dep_indices), len(time_indices))
    dep_core = dep_indices[:n].astype(int) + int(offset)
    dep_core = np.clip(dep_core, DEP_MIN, DEP_MAX)

    t_idx = time_indices[:n].astype(int) - 1
    T = phase_data.shape[2]
    t_idx = np.clip(t_idx, 0, T - 1)

    out = np.empty(n, dtype=float)
    for k in range(n):
        i0, i1, i2 = safe_triplet_indices(dep_core[k])
        v0 = phase_data[0, i0, t_idx[k]]
        v1 = phase_data[0, i1, t_idx[k]]
        v2 = phase_data[0, i2, t_idx[k]]

        # 境界で重みを再正規化
        if (i0 == i1) and (i1 == i2):
            out[k] = float(v1)
        elif (i0 == i1) or (i1 == i2):
            vals = np.array([v0, v1, v2], float)
            w = DEP_KERNEL.copy()
            if i0 == i1:
                w[0] = 0.0
            if i1 == i2:
                w[2] = 0.0
            w = w / w.sum()
            out[k] = float((vals * w).sum())
        else:
            out[k] = float((np.array([v0, v1, v2]) * DEP_KERNEL).sum())
    return out

# --- CWT (PyWavelets) ---
import pywt

def cwt_morlet_pywt(x, fs, fmin=FMIN, fmax=FMAX, voices_per_oct=VOICES_PER_OCT, cmor_B=CMOR_B, cmor_C=CMOR_C):
    """
    CWT using PyWavelets complex Morlet ('cmorB-C').
    Returns:
        freqs [Hz] (ascending), power [time x freq]
    """
    # 対数等間隔の周波数グリッド
    n_freqs = int(np.log2(fmax/fmin) * voices_per_oct)
    n_freqs = max(n_freqs, 8)
    freqs = np.geomspace(fmin, fmax, n_freqs)
    wavelet = f"cmor{cmor_B}-{cmor_C}"
    # PyWaveletsの換算： f = fc * fs / scale
    fc = pywt.central_frequency(wavelet)
    scales = (fc * fs) / freqs
    coeffs, _ = pywt.cwt(x, scales=scales, wavelet=wavelet, sampling_period=1.0/fs)
    power = (np.abs(coeffs)**2).T  # time x freq
    return freqs, power

# --- 図保存ユーティリティ ---
def save_timeseries(t, x, outpng, title):
    plt.figure(figsize=(9, 3))
    plt.plot(t, x)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()

def save_scalogram(t, freqs, power, outpng, title):
    # 主周波数リッジ（最大パワーの周波数）
    ridge_idx = np.nanargmax(power, axis=1)
    ridge_f = freqs[ridge_idx]
    extent = [t[0], t[-1], freqs[0], freqs[-1]]

    plt.figure(figsize=(9, 4))
    plt.imshow(power.T, aspect='auto', origin='lower', extent=extent)
    plt.plot(t, ridge_f, linewidth=1.2)
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()

def save_bandpower_2hz_mean(freqs, power, outpng, title, fmax=FMAX, binw=2.0):
    """
    時間平均の線形パワーを 2 Hz ビンで周波数積分 → 相対dBで図示（可視化用）
    """
    p_mean = np.nanmean(power, axis=0)
    edges = np.arange(0.0, fmax + binw, binw)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vals = np.full_like(centers, np.nan, dtype=float)

    for i, (L, R) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (freqs >= L) & (freqs < R)
        if np.any(sel):
            vals[i] = np.trapz(p_mean[sel], freqs[sel])

    eps = np.finfo(float).tiny
    ref = np.nanmedian(vals[np.isfinite(vals)]) if np.any(np.isfinite(vals)) else 1.0
    vals_db = 10.0 * np.log10((vals + eps) / ref)

    plt.figure(figsize=(9, 3))
    plt.plot(centers, vals_db)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Band power (dB, rel.)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close()

# =====================================================================
# 既存の「名前パーサ」「分割」関数（元のまま）
# =====================================================================

def parse_condition_name(name):
    parts = name.lower().split('_')
    return {
        'date': parts[0],
        'time': parts[1],
        'participant': parts[2],
        'location': parts[4],
        'texture': parts[6],
        'cover': parts[5],
        'frequency': parts[7]
    }

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

# =====================================================================
# main（元の選択・読み込みはそのまま）＋ CWT 図保存の最小追加
# =====================================================================

def main():
    # --- 入力パス決定（元のまま） ---
    db_path = path_tools.define_OCT_database_path(data_external_hdd)
    db_path_input = os.path.join(db_path, dataset, "2_processed", "oct")
    input_foldernames, input_foldernames_abs, _ = path_tools.get_folders_with_file(
        db_path_input, target_file, automatic=set_path_automatic, select_multiple=False, verbose=True
    )

    if not input_foldernames_abs:
        print("❌ No folders found that contain", target_file, "under:", db_path_input)
        return

    filepaths = [os.path.join(folder, npyname) for folder in input_foldernames_abs]
    corrpaths = [os.path.join(folder, target_file) for folder in input_foldernames_abs]

    phase_change_data_list = []
    for f in filepaths:
        try:
            data = np.load(f)
            phase_change_data_list.append(data)
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
            phase_change_data_list.append(None)

    csv_data_list = []
    for f in corrpaths:
        try:
            csv_data_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ Failed to read CSV {f}: {e}")
            csv_data_list.append(pd.DataFrame())

    # --- ループ（元の構造を維持） ---
    for idx, cond in enumerate(input_foldernames):
        parsed = parse_condition_name(cond)
        cover = parsed['cover']

        phase_data = phase_change_data_list[idx]
        csv_data = csv_data_list[idx]

        if cover not in ['bare', 'tegaderm']:
            print(f"⚠️ Skip (cover not in ['bare','tegaderm']): {cond}")
            continue
        if phase_data is None or csv_data.empty:
            print(f"⚠️ Missing data for: {cond}")
            continue

        # 保存ディレクトリ（サンプルごと）
        fig_dir = Path(input_foldernames_abs[idx]) / "figs_cwt"
        fig_dir.mkdir(exist_ok=True)

        # 既存の分割ロジックに従って before/after を得る
        dfs = split_dataframe(csv_data)

        # 既存の深さオフセット配列（元コード準拠）
        offsets = [2, 20, 40, 160] if cover == 'bare' else [22, 40, 60, 180]

        # 各区間 × 各オフセットで CWT 図を保存
        for label, df_part in dfs:
            if df_part.empty:
                continue

            dep_indices = df_part.iloc[:, 0].astype(int).values
            dep_indices = np.clip(dep_indices, 0, 1023)   # 元のクリップ
            time_indices = df_part.index.values
            time_indices = time_indices[time_indices > 0] # 元の整合

            for offset in offsets:
                trace = build_trace(phase_data, dep_indices, time_indices, offset)
                if trace.size < int(0.5 * sampling_rate):
                    # 0.5 s 未満は不安定なのでスキップ
                    print(f"⚠️ too short segment: {cond} | {label} | off{offset}")
                    continue

                # 時刻ベクトル（区間頭を0秒に）
                t = np.arange(trace.size) / sampling_rate

                # --- CWT ---
                freqs, power = cwt_morlet_pywt(
                    trace, sampling_rate,
                    fmin=FMIN, fmax=FMAX, voices_per_oct=VOICES_PER_OCT,
                    cmor_B=CMOR_B, cmor_C=CMOR_C
                )

                # --- 図の保存 ---
                base = f"{parsed['participant']}_{parsed['location']}_{parsed['texture']}_{parsed['cover']}_{parsed['frequency']}_{label}_off{offset}"
                save_timeseries(t, trace, fig_dir / f"{base}_timeseries.png", f"{base} time series")
                save_scalogram(t, freqs, power, fig_dir / f"{base}_scalogram.png", f"{base} CWT (Morlet)")
                save_bandpower_2hz_mean(freqs, power, fig_dir / f"{base}_bandpower2Hz.png", f"{base} mean band power")

                print(f"✅ Saved figures: {fig_dir} / {base}_*.png")

    print("🎉 Done. CWT quicklook figures are saved under each sample's 'figs_cwt' folder.")

if __name__ == "__main__":
    main()
