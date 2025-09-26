# OCT_brush

This repository contains processing pipelines for **Optical Coherence Tomography (OCT)** data acquired in brush-related experiments.  
It includes scripts to process structural scans, vibration videos, morph data, phase-change maps, and skin-surface tracking.

---

## Folder Structure
```OCT_brush/
├─ 1_process/
│ └─ OCT/
│ ├─ 1_0_0_process_oct_structural_images.py
│ ├─ 1_1_1_process_oct_video_vibrations.py
│ ├─ 1_1_1_process_oct_video_vibrations_4saito.py
│ ├─ 1_1_3_process_oct_video_vibrations_show.py
│ ├─ 1_2_1_process_oct_video_vibration_as_image.py
│ ├─ 1_3_0_process_oct_phase-change.py
│ ├─ 1_3_1_process_oct_morph-n-phase-change_show.py
│ └─ 1_4_0_process_oct_skin-surface_tracking.py
└─ library_python/ (utility modules used across scripts)
```


---

## Scripts Overview (OCT)

- **1_0_0_process_oct_structural_images.py**  
  Processes structural OCT acquisitions into normalized 16-bit grayscale morph images.

- **1_1_1_process_oct_video_vibrations.py**  
  Computes morphological data for non-structural vibration recordings and saves results.

- **1_1_1_process_oct_video_vibrations_4saito.py**  
  Estimates skin displacement from vibration videos by depth filtering and boundary detection; exports CSV and optional figures.

- **1_1_3_process_oct_video_vibrations_show.py**  
  Loads processed vibration data (`morph.pkl`), applies optional resizing/smoothing/detrend/flip/normalization, and visualizes A-line surfaces or subplots.

- **1_2_1_process_oct_video_vibration_as_image.py**  
  Converts vibration volumes to per-A-line images with optional downsampling; exports a multi-page TIFF stack.

- **1_3_0_process_oct_phase-change.py**  
  Runs phase-change computation on vibration recordings and saves the results as `phase_change_data.npy`.

- **1_3_1_process_oct_morph-n-phase-change_show.py**  
  Visualizes amplitude (dB), phase, and phase-change images per A-line; exports PNGs (packable into `_all_morph_and_phase/`).

- **1_4_0_process_oct_skin-surface_tracking.py**  
  Tracks the skin surface in vibration videos by denoising and binarizing depth-time slices; saves positions to `skin_displacement_estimation.csv`.

---

## Requirements
- Python 3.9+  
- Core libraries: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `scipy`, `scikit-image`, `tifffile`  
- Custom modules: `library_python` (included in this repo)

Install dependencies:
```bash
pip install -r requirements.txt
