# How Far Can a PSRAM-less ESP32 Go? A TinyML Deployment Study for WiFi CSI HAR

Submission and reproducibility package.

**Authors:** Annas W. Malik and Muhammad Ahmad, Faculty of Information Technology & Computer Science, University of Central Punjab, Lahore 54000, Pakistan.
**Target venue:** IEEE Access (primary), IEEE Sensors Journal (alternative).
**Paper type:** systems / TinyML deployment-characterization (not a new-model paper).

## What this paper shows
On the cheapest and most constrained commodity WiFi microcontroller, the classic
PSRAM-less ESP32 (about 150 kB of usable contiguous SRAM), we map the deployability
frontier for WiFi CSI human activity recognition across three model tiers (classical
learners, tiny neural networks, and five standard deep architectures) on two public
datasets (UT-HAR and CSI-HAR), with NTU-Fi_HAR as a third-dataset check. We also measure
per-inference energy directly, compare against the ESP32-S3, run the full capture-to-
prediction pipeline live on-device, and reproduce the cross-subject gap on two larger
public corpora. The two-part conclusion: on this hardware deployment is not the
bottleneck, but cross-subject generalization is.

Headline findings, all backed by the evidence in `results/`:
1. **The board runs more than expected.** Flashing every convertible model on-device shows a
   deep CNN (10 to 12 kB arena) and Transformer (24 to 54 kB) both run on the bare classic
   ESP32, so there is NO general memory wall. Only an export-pipeline compatibility boundary
   is real: recurrent and state-space models do not convert to TensorFlow Lite for
   Microcontrollers, and the Chebyshev-KAN converts but fails to allocate.
2. **Classical learners and tiny CNNs fit and run cheaply.** Measured on the board at 240 MHz:
   tiny networks use 8 to 13 kB of arena and 9 to 68 ms per inference (deep CNN and
   Transformer 170 to 321 ms). The classical tier's end-to-end cost is dominated by the
   shared five-statistic feature extraction (1.6 ms UT-HAR, 0.9 ms CSI-HAR), not the
   microsecond tree traversal, so it runs in about 1 to 2 ms, roughly an order of magnitude
   faster than the tiny CNNs.
3. **On this dataset, cross-subject generalization is the harder problem (exploratory).** Under
   leave-one-user-out on CSI-HAR the same six models drop a mean of 29 points (best 96 to 63
   percent) versus a random subject-overlapping split. The gap reproduces on CSI-Bench (35 users):
   84.1 to 51.7 percent, a 32-point drop that stays above chance. (MM-Fi was explored but is
   excluded from the manuscript: with the deliberately light amplitude-statistics classifier its
   27-class accuracy collapses to chance under a subject-independent split, a floor effect rather
   than a measurable graded gap; the raw results remain in `results/gengap_external/`.)
   On the saturated UT-HAR and NTU-Fi_HAR benchmarks the small models stay competitive
   (tiny CNN 96.7 / 100 percent, statistics MLP 95.3 percent).
4. **Measured energy (INA219, not estimated).** Per-inference energy on the classic ESP32
   (5 V board input, radio off) ranges from 3.5 mJ (tiny CNN, CSI-HAR) to 117.8 mJ (deep
   Transformer, CSI-HAR); idle baseline 254 mW. The measured active current exceeds the
   68 mA datasheet figure for several models, so the earlier 50 mA estimate was not a
   conservative bound.
5. **ESP32-S3 comparison.** The same int8 models on the PSRAM-equipped ESP32-S3 (esp-nn
   SIMD kernels) run on average 4.8x faster and use about 70 percent less energy per
   inference, the convolutional models most (up to 7.8x). The classic ESP32 is the
   conservative floor of the family.
6. **Live deployment.** The full acquisition-to-prediction pipeline (2.4 GHz station, CSI RX
   callback, windowing, normalization, int8 quantization, inference) ran continuously for
   30 minutes on the classic ESP32: 180,815 CSI packets, 2,825 windows, 0.00 percent loss,
   37.6 ms end-to-end latency, ~2.5 kB heap jitter (no leak), zero crashes.
7. **What else governs on-device speed and fit (measured):**
   - Optimized kernels dominate: ESP-NN gives a 13.6x speedup over the reference kernels;
     without it the deep CNN cannot finish one inference.
   - CPU clock: 240 MHz is about 1.5x faster than the 160 MHz ESP-IDF default.
   - Compiler flag (-O2 vs -Os): under 3 percent. Not a useful lever.
   - WiFi active costs about 44 kB of internal RAM (152 kB free block drops to 108 kB), but
     every model still fits and latency is unchanged.
   - The firmware image is dominated by the TensorFlow Lite Micro runtime (~232 kB of flash),
     not the model file; static SRAM use is ~12 kB, leaving ~168 kB for heap and arena.
   - sklearn-to-emlearn-C parity: Decision Tree and MLP are bit-exact; Random Forest matches
     100 percent (CSI-HAR) / 98.6 percent (UT-HAR) of the real test set (float export).

## Folder layout
```
manuscript/   LaTeX source + classes + figures (compiles as-is on Overleaf, pdfLaTeX)
code/         Executed Kaggle notebooks + generators; ESP32 firmware + measurement scripts
results/      All measured evidence: tables, JSON, figures, on-device logs (incl. energy,
              ESP32-S3, live pipeline, RAM breakdown, emlearn parity, external gen-gap)
models/       Trained artifacts: .joblib (sklearn), int8 .tflite, emlearn C headers
```

## How to reproduce
1. **Off-device** (accuracy, int8 sizes, convertibility, figures): run the notebooks in
   `code/kaggle_notebooks/` on Kaggle with the datasets attached (see below), GPU T4,
   Internet on. Outputs match `results/`. See `code/README.md`.
2. **On-device latency / RAM / energy**: flash the firmware in `code/firmware/` to a classic
   ESP32 (`csi_bench_main.cc` + `csi_bench_main_CMakeLists.txt`, ESP-IDF v5.1). Build-time
   switches select the mode: default (latency/RAM), `ENERGY=1` (INA219 marker), `RECORD_ALLOC=1`
   (arena breakdown), `REPLAY_DEMO=1` (stored-window replay), and `LIVE=1` (live CSI pipeline;
   needs `wifi_creds.h`, see `csi_bench_wifi_creds.h.template`, and `sdkconfig.live`). Capture
   scripts: `measure_tiny.sh`, `measure_record.sh`, `measure_replay.sh`, `measure_arena.sh`.
3. **Cross-subject external validation (MM-Fi 40 subjects)**: `code/kaggle_notebooks/mmfi_colab_gengap.py`
   reproduces the MM-Fi gen-gap in Google Colab (bypasses the dataset's Drive download quota
   via a server-side copy).
4. **Manuscript PDF**: open `manuscript/` on Overleaf and compile `frontier.tex` with
   pdfLaTeX. See `manuscript/COMPILE.md`.

## Kaggle notebooks (public, runnable)
Every off-device experiment is a public Kaggle notebook. The executed `.ipynb` files (with
outputs) are in `code/kaggle_notebooks/`.

| Experiment | Notebook |
|---|---|
| Deployability frontier (classical + tiny) | https://www.kaggle.com/code/muhammadahmad3/csi-deployability-frontier |
| Five-architecture benchmark | https://www.kaggle.com/code/muhammadahmad3/csi-architecture-benchmark |
| Deep tier on CSI-HAR | https://www.kaggle.com/code/muhammadahmad3/csi-deeptier-csihar |
| Float vs int8 accuracy validation | https://www.kaggle.com/code/muhammadahmad3/csi-accval |
| sklearn vs emlearn-C parity | https://www.kaggle.com/code/muhammadahmad3/csi-emlearn-parity |
| Reproducibility (multi-seed LOUO, resampling) | https://www.kaggle.com/code/muhammadahmad3/csi-repro |
| Unrolled-GRU convertibility | https://www.kaggle.com/code/muhammadahmad3/csi-gru-export |
| Cross-subject generalization gap (CSI-HAR) | https://www.kaggle.com/code/muhammadahmad3/csi-gengap |
| Cross-subject on CSI-Bench (35 users) | https://www.kaggle.com/code/muhammadahmad3/csi-bench-gengap |
| Confusion matrices, precision/recall/F1 | https://www.kaggle.com/code/muhammadahmad3/csi-metrics-cpu |
| Hyperparameter sweep (LR x batch) | https://www.kaggle.com/code/muhammadahmad3/csi-hpsweep-cpu |
| Per-user personalization curve | https://www.kaggle.com/code/muhammadahmad3/csi-personalization |
| On-device replay demo assets | https://www.kaggle.com/code/muhammadahmad3/csi-replay-demo-cpu |
| Third dataset (NTU-Fi_HAR) | https://www.kaggle.com/code/muhammadahmad3/csi-ntufi-frontier |

MM-Fi (40 subjects) was run in Google Colab, not Kaggle; see `code/kaggle_notebooks/mmfi_colab_gengap.py`.

## Datasets (public, not redistributed here)
All are public datasets from their original authors. We list the authoritative source/paper
first, then the Kaggle mirror actually attached to our notebooks where applicable.

**UT-HAR** (Intel 5300, 7 activities) - the most widely used public CSI HAR benchmark.
- Original: https://github.com/ermongroup/Wifi_Activity_Recognition
- Paper: S. Yousefi et al., "A survey on behavior recognition using WiFi channel state
  information," *IEEE Communications Magazine*, 55(10):98-104, 2017.
  https://doi.org/10.1109/MCOM.2017.1700082
- Loaders: SenseFi, https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- Mirror: https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-ut-har

**CSI-HAR** (ESP32-collected, 7 activities, 3 users) - our subject-independent dataset.
- Original: https://github.com/parisafm/CSI-HAR-Dataset
- Paper: P. Fard Moshiri et al., "A CSI-based human activity recognition using deep
  learning," *Sensors*, 21(21):7225, 2021. https://doi.org/10.3390/s21217225
- Mirror: https://www.kaggle.com/datasets/sayakghorai34/csi-har-dataset

**NTU-Fi_HAR** (6 activities, 3 antennas x 114 subcarriers) - third-dataset frontier check.
- Original / paper: J. Yang et al., "SenseFi," *Patterns*, 2023.
  https://doi.org/10.1016/j.patter.2023.100703
- Mirror: https://www.kaggle.com/datasets/imhoangt/ntu-fi-dataset

**MM-Fi** (40 subjects, 4 environments, 27 daily-activity classes) - cross-subject validation.
- Paper: J. Yang et al., "MM-Fi: Multi-modal non-intrusive 4D human dataset for versatile
  wireless sensing," *NeurIPS Datasets and Benchmarks*, 2023.
- Project: https://github.com/ybhbingo/MMFi_dataset  (License CC BY-NC; not redistributed)

**CSI-Bench** (35 users, in-the-wild multi-task) - cross-subject validation.
- Paper: G. Zhu et al., "CSI-Bench: A large-scale in-the-wild dataset for multi-task WiFi
  sensing," *NeurIPS Datasets and Benchmarks*, 2025. arXiv:2505.21866
- Project: https://github.com/guozhenjennzhu/CSI-Bench  (License CC BY-NC-ND; not redistributed)

## Hardware
Classic ESP32-D0WD-V3 (dual-core Xtensa LX6 at 240 MHz, 520 kB internal SRAM, no PSRAM,
4 MB flash). Latency/RAM measured with WiFi/Bluetooth disabled; energy measured with an
INA219 in series with the 5 V board input; the live-pipeline experiment runs the radio.
The ESP32-S3 comparison uses an ESP32-S3-DevKitC-1 (N16R8, 8 MB PSRAM).

## License and third-party code
Our own code, firmware, notebooks, and results are released under the MIT License
(see `LICENSE`). One file is not ours: `code/firmware/esp-tflite-micro_CMakeLists.txt` is a
**modified copy** of a build file from Espressif's `esp-tflite-micro` component
(https://github.com/espressif/tflite-micro-esp-examples), under the **Apache License 2.0**,
which it remains under. Our change only adds an environment-guarded switch (`NO_ESP_NN`) to
keep the portable reference kernels for the ESP-NN speedup measurement; the default build
path is unchanged. The datasets are not redistributed; see the dataset section for sources
and licenses.
