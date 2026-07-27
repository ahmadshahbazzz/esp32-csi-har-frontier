# How Far Can a PSRAM-less ESP32 Go? A TinyML Deployment Study for WiFi CSI HAR

Submission and reproducibility package.

**Authors:** Annas Waseem Malik and Muhammad Ahmad, Faculty of Information Technology, University of Central Punjab, Lahore, Pakistan.
**Target venue:** IEEE Access (primary), IEEE Sensors Journal (alternative).
**Paper type:** systems / TinyML deployment-characterization (not a new-model paper).

## What this paper shows
On the cheapest and most constrained commodity WiFi microcontroller, the classic
PSRAM-less ESP32 (about 150 kB of usable contiguous SRAM), we map the deployability
frontier for WiFi CSI human activity recognition across three model tiers (classical
learners, tiny neural networks, and five standard deep architectures) on two public
datasets (UT-HAR and CSI-HAR), with NTU-Fi_HAR as a third-dataset check. The two-part conclusion: on this hardware deployment is
not the bottleneck, but cross-subject generalization is.

Headline findings, all backed by the evidence in `results/`:
1. The board runs more than expected. Flashing every convertible model on-device shows a
   full deep CNN (10 to 12 kB arena) and Transformer (24 to 54 kB) both run on the bare
   classic ESP32, so there is NO general memory wall. Only a convertibility wall is real:
   recurrent and state-space models do not convert to TensorFlow Lite for Microcontrollers,
   and the Chebyshev-KAN converts but fails to allocate.
2. Classical learners and tiny CNNs fit and run cheaply. Measured on the board at 240 MHz:
   tiny networks use 8 to 13 kB of arena and 9 to 68 ms per inference (deep CNN and
   Transformer 170 to 321 ms); the emlearn Decision Tree and Random Forest run in about 1
   and 45 microseconds with under 1 kB of RAM. They are the efficiency choice, not the only
   option.
3. The binding constraint is cross-subject generalization. Under leave-one-user-out on
   CSI-HAR the same six models drop a mean of 29 points (best 96 to 63 percent) versus a
   random split that leaks users across train and test. On the saturated UT-HAR benchmark
   the small models stay competitive (tiny CNN 96.7, statistics MLP 95.3 percent), and the
   same holds on a third dataset, NTU-Fi_HAR (tiny CNN 100, Random Forest 99.7 percent).
4. What actually governs on-device speed and fit (measured):
   - Optimized kernels dominate: ESP-NN gives a 13.6x speedup over the reference kernels
     (tiny CNN 20.2 vs 275.7 ms); without it the deep CNN cannot finish one inference.
   - CPU clock: 240 MHz is about 1.5x faster than the 160 MHz ESP-IDF default.
   - Compiler flag (-O2 vs -Os): under 3 percent. Not a useful lever.
   - WiFi active costs about 44 kB of internal RAM (152 kB free block drops to 108 kB), but
     every model still fits and latency is unchanged.
   - Tensor arena is a sharp threshold at `arena_used_bytes`, and extra arena buys no speed.
   - Continuous operation is stable: 12,000 back-to-back inferences with constant heap and
     latency (no leak, no fragmentation).


## Folder layout
```
submission/
  manuscript/   LaTeX source + classes + figures (compiles as-is on Overleaf)
  code/         Executed Kaggle notebooks + generators, and ESP32 firmware + scripts
  results/      All measured evidence: tables, JSON, figures, on-device logs
  models/       Trained artifacts: .joblib (sklearn), int8 .tflite, emlearn C headers
```

## How to reproduce
1. Off-device (accuracy, int8 sizes, convertibility, figures): run
   `code/kaggle_notebooks/csi-deployability-frontier.ipynb` on Kaggle with the two datasets
   attached (`hylanj/wifi-csi-dataset-ut-har`, `sayakghorai34/csi-har-dataset`), GPU T4,
   Internet on. Outputs match `results/`. See `code/README.md`.
2. On-device (latency, RAM): flash the firmware in `code/firmware/` to a classic ESP32 and
   capture the serial output. See `code/README.md`.
3. Manuscript PDF: open `manuscript/` on Overleaf and compile `frontier.tex`. See
   `manuscript/COMPILE.md`.

## Kaggle notebooks (public, runnable)
Every off-device experiment is a public Kaggle notebook. The `.ipynb` files are also in
`code/kaggle_notebooks/`; the links below are the executed versions with outputs.

| Experiment | Notebook |
|---|---|
| Deployability frontier (classical + tiny) | https://www.kaggle.com/code/muhammadahmad3/csi-deployability-frontier |
| Five-architecture benchmark | https://www.kaggle.com/code/muhammadahmad3/csi-architecture-benchmark |
| Deep tier on CSI-HAR | https://www.kaggle.com/code/muhammadahmad3/csi-deeptier-csihar |
| Cross-subject generalization gap | https://www.kaggle.com/code/muhammadahmad3/csi-gengap |
| Confusion matrices, precision/recall/F1 | https://www.kaggle.com/code/muhammadahmad3/csi-metrics-cpu |
| Hyperparameter sweep (LR x batch) | https://www.kaggle.com/code/muhammadahmad3/csi-hpsweep-cpu |
| Per-user personalization curve | https://www.kaggle.com/code/muhammadahmad3/csi-personalization |
| On-device replay demo assets | https://www.kaggle.com/code/muhammadahmad3/csi-replay-demo-cpu |
| Third dataset (NTU-Fi_HAR) | https://www.kaggle.com/code/muhammadahmad3/csi-ntufi-frontier |

## Datasets (public, not redistributed here)
All three are public datasets from their original authors. We list the authoritative
source and paper first, then the Kaggle mirror actually attached to our notebooks.

**UT-HAR** (Intel 5300, 7 activities) - the most widely used public CSI HAR benchmark.
- Original: https://github.com/ermongroup/Wifi_Activity_Recognition
- Paper: S. Yousefi et al., "A survey on behavior recognition using WiFi channel state
  information," *IEEE Communications Magazine*, vol. 55, no. 10, pp. 98-104, 2017.
  https://doi.org/10.1109/MCOM.2017.1700082
- Loaders: SenseFi benchmark, https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- Mirror used here: https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-ut-har

**CSI-HAR** (ESP32-collected, 7 activities, 3 users) - our subject-independent dataset.
- Original: https://github.com/parisafm/CSI-HAR-Dataset
- Paper: P. Fard Moshiri et al., "A CSI-based human activity recognition using deep
  learning," *Sensors*, vol. 21, no. 21, p. 7225, 2021.
  https://doi.org/10.3390/s21217225
- Mirror used here: https://www.kaggle.com/datasets/sayakghorai34/csi-har-dataset

**NTU-Fi_HAR** (6 whole-body activities, 3 antennas x 114 subcarriers) - third dataset,
used as an independent check of the frontier only.
- Original: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- Paper: J. Yang et al., "SenseFi: a library and benchmark on deep-learning-empowered
  WiFi human sensing," *Patterns*, Cell Press, 2023.
  https://doi.org/10.1016/j.patter.2023.100703
- Mirror used here: https://www.kaggle.com/datasets/imhoangt/ntu-fi-dataset

## Hardware
Classic ESP32-D0WD-V3 (dual-core Xtensa LX6 at 240 MHz, 520 kB internal SRAM, no PSRAM,
4 MB flash), CP210x USB bridge. Measurements taken with WiFi and Bluetooth disabled.
