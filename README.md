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
datasets (UT-HAR and CSI-HAR). The two-part conclusion: on this hardware deployment is
not the bottleneck, but cross-subject generalization is.

Headline findings, all backed by the evidence in `results/`:
1. The board runs more than expected. Flashing every convertible model on-device shows a
   full deep CNN (10 to 12 kB arena) and Transformer (24 to 54 kB) both run on the bare
   classic ESP32, so there is NO general memory wall. Only a convertibility wall is real:
   recurrent and state-space models do not convert to TensorFlow Lite for Microcontrollers,
   and the Chebyshev-KAN converts but fails to allocate.
2. Classical learners and tiny CNNs fit and run cheaply. Measured on the board: tiny
   networks use 8 to 13 kB of arena and 14 to 117 ms per inference; the emlearn Decision
   Tree and Random Forest run in about 1 and 45 microseconds with under 1 kB of RAM. They
   are the efficiency choice, not the only option.
3. The binding constraint is cross-subject generalization. Under leave-one-user-out on
   CSI-HAR the same six models drop a mean of 29 points (best 96 to 63 percent) versus a
   random split that leaks users across train and test. On the saturated UT-HAR benchmark
   the small models stay competitive (tiny CNN 96.7, statistics MLP 95.3 percent).


## Folder layout
```
submission/
  manuscript/   LaTeX source + classes + figures (compiles as-is on Overleaf)
  code/         Kaggle notebook + generator, and ESP32 firmware sources + scripts
  results/      All measured evidence: tables, JSON, figures, on-device logs
  models/       Trained artifacts: .joblib (sklearn), int8 .tflite, emlearn C headers
```

## How to reproduce
1. Off-device (accuracy, int8 sizes, convertibility, figures): run
   `code/kaggle_notebooks/csi_frontier_kaggle.ipynb` on Kaggle with the two datasets
   attached (`hylanj/wifi-csi-dataset-ut-har`, `sayakghorai34/csi-har-dataset`), GPU T4,
   Internet on. Outputs match `results/`. See `code/README.md`.
2. On-device (latency, RAM): flash the firmware in `code/firmware/` to a classic ESP32 and
   capture the serial output. See `code/README.md`.
3. Manuscript PDF: open `manuscript/` on Overleaf and compile `frontier.tex`. See
   `manuscript/COMPILE.md`.

## Datasets (public, not redistributed here)
- UT-HAR: Kaggle `hylanj/wifi-csi-dataset-ut-har` (via the SenseFi loaders).
- CSI-HAR: Kaggle `sayakghorai34/csi-har-dataset`.

## Hardware
Classic ESP32-D0WD-V3 (dual-core Xtensa LX6 at 240 MHz, 520 kB internal SRAM, no PSRAM,
4 MB flash), CP210x USB bridge. Measurements taken with WiFi and Bluetooth disabled.
