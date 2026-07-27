# Code: reproducing the results

## 1. Off-device (Kaggle): accuracy, int8 sizes, convertibility, figures
`kaggle_notebooks/csi-deployability-frontier.ipynb` is the notebook that produces every off-device
number in the paper and the figures in `results/`. It is included **with the executed cell
outputs** from the actual Kaggle run (stdout training logs, the results table, and the three
figures embedded), so it can be read end-to-end without re-running. `gen_frontier_notebook.py`
regenerates a clean (output-free) copy of the same notebook from source.

- Attach datasets: `hylanj/wifi-csi-dataset-ut-har` and `sayakghorai34/csi-har-dataset`.
- Accelerator: GPU T4. Internet: on. Then Run All.
- Protocol: UT-HAR uses its fixed split over three seeds; CSI-HAR uses subject-independent
  leave-one-user-out (three folds). Tiny networks are quantized to int8 with TensorFlow Lite.
- Outputs: `frontier_table.csv`, `frontier_results.json`, `frontier_tidy.csv`,
  `fig_*.png`, the `*.joblib` classical models, and `tflite/*_int8.tflite`.

`gen_frontier_notebook.py` is the generator that writes the notebook
(`python gen_frontier_notebook.py`), so the notebook itself is reproducible from source.

## 2. On-device (classic ESP32): latency and RAM
Requires ESP-IDF v5.1 and a classic ESP32 on a serial port. The classic-ESP32 flash
offsets and a dynamic arena from the largest free internal block are already configured.

### Tiny neural networks (TensorFlow Lite Micro)
The firmware loads one int8 model as a C array and times 1000 invocations.
1. Convert a model to a C array:
   `python tflite_to_c.py <model_int8.tflite> csi_bench/main/model_tiny.cc model_data_tflite`
   (`csi_bench_main.cc` and `csi_bench_main_CMakeLists.txt` here are the firmware `main/`
   files; `model_tiny.cc` is generated, not checked in.)
2. Build and flash: `measure_tiny.sh <model_int8.tflite> <label>` auto-detects the serial
   port, flashes at 115200 baud, and captures `latency_ms` and `arena_used_bytes`.
3. `measure_all_tiny.sh` loops over every tiny model.

### Classical models (emlearn C)
`emlearn_bench/` is a standalone ESP-IDF project that times the Decision Tree and Random
Forest.
1. Export the trees to C: `python export_emlearn.py <dir-with-joblibs> emlearn_out`
   (produces the `*.h` headers, also provided in `models/emlearn_c/`).
2. Put the chosen header in `emlearn_bench/main/`, then `idf.py set-target esp32`,
   `idf.py build`, `idf.py -p <port> -b 115200 flash`, and capture the serial output.

## Notes
- Flash at 115200 baud; higher rates can fail on some CP210x cables.
- The board may re-enumerate (ttyUSB0 to ttyUSB1) under heavy reflashing; the measure
  scripts auto-detect the current port each run.
- Energy is not reported: the board has no current-sense instrumentation.
