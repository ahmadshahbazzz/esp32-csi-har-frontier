CSI-Bench external cross-subject generalization result (Kim Major #5).
Dataset: CSI-Bench (guozhenjennzhu/csi-bench, CC BY-NC-ND), MotionSourceRecognition task,
4 classes (Fan/Human/IRobot/Pet), 35 users, 10 environments. RandomForest on per-subcarrier
amplitude stats (mean/std/min/max/median of CSI_amps [232,500]). Kaggle kernel muhammadahmad3/csi-bench-gengap.
KEY: random (subject-overlapping) 84.1% acc vs user-independent (held-out users) 51.7% -> 32.4 pt gap
(macroF1 64.0 -> 23.4). Confirms the ~29pt gen-gap seen on 3-user CSI-HAR now holds on a 43-user dataset.
Difficulty tiers (train_id->test): easy 76.9 / medium 85.2 / hard 79.5 acc (noisy due to class imbalance;
lead with the random-vs-user-independent contrast).

MM-Fi external cross-subject result (Kim Major #5), added 2026-08-08.
Dataset: MM-Fi (official, CC BY-NC), WiFi-CSI modality only (CSIamp [3,114,10]), fetched via
Kaggle kernel muhammadahmad3/mmfi-csi-prep (gdown skip_download listing -> wifi-csi files only).
Scope actually fetched = environment E01 = 10 subjects (S01-S10) x 27 daily ACTIONS (A01-A27),
80190 wifi-csi frames. Gen-gap kernel muhammadahmad3/mmfi-gengap: RandomForest on per-subcarrier
amplitude stats (mean/std/min/max over antenna x packet), 5400 frames (cap 20/subject/action).
KEY: random (subject-overlapping) 47.0% acc vs subject-independent (held-out subjects) 6.7%
(near 27-class chance 3.7%) -> 40.3 pt gap (macroF1 45.2 -> 6.6). Real activity-HAR confirmation.
NOTE: absolute acc is modest (single-frame RF on 27-class HAR); the GAP is the finding. Could
extend to more MM-Fi environments (E02-E04) for 40 subjects if desired; 10 subjects already >3.

--- MM-Fi 40-subject (full, 2026-08-09, via Colab server-side Drive copy) ---
All 4 environments E01-E04 (40 subjects, 10/env), 27 actions, wifi-csi CSIamp[3,114,10].
320760 .mat extracted; RF on per-subcarrier amp-stats (mean/std/min/max = 456-dim), CAP=20/bucket -> 21600 samples.
RANDOM(overlap):    acc=35.8  macroF1=34.3
SUBJECT-INDEP(GSS): acc= 4.8  macroF1= 4.6   (27-class chance ~3.7)
>>> 40-subject MM-Fi cross-subject gen-gap = 31.0 pts
Family: CSI-HAR(3)=30.0, MM-Fi(10,1env)=40.3, MM-Fi(40,4env)=31.0, CSI-Bench(35)=32.4.
