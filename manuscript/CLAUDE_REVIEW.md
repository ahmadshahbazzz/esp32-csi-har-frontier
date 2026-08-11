# Peer Review — IEEE Access (simulated)

**Manuscript title:** *How Far Can a PSRAM-less ESP32 Go? A TinyML Deployment Study for WiFi CSI Human Activity Recognition*

Reviewer: senior embedded/TinyML + WiFi-CSI HAR peer reviewer (simulated).
Basis: full read of `frontier.tex` (833 lines), all tables/captions/figures cross-checked.

---

## 1. Overall verdict

This is a genuinely useful, honest deployment-characterization study that has clearly survived several revision rounds: it now includes a live 30-minute pipeline run, INA219-measured energy, an ESP32-S3 comparison, CPU-frequency and compiler-flag ablations, an arena-size ablation and arena head/tail breakdown, an unsupported-operator table, a concrete KAN failure diagnosis (`FILL` prepare), per-class confusion analysis, and appropriately downgraded (exploratory) statistics for n=3. On the central engineering claim — that a deep CNN and a convolution-augmented Transformer *do* fit and run on the bare classic ESP32, so there is no general memory wall — the evidence is strong and directly measured. Where the paper still outruns its evidence is (a) the cross-subject "generalization is the bottleneck" thesis, which on-device rests entirely on **one three-user dataset**, and whose external validation collapses to **near-chance accuracy** on MM-Fi (making "gap" the wrong framing); and (b) a set of **unreconciled numerical contradictions across tables** for what are nominally the same models on the same dataset, which a skeptical reviewer will read as a trust problem for the entire results section. Most fixes are reconciliation and reframing rather than new experiments, but they are load-bearing.

**Recommended decision: Major revision before submission.**

## 2. Assessment summary

- **Practical relevance: 9/10** — the exact question a cost-sensitive practitioner holding a bare WROOM-32 asks; the "what actually runs" map is directly actionable.
- **Potential originality: 6/10** — honestly positioned as characterization, not a new model; the novelty is the bare-classic-ESP32 target and the negative/positive deployment findings, which is real but modest.
- **Technical soundness: 6/10** — measurement methodology is careful and well-instrumented, but internal numerical inconsistencies and the near-chance external "gap" numbers pull this down until reconciled.
- **Experimental completeness: 8/10** — unusually thorough for this class of paper; the remaining hole is a many-subject *on-hardware* cross-subject study, which they acknowledge.
- **Reproducibility: 6/10** — toolchain/versions/firmware are specified well, but the artifact is claimed "released" with **no URL/DOI anywhere in the paper**, and key deep UT-HAR accuracies come from an undescribed "companion benchmark."
- **Writing and organization: 7/10** — clear and well-structured, but the abstract is a single ~430-word paragraph, and there is table-number sprawl the reader must hold in their head.
- **Current submission readiness: 5/10** — blank corresponding-author block, missing artifact link, and the numerical contradictions must be closed before this goes to a reviewer.

## 3. Strengths

1. **The core measured result is convincing and corrects a real misconception.** Flashing every convertible model and reporting measured arenas (Table `tab:deep`, `tab:ondevice`, `tab:arenabreakdown`) is exactly the right method, and the "no general memory wall for feed-forward models" conclusion is earned by data, not asserted.
2. **The deployment-failure analysis is specific, not hand-wavy.** The unsupported-operator table (`tab:unsupported`), the `CudnnRNNV3`/`TensorListReserve`/`While` identification, and especially the KAN `FILL`-node prepare failure diagnosis are the kind of concrete detail that makes a systems paper reproducible and credible. The unrolled-GRU counter-experiment (converts at 568 kB) turns "does not convert" into "does not convert *through this route*, and here is the cost of the alternative."
3. **Honest evaluation protocol.** Subject-independent leave-one-user-out as the default, explicit per-window normalization with a stated no-leakage argument, calibration windows drawn from training users only, and a-priori fixed hyperparameters — more rigor than most WiFi-CSI HAR papers show.
4. **Statistics are correctly downgraded.** Friedman χ²=4.44, p=0.22 and Wilcoxon p=0.25 are reported as *non-significant and explicitly exploratory* (Section `sec:results-perclass`), leaning on effect-size consistency instead. Right call for n=3 and pre-empts the obvious attack.
5. **The systems ablations are the paper's best material.** ESP-NN vs reference kernels (13.6× / >20×, watchdog trip), 160 vs 240 MHz (`tab:cpufreq`), −O2 vs −Os, and the byte-addressability `MALLOC_CAP_8BIT` gotcha are genuinely useful bare-device findings a PSRAM-equipped study would never surface.
6. **The prior-work table (`tab:priorcompare`) separates acquisition device from inference device**, correctly declining to conflate STAR (RV1126 NPU) and ESP-Fi (host PyTorch) with on-chip inference.

## 4. Major recommendations

**1. RECONCILE-CONTRADICTORY-ACCURACIES.**
(a) *Wrong:* Same model, same dataset, different accuracies across tables. Worst: `tab:frontier` reports "`Tiny MLP (net) ... 87.60 $\pm$ 1.77`" (UT-HAR) while `tab:accval` reports the identical model as "`Tiny MLP (net) & 92.13 & 92.33`" (float/int8, UT-HAR) — a **4.5-point gap outside ±2σ**, unexplained. Also `tab:frontier` "`Tiny CNN (32 ch) ... 96.67 $\pm$ 0.50`" vs `tab:accval` float 95.13. Deep CNN: 96.87 (`tab:deep`) / 96.00 (`tab:accval`) UT-HAR; 64.05 (`tab:deep`) / 63.81 (`tab:accval`) / 66.19 (`tab:prf`) CSI-HAR.
(b) *Why:* Both `tab:frontier` and `tab:accval` are labelled "float, UT-HAR fixed split, three seeds" — they should agree. A reviewer who spots the Tiny-MLP discrepancy will stop trusting Section 5.
(c) *Fix:* Regenerate accuracy tables from one run manifest; where numbers differ by provenance, add one sentence per table. The Tiny-MLP 87.60-vs-92.13 gap must be traced/fixed.

**2. PER-USER-VS-FRONTIER-MISMATCH.**
(a) *Wrong:* Under LOUO the three fold accuracies *are* the per-user accuracies, so `tab:peruser` and `tab:frontier` must average equal. They don't: `tab:peruser` TinyCNN-16 = (59.05, 57.86, 55.48) → mean **57.46**, but `tab:frontier` = "`Tiny CNN (16 ch) ... 59.52 $\pm$ 2.36`."
(b) *Why:* A ~2-pt mismatch between arithmetically-linked tables signals different seeds or a bookkeeping error; undercuts the "62.1% best" claim.
(c) *Fix:* State run provenance/seeds for each, or regenerate one from the other.

**3. MM-FI-NEAR-CHANCE-FRAMING.**
(a) *Wrong:* External validation reports "`MM-Fi (10) 47.0 → 6.7`" and "`MM-Fi (40) 35.8 → 4.8`," with the paper noting these are "near the 3.7\% chance level" (`tab:gengap-ext`). Calling 35.8 → 4.8 a "31-point gap" is a category error: the classifier does not degrade, it **completely fails to transfer** (chance).
(b) *Why:* Offered as evidence the three-user gap generalizes, but a near-chance result shows a light amplitude-statistics RF learns nothing subject-independent there — as much about the weak model/27-class difficulty as a universal "gap."
(c) *Fix:* Reframe as "subject-independent accuracy collapses to chance for this light classifier"; do not average a chance-level result into a "gap." Ideally add one above-chance cross-subject model on MM-Fi/CSI-Bench.

**4. MISSING-ARTIFACT-LINK.**
(a) *Wrong:* Abstract and conclusion promise "all code, models, and measurement firmware released," but there is **no repo URL, no Zenodo DOI, no data-availability statement** anywhere.
(b) *Why:* Reproducibility is contribution #5 and a scored criterion; an unciteable artifact is not released. Hard submission blocker.
(c) *Fix:* Add a Data/Code Availability paragraph with GitHub URL + minted Zenodo DOI; cite the exact commit for the reported numbers.

**5. LIVE-PIPELINE-VALIDATES-STABILITY-NOT-CORRECTNESS, AND OVERREACHES.**
(a) *Wrong:* From a 30-min run, Section `sec:discussion` concludes "there is no memory to leak and no arena to fragment ... Runtime stability under sustained sensing is therefore not a concern on this device." 30 min is not long-term; the no-leak argument holds only for the TFLM static arena, not the WiFi driver churning packet buffers (the likely 2.5 kB jitter source).
(b) *Why:* "Not a concern" is a universal claim from one short run.
(c) *Fix:* Soften to the 30-min observation; attribute jitter to the WiFi stack; list multi-hour/thermal stability as future work. Add an 8–12 h run if feasible.

**6. RECONCILE-LIVE-MEMORY-WITH-DISCUSSION.**
(a) *Wrong:* Section `sec:discussion` says WiFi-on "reduces the largest free internal block from 152\,kB to 108\,kB," but `tab:live` reports "Largest free block 88.0–92.0 kB."
(b) *Why:* Two "radio-on" budgets with no bridge look contradictory.
(c) *Fix:* One sentence — 108 kB is post-WiFi-init (driver only); the live ~90 kB also holds the app, window buffers, and arena. Make ~90 kB the realistic runtime figure.

**7. ENERGY-ACCOUNTING-CONFLATES-TOTAL-AND-MARGINAL.**
(a) *Wrong:* Neural energy = full bus power × latency (301–381 mW) against idle "254\,mW / 50.6\,mA," so marginal compute is only ~50–127 mW and the 5–25 mJ figures are baseline-dominated. Classical models reported as "$<$0.01\,mJ" is effectively the *marginal* number.
(b) *Why:* "Classical models are effectively free" vs neural "~25 mJ" mixes accounting conventions.
(c) *Fix:* Report both total-during-inference and marginal (active − idle) × latency, or state clearly the neural energy includes idle baseline.

**8. PRIOR-COMPARISON-TABLE-MIXES-CLOCKS.**
(a) *Wrong:* `tab:priorcompare` lists "This work ... 9--476\,ms." The 476 ms exists only at **160 MHz** (`tab:cpufreq`); at 240 MHz the max is 320.7 ms.
(b) *Why:* The headline table advertises a slower, off-operating-point number and mixes clocks.
(c) *Fix:* Use "9–321 ms (240 MHz)."

**9. "TINY" MLP IS THE LARGEST-PARAMETER MODEL IN THE PAPER.**
(a) *Wrong:* "Tiny MLP (net)" = 184.6 K params / 184 kB int8 (UT-HAR) — more than every "deep" model, including deep CNN (91.0 K, `tab:params`).
(b) *Why:* "Tiny" carries framing weight; the tiny tier containing the biggest-parameter model is confusing.
(c) *Fix:* Rename (e.g., "flattened MLP"), note tier-by-arena not params, or drop it.

**10. DEEP UT-HAR ACCURACIES DEPEND ON AN UNDESCRIBED "COMPANION BENCHMARK."**
(a) *Wrong:* `tab:deep` caption: "UT-HAR accuracy is from the companion single-pipeline benchmark" — not described in `sec:setup`, not cited, not in the artifact description.
(b) *Why:* Headline deep accuracies (96.87 CNN, 98.27 Transformer) come from an experiment the reader cannot inspect.
(c) *Fix:* Re-run deep UT-HAR in this pipeline (preferred), or describe the companion benchmark and include it in the release.

## 5. Minor recommendations

1. Abstract is one ~430-word paragraph; IEEE Access convention ≤250 words. Cut and split.
2. Blank submission fields: `\corresp{... (e-mail: ).}` and `\tfootnote{}` (lines 43, 37).
3. "Run-to-run noise" is imprecise: for a fixed model + fixed test set, float and int8 are each deterministic, so int8 > float (e.g., Transformer CSI int8 58.81 > float 58.10) is a real quantization effect, not noise (`sec:results-quant`).
4. `tab:live` "37.6\,ms (35.9--37.7)" places the mean at the max — verify not a typo.
5. `fig:pipeline` and `fig:live` overlap; merge or differentiate.
6. `fig:cm` — three matrices at 0.32\textwidth risk small labels in print.
7. `\cite{hernandez2025tools}` renders "Armenta-Garcia et al." but the key is named after a different author — confusing for `.bib` editors.
8. "0.00\% dropped" over 180,815 packets: clarify this is buffer-level (CSI callbacks delivered), not 802.11 link-level.
9. State the 7-class chance level (14.3%) once near the CSI-HAR results.
10. Reconcile `tab:priorcompare` "Live = P / Preproc = P" with the fuller Section `sec:results-live`.
11. Double-check future-dated citations (espfihar2026, csibench2025, neurosym2025, greensensing2026) before submission.

## 6. Consolidated action checklist

- Reconcile Tiny-MLP UT-HAR **87.60 vs 92.13** (`tab:frontier` vs `tab:accval`) **← very important**
- Reconcile `tab:peruser` (mean 57.46) vs `tab:frontier` (59.52) for Tiny CNN 16 ch **← very important**
- Add per-table provenance notes for deep-CNN 96.87/96.00 and 64.05/63.81/66.19 discrepancies
- Reframe MM-Fi 6.7%/4.8% as **chance-level failure, not a "gap"**; add an above-chance cross-subject classifier if possible **← very important**
- Add Data/Code Availability statement with GitHub URL + minted **Zenodo DOI** **← very important**
- Fill blank corresponding-author and email fields **← very important**
- Change `tab:priorcompare` latency to **9–321 ms (240 MHz)**; remove the 160 MHz 476 ms value
- Soften "runtime stability is not a concern" to a 30-min observation; attribute heap jitter to the WiFi stack; add a multi-hour run if feasible
- Bridge the **108 kB (WiFi-init) vs 88–92 kB (live)** memory figures
- Report **marginal (active − idle) energy** alongside total
- Rename/justify the "Tiny MLP" tier (largest-parameter model in the paper)
- Describe or re-run the "companion single-pipeline benchmark" behind deep UT-HAR accuracies
- Shorten the abstract to ≤250 words, ≤2 paragraphs
- Correct "run-to-run noise" wording for the deterministic float-vs-int8 comparison
- State the 7-class chance level (14.3%) once near the CSI-HAR results

**Bottom line:** the science is largely sound and the systems work is a real contribution; the blocking issues are internal-consistency reconciliation, one framing correction (MM-Fi), and submission hygiene (artifact link, author fields) — not missing experiments — so this *major revision before submission* can likely be turned around quickly.
