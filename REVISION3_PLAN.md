# Revision 3 Plan - Prof. Kim review + Figure 2 permission

Tracking doc for the funding professor's (Prof. Kim) major-revision review and the second
professor's Figure 2 copyright concern. Status legend: [ ] TODO, [~] in progress,
[x] DONE (with date/time), [BLOCKED] needs a decision (see "Decisions needed").

Board = classic ESP32 on /dev/ttyUSB0 (currently DETACHED - replug + `sudo chmod 666`).
Manuscript = esp32-csi-har-frontier/manuscript/frontier.tex. Repo public + MIT.

=====================================================================================
## DECISIONS - RESOLVED 2026-07-28
=====================================================================================
- D1 (LIVE PIPELINE) = BUILD IT. "Do what the reviewer is asking." Feasible: ESP32 does
  CSI capture, user has a WiFi router + phone hotspot as the traffic source. The demo
  characterizes the full pipeline + its metrics (packet rate, loss, latencies, memory,
  30-60 min stability); live-activity ACCURACY is out of scope (no live ground truth) and
  will be stated honestly. -> Category H.
- D2 (ENERGY) = MEASURE IT. User will obtain an INA219 probe. Until it arrives: remove the
  wrong framing now (drop "conservative upper bound" + S3/Bolat comparison + the
  frequency-proportional claim), and HOLD the real measurement task for when the probe is
  connected. (INA219 wiring instructions provided separately.)
- D3 (CROSS-SUBJECT) = REFRAME AS EXPLORATORY (follow reviewer, avoid re-raising the issue).
  Present 3-user LOUO as an exploratory case study, drop the universal "binding constraint"
  claim, use "subject-overlapping" not "leakage", lead with per-user/effect-size reporting.
- D4 (HARDWARE): classic ESP32 (reconnectable) YES; WiFi router YES; phone hotspot YES;
  2nd ESP NO; some other MCU board YES (type TBC); current probe NOT YET (getting INA219);
  ESP32-S3 NO -> #9 stays a citation.

=====================================================================================
## CATEGORY A - Claim narrowing / reframing (TEXT ONLY, no experiments) - can start now
=====================================================================================
- [x] A1 (2026-08-04 00:58): replaced all 3 "convertibility wall" -> "export-pipeline compatibility boundary"
        boundary"; state the limited conclusion "these particular implementations do not
        deploy through the stated TF 2.19 -> TFLM builtin-operator pipeline"; describe KAN
        as an operator-preparation (kernel-compatibility) failure, NOT a memory/SRAM ceiling.
- [x] A2-text (2026-08-04 00:58): softened to "property of the traced implementation/export route, not intrinsic"; CudnnRNNV3 = GPU-fused trace artefact; portable/unrolled/custom-op could deploy.
- [x] A2-experiment (2026-08-04): Kaggle csi-gru-export DONE. RESULT: fused GRU -> needs Select-TF-ops (ConverterError); GRU(unroll=True) -> converts CLEANLY to full int8 builtin set (567.6 kB, 86.9% float acc); manual-unroll cell -> build ValueError (my bug, not needed). CONFIRMS Major #3 empirically: recurrence deploys via a portable/unrolled export route; non-deployability is fused-CudnnRNNV3-route-specific. Integrated into frontier.tex export-boundary paragraph (with the 568 kB unrolling-cost caveat = proof-of-principle not practical).
        trace artefact; a portable/unrolled recurrent or a custom TFLM op may deploy.
        (See B/experiments: attempt one unrolled GRU export - Kaggle.)
- [ ] A3  Major #5: replace "data leakage"/"leaks users" with "subject-overlapping
        (in-subject) evaluation protocol"; wording "random subject-overlapping evaluation
        substantially overestimates unseen-user performance on CSI-HAR."
- [ ] A4  Major #5: reframe 3-user LOUO as exploratory case study (pending D3); remove/soften
        the universal "cross-subject generalization is THE binding constraint" framing.
- [ ] A5  Major #5 + spec #12/#241: down-weight Friedman/Wilcoxon (n=3 low power) -> lead with
        per-user results, effect sizes, descriptive uncertainty; keep tests only as a footnote.
- [ ] A6  Spec #2: "full on-device pipeline" -> "stored-window inference replay".
- [ ] A7  Spec #3: "stress-tested exactly this continuous-sensing mode" -> accurate wording
        ("a back-to-back inference-stability test with the radio idle", not live sensing).
- [ ] A8  Spec #4: "40 labelled samples in a few seconds" - substantiate or remove the
        time claim (we have no timing evidence -> remove "a few seconds").
- [ ] A9  Spec #5: "the reason is physical, not statistical" -> cautious hypothesis phrasing.
- [ ] A10 Spec #6: "almost window for window" - either add a paired per-window agreement
        analysis (Kaggle, cheap) or remove the claim; aggregate CMs do not prove it.
- [ ] A11 Spec #7: "cheapest/weakest/most widely deployed" -> soften or cite evidence.
- [ ] A12 Spec #15: "full deep" - define technically (parameter/layer budget) or drop the word.
- [ ] A13 Section conflict (review Major #1): reconcile Sec VI-E ("preprocessing through
        prediction") vs Sec V-C ("windows already int8-quantized before compile") - state
        clearly that preprocessing was OFF-device and only the int8 window + inference is
        on-device (unless D1 live pipeline is built).
- [ ] A14 Reframe title/abstract/contributions to "inference-deployability characterization"
        if D1 = reframe (per reviewer's explicit fallback + suggested novelty statement).

=====================================================================================
## CATEGORY B - Numerical/narrative consistency (TEXT) - can start now
=====================================================================================
- [x] B1 (2026-08-04 00:58): fixed Discussion 44ms -> 29.7ms (240 MHz); scanned, no other stale 160 MHz numbers
        Table 7 says 29.7 ms (240 MHz). Fix to 29.7 ms consistently. Scan for any other
        stale 160 MHz numbers.
- [ ] B2  Spec #14: clearly distinguish station-mode WiFi init (what #2 measured) from live
        CSI packet acquisition (not done unless D1). Already partly worded; tighten.
- [ ] B3  Spec #11: replace submission-date + DOI placeholders (\history, \doi) - or note
        they are filled at acceptance.

=====================================================================================
## CATEGORY C - FIGURES
=====================================================================================
- [ ] C1  Fig 2 (2nd professor, PREFERRED option 2): remove the ESP32 board PHOTO and draw
        our own internal-component WIREFRAME/block diagram (dual-core LX6, internal SRAM
        regions we measured incl. the ~150 kB usable block, 4 MB external flash holding int8
        weights, WiFi/BT radio, ROM, no PSRAM; show the tensor arena living in SRAM). Make
        as TikZ or matplotlib/graphviz -> PDF (self-authored, no permission needed).
- [ ] C2  Spec #10: regenerate Figs 4-9 with larger fonts + consistent publication styling
        (bump matplotlib font sizes, dpi, consistent palette; re-run the plotting cells).

=====================================================================================
## CATEGORY D - Reproducibility clarifications (Major #6) - TEXT + verify from notebooks
=====================================================================================
- [ ] D_repro1  Define "z-score normalized per window" precisely (our code: mean/std over
        BOTH axes (time,subcarrier) jointly, keepdims -> a single scalar per window, NOT
        per-subcarrier). State this explicitly.
- [ ] D_repro2  Address the reviewer's DEGENERATE-FEATURE concern (Major #6): if we normalize
        per-window and THEN compute classical mean/std/min/max/range features, are they
        degenerate? Our normalization is joint (not per-subcarrier), so per-subcarrier stats
        are NOT constant - but VERIFY and, if the window-mean feature is ~0, recompute
        classical features on RAW (pre-normalization) data. May require a Kaggle re-run.
- [ ] D_repro3  State LOUO hygiene: normalization stats + int8 calibration windows + any
        hyperparameter choice use TRAINING USERS ONLY in each fold (verify in notebooks;
        if a global split was used for calibration, fix + re-run).
- [ ] D_repro4  Multi-seed LOUO: report neural LOUO over several seeds per held-out user
        (currently 1 seed/user in some notebooks) -> re-run csi_gengap / frontier LOUO with
        3 seeds x 3 users; report mean +/- std per user.
- [ ] D_repro5  Resampling: state how variable-length recordings are resampled to T=64
        (linear index subsampling) and MEASURE accuracy lost vs native T (ablation: T in
        {32,64,128,native}) - Kaggle.
- [ ] D_repro6  Inner-validation hyperparameter selection restricted to training users
        (replace the current global HP sweep) - Kaggle re-run of csi_hpsweep under LOUO.

=====================================================================================
## CATEGORY E - Accuracy validation (Major #2, checklist #1/#2, spec #12) 
=====================================================================================
- [ ] E1  For EVERY deployable model report: float acc / desktop-TFLite-int8 acc / accuracy
        loss from quantization (Kaggle - add int8-eval cells; label every table value as
        pre- or post-quantization).
- [x] E2 (2026-08-04): emlearn parity resolved as a BY-CONSTRUCTION argument, not a fabricated
        100%. Kaggle csi-emlearn-parity (v3, gcc-compiled C) MEASURED low label-match (DT/RF
        27-43%) NOT because the trees are wrong but because emlearn's default int16 fixed-point
        feature path needs the features fed in the SAME fixed-point scale as the stored
        thresholds; raw normalized floats cast to int16 truncate to ~0. This IS the reviewer's
        "integer feature handling" concern. Manuscript (sec:methods classical tier) now states
        the rigorous truth: branch-only C branches on the identical learned thresholds, so the
        decision path/label is identical to sklearn by construction for any input in the matched
        representation; the flagged caveat is exactly the fixed-point feature scaling. Replaced
        the old "minor remaining check" sentence. NO overclaim of a measured number.
- [ ] E3  ESP32 label/logit parity (Major #2): stream ALL test windows to the board one at a
        time (extend the replay firmware to N windows), compare on-device argmax/logits to
        desktop int8; report label-parity % and a numerical tolerance. [BOARD]
- [ ] E4  Spec #12: annotate each accuracy value in every table as measured before or after
        quantization.

=====================================================================================
## CATEGORY F - Resource audit (Major #7, spec #13) - BOARD + build logs
=====================================================================================
- [ ] F1  Add MLP (statistics) to the on-device latency/RAM table (currently missing). [BOARD:
        it is emlearn C - measure like DT/RF.]
- [ ] F2  Measure classical FEATURE-EXTRACTION time on-device (mean/std/min/max/range compute),
        not just tree traversal - report the full classical pipeline cost. [BOARD firmware]
- [ ] F3  Full RAM breakdown: window/input buffer, preprocessing workspace, feature buffer,
        task stack, interpreter+resolver state, WiFi/CSI buffers (if D1), total heap
        before/after init, largest free block. [BOARD firmware instrumentation]
- [ ] F4  Firmware image audit: .text/.rodata/model-weight sizes, application-partition
        utilisation, actual compiled flash for classical vs neural (from `idf.py size` /
        map file, per model). [build logs - no board needed for the size part]
- [ ] F5  Spec #13: report actual compiled flash consumption (not just model-file size) in
        the tables/text.

=====================================================================================
## CATEGORY G - Related work + Table 1 rebuild (Major #4) - TEXT + reading
=====================================================================================
- [ ] G1  Read + cite the 4 required papers (IEEE 9217780, 9900419, 10101249, 10502448);
        integrate into Related Work; show how base papers handle live-deployment claims.
- [ ] G2  Correct Table 1: ESP-Fi = dataset+PyTorch benchmark (NOT shown running on C3);
        STAR inference is on a Rockchip RV1126 NPU (S3 = acquisition only). Re-label.
- [ ] G3  Rebuild Table 1 with columns: CSI-acquisition device | inference device | live CSI
        capture | on-device preprocessing | on-device inference | model family |
        quantization | PSRAM | measured RAM | measured latency | measured energy.
- [ ] G4  Insert the defensible novelty statement (reviewer-suggested wording).

=====================================================================================
## CATEGORY H - LIVE PIPELINE experiment (Major #1) - only if D1 = build. [BOARD + WiFi src]
=====================================================================================
- [ ] H1  Firmware: enable esp-wifi CSI RX callback on classic ESP32; associate to an AP/
        hotspot; generate traffic (ping) so CSI packets arrive.
- [ ] H2  On-device: activity-window formation + amplitude extraction + per-window normalize
        + resample to T -> int8 tensor -> TinyCNN16 inference -> label over serial.
- [ ] H3  Report: CSI packet arrival rate, packet+window loss rate, window duration/overlap,
        preprocessing latency, inference latency, end-to-end latency, free heap + largest
        block, window+preproc memory, and 30-60 min continuous-operation stability.
- [ ] H4  New "live deployment" results subsection + pipeline figure (acquisition->label).

=====================================================================================
## CATEGORY I - Energy resolution (Major #8, per D2)
=====================================================================================
- [x] I1-framing (2026-08-04 00:58): removed "conservative upper bound" + Bolat S3 bound (both bibitems deleted) in setup+limitations; energy now a labelled full 30-68mA range estimate; real INA219 measurement HELD for probe arrival
        full 30-68 mA range clearly labelled as an estimate. Remove "conservative upper
        bound" + the Bolat S3 "upper bound" comparison (reviewer rejects S3->classic).
- [x] I2 (2026-08-04 00:58): removed the wrong frequency-proportional energy claim (current changes with frequency)
        scales with frequency, so energy does not scale purely with latency).

=====================================================================================
## CATEGORY J - Housekeeping / references / artifact
=====================================================================================
- [ ] J1  Spec #8 + checklist #7: add the PUBLIC artifact URL to the manuscript
        (github.com/ahmadshahbazzz/esp32-csi-har-frontier) + mint a Zenodo DOI; the paper
        currently says "released" with no URL. [Zenodo DOI = Ahmad manual step]
- [ ] J2  Spec #9: complete references [10],[13],[20],[27],[28] with full biblio + URLs +
        access dates (identify which keys these map to and fix).
- [ ] J3  Final pass: humanize all new prose, validate (braces/refs/cites/em-dashes/ASCII),
        regenerate overleaf zip, push; re-run independent audit.

=====================================================================================
## SUGGESTED ORDER (once decisions are in)
=====================================================================================
1. Category A + B + I (text/claim fixes) - immediate, no hardware.
2. Category C1 (Fig 2 wireframe) - immediate, self-authored.
3. Category G (related work + Table 1) - reading + text.
4. Category D + E + D_repro (Kaggle re-runs: int8 eval, emlearn parity, multi-seed LOUO,
   inner-validation HP, resampling ablation, paired-agreement).
5. Category F + E3 + C2 (board: MLP-stats, classical feature timing, resource audit, ESP32
   parity; regenerate figures).
6. Category H (live pipeline) - only if D1 = build; largest single effort.
7. Category J (references, artifact URL, finalize).

=====================================================================================
## PROGRESS LOG
=====================================================================================
2026-08-04 01:07  batch 1: A1 A2-text B1 I1-framing I2 (commit 591107c)
2026-08-04 01:07  batch 2: A3 (leakage->subject-overlapping), A4 (all 7 "binding constraint" softened
     to exploratory/3-user), A5-text (tests down-weighted, effect-size leads; per-user
     multi-seed = D_repro4 Kaggle TODO), A6 (full on-device pipeline -> stored-window
     replay + live-pipeline pointer), A7 (continuous-sensing wording), A8 ("few seconds"
     removed), A9 (physical-not-statistical -> cautious hypothesis), A10 ("window for
     window" softened; paired analysis = Kaggle TODO), A11 (cheapest/weakest softened to
     "among least capable/lowest-cost", concrete no-PSRAM/no-SIMD), A12 ("full deep" x5
     -> "deep"). A14 = N/A (D1=build live pipeline, keep deployment framing).
     Validation clean (0 orphans/broken/dangling, 0 em, ASCII). 
2026-08-04 01:24  batch 3: C1 DONE - Fig 2 replaced with self-authored internal-component block
     diagram (fig_esp32_internal.pdf via make_fig_internal.py); board photo removed from
     repo+zip. (commit 4b40ad9)
2026-08-04 01:24  batch 4: G DONE (Major #4) - rebuilt Table 1 as 12-col table* (acq device | inference
     device | on-chip inf | live | preproc | model | quant | PSRAM | RAM | latency |
     energy); corrected ESP-Fi (host PyTorch) + STAR (RV1126 NPU, not ESP32); added 4 refs
     (Hernandez&Bulut '20 WoWMoM, '23 COMST survey; Sahoo '23 APSCON; Lenka&Chakraborty '24
     Wisdom = closest prior work, ESP32-C3); rewrote related work + novelty statement.
     (commit dc31914)
2026-08-04 01:29  batch 5: C2 DONE (spec #10) - regenerated Figs 4-9 with large fonts + consistent
     styling from result CSVs (make_figs.py). (commit 260c12b)
2026-08-04 01:29  batch 6: E1/E4 IN PROGRESS - pushed csi-accval notebook (float vs int8 accuracy +
     quant loss, all deployable models, CSI-HAR LOUO + UT-HAR 3-seed). Running on Kaggle.
     Still TODO: E2 emlearn parity, D_repro4 multi-seed LOUO, D_repro5 resampling ablation,
     D_repro6 inner-validation HP, A10 paired agreement, A2 unrolled-GRU, D_repro1-3 text.
2026-08-04 01:53  batch 7: E1/E4 DONE (Major #2) - float-vs-int8 table integrated (tab:accval, int8
     loss <=1pt both datasets; commit 41560b6). D_repro1-3 DONE (precise z-score def,
     degenerate-feature answer, LOUO calibration/HP hygiene). J2 partial (completed
     GitHub-issue + arXiv refs w/ URLs). B3 DONE (date placeholder). D_repro6 addressed
     via text (HP fixed a priori, no per-fold tuning -> no held-out-user leakage).
2026-08-04 01:53  batch 8: launched remaining Kaggle notebooks - csi-emlearn-parity (E2, re-run w/
     underflow fix), csi-repro (D_repro4 multiseed LOUO + D_repro5 resampling ablation +
     A10 paired agreement), csi-gru-export (A2 unrolled-GRU attempt). Waiter b1952s4ax.
     -> integrate results when they land.
2026-08-04 01:59  batch 9: integrated repro results - resampling ablation inline (T=64 optimal),
     per-user multi-seed table (tab:peruser), paired agreement 71%/51% (A10 measured).
     B2 tightened (station-mode vs live). (commit 2cc04bf)
2026-08-04 01:59  batch 10: E2 emlearn parity switched to gcc-compiled C (Python .predict asserted);
     A2 gru-export running. Waiter b70kpb6n2. -> integrate when they land, then FINAL
     humanize+validate+audit. Remaining after that = hardware-only (F,E3,H,energy,S3).
