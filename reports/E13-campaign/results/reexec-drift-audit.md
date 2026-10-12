# Re-execution drift audit - committed readings against the readings of 2026-10-09

Audit date 2026-10-10. The audit compares the metrics files of commit `bd0002e` with the working-tree files that the re-execution of 2026-10-09 wrote. It covers the batches E04, E05, E07, E08, E09, E10 and E12-H64 to E12-H68.

- 25 hypotheses of these batches have at least one number that moved by more than 1%
- 18 of the 25 have a drifted number that is not a timing or a load number
- Of these 18 recorded verdicts, 15 hold on the re-executed values, 2 hold with a smaller margin (E05-H24, E08-H47) and 1 does not hold on one clause of its bar (E08-H44)
- The E08-H44 clause fails on the values of commit `bd0002e` as well, so the re-execution did not change any of the 18 verdicts
- The E04 CPU diagnostic did not reproduce the committed E04-H15 reading: it read best `k` 1 at 6.03x, and the committed file reads best `k` 2 at 3.39x
- 33 markdown sentences in 8 notebooks contradicted the stored output and were corrected through `nbformat`

## Sources and conventions

- **HEAD** - the file content of commit `bd0002e`, read with `git show HEAD:<path>`
- **After** - the working-tree file, written by the re-execution of 2026-10-09
- **Drift list** - the output of `.venv/bin/python reports/E13-campaign/drift.py --tol 0.01 --show 1000`, run on 2026-10-10
- **Ledger A** - `docs/experiments/wmd-docdistance-experiments.md`, which holds the records of E04, E05 and E12
- **Ledger B** - `docs/experiments/wmd-structure-distance-experiments.md`, which holds the records of E07 to E10; these records have a Verdict line and no Result line
- **Cell N** - the stored output of code cell N of the working-tree notebook, unless the text says HEAD
- Numbers from a metrics file are rounded to four decimals; numbers from a cell output are copied as printed
- Key offset: the keys `E04-H1` to `E04-H5` of `reports/E04-source-conditioned-performance-metrics.json` are the ledger ids E04-H15 to E04-H19
- No other offset: the E05 keys `H20` to `H24`, the E07 to E10 keys and the E12 files carry the ledger ids
- In the E04 file the key `killed` stores the value of `kill_pass`: True means that the kill-gate passed

## 1. Drift per hypothesis

### 1.1 Stored environment fields of the two runs

| Batch | Field and where it is stored | HEAD | After |
|---|---|---|---|
| E04 | `device` of the metrics file | cpu | cuda |
| E04 | cell 8: mmBERT encoder, reranker, NLI | openvino INT8, `stellars/bge-reranker-v2-m3-openvino-int8`, `stellars/mdeberta-v3-base-mnli-xnli-openvino-int8` | torch fp16, `BAAI/bge-reranker-v2-m3`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| E04 | cell 4: package version | docdistance 1.1.2 | docdistance 1.1.5 |
| E05 | cell 4: GPU of the fidelity grids | NVIDIA RTX 5000 Ada Generation | NVIDIA RTX PRO 4000 Blackwell |
| E07 | cell 2: GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | NVIDIA RTX PRO 4000 Blackwell |
| E08, E09, E10 | `config.device` of the metrics file | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | NVIDIA RTX PRO 4000 Blackwell |
| E07 to E10 | cell 2 (torch) and cell 4 (POT) | torch 2.11.0+cu130, POT 0.9.6.post1 | torch 2.11.0+cu130, POT 0.9.6.post1 |
| E12-H64 to H68 | imports cell: POT | POT 0.9.7.post1 | POT 0.9.6.post1 |
| E12-H65 to H68 | GPU or imports cell: torch | torch 2.12.0+cu130 | torch 2.11.0+cu130 |
| E12-H64, H66, H67, H68 | package version | docdistance 1.1.5 | docdistance 1.1.5 |
| E12-H67 | `environment.load_average_before_timed_runs` | min 5.97, median 7.26, max 7.97, breaches 0 | min 31.61, median 36.46, max 44.24, breaches 56 |
| E12-H68 | `load_average` | precondition 4.25, 3 of 11 readings above 8.0, `precondition_ok` True | precondition 19.07, 11 of 11 readings above 8.0, `precondition_ok` False |

- The E05 and E07 to E10 files store no package version and no model revision
- `logs/reexec-model-fetch.log` lists the torch model snapshots that were fetched on 2026-10-09, so the model revisions of the HEAD runs are not recorded
- The E04 notebook sets `USE_GPU = True` in both runs; at HEAD cell 4 printed `device cpu`, and cell 8 then selected the OpenVINO INT8 models, not torch on the CPU
- `reports/E13-campaign/README.md` states that the load average of this host was not a measure of CPU contention on 2026-10-09 and 2026-10-10
- `reports/E13-campaign/README.md` states that the re-executed E12-H67 notebook carries `LOAD_MAX_TRIES` 2 where the HEAD notebook carried 20

### 1.2 Hypotheses with a drifted number that is not a timing or a load number

Column "Ledger = HEAD" says whether the ledger line equals the HEAD artefact. A "no" in that column is a difference that existed before the re-execution.

| Id | Ledger line states | HEAD | After | Bar or kill-gate | Ledger = HEAD | Verdict holds on the values after | Cause |
|---|---|---|---|---|---|---|---|
| E04-H15 | Confirmed. Kill-gate cosine 0.736. `best_k` 2, DR 0.0370 → 0.1253 = 3.39x, 0 violations, contrast 4.34 → 1.91 | `aniso` 0.7178, `best_k` 2, `dr` 0.1253, `dr_ratio` 3.3908, `viol` 0, `contrast` 1.9135, baseline `d_sel_DR` 0.0370, `d_sel_contrast` 4.3387 | `aniso` 0.7177, `best_k` 1, `dr` 0.2082, `dr_ratio` 6.4650, `viol` 0, `contrast` 2.2912, baseline `d_sel_DR` 0.0322, `d_sel_contrast` 3.9866 | DR ≥ 1.5x at 0 violations; kill-gate cosine ≥ 0.6 | yes, except the cosine (0.736 against 0.7178) | Yes: 3.39x → 6.47x against 1.5x. The ledger values `best_k` 2 and 3.39x do not equal the file after the re-execution | Not the device alone: the CPU diagnostic reads `best_k` 1 at 6.03x (section 2) |
| E04-H16 | Refuted. DR 0.0193 at tau 0.1 → 0.0487 at 0.01, contrast 2.31x → 1.47x, 1 violation at tau 0.02, 2 at tau 0.01 | `con_range` 1.6631, `passed` False; HEAD cell 18: DR 0.0370 → 0.1022, contrast 4.34x → 2.91x, 0 violations in all 5 rows | `con_range` 1.5566, `passed` False; cell 18: DR 0.0322 → 0.0940, contrast 3.99x → 2.89x, 0 violations in all 5 rows | Some tau < 0.1 with contrast ≥ 1.2x and DR up and 0 violations; kill-gate: the contrast varies across tau | no | Yes: `best_tau` is None in both files | Not isolated: the CPU diagnostic reads `con_range` 1.7598, which equals neither run |
| E04-H17 | Refuted (killed at gate). recall@3 0.547, Spearman 0.70, cut 2.64x, 1 new gold intrusion | `recall3` 0.5680, `spearman` 0.6121, `speed_cut` 1.9392, `intrusions` 3 against baseline 6 | `recall3` 0.5544, `spearman` 0.6121, `speed_cut` 1.2612, `intrusions` 3 | Kill-gate recall@3 ≥ 0.90; bar Spearman ≥ 0.95 and cut ≥ 3x and 0 new intrusions | no | Yes: 0.5544 is below 0.90 | Device: the CPU diagnostic reads `recall3` 0.5680, equal to HEAD. `speed_cut`, `t_big`, `t_base` and `t_mini` are timings |
| E04-H19 | Refuted (killed at gate). No speed lever, DR 3.39x, 0 violations, 0 gold intrusions, blend Set2>Set1, latency cut 1.0x | `kill_pass` True, `speed_pick` H4 MiniLM cascade, `dr_ratio` 3.3908, `latency_cut` 3.3089, `intrusions` 6, `severity` Set1>Set2, `verdict` Refuted | `kill_pass` False, `speed_pick` baseline v2-m3, `dr_ratio` 6.4650, `latency_cut` 1.0, `intrusions` 6, `severity` Set1>Set2, `verdict` Refuted (killed at gate) | Kill-gate: one resolution lever and one speed lever land; bar: DR up, latency down, 0 violations, 0 gold intrusions, Set2>Set1 | no: the HEAD file reads `verdict` Refuted with the kill-gate passed | Yes: the ledger label equals the file after the re-execution. The ledger clauses "0 gold intrusions" and "blend Set2>Set1" are contradicted at HEAD and after (6, Set1>Set2) | Device for the kill-gate: the CPU diagnostic reads `kill_pass` True. `dr_ratio` follows E04-H15 |
| E05-H20 | Refuted on the replacement bar. jina recall@3 0.599, Spearman 0.9636, 4 gold intrusions, recall@m 0.95 at m = 34, CPU cut 85-88% | jina `recall3` 0.5986, `spearman` 0.9636, `m_hit` 26, `cpu_cut` 0.8250; mmarco `cpu_cut` 0.9011, `curve.1` 0.2109; `verdicts.E05-H20` Confirmed (cascade) | Same, except jina `cpu_cut` 0.8693, mmarco `cpu_cut` 0.9127, mmarco `curve.1` 0.2143 | Replacement: Spearman ≥ 0.95 and recall@3 ≥ 0.90 and cut ≥ 50%; cascade: recall@m ≥ 0.95 at m < 35 | yes, except m (34 against 26); the note under the ledger record names this | Yes: the replacement reading 0.5986 did not move. The artefact label Confirmed (cascade) is the same at HEAD and after | `cpu_cut` is a timing. `curve.1` (recall at m = 1): not isolated |
| E05-H24 | Refuted. Redundancy 91%, k_eff 10. At the 50% cut (k ≤ 35) recall 0.762 and Spearman 0.806; at k = 45 recall 0.936 and Spearman 0.842 | k = 35: `recall` 0.7360, `spearman` -0.0788; k = 45: 0.8502, 0.7212; `ships` False for all 7 k | k = 35: `recall` 0.8400, `spearman` 0.2970; k = 45: 0.9064, 0.7576; `ships` False for all 7 k | Cut ≥ 50% and Spearman ≥ 0.95 and recall ≥ 0.95 | no | Holds with a smaller margin: at k = 35 recall 0.7360 → 0.8400 and Spearman -0.0788 → 0.2970, both against 0.95 | Not isolated. The GPU model differs (section 1.1) |
| E07-H34 | Confirmed (reorder-invariant). GW 0.0000 on a reorder against 0.0134 on cross-summary pairs | `gw_reorder_mean` 3.31e-10, `gw_cross_mean` 0.0249 | `gw_reorder_mean` 9.15e-11, `gw_cross_mean` 0.0249 | GW ≈ 0 on a reorder | no for 0.0134 (file 0.0249) | Yes: `verdicts.E07-H34.detail` prints 0.0000 in both files | Not isolated. The GPU model differs |
| E07-H36 | Exploratory. Pearson rho 0.657, 95% CI [0.540, 0.757]; the CI lies above the 0.3 margin | `rho` 0.6572, `ci` 0.5403 and 0.7574 | `rho` 0.6633, `ci` 0.5482 and 0.7630 | \|rho\| ≤ 0.3, registered as not confirmable | yes | Yes: the lower CI bound 0.5482 is above 0.3 | Not isolated. The GPU model differs |
| E07-H37 | Reported. Triangle violation rate 4%; cost tau 0.33 ms, SMD 0.08 ms, GW 0.43 ms | `triangle_violation_rate` 0.075; `ms_tau` 0.3208, `ms_smd` 0.0795, `ms_gw` 0.2998 | `triangle_violation_rate` 0.08; `ms_tau` 0.5638, `ms_smd` 0.1589, `ms_gw` 0.5608 | No threshold on the rate; the verdict is a report | no (4% against 0.075) | Yes: no threshold applies. The rate reads 0.075 → 0.08 | Rate: not isolated. The three `ms_` numbers are timings |
| E08-H44 | Confirmed. 0% triangle, Spearman 1.00, range ratio 0 → 4.1 → 6.7 → 8.8 over lambda, above footrule 6.4 and off-diagonal 7.4 at lambda 0.5 | `triangle_viol` 0.0, `mono_rho` 1.0, `range_by_lambda` -4.1e-06, 1.6627, 2.5273, 3.1428; `range_footrule` 5.7993, `range_offdiag` 6.6820 | `triangle_viol` 0.0, `mono_rho` 1.0, `range_by_lambda` -1.81e-05, 1.6623, 2.5269, 3.1424; `range_footrule` 5.7924, `range_offdiag` 6.6787 | Triangle ≤ 1%; Spearman ≥ 0.9 with 0 inversions; range ratio ≥ the E07 footrule and off-diagonal baselines | no | No on the range clause, at HEAD and after: 3.1424 at lambda 0.5 against 5.7924 and 6.6787 (HEAD 3.1428 against 5.7993 and 6.6820). The triangle and Spearman clauses hold. The only number above 1% is the range at lambda 0 | Not isolated. The GPU model differs. The failing clause is not an effect of the re-execution |
| E08-H45 | Refuted. Pure shift 0.0000 against 0.134, 0% triangle, collapse at full scramble, peak bin 4, Spearman 0.43 | `mono_rho` 0.3714, `mono_inv` 2, `peak_bin` 3, `collapse_at_full_scramble` True, `pure_shift_paug` 0.1455, `alpha_curve.0.0` 0.000175 | Same, except `alpha_curve.0.0` 0.000184 | Monotone rise with reorder; 0 triangle violations; ≈ 0 on a pure shift | no (0.43 against 0.3714, bin 4 against 3) | Yes: `collapse_at_full_scramble` is True in both files | Not isolated. The GPU model differs |
| E08-H47 | Refuted. Row-entropy 0.129 → 0.130, per-pair 0.135, naive margin +0.005 → −0.009, Pearson 0.94 | `entropy_median_raw` 0.1228 → `entropy_median_offline` 0.1204, per-pair 0.1237; `naive_margin_raw` 0.0927 → `naive_margin_offline` 0.0726; Pearson 0.9772; `verdicts.E08-H47.verdict` Kept | 0.1216 → 0.1204, per-pair 0.1237; `naive_margin_raw` 0.0908 → `naive_margin_offline` 0.0818; Pearson 0.9772; `verdicts.E08-H47.verdict` Kept | Row-entropy falls with SMD fidelity held; dynamic range and naive margin both widen | no | Holds with a smaller margin on the naive-margin clause: the margin falls 0.0927 → 0.0726 at HEAD and 0.0908 → 0.0818 after. The entropy clause reads a fall in both files, where the ledger states no sharpening | Not isolated. The GPU model differs |
| E08-H48 | Refuted. Spearman 1.00, range 8.1 unweighted and 7.2 weighted, 6.7% triangle violations against about 4% for the tau-footrule | `triangle_viol` 0.06, `range_ratio_unweighted` 7.1996, `range_ratio_weighted` 4.1301 | `triangle_viol` 0.0667, `range_ratio_unweighted` 7.2009, `range_ratio_weighted` 4.1320 | Spearman ≥ 0.9; the triangle rate is reported against the tau-footrule rate | no at HEAD (6.7% against 0.06); the rate equals the file after | The label holds: the rate is above 0, so the lever is not a metric. The clause 'more than the tau-footrule' does not hold on the stored files: cell 24 compares with the constant 0.035, and the E07 file reads 0.075 at HEAD and 0.08 after, above this lever's 0.06 and 0.0667 (corrected 2026-10-10 after the final review) | Not isolated. The GPU model differs |
| E09-H49 | Refuted. Row-entropy 0.129 → 0.135, fidelity 0.741, 1 of 24 tier violations, CMRR −0.19 | `entropy_raw` 0.1228 → `entropy_cmr` 0.1237, `smd_fidelity` 0.8695, `tier_viol` 0, `cmrr` -0.0568 | 0.1216 → 0.1237, `smd_fidelity` 0.8695, `tier_viol` 0, `cmrr` -0.1295 | CMRR > 1 with a structural gain and 0 violations | no | Yes: -0.1295 is below 1 | Not isolated. The GPU model differs. `entropy_raw` moved by 0.9%, and `cmrr` divides the relative entropy change by 1 − fidelity |
| E09-H50 | Inconclusive. Offline-frozen fidelity 0.921 against per-pair 0.741; CMRR −0.11 against −0.19 | offline-frozen `cmrr` 0.6387, fidelity 0.9695; per-pair `cmrr` -0.0568; `winner` offline-frozen (k=1) | offline-frozen `cmrr` 0.3358, fidelity 0.9695; per-pair `cmrr` -0.1295; `winner` offline-frozen (k=1) | The arm with the highest CMRR; a lever ships at CMRR > 1 | no | Yes: the winner is the same and 0.3358 is below 1 | Not isolated. The GPU model differs |
| E09-H51 | Inconclusive. Best D 0; CMRR by depth 0.00, −0.19, −0.27, −0.16; stability 0.77, 0.86, 0.83 | `cmrr_by_d` 0.0, -0.0568, -0.1255, -0.5598; `stability_by_d` 0.7355, 0.8783, 0.8338; `best_d` 0 | `cmrr_by_d` 0.0, -0.1295, -0.1627, -0.5933; `stability_by_d` 0.7354, 0.8783, 0.8337; `best_d` 0 | The D with the highest CMRR; stability bar 0.9 | no | Yes: `best_d` is 0 in both files | Not isolated. The GPU model differs |
| E09-H52 | Refuted. CMRR default −0.19 → fine +0.12; stability 0.77 → 0.86 | `cmrr` coarse -0.0119, default -0.0568, fine 0.6953 | `cmrr` coarse 0.0018, default -0.1295, fine 0.6034 | A granularity with CMRR > 1 | no | Yes: the best value 0.6034 is below 1 | Not isolated. The GPU model differs |
| E10-H56 | Inconclusive. Pure shift 0, Spearman 1.00, 0% triangle on equal length, paraphrase-proxy median 0.1, 22.7% triangle on diffuse pairs | `shift_value` 0.0, `spearman` 1.0, `paraphrase_median` 0.0460, `paraphrase_p95` 0.2893, `content_pass` False, `triangle_viol_diffuse` 0.2067 | Same, except `paraphrase_p95` 0.3116 | Spearman ≥ 0.9; ≈ 0 on the paraphrase control; 0 triangle violations on equal-length ranks; shift ≤ paraphrase p95 | no (0.1 against 0.0460, 22.7% against 0.2067) | Yes: `content_pass` is False in both files, and the shift 0.0 is below both p95 values | Not isolated. The GPU model differs |

### 1.3 Hypotheses whose drift above 1% is timing or load only

| Id | Ledger line states | HEAD | After | Bar or kill-gate | Ledger = HEAD | Verdict holds on the values after | Cause |
|---|---|---|---|---|---|---|---|
| E04-H18 | Refuted (killed at gate). Spearman 0.976, cut 2.28x above the 2x bar, recall@15 0.867, 1 gold intrusion added | `recall_m` 0.8537, `spearman` 0.9879, `e2e_cut` 3.3089, `intrusions` 6 against baseline 6, `passed` True | `recall_m` 0.8469, `spearman` 0.9879, `e2e_cut` 1.3671, `intrusions` 6, `passed` False | Kill-gate recall@15 ≥ 0.95; bar Spearman ≥ 0.95 and latency cut ≥ 2x and 0 new intrusions | no | Yes for the label: 0.8469 is below 0.95. The ledger clause that the cascade clears the speed bar does not hold after: 1.3671 against 2 (HEAD 3.3089). `recall_m` moved by 0.8% | Device: the CPU diagnostic reads `e2e_cut` 2.5030 and `passed` True |
| E05-H23 | Refuted. 6.38 s → 6.28 s, a 1.7% cut, Spearman 1.0000 | `cur_s` 6.3843, `bucketed_s` 6.2753, `cpu_cut` 0.0171 | `cur_s` 6.4931, `bucketed_s` 6.3720, `cpu_cut` 0.0186 | Cut ≥ 50% | yes | Yes: 0.0186 is below 0.50 | Timing on the CPU. The file stores no load field |
| E12-H64 | Confirmed. Arm a recovers 103.7% of the full gain; arm b −4.5% | `arm_a_center_renorm_recovery` 1.0366, `arm_b_project_only_recovery` -0.0451; `transform_cost_ms_indicative` 0.0963 and 168.81 | Recoveries equal to 14 digits; `transform_cost_ms_indicative` 0.0734 and 506.99 | Exactly one arm at or above 60% of the full gain at 0 of 24 violations | yes | Yes: the verdict readings did not move | Timing. The file stores no load field. The notebook calls this panel indicative |
| E12-H65 | Confirmed. d' 5.020 → 7.161 over pool sizes 2 → 11 | `sweep_by_pool` d' means 5.020, 5.908, 5.950, 6.357, 6.683, 7.161; `timings_s` segment 19.36, embed 24.87, sweep 53.74 | The same d' means; `timings_s` segment 47.32, embed 83.80, sweep 20.70 | d' crosses the baseline or rises monotonically; Spearman ≥ 0.8 | yes | Yes: the verdict readings did not move | Timing. The file stores no load field |
| E12-H66 | Refuted (null). d' delta −0.011, margin delta +0.012 points, latency 0.877x | `deltas.dprime` -0.0109, `deltas.latency_ratio` 0.8770, `ms_per_pair` 0.0962 and 0.0844 | `deltas.dprime` -0.0109, `deltas.latency_ratio` 0.9560, `ms_per_pair` 0.1109 and 0.1060 | 0 of 24 violations, margin ≥ +4.58, d' ≥ 4.60; the verdict follows the delta | yes | Yes: the d' and margin deltas did not move | Not isolated between `config.pot_version` (0.9.7.post1 → 0.9.6.post1) and the load on the host, which the file does not store |
| E12-H67 | Promoted. 9.94x median on the 55-row sweep (11.032 s → 1.110 s), disjoint control −0.63%, 9 of 9 arms byte-identical, 56 timed runs at load 5.97 to 7.97 with 0 breaches | `speedup_allpairs_median` 9.9361, `disjoint_regression` -0.006271, `jsonl_byte_identical_arms` 9/9, breaches 0 | `speedup_allpairs_median` 10.1196, `disjoint_regression` -0.003765, `jsonl_byte_identical_arms` 9/9, breaches 56 | Byte-identical output; speedup ≥ 3x; disjoint regression < 2% | yes | Yes: 10.1196 is above 3 and -0.003765 is below 0.02. All 56 timed runs of the re-execution started with the load reading above the ceiling 8.0 | Load reading on the host (section 1.1). The payload digests differ between the two runs (`end_to_end_cli.sha256_on` 682ad33b → 4bec474f): not isolated, `environment.pot` differs |
| E12-H68 | Promoted. 10.03x on the 55-pair sweep (2671.72 s → 266.32 s), 55 of 55 payloads byte-identical, no-fanout control +0.22% | `speedup_allpairs_55` 10.0319, `wall_off_s` 2671.717, `wall_on_s` 266.322, `pairs_byte_identical` 55/55, `nofanout_control_delta_pct` 0.22 | `speedup_allpairs_55` 9.6638, `wall_off_s` 2623.567, `wall_on_s` 271.484, `pairs_byte_identical` 55/55, `nofanout_control_delta_pct` -1.31 | Byte-identical payloads; speedup ≥ 5x; a speedup below 3x fails | yes | Holds with a smaller margin: 10.0319 → 9.6638 against 5. `load_average.precondition_ok` reads False after | Load reading on the host (section 1.1). The payload digests differ between the two runs: not isolated |

### 1.4 E06 and E12-H61

- E06: the 're-run 2026-10-09' text of ledger A covers every verdict-bearing number of the drift list - E06-H25 433.86 s → 251.12 s, E06-H26 recall@3 0.5408, Spearman 0.721, 4 gold intrusions and m = 36, E06-H27 subprocess time 30.7 s
- E12-H61: the count of removed statements flagged `moved` on the text shrinkage fixture (1 → 2, three leaves of the file) is recorded in the E15 introduction and in the Evidence line of E15-H80 of ledger A, and the `### E12-H61` record itself has no line for it; the verdict-bearing reading `kept_moved_conjunction` is 0 at HEAD and after

### 1.5 Ledger lines without drift above 1% that the artefact contradicts at HEAD and after

These lines were seen while the records were read. They are not the result of a complete comparison.

- E05-H21: ledger A states MaxSim recall@3 0.469 and Spearman 0.079, and the file reads `H21.recall3` 0.4320 and `H21.spearman` 0.9636
- E05-H22: ledger A states recall@15 0.344 and recall@3 0.078, and the file reads `H22.recall15` 0.3912 and `H22.recall3` 0.1156
- E08-H46: ledger B states a range ratio 6.2 that is narrower than the 6.7 of E08-H44, and the file reads `range_ratio` 5.3929 against `range_h1` 2.5269 with `narrower_than_h1` False
- E09-H53: ledger B states Refuted with d' 2.81 → 2.28 and 1 of 24 tier violations, and the file reads `verdicts.E09-H53.verdict` Confirmed with `tier_dprime_raw` 5.9731, `tier_dprime_cmr` 8.0237 and `tier_viol_cmr` 0

## 2. E04 CPU diagnostic

### 2.1 Procedure

- Scratch folder: `/tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/drift-audit/`
- Copy: `E04-cpu-diagnostic.ipynb` in the scratch folder, built from the working-tree notebook by `make_e04_cpu_copy.py`
- Two edits in the copy: `USE_GPU = True` → `USE_GPU = False` in code cell 2, and the metrics path of code cell 28 → `E04-cpu-diagnostic-metrics.json` in the scratch folder; all outputs were removed before the run
- Command: `CI=true HF_HUB_OFFLINE=1 setsid nohup .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 <copy>`, with the log in `E04-cpu-diagnostic.log`
- The kernel name `python3` was added because the kernel name `docdistance` of the notebook metadata is not registered in `.venv`; `reports/E13-campaign/reexec.sh` passes the same option
- Start 2026-10-10T04:29:42Z; the executed copy was written at 2026-10-10T04:44:17Z; 0 error outputs and 0 stderr streams
- Cell 2 printed `backend: CPU OpenVINO INT8; CUDA_VISIBLE_DEVICES=''`
- Cell 4 printed `docdistance 1.1.19 | torch 2.11.0+cu130 | device cpu | CPU`: the copy ran on the package of the working tree, not on 1.1.5 (re-execution) and not on 1.1.2 (HEAD run)
- The 9 files `src/docdistance/*.py` had equal SHA-256 digests before and after the run
- Cell 12 printed the same statement counts as the HEAD run and the re-execution: 10, 12, 11, 11, 11, 9, 13, 8, 9, 1, 3
- `uptime` read a 1-minute load average of 34.97 before the start and 60.92 at the end, so the latency numbers of the copy are not a measurement

### 2.2 H1 sweep (ledger id E04-H15)

| Reading | HEAD (cpu, OpenVINO INT8, docdistance 1.1.2) | After (cuda, torch fp16, docdistance 1.1.5) | CPU diagnostic (cpu, OpenVINO INT8, docdistance 1.1.19) |
|---|---|---|---|
| Kill-gate cosine | 0.718 | 0.718 | 0.715 |
| Baseline DR, violations, contrast | 0.0370, 0, 4.34x | 0.0322, 0, 3.99x | 0.0345, 0, 4.51x |
| k = 1: DR, ratio, violations, contrast | 0.2191, 5.93x, 1, 2.33x | 0.2082, 6.47x, 0, 2.29x | 0.2084, 6.03x, 0, 2.33x |
| k = 2: DR, ratio, violations, contrast | 0.1253, 3.39x, 0, 1.91x | 0.1313, 4.08x, 0, 1.93x | 0.1247, 3.61x, 0, 1.98x |
| k = 3: DR, ratio, violations, contrast | 0.1115, 3.02x, 0, 1.85x | 0.1280, 3.98x, 0, 1.93x | 0.1238, 3.58x, 0, 2.00x |
| Best k and its ratio | 2, 3.3908 | 1, 6.4650 | 1, 6.0340 |

The first five rows are copied from cell 16 of the three notebooks. The last row is copied from `E04-H1.best_k` and `E04-H1.dr_ratio` of the three metrics files.

### 2.3 Other readings of the three runs

| Reading | HEAD | After | CPU diagnostic |
|---|---|---|---|
| `E04-H2.con_range` | 1.6631 | 1.5566 | 1.7598 |
| `E04-H3.recall3` | 0.5680272108843537 | 0.554421768707483 | 0.5680272108843537 |
| `E04-H3.spearman` | 0.6121 | 0.6121 | 0.6121 |
| `E04-H3.speed_cut` | 1.9392 | 1.2612 | 1.2272 |
| `E04-H4.recall_m` | 0.8537414965986395 | 0.8469387755102041 | 0.8537414965986395 |
| `E04-H4.e2e_cut`, `E04-H4.passed` | 3.3089, True | 1.3671, False | 2.5030, True |
| `E04-H5.kill_pass`, `E04-H5.verdict` | True, Refuted | False, Refuted (killed at gate) | True, Refuted |
| D_grd tier means (cell 14): gold, Set 1, Set 2 | 0.0446, 0.2914, 0.0029 | 0.0416, 0.2231, 0.0027 | 0.0446, 0.2914, 0.0029 |
| SMD tier means (cell 14): gold, Set 1, Set 2 | 0.2083, 0.4521, 0.3677 | 0.2025, 0.4381, 0.3556 | 0.2106, 0.4514, 0.3689 |

### 2.4 Result

- The CPU diagnostic does not reproduce the HEAD reading of E04-H15: HEAD reads 1 violation at k = 1 and therefore best k 2 at 3.39x, and the diagnostic reads 0 violations at k = 1 and best k 1 at 6.03x
- The device therefore does not explain the change of the best k from 2 to 1
- The diagnostic reproduces the readings that come from the reranker and the NLI model: `E04-H3.recall3`, `E04-H3.spearman`, `E04-H4.recall_m`, `E04-H4.spearman`, the gold-intrusion counts and the D_grd tier means equal the HEAD values
- The diagnostic does not reproduce the readings that come from the mmBERT embeddings: baseline DR 0.0345 against 0.0370 at HEAD, and the SMD tier means differ in the third decimal
- The diagnostic values at k = 1 equal the `e04_pool` block of `reports/E14-H76-conditioned-centering-probe-metrics.json` to all stored digits: `arms.raw.DR` 0.03453534218694171, `arms.shipped.DR` 0.20838521405766122, `dr_ratio_shipped_over_raw` 6.0339698657004925; that file stores `fixture.docdistance_version` 1.1.5
- The CPU reading is therefore the same on docdistance 1.1.5 and on 1.1.19, and it differs from the CPU reading of the HEAD run on docdistance 1.1.2
- The stored field that differs between the HEAD run and the two later CPU readings is the package version; the HEAD notebook stores no revision of the INT8 model and no OpenVINO version, so the cause is not isolated further
- Between the CPU reading on 1.1.5 and the re-executed GPU reading on 1.1.5 the ratio at k = 1 reads 6.03x against 6.47x, and the baseline DR reads 0.0345 against 0.0322; this difference coincides with the device and the backend
- The latency-based fields follow the device: `E04-H4.passed` and `E04-H5.kill_pass` are True on both CPU runs and False on the GPU run

### 2.5 Metrics file of the diagnostic

The scratch folder is removed with the session, so the content of `E04-cpu-diagnostic-metrics.json` is copied here.

```json
{"device": "cpu",
 "baseline": {"d_sel_DR": 0.03453534218694171, "d_sel_violations": 0, "d_sel_contrast": 4.506649139105541, "d_grd_gold_intrusions": 6, "symmetric_severity": "Set1>Set2"},
 "E04-H1": {"aniso": 0.7154906392097473, "kill_pass": true, "best_k": 1, "dr_ratio": 6.0339698657004925, "dr": 0.20838521405766122, "viol": 0, "contrast": 2.3290067092382523, "passed": true, "killed": true, "verdict": "Confirmed"},
 "E04-H2": {"kill_pass": true, "con_range": 1.7598403765448394, "best_tau": null, "con_ratio": null, "dr": null, "dr_ratio": null, "viol": null, "passed": false, "killed": true, "verdict": "Refuted"},
 "E04-H3": {"kill_pass": false, "recall3": 0.5680272108843537, "spearman": 0.6121212121212121, "speed_cut": 1.2271949393308301, "intrusions": 3, "t_big": 6.543581240335091, "t_base": 5.332144902669825, "t_mini": 1.2120898073286905, "passed": false, "killed": false, "verdict": "Refuted (killed at gate)"},
 "E04-H4": {"kill_pass": false, "recall_m": 0.8537414965986395, "spearman": 0.9878787878787878, "call_cut": 4.666666666666667, "e2e_cut": 2.503009147611851, "intrusions": 6, "m": 15, "passed": true, "killed": false, "verdict": "Refuted (killed at gate)"},
 "E04-H5": {"kill_pass": true, "dr_pick": "H1 anisotropy k=1", "speed_pick": "H4 MiniLM cascade", "dr_ratio": 6.0339698657004925, "latency_cut": 2.503009147611851, "dsel_viol": 0, "intrusions": 6, "severity": "Set1>Set2", "sym_severity": "Set1>Set2", "passed": false, "killed": true, "verdict": "Refuted"}}
```

## 3. Markdown sentences that contradicted the stored output

### 3.1 Corrected sentences

33 sentences in 8 notebooks were corrected with 35 text replacements. Each replacement went through `nbformat`: read without conversion, change of `cell.source`, write with nbformat 4.5. After the write, every other cell, the metadata and the nbformat version of each notebook equal the copy taken before the write. No cell was executed.

**E04 notebook, cell 29 (Conclusions) - 6 sentences; all six also contradicted the HEAD output**

1. Stored: cell 16 prints baseline DR 0.0322, k=1 0.2082 and 6.47x with 0 violations, k=2 4.08x, k=3 3.98x, kill-gate cosine 0.718
   - Before: widens the conditioned `D_sel` dynamic range to 7.45x (0.0193 → 0.144 at k=3, already 7.42x at k=1) while holding 0 ordinality violations; the kill-gate confirmed the coverage cosines are anisotropic (mean off-diagonal 0.736).
   - After: widens the conditioned `D_sel` dynamic range to 6.47x (0.0322 → 0.2082 at k=1; 4.08x at k=2, 3.98x at k=3) while holding 0 ordinality violations; the kill-gate confirmed the coverage cosines are anisotropic (mean off-diagonal 0.718).
2. Stored: cell 16 prints contrast 3.99x for the baseline and 2.29x at k=1
   - Before: the gold/adversarial contrast slips slightly (2.31x → 2.04x) because the within-tier band spreads too.
   - After: the gold/adversarial contrast falls (3.99x → 2.29x at k=1) because the within-tier band spreads too.
3. Stored: cell 18 prints DR 0.0322 at tau 0.10 and 0.0940 at tau 0.01, contrast 3.99x and 2.89x, 0 violations in all 5 rows
   - Before: Lowering the coverage temperature raises dynamic range (0.0193 → 0.049) but lowers the contrast (2.31x → 1.47x) and breaks ordinality (V 0 → 2); temperature trades contrast for range rather than buying both
   - After: Lowering the coverage temperature raises dynamic range (0.0322 → 0.0940 at tau 0.01) but lowers the contrast (3.99x → 2.89x), with 0 ordinality violations at every temperature; temperature trades contrast for range rather than buying both
4. Stored: cell 20 prints recall@3 0.554, Spearman 0.612, latency cut 1.26x, gold intrusions 3 (baseline 6)
   - Before: `bge-reranker-base` recalls only 0.55 of the v2-m3 top-3, gives Spearman 0.70 and only 2.6x, and adds a gold intrusion;
   - After: `bge-reranker-base` recalls only 0.55 of the v2-m3 top-3, gives Spearman 0.61 and only 1.26x, and reads 3 gold intrusions against a baseline of 6;
5. Stored: cell 22 prints recall@15 0.847, Spearman 0.988, latency cut 1.37x (bar >= 2.0x), gold intrusions 6 (baseline 6), `H4 FAIL`
   - Before: A tiny MiniLM pre-filter clears fidelity (Spearman 0.976) and speed (2.28x, v2-m3 calls 70 → 15) - far better than the E03 bi-encoder (0.55) - but recall@15 of the v2-m3 top-3 is 0.87 (gate 0.95) and the dropped evidence adds one gold intrusion.
   - After: A tiny MiniLM pre-filter clears fidelity (Spearman 0.988) but not speed (1.37x against the 2.0x bar, v2-m3 calls 70 → 15) - far better than the E03 bi-encoder (0.55) - and recall@15 of the v2-m3 top-3 is 0.85 (gate 0.95), with gold intrusions unchanged at 6.
6. Stored: cell 24 prints DR ratio 6.47x, latency cut 1.00x, 0 violations, gold intrusions 6 (baseline 6), blend severity Set1>Set2 (symmetric Set1>Set2)
   - Before: DR 7.45x, 0 violations, 0 gold intrusions, blend Set2>Set1 (vs symmetric Set1>Set2), every guardrail held, but no latency improvement
   - After: DR 6.47x, 0 violations, 6 gold intrusions (baseline 6), blend Set1>Set2 (symmetric Set1>Set2), so the Set2>Set1 severity guardrail is not met, and no latency improvement

**E05 notebook, cell 30 (Conclusions) - 6 sentences, 7 replacements; all also contradicted the HEAD output**

7. Stored: cell 16 prints recall@3 0.405 and 0.599, cell 18 prints 0.432, cell 20 prints recall@3 0.116
   - Before: mmarco-MiniLM (117M) recall@3 `0.45`, jina-reranker-v2 (278M) recall@3 `0.63`, late-interaction MaxSim `0.47`, the SPLADE proxy `0.08`
   - After: mmarco-MiniLM (117M) recall@3 `0.405`, jina-reranker-v2 (278M) recall@3 `0.599`, late-interaction MaxSim `0.432`, the SPLADE proxy `0.116`
8. Stored: cell 16 prints jina Spearman 0.964, CPU cut 87%, m@0.95 26, 4 intrusions, replace no, cascade YES, and mmarco Spearman -0.042; cell 12 prints 6 gold intrusions for the shipped chain; cell 29 prints `Confirmed (cascade)`
   - Before: `D_grd` Spearman `0.927` (bar `0.95`) at an `85%` CPU cut, missing the fidelity bar by `0.023`; its recall@m reaches `0.95` only at `m = 34` (half the source) and it adds `1` gold intrusion, so it neither replaces nor cascades cleanly, but it is far ahead of the 117M MiniLM (Spearman `0.58`) and the E04 candidates
   - After: `D_grd` Spearman `0.964` (bar `0.95`, cleared) at an `87%` CPU cut; its recall@3 is `0.599` (replacement bar `0.90`), its recall@m reaches `0.95` at `m = 26` (cascade bar `m < 34`) and it reads `4` gold intrusions against a baseline of `6`, so the stored H20 table marks it `no` for replacement and `YES` for cascade (stored verdict `Confirmed (cascade)`), and it is far ahead of the 117M MiniLM (Spearman `-0.042`) and the E04 candidates
9. Stored: cell 22 prints p50=64 and 2.54x waste; cell 23 prints 6.5s → 6.4s = 2% cut on the one-doc grid (70 pairs), Spearman 1.0000
   - Before: **H23 length-bucketing is a real 43% CPU cut at bit-identical scores** - the pairs carry `2.82x` padded-token waste (median 57 tokens, padded to 256); length-sorting the grid and tightening `max_length` to the 100th percentile cuts the shipped CPU OpenVINO INT8 reranker `73.2 s -> 42.0 s` (`43%`) with the relevance ranking unchanged (Spearman `1.0000`).
   - After: **H23 length-bucketing is a 2% CPU cut at bit-identical scores** - the pairs carry `2.54x` padded-token waste (median 64 tokens, padded to 256); length-sorting the grid and tightening `max_length` to the 100th percentile cuts the shipped CPU OpenVINO INT8 reranker `6.5 s -> 6.4 s` (`2%`) on the one-document grid of 70 pairs, with the relevance ranking unchanged (Spearman `1.0000`).
10. Stored: cell 23 prints `cut 2% (bar 50%)`; 48 is 50 minus 2, computed
    - Before: It misses the `50%` bar by 7 points
    - After: It misses the `50%` bar by 48 points
11. Stored: cell 25 prints k=35 recall 0.840 and Spearman 0.297 (same sentence as 12)
    - Before: drops top-3 recall to `0.76` and `D_grd` Spearman to `0.81`
    - After: drops top-3 recall to `0.84` and `D_grd` Spearman to `0.30`
12. Stored: cell 25 prints k=45 recall 0.906, Spearman 0.758, column cut 36%
    - Before: the safe ceiling is `~36%` (k=45, recall `0.94`)
    - After: the smallest cut in the sweep is `36%` (k=45, recall `0.91`, Spearman `0.76`), below both `0.95` bars
13. Stored: cell 23 prints a 2% cut
    - Before: ship the H23 length-bucketing as a portable `~43%` CPU speedup
    - After: ship the H23 length-bucketing as a portable `~2%` CPU speedup

**E07 notebook, cell 11 - 1 sentence; also contradicted the HEAD output**

14. Stored: cell 12 prints `bases 10 x 6 bins`. The words "gold-tier" were removed because `config.bases` of the metrics file lists three `werg-` labels beside the 7 summary labels
    - Before: all 7 gold-tier bases
    - After: all 10 bases

**E08 notebook, cells 13 and 29 - 4 sentences, 5 replacements; all also contradicted the HEAD output**

15. Cell 13. Stored: cell 14 prints `sweep rows: 720`, cell 8 prints 6 bins at 12/bin/base, and `config.bases` of the metrics file holds 10 labels
    - Before: of all 7 bases x 6 displacement bins
    - After: of all 10 bases x 6 displacement bins
16. Cell 29. Stored: cell 16 prints `max|delta|=2.5e-12` (same sentence as 17)
    - Before: `lam=0` reproduces SMD to 1e-12,
    - After: `lam=0` reproduces SMD to 2.5e-12,
17. Cell 29. Stored: cell 14 prints `paug0.5` 3.14, `footrule` 5.79, `offdiag` 6.68; cell 16 prints `clears both E07 baselines ... at lambda []`
    - Before: at `lambda=0.5` it exceeds **both** saturating E07 baselines (footrule and off-diagonal mass) while staying a metric.
    - After: at `lambda=0.5` it reads 3.14, below **both** saturating E07 baselines (footrule 5.79 and off-diagonal mass 6.68), while staying a metric.
18. Cell 29. Stored: cell 18 prints `Spearman 0.37/2 inv`
    - Before: breaking monotonicity (Spearman 0.43)
    - After: breaking monotonicity (Spearman 0.37)
19. Cell 29. Stored: cell 22 prints `E08-H47: Kept | row-entropy raw 0.122 -> offline 0.120 (perpair 0.124); sharpened; SMD fidelity Pearson 0.98 (held); naive margin +0.091 -> +0.082`
    - Before: does **not** sharpen the cross-summary `T` (already concentrated at row-entropy ~0.13, little room) - row-entropy is flat (0.129 -> 0.130) and per-pair pooled is no better; SMD fidelity holds (Pearson 0.94) but the naive margin does not widen.
    - After: lowers the cross-summary `T` row-entropy from 0.122 to 0.120 (already concentrated, little room; the stored verdict line reads `sharpened`) and per-pair pooled is no better (0.124); SMD fidelity holds (Pearson 0.98) but the naive margin does not widen (+0.091 -> +0.082).

**E09 notebook, cell 7 - 1 sentence; also contradicted the HEAD output**

20. Stored: cell 8 prints `58 cross-summary pairs (independent summaries, T diffuse)`
    - Before: 55 cross-summary pairs (every independent summary against every other)
    - After: 58 cross-summary pairs (independent summaries, `T` diffuse)

**E10 notebook, cells 13 and 15 - 2 sentences; both also contradicted the HEAD output**

21. Cell 13. Stored: cell 14 prints `sweep rows: 720`, cell 8 prints 12/bin/base, and `config.bases` of the metrics file holds 10 labels
    - Before: (7 bases x 6 displacement bins)
    - After: (10 bases x 6 displacement bins)
22. Cell 15. Stored: cell 8 prints `embedded 25 docs`, and code cell 16 sets `DOCS = list(EMB.values())`
    - Before: over the 12 fixture documents
    - After: over the 25 fixture documents

**E12-H66 notebook, cell 31 (Conclusions) - 1 sentence; the HEAD output agreed with the old text**

23. Stored: cell 26 prints baseline 0.1109 ms/pair, norm-mass 0.1060 ms/pair, 0.956x
    - Before: Latency is free as predicted - 0.0844 against 0.0962 ms/pair, which is a 0.877x reading
    - After: Latency is free as predicted - 0.1060 against 0.1109 ms/pair, which is a 0.956x reading

**E12-H67 notebook, cell 44 (Conclusions) - 12 sentences; the HEAD output and the HEAD metrics file agreed with the old text**

24. Stored: cell 22 prints `off 10.985s on 1.085s 10.12x`
    - Before: **11.032 s to 1.110 s**, a **9.94x** reduction against a pre-registered bar of 3x
    - After: **10.985 s to 1.085 s**, a **10.12x** reduction against a pre-registered bar of 3x
25. Stored: `arms.all-pairs 55.off_times_s` 11.044103, 10.819476, 10.984628, `on_times_s` 1.062703, 1.584054, 1.085481 and `speedup_min` 10.181095 of the metrics file; no cell prints the three repeats
    - Before: off 11.000 / 11.129 / 11.032 s against on 1.137 / 1.110 / 1.096 s, with the minimum-of-three speedup 10.03x agreeing with the median 9.94x.
    - After: off 11.044 / 10.819 / 10.985 s against on 1.063 / 1.584 / 1.085 s, with the minimum-of-three speedup 10.18x agreeing with the median 10.12x.
26. Stored: cell 34 prints `sha256 off 4bec474fbddab0aa on 4bec474fbddab0aa`
    - Before: the same `sha256 682ad33b...` under both
    - After: the same `sha256 4bec474f...` under both
27. Stored: cell 36 prints `memo key bookkeeping 15.42 ms over 110 slots` and median 17.702s; 0.09% is 15.42 ms divided by 17.702 s, computed
    - Before: pays only its key construction: 17.50 ms of `Path.resolve()` and `stat()` over 110 slots, or **0.10%** of the 17.856 s batch.
    - After: pays only its key construction: 15.42 ms of `Path.resolve()` and `stat()` over 110 slots, or **0.09%** of the 17.702 s batch.
28. Stored: cell 36 prints `median 17.702s -> 17.635s, delta -0.38%`
    - Before: time moves **-0.63%** (median 17.856 s to 17.745 s)
    - After: time moves **-0.38%** (median 17.702 s to 17.635 s)
29. Stored: cell 26 prints 1.72x, 3.40x and 5.13x; cell 22 prints 10.12x
    - Before: returns 1.81x at N=2 (1 of 4 slots reusable), 3.24x at N=5 (4 of 10) and 5.18x at N=11 (10 of 22), against 9.94x for the all-pairs sweep
    - After: returns 1.72x at N=2 (1 of 4 slots reusable), 3.40x at N=5 (4 of 10) and 5.13x at N=11 (10 of 22), against 10.12x for the all-pairs sweep
30. Stored: cell 30 prints 9.83x; cell 32 prints 9.93x and 9.93x
    - Before: structural rerun (9.47x), the filtered arm (9.84x) and the markdown-segmentation arm (9.95x) all land
    - After: structural rerun (9.83x), the filtered arm (9.93x) and the markdown-segmentation arm (9.93x) all land
31. Stored: cell 14 prints `encode share 50.2%` and `median pair latency 209.4 ms`
    - Before: segment-and-encode of one document is **54.3%** of the 198.7 ms median pair latency
    - After: segment-and-encode of one document is **50.2%** of the 209.4 ms median pair latency
32. Stored: cell 34 prints `off 20.06s on 10.07s 1.99x`; cell 14 prints `loaded in 13.2 s`
    - Before: the same sweep runs 20.54 s uncached against 10.36 s cached, **1.98x**, because a 13.5 s model load now dominates a 1.1 s batch.
    - After: the same sweep runs 20.06 s uncached against 10.07 s cached, **1.99x**, because a 13.2 s model load now dominates a 1.1 s batch.
33. Stored: cell 36 prints `min 31.61, median 36.46, max 44.24 (ceiling 8.0); breaches: 56`
    - Before: **Caveats.** Wall time was measured on a machine running this notebook alone, with the 1-minute load average checked before each of the 56 timed runs (min 5.97, median 7.26, max 7.97, ceiling 8.0, zero breaches); the load ceiling forced idle gaps between runs, which makes each measurement quieter but does not remove the platform dependence of the ratio.
    - After: **Caveats.** The 1-minute load average was read before each of the 56 timed runs (min 31.61, median 36.46, max 44.24, ceiling 8.0, 56 breaches): every timed run started with the load reading above the ceiling, and the ratio stays platform dependent.
34. Stored: cell 34 prints 1.99x
    - Before: the end-to-end 1.98x is reported alongside.
    - After: the end-to-end 1.99x is reported alongside.
35. Stored: cell 36 prints median 17.702s
    - Before: which is why its uncached batch is 17.9 s rather than 11.0 s.
    - After: which is why its uncached batch is 17.7 s rather than 11.0 s.

Counts per notebook: E04 6, E05 6, E07 1, E08 4, E09 1, E10 2, E12-H64 0, E12-H65 0, E12-H66 1, E12-H67 12, E12-H68 0. The Conclusions cell of E12-H68 is rendered by code from the measured variables. No notebook has more than 15 contradicting sentences.

### 3.2 Sentences that were not changed

These sentences state no number that contradicts the stored output, but the stored output does not support their claim. They were left for a decision.

- E04 cell 29: "the near-miss" and "The closest faithful speed-up" for H4 - cell 22 prints `H4 FAIL` with the latency cut 1.37x below the 2.0x bar
- E05 cell 30, first paragraph: "No lever clears the pre-registered gate" - cell 29 prints `Confirmed (cascade)` for E05-H20 and `survivors: ['E05-H20']`
- E05 cell 30, Method note: "it confirms the refutations independently, and only jina approaches the bar" - cell 16 prints jina Spearman 0.964 against the 0.95 bar, and cell 18 prints MaxSim Spearman 0.964
- E05 cell 30: "worth shipping as a portable default" for H23 - cell 23 prints a 2% cut
- E08 cell 29: "This is the metric, wide-range structure distance the `tau`-footrule (4% triangle violations, narrow range) is not" - cell 14 prints a range ratio 3.14 for E08-H44 at lambda 0.5 and 5.79 for the footrule
- E08 cell 29: E08-H46 "with a range narrower than E08-H44 as designed" - cell 14 prints 5.39 for the optimal-assignment footrule and 2.53 for E08-H44 at lambda 0.25
- E08 cell 29: "The lever the design hoped would help adds nothing on an already-sharp `T`" - cell 22 prints `E08-H47: Kept` and `sharpened`
- E12-H66 cell 31: "since the 64-thread CPU was shared with concurrent campaign executors during this run" - the notebook stores no load reading, so this clause is not verified for the re-execution
- E10 code cell 16: the comment `# 12 fixture docs (incl. source)` stands beside `DOCS = list(EMB.values())`, and cell 8 prints `embedded 25 docs`

Device statements in markdown cells that name another GPU than the stored output:

- E05 cell 1: "the RTX 5000 Ada (32 GB, idle) is index 2" - cell 4 prints `NVIDIA RTX PRO 4000 Blackwell`
- E07 cell 1, E08 cell 1, E09 cell 1, E10 cell 1: "runs on the RTX 5000 Ada (GPU 2)" - cell 2 of each notebook prints `NVIDIA RTX PRO 4000 Blackwell`
- E10 cell 1 adds "so the embeddings are byte-comparable"; the E07 to E10 readings of the two runs are not equal (section 1.2)

### 3.3 Consequence for the ledger

- The SHA-256 digests of the 8 corrected notebooks changed
- `hypothesis-tools check docs/experiments/wmd-docdistance-experiments.md` printed 1 warning before the corrections and prints 13 after them: the notebook attachment lines of E04-H15 to E04-H19, E05-H20 to E05-H24, E12-H66 and E12-H67 carry the digests of the files before the corrections
- The ledger was not edited; `hypothesis-tools attach` refreshes these lines

## 4. After the audit

Added by the orchestrator on 2026-10-10.

- This report was written by an audit agent and saved to this path by the orchestrator
- The readings of sections 1.2, 1.3 and 1.5 are recorded as log lines under 12 records of ledger A and 14 records of ledger B; no recorded Result or Verdict line was changed
- The sentences of section 3.2 were corrected in two follow-ups, with the same method as section 3.1: 16 sentences in 7 notebooks (E04, E05, E07, E08, E09, E10, E12-H66), then 6 sentences in 4 notebooks (E04, E06, E08, E12-H66)
- The device statements now name the GPU that the stored output prints, without an index: `nvidia-smi` lists the NVIDIA RTX PRO 4000 Blackwell at index 0 on 2026-10-10, and the E05 and E07 to E10 notebooks select the GPU by UUID
- The comment in code cell 16 of the E10 notebook was left as it is (code cell)
- The ledger attachments of the changed notebooks were refreshed with `hypothesis-tools attach`
- `docs/solution/wmd-source-conditioned-docdistance-solution-sota.md` was corrected in two sentences: the E05 recall readings (0.405, 0.599, 0.432) and the backend and contrast of the E04-H15 record
- The final review (run `wf_73bb512c-5af`) corrected further sentences of that document to the stored files (E04-H17, E04-H18, E05-H24 and the device of the E04 numbers) and found the E11 readings of `docs/solution/wmd-structure-distance-sota.md` and `README.md` different from `reports/E11-structure-sota-decision.json`; E11 is outside sections 1 to 3 of this audit, and the note at the top of ledger B states its stored readings
- The row of E08-H48 in section 1.2 was corrected: its clause 'more than the tau-footrule' does not hold on the stored files

## Not checked

- No run isolates the cause of the E05 and E07 to E10 drift; the audit ran one diagnostic, for E04
- The E04 diagnostic ran on docdistance 1.1.19; no CPU run on 1.1.2 exists, so no run of this audit reproduced the HEAD reading of E04-H15
- Numbers that moved by 1% or less were not compared, except where a table row quotes them
- The batch headline paragraphs, the results tables and the benchmark lists of the two ledgers were not compared with the artefacts; only the `###` hypothesis records of the 25 rows and the 4 records of section 1.5 were read
- The solution documents under `docs/solution/` that quote these readings were not read
- Every markdown cell of the E04, E05, E07, E08, E09 and E10 notebooks was read against the stored output
- For the five E12 notebooks a script compared every decimal number, every percentage and every `x` ratio of every markdown cell with the stored outputs; the cells it listed and the Conclusions cells were read; integer counts in the other markdown cells of these five notebooks were not read
- Figures and other image outputs of the notebooks were not compared with the markdown text
- The files under `reports/figures/` were not read
- `/proc/pressure/cpu` was not recorded during the E04 diagnostic
- The model revisions that the HEAD runs loaded are not stored in any file that was read
