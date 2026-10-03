# Evaluation methodology

This project reports **no accuracy numbers yet**. Real numbers require labelled driver video, and this document describes how to get them reproducibly.

The scenario tests (`tests/scenarios/`) and `scripts/run_synthetic_scenarios.py` use **synthetic feature streams**. They show that the decision logic behaves as specified. They say nothing about how well the system works on real drivers.

## 1. Data

Candidate public datasets (check each licence; most are research-only and require a request form):

| Dataset | What it gives | Caveats |
|---|---|---|
| **UTA-RLDD** (Real-Life Drowsiness Dataset) | ~60 participants, ~30 h of RGB video, alert / low-vigilant / drowsy | Labels are per-video, from self-reported state, not per-second. Recorded at desks, not in cabs. |
| **NTHU-DDD** | Simulated driving, day/night, glasses/sunglasses, frame-level drowsy labels | Acted drowsiness, not real fatigue. |
| **YawDD** | Yawning, talking and singing in a car | Good for yawn false-positive testing; no sleep events. |
| **DROZY** | Sleep-deprivation sessions with physiological signals, KSS | Small; near-IR. |

Write a manifest and one label file per video (formats in `src/drowsiness/evaluation/dataset.py`). Label each interval as `drowsy`, `alert` or `ignore`. Mark transitions and ambiguous seconds `ignore`.

## 2. Splits: by subject, fixed, never mixed

```
DEVELOPMENT  (~60% of subjects)  explore, tune thresholds, debug
VALIDATION   (~20% of subjects)  choose between configs, check the tuning generalises
TEST         (~20% of subjects)  report once, after the config is frozen
```

* Splits are assigned by a deterministic hash of the **subject ID** (`split_for_subject`). Frames and clips from one person never cross splits. `check_no_subject_leakage` enforces this.
* `scripts/evaluate.py` refuses to score the test split without `--i-am-not-tuning`.
* Thresholds tuned on the same videos they are scored on produce inflated numbers. That is the main failure of most drowsiness demos.

## 3. Running it

```bash
# Predictions: run the full CV pipeline headless, one timeline per video.
python scripts/process_dataset.py --manifest data/manifest.json --split development --out runs/dev --record-features

# Fast re-tuning: change thresholds in a YAML file and replay the recorded features (no CV re-run).
python scripts/replay_features.py runs/dev/logs/<video>/features.jsonl --config my.yaml --timeline runs/dev_tuned/<video>.jsonl

# Score.
python scripts/evaluate.py --manifest data/manifest.json --split validation --predictions runs/val --out runs/val_report.json
```

## 4. Metrics

**Frame level**: each prediction frame is compared with the label at that time.

* A positive prediction defaults to `HIGH_DROWSINESS_RISK` or `CRITICAL_SLEEP_RISK`. Set `--positive` to also include `DROWSINESS_WARNING`.
* Reported: accuracy, precision, recall, F1, false-positive rate and false-negative rate.
* **Abstentions** (`UNKNOWN`, `CAMERA_UNAVAILABLE`) are not silently dropped. **Coverage** is reported, and recall is given twice:
  * on covered frames;
  * **conservatively**, with abstentions during drowsy intervals counted as misses. A system must not look good by hiding in UNKNOWN.

**Event level**: this is what matters for a fleet.

* **Event recall**: the fraction of labelled drowsy episodes with any positive prediction within ±2 s. This is the primary safety metric.
* **Detection latency**: the median and p90 time from labelled onset to the first positive prediction.
* **False alarms per hour**: predicted positive episodes that overlap no drowsy label, per hour of labelled-alert driving. This drives driver acceptance.

**Compute latency**: `scripts/run_synthetic_scenarios.py` prints per-frame time for the decision layers (well under 1 ms). For end-to-end throughput, watch the FPS counter in the demo; MediaPipe on a laptop CPU is the dominant cost.

## 5. Choosing thresholds for a safety system

Optimise event recall subject to a false-alarm budget, for example "maximise recall with ≤ 1 false alarm per hour", on **validation**. Report the chosen operating point and its test result once. Accuracy alone is misleading: drowsy time is a small fraction of driving, so "always ALERT" scores high accuracy.

## 6. Domain shift and dataset limitations

* **Acted vs real fatigue.** Simulated yawns and closed eyes are exaggerated; real drowsiness is subtler (slower blinks, longer closures, gaze fixation).
* **Desk vs cab.** Vibration, sunlight through the windscreen, steering-wheel occlusion and camera mounting angles differ.
* **RGB vs IR.** Commercial driver-facing cameras use near-IR at night. RGB webcam results do not transfer directly; recalibrate quality thresholds for IR.
* **Population.** Eye shape, glasses, head coverings, skin tone and age affect landmark quality. Report metrics per subgroup where labels allow, not just in aggregate.
* **Label granularity.** RLDD labels whole videos. Per-second metrics on it measure "drowsy session" detection, not event timing.
