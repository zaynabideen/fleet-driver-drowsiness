# False positives, false negatives and how the design handles them

In a driver-monitoring system both error types are costly:

* A **false negative** (missed drowsiness) can mean a crash.
* **False positives** (nuisance alarms) teach drivers to ignore or cover the camera, which then causes false negatives later.

The table lists each situation, the risk it creates, and the specific mechanism that addresses it. "Test" names the test that exercises it (in `tests/`).

| Situation | Risk | Mechanism | Test |
|---|---|---|---|
| **Normal blinking** (100-400 ms) | FP | Blinks are timed episodes. Only closures ≥ 1.0 s are evidence; a single closure ≥ 1 s still needs 0.5 s of persistence before WARNING. | `test_scenario_1_normal_blinking_is_alert` |
| **Single-frame tracker glitch** (EAR dips for 1 frame) | FP | Closures < 50 ms are discarded as noise. Per-frame EAR hysteresis stops threshold chatter. | `test_noisy_eye_signal_does_not_trigger`, `test_eye_state_hysteresis_prevents_chatter` |
| **Looking down at dashboard / phone** | FP | Head-down only counts as drowsiness **while eyes are not confirmed open**. Head-down with open eyes is reported as `eyes_off_road_s` (distraction context), never as drowsiness risk. A single quick head drop ("nod") is ignored; nods count from 2 in 60 s. | `test_scenario_7_*`, `test_head_down_with_eyes_open_is_distraction_not_drowsiness` |
| **Downcast eyelids when looking down** (EAR falls without real closure) | FP | Short, and the eyes reopen when the head comes back; temporal confirmation stops it escalating. At > 35° pitch, EAR is declared invalid and eyes become UNOBSERVABLE. | `test_scenario_7_brief_look_down_is_not_sleep` |
| **Driver turning head** (mirror check, junction) | FP | Beyond 35° yaw EAR is geometrically invalid (the eye is foreshortened), so eyes are UNOBSERVABLE, not CLOSED. Looking away is reported but not counted as drowsiness. | `test_driver_turning_head_to_check_mirror_is_not_drowsiness`, `test_ear_drops_when_yawed_far_*` |
| **Talking, laughing, singing** | FP | A yawn needs a wide inner-lip opening (MAR ≥ 0.5) sustained for ≥ 2 s. Speech is narrower and shorter; laughs are short. | `test_talking_and_laughing_stay_alert`, `test_talking_is_not_yawning` |
| **Single yawn** | FP | Yawning alone is capped at severity 1 (MODERATE). It starts counting at 2 yawns in 5 minutes. | `test_scenario_8_single_yawn_is_not_critical`, `test_yawning_alone_never_exceeds_moderate` |
| **Facial expressions / squinting in sun** | FP | Partial closure stays above the closed threshold (65% of the driver's own open-eye EAR). A long squint can still read as closure; this is a known limitation (see below). | `test_threshold_is_relative_to_driver` |
| **Naturally narrow eyes** | FP | EAR thresholds are relative to a per-driver calibrated baseline, not a population number. | `test_threshold_is_relative_to_driver` |
| **Sunglasses** | FN (eyes unseen) | Eye-region brightness much darker than the face makes eyes UNOBSERVABLE. The state is **MONITORING** (never ALERT). Head nods still drive risk; repeated nodding can reach HIGH. | `test_sunglasses_make_eyes_unobservable`, `test_sunglasses_driver_is_monitoring_not_alert` |
| **Clear glasses / reflections** | FP/FN | Landmarks usually still track. Glare lowers frame quality; left/right EAR asymmetry lowers confidence. | `test_one_eye_occluded_flags_asymmetry` |
| **Low light / night** | FN | Face brightness and sharpness checks reduce quality and confidence. An RGB webcam at night is a real limitation; fleet units use IR illumination. | `test_low_light_reduces_quality` |
| **Camera covered / disconnected / frozen** | FN | Black or flat frames, missing frames and frozen (identical) frames → CAMERA_UNAVAILABLE after 2 s. This raises a system alert and is **never** ALERT. | `test_scenario_6_*`, `test_camera_*` |
| **Face temporarily lost** (detector flicker) | FP (state flapping) | < 1 s: hold the last state with decaying confidence. ≥ 1 s: UNKNOWN. | `test_face_loss_within_grace_*`, `test_scenario_5_*` |
| **Driver slumps out of view** | **FN** | If the face or eyes disappear within 3 s of an eye closure, nod or head drop while an elevated state is active, the elevated state is **held**, not reset to UNKNOWN. | `test_slump_out_of_view_holds_elevated_state` |
| **Microsleep without head movement** | FN | A 3 s continuous closure is CRITICAL on its own. A 2 s closure with the head down is CRITICAL. | `test_microsleep_without_head_movement_is_critical` |
| **Gradual drowsiness** (no single dramatic event) | FN | PERCLOS over 60 s plus slow-blink and long-closure counts accumulate evidence. | `test_gradual_drowsiness_via_perclos` |
| **Low frame rate** (laptop under load) | FP/FN | All durations use timestamps, never frame counts. | `test_critical_detection_is_frame_rate_independent` |
| **Driver already tired at start** | FN | The calibration baseline uses the 85th percentile of EAR and is clamped to a plausible range. | `test_calibration_clamps_implausible_baseline` |
| **Noisy oscillating risk** | FP (alert spam) | Escalation needs persistence. De-escalation needs 5-10 s holds and steps down one level at a time. Alerts have per-type cooldowns. | `test_hysteresis_no_flapping_*`, `test_alert_cooldown_prevents_spam` |

## Known limitations (not solved here)

* **Sustained squinting** (low sun, smiling hard) can look like partial closure for a long time.
* **Microsleeps with eyes open** exist and cannot be seen by an eyelid-based system.
* **Sunglasses plus a still head** leaves almost no drowsiness signal. The system honestly reports MONITORING, but it cannot detect drowsiness.
* **Sustained head-down with eyes hidden** (sunglasses) counts as drowsiness evidence up to WARNING. That is a deliberately conservative choice that can cause nuisance warnings.
* **Unvalidated thresholds.** All thresholds are literature-informed defaults until tuned and validated on labelled data (see [EVALUATION.md](EVALUATION.md)).
