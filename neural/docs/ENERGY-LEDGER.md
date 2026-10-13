# ENERGY LEDGER (Stage 5b)

Generated automatically by `neural/powerlog.py`. Do not hand-edit.

Idle baseline: **4.07 W** GPU over 60.0 s (2026-10-12T00:14:49) — used for idle-subtracted ("net") energy.

CPU package energy comes from Intel RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`), which this kernel exposes to root only. It became readable during the night, so rows after that carry `cpu_wh` and earlier rows do not (currently RAPL is readable); the sampler re-checks readability at the start of every run. Battery counters are sampled but read 0 W while on AC (this laptop is on mains), and whole-wall power is unavailable, so screen, fans, SSD and RAM are not captured.


## All measured runs since the Stage 5b logger was switched on

| run_id | purpose | parent | duration (s) | GPU Wh | GPU net Wh | mean W | peak W | mean °C | CPU Wh | throttle |
|---|---|---|---|---|---|---|---|---|---|---|
| det_v1 | train |  | 14.5 | 0.0533 | 0.0369 | 13.51 | 30.55 | 48.1 | 0.204 | - |
| det_v2 | train | det_v1 | 16.0 | 0.0735 | 0.0554 | 16.13 | 31.95 | 48.9 | 0.2456 | - |
| det_v3 | train | det_v2 | 18.6 | 0.1609 | 0.1399 | 31.45 | 79.22 | 51.4 | 0.3051 | - |
| det_v4 | train | det_v2 | 10.5 | 0.0434 | 0.0315 | 17.19 | 71.75 | 49.2 | 0.1131 | - |
| validate_judges | eval |  | 1.2 | 0.0011 | 0.0 | 4.03 | 4.03 | 43.0 | 0.005 | - |
| diag_motion | eval |  | 48.4 | 0.6276 | 0.5729 | 47.49 | 51.73 | 61.7 | 0.6951 | - |
| validate_judges | eval |  | 330.5 | 4.2848 | 3.9112 | 46.69 | 54.44 | 60.2 | 3.6843 | - |
| score_v0 | eval | flow96 | 727.7 | 10.4223 | 9.5996 | 51.57 | 125.27 | 64.9 | 9.4813 | sw_power_cap |
| score_v0 | eval |  | 730.1 | 10.4455 | 9.6201 | 51.5 | 96.89 | 64.9 | 9.5296 | sw_power_cap |
| flow96c | train | flow96 | 1.3 | 0.0026 | 0.0011 | 9.22 | 14.21 | 49.0 | 0.0052 | - |
| flow96c | train | flow96 | 692.2 | 18.6782 | 17.8956 | 97.16 | 106.32 | 84.7 | 8.7972 | sw_power_cap,sw_thermal_slowdown |
| flow96c | train | flow96 | 1179.3 | 32.1349 | 30.8016 | 98.13 | 104.59 | 86.0 | 13.2542 | sw_power_cap,sw_thermal_slowdown |
| score_v0 | eval | flow96 | 750.1 | 10.7059 | 9.8579 | 51.37 | 79.88 | 64.8 | 7.6606 | - |
| score_v0b | eval |  | 751.8 | 10.716 | 9.866 | 51.3 | 80.07 | 64.8 | 7.7501 | - |
| score_flow96c | eval | flow96c | 1153.7 | 24.317 | 23.0127 | 75.84 | 105.19 | 73.4 | 11.0799 | sw_power_cap |
| score_flow96c | eval |  | 1155.4 | 24.2847 | 22.9785 | 75.68 | 105.19 | 73.5 | 11.0879 | sw_power_cap |
| reveals_smoke | eval |  | 19.3 | 0.4457 | 0.4239 | 84.27 | 89.41 | 86.1 | 0.5124 | sw_power_cap,sw_thermal_slowdown |
| flow96d | train |  | 2407.3 | 68.0167 | 65.2951 | 101.68 | 105.44 | 85.7 | 23.4744 | sw_power_cap,sw_thermal_slowdown |
| score_flow96d | eval |  | 2.0 | 0.0025 | 0.0002 | 9.06 | 14.03 | 49.0 | 0.0053 | - |
| reveals_all | eval |  | 3.8 | 0.0411 | 0.0368 | 47.54 | 65.52 | 59.0 | 0.0324 | - |
| reveals_all | eval |  | 11.8 | 0.1066 | 0.0933 | 34.26 | 65.52 | 54.9 | 0.1335 | - |
| score_flow96d | eval |  | 1.9 | 0.0012 | 0.0 | 4.2 | 4.29 | 48.0 | 0.0058 | - |
| score_flow96d | eval | flow96d | 1173.2 | 24.3511 | 23.0247 | 74.72 | 104.81 | 73.4 | 11.4223 | sw_power_cap |
| score_flow96d | eval |  | 1174.9 | 24.4027 | 23.0744 | 74.74 | 105.13 | 73.4 | 11.5702 | sw_power_cap |
| score_flow96c_confirm | eval | flow96c | 1159.2 | 24.3178 | 23.0073 | 75.51 | 104.93 | 73.6 | 11.3493 | sw_power_cap |
| confirm_flow96c | eval |  | 1161.0 | 24.316 | 23.0034 | 75.38 | 104.82 | 73.6 | 11.2915 | sw_power_cap |
| score_flow96c_test | eval | flow96c | 930.1 | 19.9068 | 18.8553 | 77.0 | 111.62 | 73.8 | 8.7879 | sw_power_cap |
| test_flow96c | eval |  | 931.6 | 19.929 | 18.8758 | 76.97 | 111.62 | 73.8 | 8.7699 | sw_power_cap |
| review_images | eval |  | 9.5 | 0.1038 | 0.0931 | 44.16 | 69.46 | 64.1 | 0.0834 | - |
| score_v0_confirm | eval | flow96 | 762.1 | 10.6838 | 9.8222 | 50.42 | 77.84 | 63.0 | 8.1676 | - |
| confirm_v0 | eval |  | 763.9 | 10.7041 | 9.8405 | 50.5 | 79.9 | 63.0 | 8.2023 | - |
| play_flow96 | play | player session, model flow96 | 22.3 | 0.1088 | 0.0836 | 18.43 | 18.85 | 49.4 | None | - |
| play_flow96d | play | player session, model flow96d | 12.7 | 0.0696 | 0.0552 | 21.31 | 22.96 | 51.0 | None | - |
| play_flow96c | play | player session, model flow96c | 13.0 | 0.0756 | 0.0609 | 21.19 | 21.8 | 49.0 | None | - |
| play_flow96 | play | player session, model flow96 | 23.6 | 0.0869 | 0.0602 | 13.74 | 15.61 | 42.3 | None | sw_power_cap |
| play_flow96 | play | player session, model flow96 | 14.5 | 0.0536 | 0.0372 | 13.55 | 15.38 | 42.2 | None | sw_power_cap |
| play_flow96c | play | player session, model flow96c | 5.9 | 0.0232 | 0.0165 | 16.58 | 17.9 | 44.2 | None | sw_power_cap |
| play_flow96c | play | player session, model flow96c | 23.8 | 0.1055 | 0.0786 | 16.1 | 16.68 | 45.2 | None | sw_power_cap |
| play_flow96d | play | player session, model flow96d | 11.4 | 0.0521 | 0.0392 | 16.42 | 17.53 | 43.0 | None | sw_power_cap |
| play_flow96d | play | player session, model flow96d | 10.0 | 0.0431 | 0.0318 | 16.64 | 17.5 | 44.6 | None | sw_power_cap |

### Totals

- Runs recorded: **40**
- GPU energy, gross: **374.8990 Wh**
- of which wasted on aborted attempts: **18.6808 Wh** over 0.19 h (flow96c, flow96c) — measured, real energy, but it produced no checkpoint
- GPU energy, idle-subtracted: **354.2901 Wh**
- CPU energy: 187.7064 Wh (0 if unavailable)
- Wall-clock measured: **5.06 h**
- train: 8 runs, 119.163 Wh GPU, 1.21 h
- eval: 23 runs, 255.117 Wh GPU, 3.82 h
- play: 9 runs, 0.618 Wh GPU, 0.04 h

### Winning model lineage (flow96c)

Runs contributing to the final checkpoint through `parent_run` links: flow96c

- Training energy of the lineage: 32.1349 Wh GPU (0.33 h)
- Idle-subtracted: 30.8016 Wh

## Notes

- Every heavy job is wrapped in `memcap.sh` and `systemd-inhibit --what=sleep:idle:handle-lid-switch`; the sampler adds ~0.1% overhead.
- Throttle column lists NVML throttle-reason bits observed during the run (idle/clock-setting bits filtered).

