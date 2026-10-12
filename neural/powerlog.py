"""Energy accounting for Stage 5b.

Samples GPU power (NVML) at ~1 Hz in a background thread, integrates with the
trapezoid rule to Wh, and writes one record per run to docs/ENERGY-LEDGER.json
plus a readable docs/ENERGY-LEDGER.md.

Usage as a wrapper (composes with memcap.sh):

    ./memcap.sh --mem 14G -- .venv/bin/python powerlog.py \
        --run-id flow96c --purpose train --parent-run flow96 -- \
        .venv/bin/python train_flow.py --steps 8000

Usage from code:

    from powerlog import energy_run
    with energy_run("g2_x", purpose="eval", parent_run="flow96"):
        ...

CPU package energy (Intel RAPL) is root-readable only on this kernel.  The
sampler re-checks readability at the start of every run, so making those files
readable later (with sudo) is picked up by later runs with no code change.
Battery counters are sampled but are meaningless while on AC (this machine
reports 0 W while plugged in); they are recorded as available-or-not, not
mixed into the GPU/CPU totals.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_JSON = os.path.join(HERE, "docs", "ENERGY-LEDGER.json")
LEDGER_MD = os.path.join(HERE, "docs", "ENERGY-LEDGER.md")

RAPL_GLOB = "/sys/class/powercap/intel-rapl:0/energy_uj"
BAT_POWER = "/sys/class/power_supply/BAT0/power_now"
BAT_STATUS = "/sys/class/power_supply/BAT0/status"

# NVML throttle reason bits -> human names (nvmlClocksThrottleReason*)
THROTTLE_BITS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}


# ---------------------------------------------------------------- integrator


def integrate_wh(times_s, values_w):
    """Trapezoid integration of a power trace to watt-hours.

    times_s: monotonically increasing timestamps in seconds.
    values_w: power samples in watts (same length).
    """
    if len(times_s) < 2:
        return 0.0
    wh = 0.0
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        if dt <= 0:
            continue
        wh += 0.5 * (values_w[i] + values_w[i - 1]) * dt
    return wh / 3600.0


def mean_of(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------- samplers


class _Nvml:
    """Thin NVML wrapper; returns None instead of raising when unavailable."""

    def __init__(self):
        self.ok = False
        self.handle = None
        try:
            import pynvml  # nvidia-ml-py

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:
            self.pynvml = None

    def sample(self):
        if not self.ok:
            return {}
        p = self.pynvml
        out = {}
        try:
            out["gpu_w"] = p.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            pass
        try:
            out["gpu_temp_c"] = float(p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            out["gpu_sm_mhz"] = float(p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_SM))
        except Exception:
            pass
        try:
            out["gpu_mem_used_mb"] = p.nvmlDeviceGetMemoryInfo(self.handle).used / (1024.0 * 1024.0)
        except Exception:
            pass
        try:
            out["throttle_mask"] = int(p.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle))
        except Exception:
            pass
        return out


def _read_int(path):
    try:
        with open(path, "r") as fh:
            return int(fh.read().strip())
    except Exception:
        return None


class _Rapl:
    """Intel RAPL package energy counter. Root-readable only on this kernel."""

    def __init__(self):
        self.path = RAPL_GLOB
        self.ok = os.access(self.path, os.R_OK)
        self._last = None
        self._max_range = _read_int("/sys/class/powercap/intel-rapl:0/max_energy_range_uj") or 262143328850
        if self.ok:
            self._last = _read_int(self.path)

    def sample_w(self):
        """Instantaneous package power in W, from consecutive energy reads."""
        if not self.ok:
            return None
        now = _read_int(self.path)
        if now is None:
            return None
        prev, self._last = self._last, now
        if prev is None:
            return None
        d = now - prev
        if d < 0:  # counter wrapped
            d += self._max_range
        return d / 1e6  # uJ over the (1 s) sample interval -> W


class _Battery:
    def __init__(self):
        self.ok = os.access(BAT_POWER, os.R_OK)

    def sample(self):
        if not self.ok:
            return {}
        w = _read_int(BAT_POWER)
        status = None
        try:
            with open(BAT_STATUS) as fh:
                status = fh.read().strip()
        except Exception:
            pass
        return {"bat_w": (w / 1e6 if w else None), "bat_status": status}


# ---------------------------------------------------------------- logger


class PowerLogger:
    """Background 1 Hz sampler.  Negligible overhead (<0.1% of a training run)."""

    def __init__(self, interval=1.0, sample_cpu=True):
        self.interval = interval
        self.nvml = _Nvml()
        self.rapl = _Rapl() if sample_cpu else None
        self.battery = _Battery()
        self._t0 = None
        self._t = []
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    @property
    def cpu_available(self):
        return bool(self.rapl and self.rapl.ok)

    def _loop(self):
        while not self._stop.is_set():
            rec = {"t": time.time()}
            rec.update(self.nvml.sample())
            if self.rapl:
                rec["cpu_w"] = self.rapl.sample_w()
            rec.update(self.battery.sample())
            self._samples.append(rec)
            self._stop.wait(self.interval)

    def start(self):
        self._t0 = time.time()
        self._t = [self._t0]
        self._samples = []
        # prime the RAPL delta
        if self.rapl:
            self.rapl.sample_w()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._t0 is None:      # never started
            self._t0 = time.time()
            return self.summary()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        return self.summary()

    def summary(self):
        end = time.time()
        s = list(self._samples)
        times = [x["t"] for x in s]
        res = {
            "start": _dt.datetime.fromtimestamp(self._t0 or end).isoformat(timespec="seconds"),
            "end": _dt.datetime.fromtimestamp(end).isoformat(timespec="seconds"),
            "duration_s": round(end - (self._t0 or end), 1),
            "samples": len(s),
            "gpu_wh": None,
            "cpu_wh": None,
            "mean_gpu_w": None,
            "peak_gpu_w": None,
            "mean_cpu_w": None,
            "peak_cpu_w": None,
            "mean_gpu_temp_c": None,
            "peak_gpu_temp_c": None,
            "gpu_energy_j": None,
            "n_frames": None,
            "parent_run": None,
            "cpu_available": self.cpu_available,
        }
        gw = [x.get("gpu_w") for x in s]
        gw_v = [v for v in gw if v is not None]
        if len(gw_v) >= 2:
            ts = [x["t"] for x in s if x.get("gpu_w") is not None]
            res["gpu_wh"] = round(integrate_wh(ts, [v for v in gw if v is not None]), 4)
            res["gpu_energy_j"] = round(res["gpu_wh"] * 3600.0, 1)
            res["mean_gpu_w"] = round(mean_of(gw_v), 2)
            res["peak_gpu_w"] = round(max(gw_v), 2)
        cw = [x.get("cpu_w") for x in s]
        cw_v = [v for v in cw if v is not None]
        if len(cw_v) >= 2:
            ts = [x["t"] for x in s if x.get("cpu_w") is not None]
            res["cpu_wh"] = round(integrate_wh(ts, cw_v), 4)
            res["mean_cpu_w"] = round(mean_of(cw_v), 2)
            res["peak_cpu_w"] = round(max(cw_v), 2)
        temps = [x.get("gpu_temp_c") for x in s if x.get("gpu_temp_c") is not None]
        if temps:
            res["mean_gpu_temp_c"] = round(mean_of(temps), 1)
            res["peak_gpu_temp_c"] = round(max(temps), 1)
        masks = [x.get("throttle_mask") for x in s if x.get("throttle_mask") is not None]
        seen = set()
        for m in masks:
            for bit, name in THROTTLE_BITS.items():
                if m & bit:
                    seen.add(name)
        seen.discard("gpu_idle")
        seen.discard("applications_clocks_setting")
        res["throttle_events"] = sorted(seen)
        bat = [x.get("bat_w") for x in s if x.get("bat_w")]
        res["battery_measured"] = bool(bat)
        if bat:
            res["mean_bat_w"] = round(mean_of(bat), 2)
        return res


class energy_run:
    """Context manager: log one job to the ledger.

    with energy_run("flow96c", "train", parent_run="flow96", config="..."):
        train()
    """

    def __init__(self, run_id, purpose, parent_run=None, notes=None, config=None, script=None, n_frames=None):
        self.run_id = run_id
        self.purpose = purpose
        self.parent_run = parent_run
        self.notes = notes
        self.config = config
        self.script = script or " ".join(sys.argv[:3])
        self.n_frames = n_frames
        self.logger = None
        self.record = None

    def __enter__(self):
        self.logger = PowerLogger().start()
        print(f"[powerlog] run '{self.run_id}' ({self.purpose}) started; cpu_energy="
              f"{'rapl' if self.logger.cpu_available else 'unavailable'}", flush=True)
        return self

    def set_frames(self, n):
        self.n_frames = n

    def __exit__(self, *exc):
        self.record = self.logger.stop()
        self.record.update(
            run_id=self.run_id,
            purpose=self.purpose,
            parent_run=self.parent_run,
            notes=self.notes,
            config=self.config,
            script=self.script,
            host=platform.node(),
        )
        if self.n_frames:
            self.record["n_frames"] = self.n_frames
            dur = self.record.get("duration_s") or 0
            if dur > 0:
                self.record["j_per_frame"] = round((self.record.get("gpu_energy_j") or 0) / self.n_frames, 4)
                self.record["fps"] = round(self.n_frames / dur, 2)
        if exc and exc[0] is not None:
            self.record["crashed"] = repr(exc[1])
        append_record(self.record)
        print(f"[powerlog] {self.run_id}: {self.record['duration_s']}s  "
              f"gpu {self.record['gpu_wh']} Wh ({self.record['mean_gpu_w']} W mean, "
              f"{self.record['peak_gpu_w']} W peak)  cpu {self.record['cpu_wh']} Wh  "
              f"throttle={self.record['throttle_events']}", flush=True)
        return False


# ---------------------------------------------------------------- ledger


def _load_ledger():
    if not os.path.exists(LEDGER_JSON):
        return {"records": [], "idle_baseline": None}
    try:
        with open(LEDGER_JSON) as fh:
            data = json.load(fh)
        data.setdefault("records", [])
        data.setdefault("idle_baseline", None)
        return data
    except Exception as exc:  # corrupt ledger should not kill a run
        print(f"[powerlog] ledger unreadable ({exc}); starting a new one", flush=True)
        return {"records": [], "idle_baseline": None}


def append_record(rec):
    if rec.get("purpose") == "idle_baseline":
        data = _load_ledger()
        data["idle_baseline"] = rec
        _save_ledger(data)
        return
    data = _load_ledger()
    data["records"].append(rec)
    _save_ledger(data)


def _save_ledger(data):
    os.makedirs(os.path.dirname(LEDGER_JSON), exist_ok=True)
    tmp = LEDGER_JSON + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, LEDGER_JSON)
    write_markdown(data)


def _net(wh, dur_s, idle_w):
    """Idle-subtracted energy in Wh."""
    if wh is None or idle_w is None or dur_s is None:
        return None
    return max(0.0, wh - idle_w * dur_s / 3600.0)


def write_markdown(data):
    idle = data.get("idle_baseline") or {}
    idle_w = idle.get("mean_gpu_w")
    recs = data["records"]
    phase = [r for r in recs if r.get("purpose") != "idle_baseline"]

    def totals(rows, field):
        vals = [r.get(field) for r in rows if r.get(field) is not None]
        return sum(vals) if vals else 0.0

    lines = []
    lines.append("# ENERGY LEDGER (Stage 5b)\n")
    lines.append("Generated automatically by `neural/powerlog.py`. Do not hand-edit.\n")
    lines.append(
        f"Idle baseline: **{idle_w} W** GPU over {idle.get('duration_s')} s "
        f"({idle.get('start')}) — used for idle-subtracted (\"net\") energy.\n"
    )
    lines.append(
        "CPU package energy (Intel RAPL) is root-readable only on this kernel; "
        "`cpu_wh` is `null` unless the user makes `/sys/class/powercap/intel-rapl*/energy_uj` "
        "readable. Battery counters are sampled but read 0 W while on AC. "
        "Whole-wall power is unavailable: screen, fans, SSD and RAM are not captured.\n"
    )

    def table(rows, title):
        lines.append(f"\n## {title}\n")
        lines.append("| run_id | purpose | parent | duration (s) | GPU Wh | GPU net Wh | mean W | peak W | mean °C | CPU Wh | throttle |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            net = _net(r.get("gpu_wh"), r.get("duration_s"), idle_w)
            lines.append(
                "| {id} | {p} | {par} | {d} | {wh} | {net} | {mw} | {pw} | {t} | {cw} | {th} |".format(
                    id=r.get("run_id"),
                    p=r.get("purpose"),
                    par=r.get("parent_run") or "",
                    d=r.get("duration_s"),
                    wh=r.get("gpu_wh"),
                    net=None if net is None else round(net, 4),
                    mw=r.get("mean_gpu_w"),
                    pw=r.get("peak_gpu_w"),
                    t=r.get("mean_gpu_temp_c"),
                    cw=r.get("cpu_wh"),
                    th=",".join(r.get("throttle_events") or []) or "-",
                )
            )

    table(phase, "All measured runs since the Stage 5b logger was switched on")

    lines.append("\n### Totals\n")
    lines.append(f"- Runs recorded: **{len(phase)}**")
    lines.append(f"- GPU energy, gross: **{totals(phase, 'gpu_wh'):.4f} Wh**")
    if idle_w is not None:
        net = sum(_net(r.get("gpu_wh"), r.get("duration_s"), idle_w) or 0.0 for r in phase)
        lines.append(f"- GPU energy, idle-subtracted: **{net:.4f} Wh**")
    lines.append(f"- CPU energy: {totals(phase, 'cpu_wh'):.4f} Wh (0 if unavailable)")
    lines.append(f"- Wall-clock measured: **{sum(r.get('duration_s') or 0 for r in phase) / 3600:.2f} h**")
    for purpose in ("train", "eval", "play", "benchmark"):
        rows = [r for r in phase if r.get("purpose") == purpose]
        if rows:
            lines.append(
                f"- {purpose}: {len(rows)} runs, {totals(rows, 'gpu_wh'):.3f} Wh GPU, "
                f"{sum(r.get('duration_s') or 0 for r in rows) / 3600:.2f} h"
            )

    # lineage: transitive closure over parent_run starting from the winner label
    winner = data.get("winner")
    if winner:
        chain, seen, stack = [], set(), [winner]
        while stack:
            cur = stack.pop()
            if cur in seen or cur is None:
                continue
            seen.add(cur)
            row = next((r for r in recs if r.get("run_id") == cur), None)
            if row:
                chain.append(row)
                stack.append(row.get("parent_run"))
        lines.append(f"\n### Winning model lineage ({winner})\n")
        lines.append("Runs contributing to the final checkpoint through `parent_run` links: "
                     + ", ".join(r.get("run_id") for r in chain) + "\n")
        lines.append(f"- Training energy of the lineage: {totals(chain, 'gpu_wh'):.4f} Wh GPU "
                     f"({sum(r.get('duration_s') or 0 for r in chain) / 3600:.2f} h)")
        if idle_w is not None:
            net = sum(_net(r.get("gpu_wh"), r.get("duration_s"), idle_w) or 0.0 for r in chain)
            lines.append(f"- Idle-subtracted: {net:.4f} Wh")

    lines.append("\n## Notes\n")
    lines.append(
        "- Every heavy job is wrapped in `memcap.sh` and `systemd-inhibit --what=sleep:idle:handle-lid-switch`; "
        "the sampler adds ~0.1% overhead.\n"
        "- Throttle column lists NVML throttle-reason bits observed during the run (idle/clock-setting bits filtered).\n"
    )
    with open(LEDGER_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- CLI


def _cmd_baseline(args):
    logger = PowerLogger().start()
    print(f"[powerlog] idle baseline sampling for {args.seconds}s ...", flush=True)
    time.sleep(args.seconds)
    rec = logger.stop()
    rec.update(run_id="idle_baseline", purpose="idle_baseline",
               notes="60 s idle baseline, nothing else running", script="powerlog.py --baseline")
    append_record(rec)
    print(f"[powerlog] idle baseline: {rec['mean_gpu_w']} W GPU mean, {rec['mean_gpu_temp_c']} °C", flush=True)
    return 0


def _cmd_wrap(args):
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2
    with energy_run(args.run_id, args.purpose, parent_run=args.parent_run,
                    notes=args.notes, config=args.config) as run:
        proc = subprocess.run(cmd)
        try:
            nf = int(os.environ.get("POWERLOG_FRAMES", "0"))
            if nf:
                run.set_frames(nf)
        except ValueError:
            pass
    return proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="measure an idle baseline instead of wrapping a command")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--run-id", default="unnamed")
    ap.add_argument("--purpose", default="eval", choices=["train", "eval", "play", "benchmark", "idle_baseline"])
    ap.add_argument("--parent-run", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--reconcile", action="store_true", help="print ledger totals and their reconciliation")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if args.reconcile:
        data = _load_ledger()
        recs = [r for r in data["records"] if r.get("purpose") != "idle_baseline"]
        wh = sum(r.get("gpu_wh") or 0.0 for r in recs)
        print(f"ledger records: {len(recs)}")
        print(f"sum of per-run gpu_wh: {wh:.4f} Wh")
        print(f"json file: {LEDGER_JSON}")
        print(f"markdown:  {LEDGER_MD}")
        return 0
    if args.baseline:
        return _cmd_baseline(args)
    return _cmd_wrap(args)


if __name__ == "__main__":
    sys.exit(main())
