"""Reads the GPU's energy, power, clock speed, temperature and memory using NVML.

Energy is measured two ways for every timed run:
  1. NVML's built-in energy counter (in millijoules). We read it right before and right
     after the run. This is the main number.
  2. Power readings taken about 10 times a second, added up over time. This is a cross-check,
     and the fallback if a GPU has no energy counter.

This only measures the GPU board, not the CPU, RAM or the wall socket.
"""
from __future__ import annotations

import threading
import time

import pynvml

def trapezoid_energy_j(times_s, powers_w):
    """Energy in joules from power readings (W) at known times (s), one trapezoid per pair."""
    return sum(0.5 * (powers_w[i] + powers_w[i - 1]) * (times_s[i] - times_s[i - 1]) for i in range(1, len(times_s)))

# Bits in NVML's "throttle reasons" number. Heat and hardware slowdowns make a run invalid.
# The normal power-limit throttling on a 4090 under load is only logged, not flagged.
HW_SLOWDOWN = 0x8
SW_THERMAL = 0x20
HW_THERMAL = 0x40
HW_POWER_BRAKE = 0x80
BAD_THROTTLE_MASK = HW_SLOWDOWN | SW_THERMAL | HW_THERMAL | HW_POWER_BRAKE


def _s(x):
    """Older pynvml returns bytes for strings, newer returns str."""
    return x.decode() if isinstance(x, bytes) else x


class GpuMonitor:
    def __init__(self, index: int = 0, interval_s: float = 0.1):
        pynvml.nvmlInit()
        self.index = index
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.name = _s(pynvml.nvmlDeviceGetName(self.h))
        self.interval = interval_s
        self._samples: list[tuple] = []  # (time, power_w, sm_mhz, temp_c, mem_used_mib, throttle_bits)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.has_energy_counter = self._probe_energy()

    # ---- reading the GPU ---------------------------------------------------------
    def _probe_energy(self) -> bool:
        """Check whether this GPU has the energy counter (Volta and newer do)."""
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            return True
        except pynvml.NVMLError:
            return False

    def energy_mj(self) -> int | None:
        if not self.has_energy_counter:
            return None
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)

    def _throttle(self) -> int:
        # The function was renamed in newer NVML versions, so try both names.
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(pynvml, fn, None)
            if f:
                try:
                    return int(f(self.h))
                except pynvml.NVMLError:
                    return 0
        return 0

    def _read(self) -> tuple:
        def safe(fn, default=0.0):
            try:
                return fn()
            except pynvml.NVMLError:
                return default

        power = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(self.h)) / 1000.0   # mW -> W
        sm = safe(lambda: pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM))
        temp = safe(lambda: pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU))
        mem = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(self.h).used) / 2**20  # bytes -> MiB
        return (time.perf_counter(), power, sm, temp, mem, self._throttle())

    def _append_sample(self) -> int:
        s = self._read()
        with self._lock:
            self._samples.append(s)
            return len(self._samples) - 1

    # ---- background sampling -----------------------------------------------------
    def start(self):
        """Start a background thread that records a reading every `interval` seconds."""
        if self._thread:
            return
        self._stop.clear()
        self._append_sample()

        def loop():
            while not self._stop.wait(self.interval):
                self._append_sample()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def mem_used_mib(self) -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20

    # ---- timed windows -----------------------------------------------------------
    def begin_window(self) -> dict:
        """Mark the start of a timed run. Pass the result to end_window()."""
        e0 = self.energy_mj()
        t0 = time.perf_counter()
        i0 = self._append_sample()
        return {"t0": t0, "e0": e0, "i0": i0}

    def end_window(self, w: dict) -> dict:
        """Finish a timed run and return its energy, power, clocks, temperature and memory."""
        e1 = self.energy_mj()
        t1 = time.perf_counter()
        self._append_sample()
        with self._lock:
            samples = self._samples[w["i0"]:]
        ts = [s[0] for s in samples]
        pw = [s[1] for s in samples]
        dur = t1 - w["t0"]
        e_int = trapezoid_energy_j(ts, pw)
        e_cnt = (e1 - w["e0"]) / 1000.0 if (e1 is not None and w["e0"] is not None) else None
        throttle = 0
        for s in samples:
            throttle |= s[5]
        return {
            "duration_s": dur,
            "energy_counter_j": e_cnt,
            "energy_integrated_j": e_int,
            "energy_j": e_cnt if e_cnt is not None else e_int,   # prefer the counter
            "energy_source": "nvml_counter" if e_cnt is not None else "integrated_power",
            "power_mean_w": (e_cnt if e_cnt is not None else e_int) / dur if dur > 0 else float("nan"),
            "power_max_w": max(pw),
            "sm_clock_mean_mhz": sum(s[2] for s in samples) / len(samples),
            "sm_clock_min_mhz": min(s[2] for s in samples),
            "temp_max_c": max(s[3] for s in samples),
            "mem_used_peak_mib": max(s[4] for s in samples),
            "throttle_mask": throttle,
            "thermal_throttled": bool(throttle & BAD_THROTTLE_MASK),
            "n_power_samples": len(samples),
        }

    def measure_idle_power_w(self, seconds: float) -> float:
        """Average power while the GPU sits idle."""
        w = self.begin_window()
        time.sleep(seconds)
        r = self.end_window(w)
        return r["power_mean_w"]

    # ---- GPU details for the log -------------------------------------------------
    def info(self) -> dict:
        d = {"gpu_name": self.name, "energy_counter": self.has_energy_counter}
        try:
            d["driver"] = _s(pynvml.nvmlSystemGetDriverVersion())
        except pynvml.NVMLError:
            pass
        try:
            d["mem_total_mib"] = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / 2**20
            d["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000.0
        except pynvml.NVMLError:
            pass
        try:
            cur, _pend = pynvml.nvmlDeviceGetMigMode(self.h)
            d["mig_enabled"] = bool(cur)
        except pynvml.NVMLError:
            d["mig_enabled"] = False  # this GPU doesn't support MIG, so it can't be a MIG slice
        return d
