"""
Background system monitoring for CPU/RAM/GPU usage.
"""
import time
import threading

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None


def start_sys_monitor(wb_run, interval: int) -> None:
    """Start a background thread that logs system stats to W&B run."""
    if wb_run is None:
        return

    gpu = pynvml.nvmlDeviceGetHandleByIndex(0) if pynvml else None

    def loop():
        while True:
            payload = {
                'sys/cpu': psutil.cpu_percent(),
                'sys/ram': psutil.virtual_memory().percent,
            }
            if gpu:
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                payload['sys/gpu'] = util.gpu
                payload['sys/vram'] = mem.used / mem.total * 100
            wb_run.log(payload)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f'[✓] System monitor started (interval={interval}s)')
