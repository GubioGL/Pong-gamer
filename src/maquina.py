"""Description of the computer and software a measurement ran on (for the Supplementary Material)."""
from __future__ import annotations

import os
import platform
from importlib import metadata

PACKAGES = ("jax", "jaxlib", "pennylane", "optax", "numpy", "pygame", "cvjax")


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        return None


def _os_name():
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME"):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def machine_info():
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": None,                      # all simulations run on CPU
        "os": _os_name(),
        "kernel": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "packages": versions,
    }
