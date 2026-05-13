
import os
import gc
import ctypes
from ctypes import wintypes
import sys
from typing import Optional, Dict

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

def get_process_rss_mb() -> float:
    """
    Obtiene la RAM residente (RSS) del proceso actual en MB (Windows).
    """
    if sys.platform != "win32":
        return float("nan")
        
    try:
        # Intentar kernel32 primero (Windows 7+)
        try:
            GetProcessMemoryInfo = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        except AttributeError:
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        
        if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024 * 1024)
    except Exception:
        pass
        
    # Fallback final: wmic (lento pero funciona en Windows sin psutil)
    try:
        pid = os.getpid()
        cmd = f"wmic process where processid={pid} get WorkingSetSize /format:list"
        import subprocess
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.splitlines():
            if "WorkingSetSize" in line:
                return float(line.split("=")[1]) / (1024 * 1024)
    except Exception:
        pass
        
    return float("nan")

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("sullAvailExtendedVirtual", ctypes.c_uint64),
    ]

def get_system_available_mb() -> float:
    """
    Obtiene la RAM disponible del sistema en MB (Windows fallback).
    """
    if sys.platform != "win32":
        return float("nan")
        
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / (1024 * 1024)
    except Exception as e:
        print(f"[WARNING] Error en get_system_available_mb: {e}")
        
    return float("nan")

class MemoryGuard:
    """
    Context manager que monitorea la RAM del proceso y aborta si se excede el budget.
    """
    def __init__(self, budget_mb: int, label: str = "MemoryGuard", abort_on_breach: bool = True):
        self.budget_mb = budget_mb
        self.label = label
        self.abort_on_breach = abort_on_breach
        self.initial_rss = 0.0
        self.peak_rss = 0.0

    def __enter__(self):
        self.initial_rss = get_process_rss_mb()
        self.peak_rss = self.initial_rss
        return self

    def check(self):
        current_rss = get_process_rss_mb()
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss
            
        if current_rss > self.budget_mb:
            msg = f"[{self.label}] RAM BREACH: {current_rss:.2f} MB > Budget {self.budget_mb} MB"
            if self.abort_on_breach:
                raise MemoryError(msg)
            else:
                print(f"[WARNING] {msg}. Ejecutando safe_collect().")
                safe_collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.check()
        delta = self.peak_rss - self.initial_rss
        print(f"[{self.label}] Exit. Peak: {self.peak_rss:.2f} MB, Delta: {delta:.2f} MB")
        safe_collect()

def safe_collect() -> Dict[str, float]:
    """
    Fuerza recolección de basura y retorna estadísticas.
    """
    rss_before = get_process_rss_mb()
    n_collected = gc.collect()
    rss_after = get_process_rss_mb()
    
    return {
        "n_collected": float(n_collected),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "freed_mb": rss_before - rss_after if not (isinstance(rss_before, complex) or rss_before != rss_before) else 0.0
    }
