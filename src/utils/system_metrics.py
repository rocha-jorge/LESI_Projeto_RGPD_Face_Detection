import psutil

def get_process_usage(sample_interval: float = 0.1) -> tuple[float, float]:
    """Return (cpu_percent, memory_mb) for the current process.

    - cpu_percent: Percentage of CPU used by this process (0-100)
    - memory_mb: Resident memory (RSS) in megabytes
    """
    proc = psutil.Process()
    cpu = proc.cpu_percent(interval=sample_interval)
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    return cpu, mem_mb

def get_system_usage(sample_interval: float = 0.1) -> tuple[float, float]:
    """Return (cpu_percent, memory_mb_used) for the whole system."""
    cpu = psutil.cpu_percent(interval=sample_interval)
    mem_mb = psutil.virtual_memory().used / (1024 * 1024)
    return cpu, mem_mb