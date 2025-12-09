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

def get_resources_snapshot(sample_interval: float = 0.1) -> dict:
    """Return a dict snapshot of process and system resource usage.

    Keys: proc_cpu, proc_ram_mb, sys_cpu, sys_ram_mb
    """
    p_cpu, p_mem = get_process_usage(sample_interval)
    s_cpu, s_mem = get_system_usage(sample_interval)
    return {
        "proc_cpu": float(p_cpu),
        "proc_ram_mb": float(p_mem),
        "sys_cpu": float(s_cpu),
        "sys_ram_mb": float(s_mem),
    }

def log_resources_snapshot(prefix: str = "Batch resources", sample_interval: float = 0.1) -> None:
    """Log a single formatted snapshot of process/system usage via logging.info.

    Example output:
    "Batch resources | Proc CPU: 0.0% | Proc RAM: 225.3 MB | Sys CPU: 1.3% | Sys RAM: 13330 MB"
    """
    snap = get_resources_snapshot(sample_interval)
    import logging
    logging.info(
        f"{prefix} | Proc CPU: {snap['proc_cpu']:.1f}% | Proc RAM: {snap['proc_ram_mb']:.1f} MB | Sys CPU: {snap['sys_cpu']:.1f}% | Sys RAM: {snap['sys_ram_mb']:.0f} MB"
    )