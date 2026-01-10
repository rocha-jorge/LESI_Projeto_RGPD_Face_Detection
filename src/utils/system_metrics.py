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

def log_resources_snapshot(
    prefix: str = "System resource usage",
    sample_interval: float = 0.1,
) -> None:
    """Log a formatted snapshot of system and current-process usage as two lines.

    Line 1: "System resource usage | CPU 41.1% | RAM 495 MB/14942 MB (3.3%)"
    Line 2: "Agent resource usage | CPU 2.4% | RAM 512 MB"
    """
    snap = get_resources_snapshot(sample_interval)
    vm = psutil.virtual_memory()
    used_mb = vm.used / (1024 * 1024)
    total_mb = vm.total / (1024 * 1024)
    import logging
    logging.info(
        f"{prefix} | CPU {snap['sys_cpu']:.1f}% | RAM {used_mb:.0f} MB/{total_mb:.0f} MB ({vm.percent:.1f}%)"
    )
    logging.info(
        f"Agent resource usage | CPU {snap['proc_cpu']:.1f}% | RAM {snap['proc_ram_mb']:.0f} MB"
    )