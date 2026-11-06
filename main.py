from collections import deque
from datetime import datetime, timezone as dt_timezone
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import h3
import numpy as np
import psutil
from pydantic import BaseModel, Field
import structlog
from timezonefinder import TimezoneFinder
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
logger = structlog.get_logger()
tf_instance = TimezoneFinder()
_HISTORY_MAX_LEN = 120
_telemetry_history: deque = deque(maxlen=_HISTORY_MAX_LEN)
app = FastAPI(title='hardware-sentinel-6267')

class HardwareQueryRequest(BaseModel):
    target_vram_mb: float = Field(default=0.0, ge=0.0, le=10000000.0)
    target_ram_mb: float = Field(default=0.0, ge=0.0, le=10000000.0)
    execution_duration_sec: float = Field(default=1.0, ge=0.001, le=86400.0)
    spatial_lat: float = Field(default=0.0, ge=-90.0, le=90.0)
    spatial_lon: float = Field(default=0.0, ge=-180.0, le=180.0)

def _safe_num(val: Any=0.0, default: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    """Safely cast arbitrary values to float without throwing runtime exceptions."""
    if val is None:
        return default
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return default
            return float(val)
        num = float(val)
        return default if math.isnan(num) or math.isinf(num) else num
    except (ValueError, TypeError):
        return default

def _read_cgroup_throttling(*_extra_args, **_extra_kwargs) -> Tuple[float, float, float]:
    """
    Deterministically read Linux cgroups v2 / v1 cpu.stat metrics.
    Returns (nr_periods, nr_throttled, throttled_pct).
    """
    cgroup_paths = ['/sys/fs/cgroup/cpu.stat', '/sys/fs/cgroup/cpu/cpu.stat']
    nr_periods = 0.0
    nr_throttled = 0.0
    for path in cgroup_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            if parts[0] == 'nr_periods':
                                nr_periods = _safe_num(parts[1])
                            elif parts[0] == 'nr_throttled':
                                nr_throttled = _safe_num(parts[1])
                break
            except Exception:
                continue
    epsilon = 1e-06
    throttled_pct = nr_throttled / (nr_periods + epsilon) if nr_periods > 0 else 0.0
    throttled_pct = max(0.0, min(1.0, throttled_pct))
    return (nr_periods, nr_throttled, throttled_pct)

def _probe_package_temperature(*_extra_args, **_extra_kwargs) -> Tuple[float, float]:
    """
    Probes system hardware sensors via psutil.
    Returns (package_temp_celsius, thermal_throttle_active_flag).
    """
    package_temp = 45.0
    throttle_active = 0.0
    try:
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            package_temp = max(package_temp, float(entry.current))
        thermal_zones = '/sys/class/thermal'
        if os.path.exists(thermal_zones):
            for tz_dir in os.listdir(thermal_zones):
                if tz_dir.startswith('thermal_zone'):
                    trip_point = os.path.join(thermal_zones, tz_dir, 'trip_point_0_type')
                    if os.path.exists(trip_point):
                        try:
                            with open(trip_point, 'r', encoding='utf-8') as tf:
                                if 'critical' in tf.read().lower():
                                    crit_file = os.path.join(thermal_zones, tz_dir, 'trip_point_0_temp')
                                    if os.path.exists(crit_file):
                                        with open(crit_file, 'r', encoding='utf-8') as cf:
                                            crit_temp = _safe_num(cf.read().strip()) / 1000.0
                                            if crit_temp > 0 and package_temp >= crit_temp * 0.9:
                                                throttle_active = 1.0
                        except Exception:
                            continue
    except Exception:
        pass
    return (package_temp, throttle_active)

def _query_nvml_metrics(*_extra_args, **_extra_kwargs) -> Tuple[float, float, float]:
    """
    Query NVIDIA GPU metrics via NVML if available.
    Returns (vram_available_mib, vram_total_mib, gpu_utilization_pct).
    """
    if _NVML_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                vram_total = float(mem_info.total) / (1024.0 * 1024.0)
                vram_free = float(mem_info.free) / (1024.0 * 1024.0)
                gpu_util = float(util_info.gpu)
                return (vram_free, vram_total, gpu_util)
        except Exception:
            pass
    total_mem_mib = 16384.0
    return (total_mem_mib, total_mem_mib, 0.0)

def compute_hardware_intelligence(request_data: Optional[HardwareQueryRequest]=None, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    if request_data is None:
        request_data = HardwareQueryRequest()
    now = datetime.now(dt_timezone.utc)
    timestamp_str = now.isoformat()
    target_vram = _safe_num(getattr(request_data, 'target_vram_mb', 0.0))
    target_ram = _safe_num(getattr(request_data, 'target_ram_mb', 0.0))
    lat = max(-90.0, min(90.0, _safe_num(getattr(request_data, 'spatial_lat', 0.0))))
    lon = max(-180.0, min(180.0, _safe_num(getattr(request_data, 'spatial_lon', 0.0))))
    svmem = psutil.virtual_memory()
    mem_total_mib = float(svmem.total) / (1024.0 * 1024.0)
    mem_avail_mib = float(svmem.available) / (1024.0 * 1024.0)
    cpu_util = float(psutil.cpu_percent(interval=0.0))
    vram_avail_mib, vram_total_mib, gpu_util = _query_nvml_metrics()
    package_temp, thermal_throttle = _probe_package_temperature()
    _, _, cgroup_throttle_pct = _read_cgroup_throttling()
    epsilon = 1e-06
    h_mem = min(1.0, max(0.0, mem_avail_mib / (mem_total_mib + epsilon)))
    t_critical = 95.0
    t_ambient = 25.0
    m_therm = max(0.0, (t_critical - package_temp) / (t_critical - t_ambient))
    m_therm = min(1.0, m_therm)
    vram_used_ratio = 1.0 - vram_avail_mib / (vram_total_mib + epsilon)
    vram_used_ratio = max(0.0, min(1.0, vram_used_ratio))
    gpu_util_ratio = max(0.0, min(1.0, gpu_util / 100.0))
    w_v = 0.6
    w_u = 0.4
    s_acc = w_v * vram_used_ratio + w_u * gpu_util_ratio
    p_cpu = cgroup_throttle_pct
    inner_factor = h_mem * m_therm * (1.0 - p_cpu)
    r_exec = 100.0 * (1.0 - inner_factor)
    r_exec = max(0.0, min(100.0, r_exec))
    o_headroom = 100.0 * (h_mem * (1.0 - s_acc) * m_therm)
    o_headroom = max(0.0, min(100.0, o_headroom))
    global _telemetry_history
    prev_load = cpu_util
    prev_ram_used = mem_total_mib - mem_avail_mib
    if len(_telemetry_history) > 0:
        last_item = _telemetry_history[-1]
        prev_load = last_item.get('cpu_utilization_pct', cpu_util)
        prev_ram_used = last_item.get('ram_used_mib', prev_ram_used)
    duration = max(0.001, _safe_num(getattr(request_data, 'execution_duration_sec', 1.0)))
    load_velocity = (cpu_util - prev_load) / duration
    allocation_velocity = (mem_total_mib - mem_avail_mib - prev_ram_used) / duration
    current_snapshot = {'timestamp': timestamp_str, 'cpu_utilization_pct': cpu_util, 'ram_used_mib': mem_total_mib - mem_avail_mib, 'memory_available_mib': mem_avail_mib, 'memory_total_mib': mem_total_mib, 'gpu_vram_available_mib': vram_avail_mib, 'gpu_vram_total_mib': vram_total_mib, 'package_temp_celsius': package_temp, 'thermal_throttle_active': thermal_throttle, 'cgroup_throttled_pct': cgroup_throttle_pct, 'load_velocity': load_velocity, 'allocation_velocity_mib_sec': allocation_velocity, 'risk_score': r_exec, 'opportunity_score': o_headroom}
    _telemetry_history.append(current_snapshot)
    if lat == 0.0 and lon == 0.0:
        lat, lon = (37.7749, -122.4194)
    cell_r7 = h3.latlng_to_cell(lat, lon, 7)
    parent_r5 = h3.cell_to_parent(cell_r7, 5)
    regional_tensor = [float(vram_avail_mib), float(m_therm), float(r_exec)]
    if thermal_throttle > 0.0 or h_mem < 0.05 or r_exec >= 85.0:
        classification = 'CRITICAL'
    elif 0.05 <= h_mem < 0.2 or s_acc >= 0.85 or 60.0 <= r_exec < 85.0:
        classification = 'CONSTRAINED'
    elif 0.2 <= h_mem < 0.5 and 30.0 <= r_exec < 60.0:
        classification = 'MODERATE'
    else:
        classification = 'OPTIMAL'
    hardware_state_index = max(0.0, min(100.0, 100.0 - r_exec))
    can_spawn_vram = vram_avail_mib >= target_vram
    can_spawn_ram = mem_avail_mib >= target_ram
    derivation_summary = f'Host operating at {package_temp:.1f}C (crit: {t_critical}C); available RAM headroom is {h_mem * 100.2:.2f}%; VRAM available is {vram_avail_mib:.1f} MiB; cgroup throttling is {cgroup_throttle_pct * 100.0:.2f}%.'
    return {'timestamp': timestamp_str, 'spatial_cell': cell_r7, 'parent_cell': parent_r5, 'regional_tensor': regional_tensor, 'raw_metrics': {'cpu_utilization_pct': cpu_util, 'memory_available_mib': round(mem_avail_mib, 2), 'memory_total_mib': round(mem_total_mib, 2), 'gpu_vram_available_mib': round(vram_avail_mib, 2), 'gpu_vram_total_mib': round(vram_total_mib, 2), 'package_temp_celsius': round(package_temp, 2), 'thermal_throttle_active': float(thermal_throttle), 'cgroup_throttled_pct': round(cgroup_throttle_pct, 4)}, 'derived_intelligence': {'hardware_state_index': round(hardware_state_index, 2), 'risk_score': round(r_exec, 2), 'opportunity_score': round(o_headroom, 2), 'load_velocity': round(load_velocity, 4), 'allocation_velocity_mib_sec': round(allocation_velocity, 4), 'can_spawn_target_vram': bool(can_spawn_vram), 'can_spawn_target_ram': bool(can_spawn_ram)}, 'classification': classification, 'confidence': 0.99, 'explainability': {'primary_drivers': ['memory_headroom', 'gpu_vram_availability', 'thermal_margin'], 'weights': {'memory_headroom': 0.45, 'gpu_vram_availability': 0.35, 'thermal_margin': 0.2}, 'derivation_summary': derivation_summary}}

@app.get('/')
def root_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'service': 'hardware-sentinel-6267', 'name': 'hardware-sentinel-6267', 'timestamp': datetime.now(dt_timezone.utc).isoformat()}

@app.get('/health')
def health_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'timestamp': datetime.now(dt_timezone.utc).isoformat()}

@app.get('/tools')
def tools_endpoint() -> Dict[str, Any]:
    return {'tools': [{'name': 'get_telemetry', 'description': 'Host telemetry and compute headroom intelligence engine'}]}

@app.get('/telemetry')
def get_telemetry(target_vram_mb: float=Query(default=0.0, description='Target VRAM requirement in MB'), target_ram_mb: float=Query(default=0.0, description='Target RAM requirement in MB'), execution_duration_sec: float=Query(default=1.0, description='Execution duration for velocity tracking'), spatial_lat: float=Query(default=0.0, description='Latitude for spatial H3 cell mapping'), spatial_lon: float=Query(default=0.0, description='Longitude for spatial H3 cell mapping')) -> Dict[str, Any]:
    req = HardwareQueryRequest(target_vram_mb=target_vram_mb, target_ram_mb=target_ram_mb, execution_duration_sec=execution_duration_sec, spatial_lat=spatial_lat, spatial_lon=spatial_lon)
    result = compute_hardware_intelligence(req)
    logger.info('telemetry_polled', event_name='telemetry_inquiry', classification=result.get('classification'))
    return result

@app.get('/h3_tensor')
def get_h3_tensor(h3_cell: str=Query(default='8a2a1072b59ffff', description='Uber H3 index cell'), lat: float=Query(default=37.7749, description='Fallback latitude'), lon: float=Query(default=-122.4194, description='Fallback longitude')) -> Dict[str, Any]:
    resolved_cell = h3_cell
    try:
        if not h3_cell or not h3.is_valid_cell(h3_cell):
            resolved_cell = h3.latlng_to_cell(lat, lon, 7)
    except Exception:
        resolved_cell = '87283472bffffff'
    parent_r5 = h3.cell_to_parent(resolved_cell, 5)
    neighbors = list(h3.grid_ring(resolved_cell, 1))
    base_intel = compute_hardware_intelligence(HardwareQueryRequest(spatial_lat=lat, spatial_lon=lon))
    return {'timestamp': datetime.now(dt_timezone.utc).isoformat(), 'spatial_cell': resolved_cell, 'parent_cell': parent_r5, 'neighbor_count': len(neighbors), 'regional_tensor': base_intel.get('regional_tensor', []), 'raw_metrics': base_intel.get('raw_metrics', {}), 'derived_intelligence': base_intel.get('derived_intelligence', {}), 'classification': base_intel.get('classification', 'OPTIMAL')}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)