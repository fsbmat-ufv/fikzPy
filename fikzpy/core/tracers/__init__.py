"""Optional raster tracing adapters for the future semantic Classic pipeline."""

from fikzpy.core.tracers.autotrace_adapter import AutoTraceAdapter
from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerConfig
from fikzpy.core.tracers.base import TracerExecutionError, TracerKind, TracerRequest, TracerResult
from fikzpy.core.tracers.potrace_adapter import PotraceAdapter
from fikzpy.core.tracers.registry import DEFAULT_REGISTRY, TracerRegistry
from fikzpy.core.tracers.registry import get_tracer_availability, list_tracer_availability
from fikzpy.core.tracers.registry import trace_image, trace_with_first_available
from fikzpy.core.tracers.vtracer_adapter import VTracerAdapter

__all__ = [
    "AutoTraceAdapter",
    "BaseRasterTracer",
    "DEFAULT_REGISTRY",
    "PotraceAdapter",
    "TracerAvailability",
    "TracerConfig",
    "TracerExecutionError",
    "TracerKind",
    "TracerRegistry",
    "TracerRequest",
    "TracerResult",
    "VTracerAdapter",
    "get_tracer_availability",
    "list_tracer_availability",
    "trace_image",
    "trace_with_first_available",
]
