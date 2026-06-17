"""Registry for optional raster tracer adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fikzpy.core.tracers.autotrace_adapter import AutoTraceAdapter
from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerConfig
from fikzpy.core.tracers.base import TracerExecutionError, TracerKind, TracerRequest, TracerResult
from fikzpy.core.tracers.base import coerce_tracer_kind
from fikzpy.core.tracers.potrace_adapter import PotraceAdapter
from fikzpy.core.tracers.vtracer_adapter import VTracerAdapter


@dataclass
class TracerRegistry:
    """Simple explicit registry for optional raster tracers."""

    tracers: dict[TracerKind, BaseRasterTracer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracers:
            self.tracers = {
                TracerKind.POTRACE: PotraceAdapter(),
                TracerKind.AUTOTRACE: AutoTraceAdapter(),
                TracerKind.VTRACER: VTracerAdapter(),
            }

    def list_tracers(self) -> tuple[TracerKind, ...]:
        """Return registered tracers in deterministic order."""
        return tuple(kind for kind in TracerKind if kind in self.tracers)

    def get(self, tracer: TracerKind | str) -> BaseRasterTracer:
        """Return a tracer adapter by explicit kind."""
        kind = coerce_tracer_kind(tracer)
        try:
            return self.tracers[kind]
        except KeyError as exc:
            raise KeyError(f"Tracer is not registered: {kind.value}") from exc

    def availability(
        self,
        tracer: TracerKind | str,
        config: TracerConfig | None = None,
    ) -> TracerAvailability:
        """Return availability for one tracer."""
        return self.get(tracer).availability(config or TracerConfig())

    def list_availability(self, config: TracerConfig | None = None) -> tuple[TracerAvailability, ...]:
        """Return availability for all registered tracers."""
        effective_config = config or TracerConfig()
        return tuple(self.tracers[kind].availability(effective_config) for kind in self.list_tracers())

    def trace(self, request: TracerRequest) -> TracerResult:
        """Execute the explicitly requested tracer."""
        return self.get(request.tracer).trace(request)

    def trace_image(
        self,
        image: Any,
        tracer: TracerKind | str,
        config: TracerConfig | None = None,
        output_directory: str | Path | None = None,
    ) -> TracerResult:
        """Trace an image with an explicit tracer."""
        kind = coerce_tracer_kind(tracer)
        request = TracerRequest(
            image=image,
            tracer=kind,
            config=config or TracerConfig(),
            output_directory=output_directory,
        )
        return self.trace(request)

    def trace_with_first_available(
        self,
        image: Any,
        tracers: tuple[TracerKind | str, ...] | None = None,
        config: TracerConfig | None = None,
        output_directory: str | Path | None = None,
    ) -> TracerResult:
        """Explicitly choose the first available tracer from a provided order."""
        effective_config = config or TracerConfig()
        ordered = tuple(coerce_tracer_kind(item) for item in tracers) if tracers else self.list_tracers()
        for kind in ordered:
            availability = self.availability(kind, effective_config)
            if availability.available:
                result = self.trace_image(image, kind, effective_config, output_directory)
                result_warnings = (*result.warnings, f"explicit first-available selection chose {kind.value}")
                return TracerResult(
                    requested_tracer=result.requested_tracer,
                    effective_tracer=result.effective_tracer,
                    success=result.success,
                    backend=result.backend,
                    svg_text=result.svg_text,
                    svg_path=result.svg_path,
                    command=result.command,
                    return_code=result.return_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=result.duration_seconds,
                    temporary_files=result.temporary_files,
                    warnings=result_warnings,
                    config=result.config,
                    version=result.version,
                    svg_sha256=result.svg_sha256,
                    svg_bytes=result.svg_bytes,
                    svg_width=result.svg_width,
                    svg_height=result.svg_height,
                    svg_viewbox=result.svg_viewbox,
                    error=result.error,
                )
        raise TracerExecutionError("No requested tracer is available.")


DEFAULT_REGISTRY = TracerRegistry()


def trace_image(
    image: Any,
    tracer: TracerKind | str,
    config: TracerConfig | None = None,
    output_directory: str | Path | None = None,
) -> TracerResult:
    """Trace an image with an explicitly selected optional tracer."""
    return DEFAULT_REGISTRY.trace_image(image, tracer, config, output_directory)


def get_tracer_availability(
    tracer: TracerKind | str,
    config: TracerConfig | None = None,
) -> TracerAvailability:
    """Return availability for one tracer from the default registry."""
    return DEFAULT_REGISTRY.availability(tracer, config)


def list_tracer_availability(config: TracerConfig | None = None) -> tuple[TracerAvailability, ...]:
    """Return availability for all tracers from the default registry."""
    return DEFAULT_REGISTRY.list_availability(config)


def trace_with_first_available(
    image: Any,
    tracers: tuple[TracerKind | str, ...] | None = None,
    config: TracerConfig | None = None,
    output_directory: str | Path | None = None,
) -> TracerResult:
    """Explicit first-available helper; no production flow calls this."""
    return DEFAULT_REGISTRY.trace_with_first_available(image, tracers, config, output_directory)


__all__ = [
    "DEFAULT_REGISTRY",
    "TracerRegistry",
    "get_tracer_availability",
    "list_tracer_availability",
    "trace_image",
    "trace_with_first_available",
]
