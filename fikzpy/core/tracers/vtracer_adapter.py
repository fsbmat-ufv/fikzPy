"""Optional VTracer adapter."""

from __future__ import annotations

from pathlib import Path

from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerConfig
from fikzpy.core.tracers.base import TracerKind, TracerRequest, TracerResult
from fikzpy.core.tracers.base import executable_version, has_vtracer_python_api
from fikzpy.core.tracers.base import import_optional_module, module_version, resolve_executable
from fikzpy.core.tracers.base import run_cli_tracer, run_python_vtracer


class VTracerAdapter(BaseRasterTracer):
    """Run VTracer as an optional color-region tracer."""

    kind = TracerKind.VTRACER

    def availability(self, config: TracerConfig | None = None) -> TracerAvailability:
        """Return VTracer Python or executable availability."""
        effective_config = config or TracerConfig()
        module = import_optional_module("vtracer")
        python_available = has_vtracer_python_api(module)
        executable = resolve_executable(effective_config.vtracer_path, "vtracer")
        cli_available = executable is not None
        if python_available:
            return TracerAvailability(
                tracer=self.kind,
                available=True,
                backend="python",
                version=module_version(module),
                python_module_available=True,
                cli_available=cli_available,
                executable_path=executable,
            )
        if cli_available:
            return TracerAvailability(
                tracer=self.kind,
                available=True,
                backend="cli",
                executable_path=executable,
                version=executable_version(executable),
                python_module_available=False,
                cli_available=True,
            )
        return TracerAvailability(
            tracer=self.kind,
            available=False,
            backend=None,
            reason="vtracer Python module and executable were not found",
            python_module_available=False,
            cli_available=False,
        )

    def _trace_available(self, request: TracerRequest, availability: TracerAvailability) -> TracerResult:
        if availability.backend == "python":
            module = import_optional_module("vtracer")
            return run_python_vtracer(request=request, availability=availability, module=module)
        return run_cli_tracer(
            request=request,
            availability=availability,
            input_mode="rgb_png",
            output_name="vtracer_output.svg",
            command_builder=_build_vtracer_command,
        )


def _build_vtracer_command(
    input_path: Path,
    output_path: Path,
    config: TracerConfig,
    executable: str,
) -> list[str]:
    return [
        executable,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--colormode",
        config.vtracer_colormode,
        "--hierarchical",
        config.vtracer_hierarchical,
        "--mode",
        config.vtracer_mode,
        "--filter_speckle",
        str(int(config.vtracer_filter_speckle)),
        "--color_precision",
        str(int(config.vtracer_color_precision)),
        "--layer_difference",
        str(int(config.vtracer_layer_difference)),
        "--corner_threshold",
        str(int(config.vtracer_corner_threshold)),
        "--length_threshold",
        _format_number(config.vtracer_length_threshold),
        "--max_iterations",
        str(int(config.vtracer_max_iterations)),
        "--splice_threshold",
        str(int(config.vtracer_splice_threshold)),
        "--path_precision",
        str(int(config.vtracer_path_precision)),
    ]


def _format_number(value: float) -> str:
    return f"{float(value):g}"


__all__ = ["VTracerAdapter"]
