"""Optional AutoTrace adapter."""

from __future__ import annotations

from pathlib import Path

from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerConfig
from fikzpy.core.tracers.base import TracerKind, TracerRequest, TracerResult
from fikzpy.core.tracers.base import executable_version, resolve_executable, run_cli_tracer


class AutoTraceAdapter(BaseRasterTracer):
    """Run AutoTrace as an optional external outline or centerline tracer."""

    kind = TracerKind.AUTOTRACE

    def availability(self, config: TracerConfig | None = None) -> TracerAvailability:
        """Return AutoTrace executable availability."""
        effective_config = config or TracerConfig()
        executable = resolve_executable(effective_config.autotrace_path, "autotrace")
        if executable is None:
            return TracerAvailability(
                tracer=self.kind,
                available=False,
                backend="cli",
                reason="autotrace executable was not found",
            )
        return TracerAvailability(
            tracer=self.kind,
            available=True,
            backend="cli",
            executable_path=executable,
            version=executable_version(executable),
            cli_available=True,
        )

    def _trace_available(self, request: TracerRequest, availability: TracerAvailability) -> TracerResult:
        return run_cli_tracer(
            request=request,
            availability=availability,
            input_mode="mask_png" if request.config.autotrace_centerline else "rgb_png",
            output_name="autotrace_output.svg",
            command_builder=_build_autotrace_command,
        )


def _build_autotrace_command(
    input_path: Path,
    output_path: Path,
    config: TracerConfig,
    executable: str,
) -> list[str]:
    command = [
        executable,
        str(input_path),
        "--output-file",
        str(output_path),
        "--output-format",
        "svg",
    ]
    if config.autotrace_centerline:
        command.append("--centerline")
    if config.autotrace_color_count is not None:
        command.extend(["--color-count", str(int(config.autotrace_color_count))])
    if config.autotrace_despeckle_level is not None:
        command.extend(["--despeckle-level", str(int(config.autotrace_despeckle_level))])
    if config.autotrace_corner_threshold is not None:
        command.extend(["--corner-threshold", _format_number(config.autotrace_corner_threshold)])
    if config.autotrace_error_threshold is not None:
        command.extend(["--error-threshold", _format_number(config.autotrace_error_threshold)])
    if config.autotrace_background_color is not None:
        command.extend(["--background-color", config.autotrace_background_color])
    return command


def _format_number(value: float) -> str:
    return f"{float(value):g}"


__all__ = ["AutoTraceAdapter"]
