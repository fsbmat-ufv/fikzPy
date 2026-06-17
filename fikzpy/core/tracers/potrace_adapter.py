"""Optional Potrace adapter."""

from __future__ import annotations

from pathlib import Path

from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerConfig
from fikzpy.core.tracers.base import TracerKind, TracerRequest, TracerResult
from fikzpy.core.tracers.base import executable_version, resolve_executable, run_cli_tracer


class PotraceAdapter(BaseRasterTracer):
    """Run Potrace as an optional external bitmap outline tracer."""

    kind = TracerKind.POTRACE

    def availability(self, config: TracerConfig | None = None) -> TracerAvailability:
        """Return Potrace executable availability."""
        effective_config = config or TracerConfig()
        executable = resolve_executable(effective_config.potrace_path, "potrace")
        if executable is None:
            return TracerAvailability(
                tracer=self.kind,
                available=False,
                backend="cli",
                reason="potrace executable was not found",
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
            input_mode="mask_pbm",
            output_name="potrace_output.svg",
            command_builder=_build_potrace_command,
        )


def _build_potrace_command(
    input_path: Path,
    output_path: Path,
    config: TracerConfig,
    executable: str,
) -> list[str]:
    command = [
        executable,
        str(input_path),
        "--svg",
        "--output",
        str(output_path),
        "--turdsize",
        str(int(config.potrace_turdsize)),
        "--alphamax",
        _format_number(config.potrace_alphamax),
        "--opttolerance",
        _format_number(config.potrace_opttolerance),
    ]
    if not config.potrace_opticurve:
        command.append("--longcurve")
    if config.potrace_turnpolicy is not None:
        command.extend(["--turnpolicy", config.potrace_turnpolicy])
    return command


def _format_number(value: float) -> str:
    return f"{float(value):g}"


__all__ = ["PotraceAdapter"]
