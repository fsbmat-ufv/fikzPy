"""Common infrastructure for optional raster tracing adapters."""

from __future__ import annotations

import importlib
import inspect
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from fikzpy.core.adaptive_preprocessing import PreprocessingResult, preprocess_image
from fikzpy.core.diagnostics import log_event
from fikzpy.core.image_classifier import ImageCategory


class TracerKind(Enum):
    """Supported optional raster tracing engines."""

    POTRACE = "potrace"
    AUTOTRACE = "autotrace"
    VTRACER = "vtracer"


class TracerExecutionError(RuntimeError):
    """Raised when a tracer cannot produce a valid SVG result."""


@dataclass(frozen=True)
class TracerAvailability:
    """Availability diagnostics for one optional tracer."""

    tracer: TracerKind
    available: bool
    backend: str | None = None
    executable_path: str | None = None
    version: str | None = None
    reason: str | None = None
    python_module_available: bool = False
    cli_available: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tracer, TracerKind):
            object.__setattr__(self, "tracer", coerce_tracer_kind(self.tracer))
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool.")
        if not isinstance(self.python_module_available, bool):
            raise TypeError("python_module_available must be a bool.")
        if not isinstance(self.cli_available, bool):
            raise TypeError("cli_available must be a bool.")

    def to_dict(self) -> dict[str, Any]:
        """Return serializable availability diagnostics."""
        return {
            "tracer": self.tracer.value,
            "available": self.available,
            "backend": self.backend,
            "executable_path": self.executable_path,
            "version": self.version,
            "reason": self.reason,
            "python_module_available": self.python_module_available,
            "cli_available": self.cli_available,
        }


@dataclass(frozen=True)
class TracerConfig:
    """Shared conservative configuration for optional raster tracers."""

    potrace_path: str | Path | None = None
    autotrace_path: str | Path | None = None
    vtracer_path: str | Path | None = None
    timeout_seconds: float = 30.0
    keep_temporary_files: bool = False
    preserve_diagnostics_on_error: bool = True
    raise_on_error: bool = False
    potrace_turdsize: int = 2
    potrace_alphamax: float = 1.0
    potrace_opticurve: bool = True
    potrace_opttolerance: float = 0.2
    potrace_invert: bool = False
    potrace_turnpolicy: str | None = None
    autotrace_centerline: bool = False
    autotrace_color_count: int | None = None
    autotrace_despeckle_level: int | None = None
    autotrace_corner_threshold: float | None = None
    autotrace_error_threshold: float | None = None
    autotrace_background_color: str | None = None
    vtracer_colormode: str = "color"
    vtracer_hierarchical: str = "stacked"
    vtracer_mode: str = "spline"
    vtracer_filter_speckle: int = 4
    vtracer_color_precision: int = 6
    vtracer_layer_difference: int = 16
    vtracer_corner_threshold: int = 60
    vtracer_length_threshold: float = 4.0
    vtracer_max_iterations: int = 10
    vtracer_splice_threshold: int = 45
    vtracer_path_precision: int = 3

    def __post_init__(self) -> None:
        _validate_timeout(self.timeout_seconds)
        _validate_bool("keep_temporary_files", self.keep_temporary_files)
        _validate_bool("preserve_diagnostics_on_error", self.preserve_diagnostics_on_error)
        _validate_bool("raise_on_error", self.raise_on_error)
        _validate_non_negative_int("potrace_turdsize", self.potrace_turdsize)
        _validate_non_negative_float("potrace_alphamax", self.potrace_alphamax)
        _validate_bool("potrace_opticurve", self.potrace_opticurve)
        _validate_non_negative_float("potrace_opttolerance", self.potrace_opttolerance)
        _validate_bool("potrace_invert", self.potrace_invert)
        if self.potrace_turnpolicy is not None:
            _validate_choice(
                "potrace_turnpolicy",
                self.potrace_turnpolicy,
                {"black", "white", "left", "right", "minority", "majority", "random"},
            )
        _validate_bool("autotrace_centerline", self.autotrace_centerline)
        _validate_optional_min_int("autotrace_color_count", self.autotrace_color_count, 2)
        _validate_optional_min_int("autotrace_despeckle_level", self.autotrace_despeckle_level, 0)
        _validate_optional_non_negative_float("autotrace_corner_threshold", self.autotrace_corner_threshold)
        _validate_optional_non_negative_float("autotrace_error_threshold", self.autotrace_error_threshold)
        _validate_optional_text("autotrace_background_color", self.autotrace_background_color)
        _validate_choice("vtracer_colormode", self.vtracer_colormode, {"color", "binary"})
        _validate_choice("vtracer_hierarchical", self.vtracer_hierarchical, {"stacked", "cutout"})
        _validate_choice("vtracer_mode", self.vtracer_mode, {"spline", "polygon", "none"})
        _validate_non_negative_int("vtracer_filter_speckle", self.vtracer_filter_speckle)
        _validate_non_negative_int("vtracer_color_precision", self.vtracer_color_precision)
        _validate_non_negative_int("vtracer_layer_difference", self.vtracer_layer_difference)
        _validate_non_negative_int("vtracer_corner_threshold", self.vtracer_corner_threshold)
        _validate_non_negative_float("vtracer_length_threshold", self.vtracer_length_threshold)
        _validate_non_negative_int("vtracer_max_iterations", self.vtracer_max_iterations)
        _validate_non_negative_int("vtracer_splice_threshold", self.vtracer_splice_threshold)
        _validate_non_negative_int("vtracer_path_precision", self.vtracer_path_precision)

    def executable_for(self, tracer: TracerKind) -> str | Path | None:
        """Return the configured executable path for a tracer."""
        kind = coerce_tracer_kind(tracer)
        if kind is TracerKind.POTRACE:
            return self.potrace_path
        if kind is TracerKind.AUTOTRACE:
            return self.autotrace_path
        return self.vtracer_path

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostic dictionary."""
        return {
            "potrace_path": str(self.potrace_path) if self.potrace_path is not None else None,
            "autotrace_path": str(self.autotrace_path) if self.autotrace_path is not None else None,
            "vtracer_path": str(self.vtracer_path) if self.vtracer_path is not None else None,
            "timeout_seconds": self.timeout_seconds,
            "keep_temporary_files": self.keep_temporary_files,
            "preserve_diagnostics_on_error": self.preserve_diagnostics_on_error,
            "raise_on_error": self.raise_on_error,
            "potrace_turdsize": self.potrace_turdsize,
            "potrace_alphamax": self.potrace_alphamax,
            "potrace_opticurve": self.potrace_opticurve,
            "potrace_opttolerance": self.potrace_opttolerance,
            "potrace_invert": self.potrace_invert,
            "potrace_turnpolicy": self.potrace_turnpolicy,
            "autotrace_centerline": self.autotrace_centerline,
            "autotrace_color_count": self.autotrace_color_count,
            "autotrace_despeckle_level": self.autotrace_despeckle_level,
            "autotrace_corner_threshold": self.autotrace_corner_threshold,
            "autotrace_error_threshold": self.autotrace_error_threshold,
            "autotrace_background_color": self.autotrace_background_color,
            "vtracer_colormode": self.vtracer_colormode,
            "vtracer_hierarchical": self.vtracer_hierarchical,
            "vtracer_mode": self.vtracer_mode,
            "vtracer_filter_speckle": self.vtracer_filter_speckle,
            "vtracer_color_precision": self.vtracer_color_precision,
            "vtracer_layer_difference": self.vtracer_layer_difference,
            "vtracer_corner_threshold": self.vtracer_corner_threshold,
            "vtracer_length_threshold": self.vtracer_length_threshold,
            "vtracer_max_iterations": self.vtracer_max_iterations,
            "vtracer_splice_threshold": self.vtracer_splice_threshold,
            "vtracer_path_precision": self.vtracer_path_precision,
        }


@dataclass(frozen=True)
class TracerRequest:
    """Input request for a raster tracer adapter."""

    image: Any
    tracer: TracerKind | str
    config: TracerConfig = field(default_factory=TracerConfig)
    output_directory: str | Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracer", coerce_tracer_kind(self.tracer))
        if not isinstance(self.config, TracerConfig):
            raise TypeError("config must be a TracerConfig.")
        if self.output_directory is not None:
            object.__setattr__(self, "output_directory", Path(self.output_directory))


@dataclass(frozen=True)
class TracerResult:
    """Structured result from an optional raster tracer."""

    requested_tracer: TracerKind
    effective_tracer: TracerKind
    success: bool
    backend: str | None
    svg_text: str | None = None
    svg_path: str | None = None
    command: tuple[str, ...] = ()
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    temporary_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: TracerConfig = field(default_factory=TracerConfig)
    version: str | None = None
    svg_sha256: str | None = None
    svg_bytes: int = 0
    svg_width: str | None = None
    svg_height: str | None = None
    svg_viewbox: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_tracer", coerce_tracer_kind(self.requested_tracer))
        object.__setattr__(self, "effective_tracer", coerce_tracer_kind(self.effective_tracer))
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool.")
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative.")
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(self, "temporary_files", tuple(str(item) for item in self.temporary_files))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    def to_dict(self, *, include_svg: bool = False) -> dict[str, Any]:
        """Return diagnostics; full SVG text is omitted unless requested."""
        data = {
            "requested_tracer": self.requested_tracer.value,
            "effective_tracer": self.effective_tracer.value,
            "success": self.success,
            "backend": self.backend,
            "svg_path": self.svg_path,
            "command": list(self.command),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "temporary_files": list(self.temporary_files),
            "warnings": list(self.warnings),
            "config": self.config.to_dict(),
            "version": self.version,
            "svg_sha256": self.svg_sha256,
            "svg_bytes": self.svg_bytes,
            "svg_width": self.svg_width,
            "svg_height": self.svg_height,
            "svg_viewbox": self.svg_viewbox,
            "error": self.error,
        }
        if include_svg:
            data["svg_text"] = self.svg_text
        return data


@dataclass(frozen=True)
class PreparedInput:
    """Temporary input file prepared for a tracer backend."""

    path: Path
    mode: str
    temporary_files: tuple[Path, ...]


@dataclass(frozen=True)
class SvgValidation:
    """Validated SVG content and lightweight metadata."""

    text: str
    path: Path
    sha256: str
    byte_count: int
    width: str | None
    height: str | None
    viewbox: str | None


class BaseRasterTracer:
    """Base class for isolated optional raster tracer adapters."""

    kind: TracerKind

    def availability(self, config: TracerConfig | None = None) -> TracerAvailability:
        """Return deterministic availability diagnostics."""
        raise NotImplementedError

    def trace(self, request: TracerRequest) -> TracerResult:
        """Execute the tracer request."""
        if request.tracer is not self.kind:
            raise ValueError(f"{self.kind.value} adapter received {request.tracer.value} request.")
        config = request.config
        availability = self.availability(config)
        log_event("Tracer", f"requested={request.tracer.value}")
        log_event("Tracer", f"backend={availability.backend or 'none'}")
        log_event("Tracer", f"available={str(availability.available).lower()}")
        log_event("Tracer", f"version={availability.version}")
        if not availability.available:
            result = self._unavailable_result(request, availability)
            if config.raise_on_error:
                raise TracerExecutionError(result.error or "tracer is unavailable")
            return result
        return self._trace_available(request, availability)

    def _trace_available(self, request: TracerRequest, availability: TracerAvailability) -> TracerResult:
        raise NotImplementedError

    def _unavailable_result(self, request: TracerRequest, availability: TracerAvailability) -> TracerResult:
        return TracerResult(
            requested_tracer=request.tracer,
            effective_tracer=self.kind,
            success=False,
            backend=availability.backend,
            warnings=(availability.reason or "tracer is unavailable",),
            config=request.config,
            version=availability.version,
            error=availability.reason or "tracer is unavailable",
        )


def run_cli_tracer(
    *,
    request: TracerRequest,
    availability: TracerAvailability,
    input_mode: str,
    output_name: str,
    command_builder: Callable[[Path, Path, TracerConfig, str], list[str]],
) -> TracerResult:
    """Run an external tracer executable with safe subprocess settings."""
    config = request.config
    workspace = _make_workspace(request.output_directory)
    temporary_files: list[Path] = [workspace]
    command: list[str] = []
    stdout = ""
    stderr = ""
    return_code: int | None = None
    start = time.perf_counter()
    success = False
    validation: SvgValidation | None = None
    error: str | None = None
    warnings: list[str] = []

    try:
        prepared = prepare_input_file(request.image, workspace, input_mode, config)
        temporary_files.extend(prepared.temporary_files)
        output_path = workspace / output_name
        temporary_files.append(output_path)
        executable = availability.executable_path
        if executable is None:
            raise TracerExecutionError("executable path is unavailable")
        command = command_builder(prepared.path, output_path, config, executable)
        log_event("Tracer", f"input={prepared.path}")
        log_event("Tracer", f"output={output_path}")
        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(config.timeout_seconds),
        )
        return_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        log_event("Tracer", f"return_code={return_code}")
        if return_code != 0:
            raise TracerExecutionError(f"tracer returned non-zero exit code {return_code}")
        validation = validate_svg_file(output_path)
        success = True
        log_event("Tracer", f"svg_bytes={validation.byte_count}")
        log_event("Tracer", f"sha256={validation.sha256}")
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        error = f"tracer timed out after {config.timeout_seconds:.3g} seconds"
        warnings.append(error)
    except Exception as exc:
        error = str(exc)
        warnings.append(error)

    duration = time.perf_counter() - start
    keep_files = bool(request.output_directory) or config.keep_temporary_files or (
        not success and config.preserve_diagnostics_on_error
    )
    if not keep_files:
        _remove_workspace(workspace)
    if error is not None and config.raise_on_error:
        raise TracerExecutionError(error)
    return TracerResult(
        requested_tracer=request.tracer,
        effective_tracer=request.tracer,
        success=success,
        backend=availability.backend,
        svg_text=validation.text if validation else None,
        svg_path=str(validation.path) if validation else None,
        command=tuple(command),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        temporary_files=tuple(str(path) for path in temporary_files),
        warnings=tuple(warnings),
        config=config,
        version=availability.version,
        svg_sha256=validation.sha256 if validation else None,
        svg_bytes=validation.byte_count if validation else 0,
        svg_width=validation.width if validation else None,
        svg_height=validation.height if validation else None,
        svg_viewbox=validation.viewbox if validation else None,
        error=error,
    )


def run_python_vtracer(
    *,
    request: TracerRequest,
    availability: TracerAvailability,
    module: Any,
) -> TracerResult:
    """Run the optional Python VTracer backend against temporary PNG files."""
    config = request.config
    workspace = _make_workspace(request.output_directory)
    temporary_files: list[Path] = [workspace]
    start = time.perf_counter()
    success = False
    validation: SvgValidation | None = None
    error: str | None = None
    warnings: list[str] = []

    try:
        prepared = prepare_input_file(request.image, workspace, "rgb_png", config)
        temporary_files.extend(prepared.temporary_files)
        output_path = workspace / "vtracer_output.svg"
        temporary_files.append(output_path)
        log_event("Tracer", f"input={prepared.path}")
        log_event("Tracer", f"output={output_path}")
        _call_vtracer_python(module, prepared.path, output_path, config)
        validation = validate_svg_file(output_path)
        success = True
        log_event("Tracer", "return_code=0")
        log_event("Tracer", f"svg_bytes={validation.byte_count}")
        log_event("Tracer", f"sha256={validation.sha256}")
    except Exception as exc:
        error = str(exc)
        warnings.append(error)

    duration = time.perf_counter() - start
    keep_files = bool(request.output_directory) or config.keep_temporary_files or (
        not success and config.preserve_diagnostics_on_error
    )
    if not keep_files:
        _remove_workspace(workspace)
    if error is not None and config.raise_on_error:
        raise TracerExecutionError(error)
    return TracerResult(
        requested_tracer=request.tracer,
        effective_tracer=request.tracer,
        success=success,
        backend=availability.backend,
        svg_text=validation.text if validation else None,
        svg_path=str(validation.path) if validation else None,
        command=(),
        return_code=0 if success else None,
        stdout="",
        stderr="",
        duration_seconds=duration,
        temporary_files=tuple(str(path) for path in temporary_files),
        warnings=tuple(warnings),
        config=config,
        version=availability.version,
        svg_sha256=validation.sha256 if validation else None,
        svg_bytes=validation.byte_count if validation else 0,
        svg_width=validation.width if validation else None,
        svg_height=validation.height if validation else None,
        svg_viewbox=validation.viewbox if validation else None,
        error=error,
    )


def prepare_input_file(image: Any, directory: Path, mode: str, config: TracerConfig) -> PreparedInput:
    """Prepare a deterministic temporary input file for a tracer."""
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "mask_pbm":
        mask = _input_to_mask(image)
        path = directory / "input.pbm"
        _write_pbm(path, mask, invert=config.potrace_invert)
        return PreparedInput(path=path, mode=mode, temporary_files=(path,))
    if mode == "mask_png":
        mask = _input_to_mask(image)
        path = directory / "input.png"
        cv2.imwrite(str(path), mask)
        return PreparedInput(path=path, mode=mode, temporary_files=(path,))
    if mode == "rgb_png":
        rgb = _input_to_rgb_or_rgba(image)
        path = directory / "input.png"
        _write_color_png(path, rgb)
        return PreparedInput(path=path, mode=mode, temporary_files=(path,))
    raise ValueError(f"Unsupported input mode: {mode}")


def validate_svg_file(path: Path) -> SvgValidation:
    """Validate that a tracer output path contains basic SVG XML."""
    if not path.exists():
        raise TracerExecutionError(f"SVG output does not exist: {path}")
    data = path.read_bytes()
    if not data:
        raise TracerExecutionError("SVG output is empty")
    text = data.decode("utf-8", errors="replace")
    lower_start = text.lstrip()[:128].lower()
    if lower_start.startswith("<html") or "<html" in lower_start:
        raise TracerExecutionError("SVG output looks like HTML")
    if "<svg" not in text.lower():
        raise TracerExecutionError("SVG output does not contain an svg element")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise TracerExecutionError(f"SVG output is malformed XML: {exc}") from exc
    if not _tag_name(root.tag).lower().endswith("svg"):
        raise TracerExecutionError("SVG output root is not svg")
    digest = sha256(data).hexdigest()
    return SvgValidation(
        text=text,
        path=path,
        sha256=digest,
        byte_count=len(data),
        width=root.attrib.get("width"),
        height=root.attrib.get("height"),
        viewbox=root.attrib.get("viewBox") or root.attrib.get("viewbox"),
    )


def coerce_tracer_kind(value: TracerKind | str) -> TracerKind:
    """Coerce strings or enum values into TracerKind."""
    if isinstance(value, TracerKind):
        return value
    normalized = str(value).strip().lower()
    for kind in TracerKind:
        if normalized in {kind.value, kind.name.lower()}:
            return kind
    raise ValueError(f"Unsupported tracer kind: {value!r}")


def resolve_executable(configured_path: str | Path | None, executable_name: str) -> str | None:
    """Resolve a configured executable path or search PATH deterministically."""
    if configured_path is not None:
        configured = Path(configured_path)
        if configured.exists():
            return str(configured)
        resolved = shutil.which(str(configured_path))
        return resolved
    return shutil.which(executable_name)


def executable_version(executable_path: str, timeout_seconds: float = 5.0) -> str | None:
    """Return a short executable version string when safe to query."""
    try:
        completed = subprocess.run(
            [executable_path, "--version"],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
        )
    except Exception:
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return None
    return text.splitlines()[0].strip()


def import_optional_module(name: str) -> Any | None:
    """Import an optional module without making it a dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def module_version(module: Any) -> str | None:
    """Return a best-effort module version."""
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def has_vtracer_python_api(module: Any | None) -> bool:
    """Return whether a vtracer module exposes a known conversion function."""
    if module is None:
        return False
    return any(
        callable(getattr(module, name, None))
        for name in ("convert_image_to_svg_py", "convert_image_to_svg")
    )


def _call_vtracer_python(module: Any, input_path: Path, output_path: Path, config: TracerConfig) -> None:
    function = getattr(module, "convert_image_to_svg_py", None) or getattr(module, "convert_image_to_svg", None)
    if function is None or not callable(function):
        raise TracerExecutionError("Python vtracer module does not expose a supported conversion function")
    kwargs = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "colormode": config.vtracer_colormode,
        "hierarchical": config.vtracer_hierarchical,
        "mode": config.vtracer_mode,
        "filter_speckle": config.vtracer_filter_speckle,
        "color_precision": config.vtracer_color_precision,
        "layer_difference": config.vtracer_layer_difference,
        "corner_threshold": config.vtracer_corner_threshold,
        "length_threshold": config.vtracer_length_threshold,
        "max_iterations": config.vtracer_max_iterations,
        "splice_threshold": config.vtracer_splice_threshold,
        "path_precision": config.vtracer_path_precision,
    }
    try:
        signature = inspect.signature(function)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        }
        function(**accepted)
    except TypeError:
        function(str(input_path), str(output_path))


def _input_to_mask(image: Any) -> np.ndarray:
    if isinstance(image, PreprocessingResult):
        return _normalize_binary_mask(image.binary_mask)
    if isinstance(image, np.ndarray) and image.ndim == 2 and _is_binary_like(image):
        return _normalize_binary_mask(image)
    result = preprocess_image(image, category=ImageCategory.BINARY_OUTLINE)
    return _normalize_binary_mask(result.binary_mask)


def _input_to_rgb_or_rgba(image: Any) -> np.ndarray:
    if isinstance(image, PreprocessingResult):
        return _normalize_color_array(image.original)
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not read image: {path}")
        if loaded.ndim == 2:
            return np.repeat(loaded[:, :, np.newaxis], 3, axis=2)
        if loaded.shape[2] == 3:
            return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        if loaded.shape[2] == 4:
            return cv2.cvtColor(loaded, cv2.COLOR_BGRA2RGBA)
        raise ValueError("Image path must contain grayscale, RGB, or RGBA data.")
    if _looks_like_pil_image(image):
        mode = getattr(image, "mode", "")
        converted = image.convert("RGBA" if "A" in mode else "RGB")
        return _normalize_color_array(np.asarray(converted))
    return _normalize_color_array(np.asarray(image))


def _normalize_color_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.size == 0:
        raise ValueError("Image must not be empty.")
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError("Image must be a 2D grayscale or 3D color array.")
    height, width, channels = values.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    if channels not in {1, 3, 4}:
        raise ValueError("Image must have 1, 3, or 4 channels.")
    numeric = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Image values must be finite.")
    if float(numeric.min(initial=0.0)) < 0.0:
        raise ValueError("Image values must be non-negative.")
    if values.dtype.kind == "f" and float(numeric.max(initial=0.0)) <= 1.0:
        numeric = numeric * 255.0
    if float(numeric.max(initial=0.0)) > 255.0:
        raise ValueError("Image values must be in the 0-255 range.")
    if channels == 1:
        numeric = np.repeat(numeric[:, :, :1], 3, axis=2)
    return np.rint(np.clip(numeric, 0.0, 255.0)).astype(np.uint8)


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("mask must not be empty.")
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.all(np.isfinite(array)):
        raise ValueError("mask values must be finite.")
    return (array > 0).astype(np.uint8) * 255


def _is_binary_like(array: np.ndarray) -> bool:
    values = np.asarray(array)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    unique = np.unique(values)
    return all(float(value) in {0.0, 1.0, 255.0} for value in unique)


def _write_pbm(path: Path, mask: np.ndarray, *, invert: bool) -> None:
    binary = mask > 0
    if invert:
        binary = ~binary
    height, width = binary.shape
    lines = ["P1", f"{width} {height}"]
    lines.extend(" ".join("1" if value else "0" for value in row) for row in binary)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_color_png(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3:
        raise ValueError("color image must be 3D.")
    if image.shape[2] == 4:
        output = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif image.shape[2] == 3:
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        output = image[:, :, 0]
    cv2.imwrite(str(path), output)


def _make_workspace(output_directory: str | Path | None) -> Path:
    if output_directory is not None:
        path = Path(output_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="fikzpy_tracer_"))


def _remove_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tag_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _looks_like_pil_image(image: object) -> bool:
    return hasattr(image, "convert") and hasattr(image, "mode") and image.__class__.__module__.startswith("PIL")


def _validate_timeout(value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive.")


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_optional_min_int(name: str, value: int | None, minimum: int) -> None:
    if value is not None and int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _validate_non_negative_float(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_optional_non_negative_float(name: str, value: float | None) -> None:
    if value is not None:
        _validate_non_negative_float(name, value)


def _validate_optional_text(name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string when provided.")


def _validate_choice(name: str, value: str, choices: set[str]) -> None:
    if str(value).strip().lower() not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}.")


__all__ = [
    "BaseRasterTracer",
    "PreparedInput",
    "SvgValidation",
    "TracerAvailability",
    "TracerConfig",
    "TracerExecutionError",
    "TracerKind",
    "TracerRequest",
    "TracerResult",
    "coerce_tracer_kind",
    "executable_version",
    "has_vtracer_python_api",
    "import_optional_module",
    "module_version",
    "prepare_input_file",
    "resolve_executable",
    "run_cli_tracer",
    "run_python_vtracer",
    "validate_svg_file",
]
