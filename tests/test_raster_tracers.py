from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import fikzpy.core.tracers.base as tracer_base
import fikzpy.core.tracers.potrace_adapter as potrace_adapter
import fikzpy.core.tracers.autotrace_adapter as autotrace_adapter
import fikzpy.core.tracers.vtracer_adapter as vtracer_adapter
from fikzpy.core.adaptive_preprocessing import preprocess_image
from fikzpy.core.tracers import AutoTraceAdapter, PotraceAdapter, TracerConfig, TracerKind
from fikzpy.core.tracers import TracerRegistry, TracerRequest, VTracerAdapter, trace_image
from fikzpy.core.tracers.base import BaseRasterTracer, TracerAvailability, TracerExecutionError
from fikzpy.core.tracers.base import TracerResult, prepare_input_file, validate_svg_file


SVG_TEXT = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="12" viewBox="0 0 10 12"></svg>'


def _image() -> np.ndarray:
    image = np.full((24, 32, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 6), (24, 18), (0, 0, 0), -1)
    return image


def _mask() -> np.ndarray:
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 5:15] = 255
    return mask


def _completed(command: list[str], returncode: int = 0, stdout: str = "out", stderr: str = "err") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _write_svg_for_command(command: list[str], text: str = SVG_TEXT) -> Path:
    output = _output_from_command(command)
    output.write_text(text, encoding="utf-8")
    return output


def _output_from_command(command: list[str]) -> Path:
    for flag in ("--output", "--output-file"):
        if flag in command:
            return Path(command[command.index(flag) + 1])
    raise AssertionError(f"output flag not found in {command}")


def _patch_cli_available(monkeypatch: pytest.MonkeyPatch, module, executable: str = "C:/Tools/tracer.exe") -> None:
    monkeypatch.setattr(module, "resolve_executable", lambda *_args, **_kwargs: executable)
    monkeypatch.setattr(module, "executable_version", lambda *_args, **_kwargs: "fake 1.2.3")


def _patch_cli_run(monkeypatch: pytest.MonkeyPatch, *, svg_text: str = SVG_TEXT, returncode: int = 0) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        assert kwargs.get("shell") is False
        commands.append(list(command))
        if returncode == 0:
            _write_svg_for_command(list(command), svg_text)
        return _completed(list(command), returncode=returncode)

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)
    return commands


def test_potrace_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter, "C:/Program Files/Potrace/potrace.exe")

    availability = PotraceAdapter().availability()

    assert availability.available
    assert availability.backend == "cli"
    assert availability.executable_path == "C:/Program Files/Potrace/potrace.exe"
    assert availability.version == "fake 1.2.3"


def test_potrace_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(potrace_adapter, "resolve_executable", lambda *_args, **_kwargs: None)

    availability = PotraceAdapter().availability()

    assert not availability.available
    assert "not found" in availability.reason


def test_autotrace_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)

    availability = AutoTraceAdapter().availability()

    assert availability.available
    assert availability.cli_available
    assert availability.version == "fake 1.2.3"


def test_autotrace_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autotrace_adapter, "resolve_executable", lambda *_args, **_kwargs: None)

    availability = AutoTraceAdapter().availability()

    assert not availability.available
    assert availability.backend == "cli"


def test_vtracer_python_available(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(__version__="0.6.0", convert_image_to_svg_py=lambda **_kwargs: None)
    monkeypatch.setattr(vtracer_adapter, "import_optional_module", lambda _name: module)
    monkeypatch.setattr(vtracer_adapter, "resolve_executable", lambda *_args, **_kwargs: None)

    availability = VTracerAdapter().availability()

    assert availability.available
    assert availability.backend == "python"
    assert availability.python_module_available
    assert availability.version == "0.6.0"


def test_vtracer_cli_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vtracer_adapter, "import_optional_module", lambda _name: None)
    monkeypatch.setattr(vtracer_adapter, "resolve_executable", lambda *_args, **_kwargs: "vtracer.exe")
    monkeypatch.setattr(vtracer_adapter, "executable_version", lambda *_args, **_kwargs: "vtracer 1.0")

    availability = VTracerAdapter().availability()

    assert availability.available
    assert availability.backend == "cli"
    assert availability.cli_available


def test_vtracer_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vtracer_adapter, "import_optional_module", lambda _name: None)
    monkeypatch.setattr(vtracer_adapter, "resolve_executable", lambda *_args, **_kwargs: None)

    availability = VTracerAdapter().availability()

    assert not availability.available
    assert "not found" in availability.reason


def test_vtracer_prefers_python_backend_over_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(convert_image_to_svg=lambda *_args: None)
    monkeypatch.setattr(vtracer_adapter, "import_optional_module", lambda _name: module)
    monkeypatch.setattr(vtracer_adapter, "resolve_executable", lambda *_args, **_kwargs: "vtracer.exe")

    availability = VTracerAdapter().availability()

    assert availability.backend == "python"
    assert availability.cli_available


def test_manual_path_with_spaces_is_supported(tmp_path: Path) -> None:
    executable = tmp_path / "Program Files" / "Potrace Tool" / "potrace fake.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    config = TracerConfig(potrace_path=executable)

    availability = PotraceAdapter().availability(config)

    assert availability.available
    assert availability.executable_path == str(executable)


def test_successful_potrace_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)
    commands = _patch_cli_run(monkeypatch)

    result = trace_image(_mask(), TracerKind.POTRACE, output_directory=tmp_path)

    assert result.success
    assert result.svg_text == SVG_TEXT
    assert result.svg_sha256
    assert result.svg_width == "10"
    assert result.svg_height == "12"
    assert result.svg_viewbox == "0 0 10 12"
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert commands[0][0] == "C:/Tools/tracer.exe"


def test_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)

    def fake_run(command, **kwargs):
        assert kwargs.get("shell") is False
        raise subprocess.TimeoutExpired(command, timeout=0.01, output="partial", stderr="late")

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    result = trace_image(_mask(), TracerKind.POTRACE, TracerConfig(timeout_seconds=0.01))

    assert not result.success
    assert "timed out" in result.error
    assert result.stdout == "partial"
    assert result.stderr == "late"


def test_nonzero_return_code_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)
    _patch_cli_run(monkeypatch, returncode=7)

    result = trace_image(_image(), TracerKind.AUTOTRACE)

    assert not result.success
    assert result.return_code == 7
    assert "non-zero" in result.error


def test_stdout_and_stderr_are_captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)

    def fake_run(command, **kwargs):
        _write_svg_for_command(list(command))
        return _completed(list(command), stdout="hello stdout", stderr="hello stderr")

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    result = trace_image(_image(), TracerKind.AUTOTRACE, output_directory=tmp_path)

    assert result.stdout == "hello stdout"
    assert result.stderr == "hello stderr"


def test_missing_svg_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)

    def fake_run(command, **kwargs):
        return _completed(list(command))

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    result = trace_image(_mask(), TracerKind.POTRACE)

    assert not result.success
    assert "does not exist" in result.error


def test_empty_svg_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)

    def fake_run(command, **kwargs):
        _write_svg_for_command(list(command), "")
        return _completed(list(command))

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    result = trace_image(_mask(), TracerKind.POTRACE)

    assert not result.success
    assert "empty" in result.error


def test_malformed_svg_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)
    _patch_cli_run(monkeypatch, svg_text="<svg><path></svg>")

    result = trace_image(_image(), TracerKind.AUTOTRACE)

    assert not result.success
    assert "malformed" in result.error


def test_html_output_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)
    _patch_cli_run(monkeypatch, svg_text="<html><body>error</body></html>")

    result = trace_image(_image(), TracerKind.AUTOTRACE)

    assert not result.success
    assert "HTML" in result.error


def test_svg_hash_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "out.svg"
    path.write_text(SVG_TEXT, encoding="utf-8")

    first = validate_svg_file(path)
    second = validate_svg_file(path)

    assert first.sha256 == second.sha256
    assert first.byte_count == len(SVG_TEXT.encode("utf-8"))


def test_version_detection_uses_safe_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(list(command), stdout="tool 9.8")

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    assert tracer_base.executable_version("tool.exe") == "tool 9.8"
    assert calls[0][1]["shell"] is False
    assert calls[0][0] == ["tool.exe", "--version"]


def test_success_cleanup_removes_temporary_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)
    _patch_cli_run(monkeypatch)

    result = trace_image(_mask(), TracerKind.POTRACE)

    assert result.success
    assert not Path(result.temporary_files[0]).exists()


def test_error_can_preserve_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)
    _patch_cli_run(monkeypatch, returncode=2)
    config = TracerConfig(preserve_diagnostics_on_error=True)

    result = trace_image(_mask(), TracerKind.POTRACE, config)

    assert not result.success
    assert Path(result.temporary_files[0]).exists()


def test_numpy_input_is_prepared_for_vtracer(tmp_path: Path) -> None:
    prepared = prepare_input_file(_image(), tmp_path, "rgb_png", TracerConfig())

    assert prepared.path.exists()
    assert prepared.path.suffix == ".png"


def test_pil_input_is_supported_when_pillow_is_available(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    pil_image = image_module.fromarray(_image())

    prepared = prepare_input_file(pil_image, tmp_path, "rgb_png", TracerConfig())

    assert prepared.path.exists()


def test_path_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), cv2.cvtColor(_image(), cv2.COLOR_RGB2BGR))

    prepared = prepare_input_file(path, tmp_path / "work", "rgb_png", TracerConfig())

    assert prepared.path.exists()


def test_preprocessing_result_input_uses_existing_mask(tmp_path: Path) -> None:
    preprocessing = preprocess_image(_mask())

    prepared = prepare_input_file(preprocessing, tmp_path, "mask_pbm", TracerConfig())

    text = prepared.path.read_text(encoding="ascii")
    assert text.startswith("P1")
    assert "1" in text


def test_binary_mask_for_potrace_writes_pbm(tmp_path: Path) -> None:
    prepared = prepare_input_file(_mask(), tmp_path, "mask_pbm", TracerConfig())

    assert prepared.path.suffix == ".pbm"
    assert prepared.path.read_text(encoding="ascii").splitlines()[0] == "P1"


def test_rgb_is_preserved_for_vtracer_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def convert_image_to_svg_py(**kwargs):
        input_path = Path(kwargs["input_path"])
        output_path = Path(kwargs["output_path"])
        pixel = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)[0, 0].tolist()
        assert pixel == [0, 0, 255]
        output_path.write_text(SVG_TEXT, encoding="utf-8")

    module = types.SimpleNamespace(__version__="py", convert_image_to_svg_py=convert_image_to_svg_py)
    monkeypatch.setattr(vtracer_adapter, "import_optional_module", lambda _name: module)
    monkeypatch.setattr(vtracer_adapter, "resolve_executable", lambda *_args, **_kwargs: None)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)

    result = trace_image(image, TracerKind.VTRACER, output_directory=tmp_path)

    assert result.success
    assert result.backend == "python"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TracerConfig(timeout_seconds=0),
        lambda: TracerConfig(potrace_turdsize=-1),
        lambda: TracerConfig(potrace_turnpolicy="sideways"),
        lambda: TracerConfig(autotrace_color_count=1),
        lambda: TracerConfig(vtracer_colormode="unknown"),
    ],
)
def test_invalid_parameters_are_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_command_construction_is_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)
    first_commands = _patch_cli_run(monkeypatch)
    first = trace_image(_mask(), TracerKind.POTRACE, output_directory=tmp_path / "one")
    second_commands = _patch_cli_run(monkeypatch)
    second = trace_image(_mask(), TracerKind.POTRACE, output_directory=tmp_path / "two")

    assert first.success and second.success
    assert first_commands[0][:1] == second_commands[0][:1]
    assert first_commands[0][2:3] == second_commands[0][2:3]
    assert "--turdsize" in first_commands[0]


def test_subprocess_is_never_invoked_with_shell_true(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_available(monkeypatch, autotrace_adapter)
    seen_shell_values = []

    def fake_run(command, **kwargs):
        seen_shell_values.append(kwargs.get("shell"))
        _write_svg_for_command(list(command))
        return _completed(list(command))

    monkeypatch.setattr(tracer_base.subprocess, "run", fake_run)

    trace_image(_image(), TracerKind.AUTOTRACE)

    assert seen_shell_values == [False]


def test_registry_lists_and_reports_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(potrace_adapter, "resolve_executable", lambda *_args, **_kwargs: None)
    registry = TracerRegistry()

    kinds = registry.list_tracers()
    availability = registry.list_availability()

    assert kinds == (TracerKind.POTRACE, TracerKind.AUTOTRACE, TracerKind.VTRACER)
    assert len(availability) == 3


def test_fallback_explicit_helper_chooses_available_tracer(tmp_path: Path) -> None:
    class UnavailableTracer(BaseRasterTracer):
        kind = TracerKind.POTRACE

        def availability(self, config=None):
            return TracerAvailability(self.kind, False, reason="no")

    class AvailableTracer(BaseRasterTracer):
        kind = TracerKind.AUTOTRACE

        def availability(self, config=None):
            return TracerAvailability(self.kind, True, backend="fake")

        def _trace_available(self, request, availability):
            return TracerResult(
                requested_tracer=request.tracer,
                effective_tracer=self.kind,
                success=True,
                backend="fake",
                svg_text=SVG_TEXT,
                config=request.config,
            )

    registry = TracerRegistry({TracerKind.POTRACE: UnavailableTracer(), TracerKind.AUTOTRACE: AvailableTracer()})

    result = registry.trace_with_first_available(_image(), (TracerKind.POTRACE, TracerKind.AUTOTRACE))

    assert result.success
    assert "explicit first-available" in result.warnings[-1]


def test_requested_unavailable_tracer_does_not_fallback_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(potrace_adapter, "resolve_executable", lambda *_args, **_kwargs: None)

    result = trace_image(_mask(), TracerKind.POTRACE)

    assert not result.success
    assert result.effective_tracer is TracerKind.POTRACE


def test_tracer_import_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.tracers; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_modules_do_not_generate_documents_or_use_svg_bridge() -> None:
    sources = [
        Path(tracer_base.__file__).read_text(encoding="utf-8").lower(),
        Path(potrace_adapter.__file__).read_text(encoding="utf-8").lower(),
        Path(autotrace_adapter.__file__).read_text(encoding="utf-8").lower(),
        Path(vtracer_adapter.__file__).read_text(encoding="utf-8").lower(),
    ]

    for source in sources:
        assert "\\\\draw" not in source
        assert "tikzpicture" not in source
        assert "svg2tikz" not in source


def test_windows_style_command_keeps_executable_path_as_single_argument(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = str(tmp_path / "Program Files" / "Trace Tools" / "autotrace.exe")
    _patch_cli_available(monkeypatch, autotrace_adapter, executable)
    commands = _patch_cli_run(monkeypatch)

    result = trace_image(_image(), TracerKind.AUTOTRACE, output_directory=tmp_path / "out")

    assert result.success
    assert commands[0][0] == executable
    assert isinstance(result.command, tuple)


def test_to_dict_omits_svg_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_cli_available(monkeypatch, potrace_adapter)
    _patch_cli_run(monkeypatch)

    result = trace_image(_mask(), TracerKind.POTRACE, output_directory=tmp_path)
    data = result.to_dict()

    assert "svg_text" not in data
    assert result.to_dict(include_svg=True)["svg_text"] == SVG_TEXT
