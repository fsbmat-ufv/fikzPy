"""LaTeX detection and compilation support."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from fikzpy.core.tikz_generator import wrap_standalone_document


LATEX_ENGINES = ("pdflatex", "xelatex", "lualatex")


@dataclass(frozen=True)
class LatexTool:
    """A detected LaTeX executable."""

    engine: str
    path: Path
    distribution: str = "Unknown"


@dataclass(frozen=True)
class LatexCompileResult:
    """Result of a LaTeX compilation run."""

    returncode: int
    output: str
    tex_path: Path
    pdf_path: Path
    command: tuple[str, ...]


def detect_latex_tools(
    *,
    distribution: str | None = None,
    engines: Iterable[str] = LATEX_ENGINES,
) -> list[LatexTool]:
    """Detect installed LaTeX executables from PATH and common locations."""
    normalized_distribution = distribution.lower() if distribution else None
    tools: list[LatexTool] = []
    seen: set[Path] = set()

    for engine in engines:
        executable = _engine_executable_name(engine)
        found = shutil.which(executable)
        if found:
            path = Path(found).resolve()
            if path not in seen and _matches_distribution(path, normalized_distribution):
                tools.append(LatexTool(engine=engine, path=path, distribution=_infer_distribution(path)))
                seen.add(path)

        for candidate in _common_latex_candidates(engine):
            if candidate.exists():
                path = candidate.resolve()
                if path not in seen and _matches_distribution(path, normalized_distribution):
                    tools.append(LatexTool(engine=engine, path=path, distribution=_infer_distribution(path)))
                    seen.add(path)

    return tools


def compile_tikz_to_pdf(
    tikz_picture: str,
    output_tex_path: str | Path,
    *,
    engine: str = "pdflatex",
    manual_path: str | Path | None = None,
    timeout: int = 60,
) -> LatexCompileResult:
    """Write a standalone document and compile it to PDF."""
    tex_path = Path(output_tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(wrap_standalone_document(tikz_picture), encoding="utf-8")
    return compile_latex_document(tex_path, engine=engine, manual_path=manual_path, timeout=timeout)


def compile_latex_document(
    tex_path: str | Path,
    *,
    engine: str = "pdflatex",
    manual_path: str | Path | None = None,
    timeout: int = 60,
) -> LatexCompileResult:
    """Compile a .tex file and return process output."""
    tex_file = Path(tex_path).resolve()
    if not tex_file.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_file}")

    executable = resolve_latex_executable(engine=engine, manual_path=manual_path)
    output_dir = tex_file.parent
    command = (
        str(executable),
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_dir}",
        tex_file.name,
    )

    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return LatexCompileResult(
        returncode=completed.returncode,
        output=output,
        tex_path=tex_file,
        pdf_path=tex_file.with_suffix(".pdf"),
        command=command,
    )


def resolve_latex_executable(
    *,
    engine: str = "pdflatex",
    manual_path: str | Path | None = None,
) -> Path:
    """Resolve a LaTeX executable path or raise a helpful error."""
    if manual_path is not None:
        candidate = Path(manual_path)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Manual LaTeX path does not exist: {candidate}")

    executable = _engine_executable_name(engine)
    found = shutil.which(executable)
    if found:
        return Path(found).resolve()

    detected = detect_latex_tools(engines=(engine,))
    if detected:
        return detected[0].path

    raise FileNotFoundError(
        f"Could not find {engine}. Install MiKTeX, TeX Live, MacTeX, "
        "or configure a manual path in fikzPy."
    )


def _engine_executable_name(engine: str) -> str:
    if engine not in LATEX_ENGINES:
        raise ValueError(f"Unsupported LaTeX engine: {engine}")
    return f"{engine}.exe" if os.name == "nt" else engine


def _common_latex_candidates(engine: str) -> list[Path]:
    executable = _engine_executable_name(engine)
    candidates: list[Path] = []

    program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    for root in filter(None, program_files):
        base = Path(root)
        candidates.extend(
            [
                base / "MiKTeX" / "miktex" / "bin" / "x64" / executable,
                base / "MiKTeX 2.9" / "miktex" / "bin" / "x64" / executable,
            ]
        )

    for year in range(2026, 2018, -1):
        candidates.append(Path("C:/texlive") / str(year) / "bin" / "windows" / executable)
        candidates.append(Path("/usr/local/texlive") / str(year) / "bin" / "x86_64-linux" / executable)

    candidates.extend(
        [
            Path("/Library/TeX/texbin") / executable,
            Path("/usr/texbin") / executable,
            Path("/opt/homebrew/bin") / executable,
            Path("/usr/local/bin") / executable,
        ]
    )

    return candidates


def _infer_distribution(path: Path) -> str:
    text = str(path).lower()
    if "miktex" in text:
        return "MiKTeX"
    if "mactex" in text or "/library/tex/" in text.replace("\\", "/"):
        return "MacTeX"
    if "texlive" in text or "tex live" in text:
        return "TeX Live"
    return "Unknown"


def _matches_distribution(path: Path, distribution: str | None) -> bool:
    if not distribution:
        return True
    inferred = _infer_distribution(path).lower()
    return distribution in inferred or inferred in distribution
