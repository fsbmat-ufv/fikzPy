"""Build a self-contained offline Windows installer from a clean build environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEX_URL = "https://github.com/rstudio/tinytex-releases/releases/download/v2026.10/TinyTeX-v2026.10.zip"
TEX_SHA256 = "dc1d6e63b21af62130ede1dcc4113bffdc184544690a23e3f489d75340f4ff61"


def sha256(path: Path) -> str:
    """Hash a large artifact without loading it entirely into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_latex() -> Path:
    """Download a pinned portable TeX distribution and verify its published digest."""
    archive = ROOT / "build/downloads/TinyTeX-v2026.10.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading {TEX_URL}", flush=True)
        urllib.request.urlretrieve(TEX_URL, archive)
    if sha256(archive) != TEX_SHA256:
        raise RuntimeError(f"TinyTeX SHA-256 mismatch: {archive}")
    destination = ROOT / "build/latex"
    if not (destination / "TinyTeX/bin/windows/pdflatex.exe").exists():
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if not (destination / entry.filename).resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError(f"Unsafe archive entry: {entry.filename}")
            zipped.extractall(destination)
    return destination / "TinyTeX"


def copy_notices(bundle: Path) -> None:
    """Ship dependency license texts, provenance, and editable application source."""
    license_root = bundle / "licenses"
    license_root.mkdir(exist_ok=True)
    packages = []
    lock = ROOT / "installer/requirements-build.lock.txt"
    for line in lock.read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        dist = metadata.distribution(name)
        if dist.version != expected:
            raise RuntimeError(f"Build environment mismatch: {name}={dist.version}, expected {expected}")
        packages.append({"name": name, "version": dist.version,
                         "license": dist.metadata.get("License-Expression") or dist.metadata.get("License", ""),
                         "project_urls": dist.metadata.get_all("Project-URL", [])})
        for file in dist.files or []:
            if any(token in str(file).lower() for token in ("license", "copying", "copyright")):
                source = Path(dist.locate_file(file))
                if source.is_file():
                    target = license_root / name / str(file).replace("../", "").replace("..\\", "")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    shutil.copy2(ROOT / "LICENSE", license_root / "fikzPy-LICENSE.txt")
    shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", license_root / "Python-LICENSE.txt")
    shutil.copytree(ROOT / "installer/licenses", license_root / "Qt-open-source", dirs_exist_ok=True)
    shutil.copy2(lock, bundle / "requirements-build.lock.txt")
    (license_root / "packages.json").write_text(json.dumps(packages, indent=2), encoding="utf-8")
    (license_root / "README.txt").write_text(
        "fikzPy: https://github.com/fsbmat-ufv/fikzPy\n"
        "Python/Qt/scientific libraries retain their respective licenses (included here).\n"
        "Qt DLLs are separate files under _internal and may be replaced by compatible builds.\n"
        "svg2tikz/inkex upstream source: https://github.com/xyz2tex/svg2tikz and "
        "https://gitlab.com/inkscape/extensions (versions in packages.json).\n"
        "TinyTeX/TeX Live license texts and package metadata remain in latex/.\n"
        "TinyTeX source: https://github.com/rstudio/tinytex-releases/releases/tag/v2026.10\n"
        "LaTeX package sources: https://ctan.org and https://tug.org/texlive/\n"
        "The included fikzPy-source.zip contains editable application and build sources.\n",
        encoding="utf-8",
    )
    # Include current working-tree sources, without private files, caches or experiments.
    with zipfile.ZipFile(bundle / "fikzPy-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("fikzpy", "installer", "scripts"):
            for source in (ROOT / directory).rglob("*"):
                if source.is_file() and source.suffix in {".py", ".iss", ".txt", ".md"} and not any(
                    part in {"__pycache__", "Output"} for part in source.relative_to(ROOT).parts
                ):
                    archive.write(source, source.relative_to(ROOT))
        for name in ("LICENSE", "README.md", "pyproject.toml", "requirements.txt"):
            archive.write(ROOT / name, name)


def validate_bundle(bundle: Path, name: str) -> Path:
    """Run all public modes with only Windows system directories on the original PATH."""
    report = ROOT / "build/installer-validation" / name / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        report.unlink()
    environment = os.environ.copy()
    windows = Path(environment.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join([str(windows / "System32"), str(windows)])
    environment["QT_QPA_PLATFORM"] = "offscreen"
    for variable in ("PYTHONPATH", "PYTHONHOME", "TEXMFCNF", "TEXMFROOT", "QT_PLUGIN_PATH"):
        environment.pop(variable, None)
    completed = subprocess.run([str(bundle / "fikzPy.exe"), "--self-test", str(report)],
                               cwd=report.parent, env=environment, timeout=240, check=False)
    if not report.is_file():
        raise RuntimeError(f"Packaged self-test did not produce {report}; exit={completed.returncode}")
    result = json.loads(report.read_text(encoding="utf-8"))
    if completed.returncode != 0 or not result["ok"]:
        raise RuntimeError(f"Packaged self-test failed: {result}")
    print(f"Validated {len(result['modes'])} public modes and PDF output: {report}", flush=True)
    return report


def validate_installation(compiler: Path, version: str) -> None:
    """Install and remove a separate test product without touching an existing fikzPy."""
    location = ROOT / "build/installer-validation/installed application"
    if location.exists():
        raise FileExistsError(f"Installation test needs a new directory: {location}")
    script = ROOT / "installer/fikzpy-offline.iss"
    subprocess.run([str(compiler), "/Q", "/DValidationBuild=1", f"/DMyAppVersion={version}",
                    "/FfikzPy-Validation-Setup", str(script)], check=True, cwd=ROOT)
    setup = ROOT / "installer/Output/fikzPy-Validation-Setup.exe"
    log = ROOT / "build/installer-validation/install.log"
    subprocess.run([str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS",
                    "/TASKS=", f"/DIR={location}", f"/LOG={log}"], check=True, timeout=240)
    try:
        validate_bundle(location, "installed")
    finally:
        uninstaller = location / "unins000.exe"
        if uninstaller.is_file():
            subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                            f"/LOG={log.with_name('uninstall.log')}"], check=True, timeout=120)
    if (location / "fikzPy.exe").exists() or (location / "latex/bin/windows/pdflatex.exe").exists():
        raise RuntimeError(f"Uninstallation left application binaries in {location}")
    # The test-only installer must never be confused with the public release asset.
    setup.unlink()
    print("Validated installation, five installed PDFs and uninstallation", flush=True)


def main() -> int:
    """Freeze, validate and wrap the application in an Inno Setup installer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, help="Path to the Inno Setup 6 compiler")
    parser.add_argument("--skip-freeze", action="store_true", help="Reuse the existing dist/fikzPy bundle")
    parser.add_argument("--validate-installation", action="store_true",
                        help="Also install/uninstall a separate test product in the build directory")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Build on Windows with 64-bit Python 3.13")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    latex = prepare_latex()
    if not args.skip_freeze:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_exe.py"), "--onedir"], cwd=ROOT, check=True)
    bundle = ROOT / "dist/fikzPy"
    shutil.copytree(latex, bundle / "latex", dirs_exist_ok=True)
    shutil.copy2(ROOT / "README.md", bundle / "README.md")
    copy_notices(bundle)
    manifest = {"version": version, "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "diff", "HEAD", "--", "fikzpy", "installer", "scripts", "pyproject.toml"], cwd=ROOT)),
        "tinytex_url": TEX_URL, "tinytex_sha256": TEX_SHA256,
        "python": sys.version, "platform": "Windows x64", "offline": True}
    (bundle / "build-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_bundle(bundle, "frozen")
    candidates = [args.iscc, Path(os.environ["LOCALAPPDATA"]) / "Programs/Inno Setup 6/ISCC.exe",
                  Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")]
    compiler = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
    if compiler is None:
        raise FileNotFoundError("Inno Setup 6 compiler not found; use --iscc")
    subprocess.run([str(compiler), "/Q", f"/DMyAppVersion={version}",
                    str(ROOT / "installer/fikzpy-offline.iss")], check=True, cwd=ROOT)
    output = ROOT / f"installer/Output/fikzPy-Setup-{version}-Windows-x64.exe"
    digest_file = output.with_suffix(".exe.sha256")
    digest_file.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    shutil.copy2(bundle / "build-manifest.json", output.with_suffix(".manifest.json"))
    if args.validate_installation:
        validate_installation(compiler, version)
    print(f"Installer: {output} ({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
