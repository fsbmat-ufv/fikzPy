# Windows offline installer — v0.1.1

Validated on 2026-10-30. The previous download-based installer required Python
and internet access. The new installer ships a frozen application and private
TinyTeX, including TikZ and standalone, so the generated documents compile offline.

## Validation performed

- All 700 regression tests passed (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, Qt offscreen).
- A clean Python 3.13 x64 build environment used the pinned build lockfile.
- The TinyTeX v2026.10 ZIP matched the upstream SHA-256 recorded in the build script.
- The frozen executable initialized the unchanged main window, exercised all five
  public modes and compiled their PDFs. Visual was required to use svg2tikz.
- The same test passed after installing the payload in a path containing spaces.
- Test installation used a separate AppId and disabled shortcuts; uninstall
  reported all installed files removed, without requiring a restart.
- The original user's installation registry values and Start Menu shortcut hash
  were compared before and after and remained unchanged.

| Public mode | Compiled PDF size (bytes) |
| --- | ---: |
| Classic Auto | 1,214 |
| Classic Filled | 1,214 |
| Visual | 2,933 |
| Contornos | 1,532 |
| Ponto a Ponto | 13,126 |

Both executable tests ran from outside the application directory with only
Windows system directories on the original PATH. The packaged launcher exposed
the private LaTeX directory to its own process. No installed Python or external
LaTeX was used. Application modes, GUI layout and compiler source were unchanged.

## Reproducing and reviewing

Follow `installer/README.md`, then run the builder with `--validate-installation`.
Reports, five TEX/PDF pairs, compilation logs and install/uninstall logs are saved
under `build/installer-validation/`. The test installer has a distinct AppId and
an uncompressed payload; the public installer uses LZMA2 compression. Both use
the same file list, application executable and private TeX distribution.

The build environment PATH is isolated as well: an initial attempt revealed that
a developer tool's versioned ICU DLL was being collected instead of the Windows
ICU API required by Qt. The isolated build resolved this packaging conflict.

## Limits

Validation used this Windows host and an isolated runtime environment, not a
fresh Windows VM or another physical computer. The supported target is Windows
10/11 x64. The binary is unsigned. Custom user-authored LaTeX can require extra
packages; optional external raster tracers are not included. The legacy online
installer remains available in source as a fallback.
