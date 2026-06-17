from __future__ import annotations

from fikzpy.core.visual_postprocessor import postprocess_visual_tikz_picture


def test_postprocessor_converts_monolithic_path_to_draw_groups() -> None:
    tikz = "\n".join(
        [
            "\\begin{tikzpicture}[scale=1]",
            "  \\path[fill=black,even odd rule] (0,0) -- (1,0) -- (1,1) -- cycle "
            "(3,0) -- (4,0) -- (4,1) -- cycle;",
            "\\end{tikzpicture}",
        ]
    )

    result = postprocess_visual_tikz_picture(tikz)

    assert result.stats.changed
    assert result.stats.subpaths == 2
    assert result.stats.output_draw_commands == 1
    assert "\\path[fill=black" not in result.tikz_picture
    assert result.tikz_picture.count("\\draw[fikzInk]") == 1


def test_postprocessor_turns_nested_hole_into_eraser_layer() -> None:
    tikz = "\n".join(
        [
            "\\begin{tikzpicture}[scale=1]",
            "  \\path[fill=black,even odd rule] (0,0) -- (4,0) -- (4,4) -- (0,4) -- cycle "
            "(1,1) -- (3,1) -- (3,3) -- (1,3) -- cycle;",
            "\\end{tikzpicture}",
        ]
    )

    result = postprocess_visual_tikz_picture(tikz)

    assert result.stats.output_draw_commands == 2
    assert result.tikz_picture.count("-- cycle") == 2
    assert "\\draw[fikzInk]" in result.tikz_picture
    assert "\\draw[fikzErase]" in result.tikz_picture


def test_postprocessor_preserves_cubic_bezier_segments() -> None:
    tikz = "\n".join(
        [
            "\\begin{tikzpicture}[scale=1]",
            "  \\path[fill=black,even odd rule] (0,0) .. controls (0.5,1) and (1.5,1) .. (2,0) "
            "-- (2,2) -- cycle;",
            "\\end{tikzpicture}",
        ]
    )

    result = postprocess_visual_tikz_picture(tikz)

    assert ".. controls" in result.tikz_picture
    assert "\\draw[fikzInk]" in result.tikz_picture
