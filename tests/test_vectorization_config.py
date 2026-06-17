from __future__ import annotations

from fikzpy.core.vectorization_config import VectorizationConfig, config_for_mode


def test_classic_config_preserves_stable_mode() -> None:
    config = VectorizationConfig.classic()

    assert config.mode == "classic"
    assert not config.preprocessing.use_bilateral_filter
    assert not config.merging.enabled
    assert not config.smoothing.enabled


def test_smooth_config_enables_optional_steps() -> None:
    config = VectorizationConfig.smooth()

    assert config.mode == "smooth"
    assert config.preprocessing.use_bilateral_filter
    assert config.merging.enabled
    assert config.smoothing.enabled


def test_config_for_mode_keeps_line_art_alias() -> None:
    assert config_for_mode("line_art").mode == "classic"
    assert config_for_mode("vector").mode == "vector"
    assert config_for_mode("fidelity").mode == "fidelity"
    assert config_for_mode("fidelidade").mode == "fidelity"
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("svg_trace").mode == "visual"
    assert config_for_mode("smooth").mode == "smooth"
    assert config_for_mode("contours").mode == "contours"
