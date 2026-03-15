"""
redteam/scenarios.py
====================
Structured red-team scenario definitions.

Each scenario applies one or more perturbations at a given severity to
the full synthetic dataset and records expected vs. actual model behaviour.

Scenarios are intentionally conservative — they test robustness properties
relevant to humanitarian monitoring (e.g. degraded sensor data, partial
occlusion from cloud cover) rather than attack vectors.

Safety note
-----------
Scenarios must only be run against synthetic data.
Results are for research and model-improvement purposes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """
    A single robustness test scenario.

    Parameters
    ----------
    name          : str   short identifier
    description   : str   human-readable description
    perturbations : list  each entry is (perturbation_name, severity_int)
    expected_min_acc : float  minimum acceptable accuracy (0–1)
                              used as pass/fail threshold in the eval report
    """

    name: str
    description: str
    perturbations: list[tuple[str, int]] = field(default_factory=list)
    expected_min_acc: float = 0.0


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="clean",
        description="Baseline — no perturbation. Establishes reference accuracy.",
        perturbations=[],
        expected_min_acc=0.0,  # no threshold on baseline
    ),
    Scenario(
        name="mild_noise",
        description="Low-level Gaussian noise simulating sensor read noise.",
        perturbations=[("gaussian_noise", 0)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="severe_noise",
        description="High Gaussian noise simulating heavily degraded imagery.",
        perturbations=[("gaussian_noise", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="transmission_artefacts",
        description="Heavy JPEG compression simulating bandwidth-limited downlink.",
        perturbations=[("jpeg_compression", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="sensor_blur",
        description="Gaussian blur simulating sensor defocus or atmospheric distortion.",
        perturbations=[("gaussian_blur", 1)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="cloud_occlusion_mild",
        description="Small black patch simulating partial cloud cover (10% of image).",
        perturbations=[("occlusion_patch", 0)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="cloud_occlusion_severe",
        description="Large black patch simulating heavy cloud cover (40% of image).",
        perturbations=[("occlusion_patch", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="brightness_overexposed",
        description="Bright shift simulating overexposed imagery.",
        perturbations=[("brightness_shift", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="low_contrast",
        description="Contrast reduction simulating hazy or foggy conditions.",
        perturbations=[("contrast_scale", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="sensor_tilt",
        description="45-degree rotation simulating off-nadir sensor angle.",
        perturbations=[("rotation", 2)],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="combined_degraded",
        description=(
            "Combined: Gaussian blur + mild noise + brightness shift. "
            "Simulates multiple simultaneous degradation sources."
        ),
        perturbations=[
            ("gaussian_blur", 1),
            ("gaussian_noise", 0),
            ("brightness_shift", 0),
        ],
        expected_min_acc=0.0,
    ),
    Scenario(
        name="worst_case",
        description=(
            "Worst case: severe noise + heavy occlusion + heavy compression. "
            "Tests model failure modes under extreme degradation."
        ),
        perturbations=[
            ("gaussian_noise", 2),
            ("occlusion_patch", 2),
            ("jpeg_compression", 2),
        ],
        expected_min_acc=0.0,
    ),
]

SCENARIO_MAP: dict[str, Scenario] = {s.name: s for s in SCENARIOS}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIO_MAP:
        raise ValueError(
            f"Unknown scenario {name!r}. Available: {sorted(SCENARIO_MAP.keys())}"
        )
    return SCENARIO_MAP[name]
