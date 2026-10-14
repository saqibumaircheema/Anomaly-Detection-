"""Preprocessing utilities for the Dark Machines channel-2b event format."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


N_PARTICLES = 8
N_PARTICLE_FEATURES = 4
LOG_EPSILON = 1e-6


def find_data_start_line(filepath: str | os.PathLike[str]) -> int:
    """Return the number of non-data lines before the first event."""
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(";")
                if len(parts) >= 5:
                    float(parts[3])
                    float(parts[4])
                    return i
            except ValueError:
                continue

    raise ValueError(f"No valid event row found in {filepath}")


def process_file(filepath: str | os.PathLike[str], skip_rows: int = 0) -> np.ndarray:
    """Convert raw events into the 39-feature representation used by the model."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    event_features: list[list[float]] = []

    with filepath.open("r", encoding="utf-8") as handle:
        for _ in range(skip_rows):
            next(handle)

        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";")
            met = float(parts[3])
            met_phi = float(parts[4])
            particle_objects = parts[5:]

            particles: list[tuple[float, float, float, float, str]] = []
            counts = {"ele": 0, "muon": 0, "jet": 0, "bjet": 0, "photon": 0}

            for object_string in particle_objects:
                if not object_string:
                    continue
                values = object_string.split(",")
                if len(values) < 5:
                    continue

                object_type = values[0]
                if object_type in ("e+", "e-"):
                    counts["ele"] += 1
                elif object_type in ("m+", "m-"):
                    counts["muon"] += 1
                elif object_type == "j":
                    counts["jet"] += 1
                elif object_type == "b":
                    counts["bjet"] += 1
                elif object_type == "g":
                    counts["photon"] += 1

                particles.append(
                    (
                        float(values[1]),
                        float(values[2]),
                        float(values[3]),
                        float(values[4]),
                        object_type,
                    )
                )

            particles.sort(key=lambda particle: particle[0], reverse=True)

            features = [
                counts["ele"],
                counts["muon"],
                counts["jet"],
                counts["bjet"],
                counts["photon"],
                np.log(met + LOG_EPSILON),
                met_phi,
            ]

            for i in range(N_PARTICLES):
                if i < len(particles):
                    energy, pt, eta, phi, _ = particles[i]
                    features.extend(
                        [
                            np.log(energy + LOG_EPSILON),
                            np.log(pt + LOG_EPSILON),
                            eta,
                            phi,
                        ]
                    )
                else:
                    features.extend([0.0] * N_PARTICLE_FEATURES)

            event_features.append(features)

    return np.asarray(event_features, dtype=np.float32)
