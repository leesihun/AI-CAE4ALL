"""Early validation for cHI-MGNflow runtime-only configuration contracts."""

from __future__ import annotations


def validate_runtime_config(config: dict) -> str:
    """Validate choices that must be settled before a runtime branch is imported.

    cHI-MGNflow currently has a data-parallel trainer only.  The historical
    ``parallelism`` directory is a copied variational-MGN pipeline and does not
    implement the flow-matching model, so exposing it as a selectable route
    would fail after launch with imports and losses from the wrong architecture.

    Returns the canonical parallel mode for the caller's status output.
    """
    parallel_mode = str(config.get("parallel_mode", "ddp")).lower().strip()
    if parallel_mode != "ddp":
        if parallel_mode == "model_split":
            raise ValueError(
                "parallel_mode=model_split is not implemented for cHI-MGNflow; "
                "use parallel_mode=ddp (one GPU runs the single-device trainer)."
            )
        raise ValueError(
            f"parallel_mode must be 'ddp' for cHI-MGNflow, got {parallel_mode!r}"
        )

    # These controls belonged to the incompatible copied model-split path.
    # Keep an explicit zero std_noise as a transition-compatible no-op because
    # older shipped configs contain it, but never pretend a non-zero value works.
    if "pipeline_microbatches" in config:
        raise ValueError(
            "pipeline_microbatches is unavailable because cHI-MGNflow does not "
            "implement parallel_mode=model_split."
        )
    for key in ("noise_gamma", "noise_std_ratio"):
        if key in config:
            raise ValueError(
                f"{key} is not consumed by the cHI-MGNflow trainer; delete it."
            )
    if "std_noise" in config:
        try:
            std_noise = float(config["std_noise"])
        except (TypeError, ValueError) as exc:
            raise ValueError("std_noise must be numeric when present.") from exc
        if std_noise != 0.0:
            raise ValueError(
                "std_noise is not implemented by the cHI-MGNflow trainer; "
                "only the legacy compatibility value 0 is accepted."
            )

    return parallel_mode
