"""Stage 0: Human data snapshot (frozen).

This stage originally pulled live data from the study API.
The pre-built snapshot is shipped in figures/data/human_snapshot.json.
This stage is frozen in dvc.yaml and cannot be re-run from the public repo.
"""

raise RuntimeError(
    "This stage requires the live study API and cannot be run from the public repo. "
    "The pre-built human_snapshot.json is shipped in figures/data/."
)
