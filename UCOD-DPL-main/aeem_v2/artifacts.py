"""Immutable experiment artifact helpers for AEEM v2."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def create_experiment_directory(artifact_root: Path, experiment_id: str) -> Path:
    if not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id must start with an alphanumeric character and only "
            "contain letters, numbers, '.', '_' or '-'"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = artifact_root / experiment_id
    experiment_dir.mkdir(exist_ok=False)
    return experiment_dir


def _run_git(repo_root: Path, arguments: Iterable[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def capture_git_state(repo_root: Path, experiment_dir: Path) -> Dict[str, Any]:
    diff_result = _run_git(repo_root, ["diff", "--no-ext-diff", "--no-color"])
    diff_path = experiment_dir / "git_diff.patch"
    diff_path.write_text(diff_result.stdout, encoding="utf-8")

    status_result = _run_git(repo_root, ["status", "--short"])
    commit_result = _run_git(repo_root, ["rev-parse", "HEAD"])

    return {
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "diff_returncode": diff_result.returncode,
        "diff_sha256": sha256_file(diff_path),
        "status_returncode": status_result.returncode,
        "status": status_result.stdout.splitlines(),
    }
