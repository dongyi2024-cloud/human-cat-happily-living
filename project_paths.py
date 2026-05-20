from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve repo-relative paths against the project root."""
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_project_dir(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    if resolved is None:
        raise ValueError("path 不能为空")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def project_relative_display(path: str | Path) -> str:
    resolved = resolve_project_path(path)
    if resolved is None:
        return ""
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
