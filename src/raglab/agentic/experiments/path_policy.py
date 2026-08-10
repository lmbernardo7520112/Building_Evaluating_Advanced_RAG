"""Experimental Path Policy Core — RAGLab V7 Experimental Readiness.

Validates artifact roots and run directories fail-closed, ensuring experiment outputs
are written strictly to durable, external, explicitly allowed directory locations.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathPolicyError(Exception):
    """Raised when an artifact root or run directory violates path policy."""


def _is_safe_identifier(identifier: str) -> bool:
    """Validate slice_id or run_id for traversal or forbidden characters."""
    if not identifier or not isinstance(identifier, str):
        return False
    if (
        ".." in identifier
        or "/" in identifier
        or "\\" in identifier
        or "\0" in identifier
    ):
        return False
    p = Path(identifier)
    return not p.is_absolute() and len(p.parts) == 1


class ExperimentalPathPolicy:
    """Enforces durable, external artifact storage for experiment runs."""

    def __init__(
        self,
        repository_root: Path | str | None = None,
        allowed_roots: list[Path | str] | tuple[Path | str, ...] | None = None,
    ) -> None:
        """Initialize path policy with repository_root and optional allowed_roots."""
        if repository_root is None:
            self.repo_root = Path(__file__).parents[3].resolve()
        else:
            self.repo_root = Path(repository_root).resolve()

        if allowed_roots is not None:
            validated_allowed: list[Path] = []
            for root in allowed_roots:
                root_str = str(root)
                if not root_str or not root_str.strip():
                    raise PathPolicyError("Allowed root cannot be empty")
                self._check_forbidden_candidate(root_str, is_allowed_root=True)
                validated_allowed.append(Path(root_str).resolve())
            self.allowed_roots: list[Path] | None = validated_allowed
        else:
            self.allowed_roots = None

    def _check_forbidden_candidate(
        self, path_str: str, is_allowed_root: bool = False
    ) -> None:
        """Validate candidate path against forbidden root rules."""
        if not path_str or not str(path_str).strip():
            raise PathPolicyError("Target directory path cannot be empty")

        p = Path(path_str)
        if not p.is_absolute():
            raise PathPolicyError(
                f"Target directory must be an absolute path, got '{path_str}'"
            )

        if ".." in p.parts:
            raise PathPolicyError(
                f"Path traversal ('..') detected in path: '{path_str}'"
            )

        resolved = p.resolve()

        if resolved == Path("/"):
            raise PathPolicyError("Root directory '/' is strictly forbidden")

        if resolved == Path.home().resolve():
            raise PathPolicyError(
                "User home directory is strictly forbidden as artifact root"
            )

        # /tmp e /var/tmp
        for tmp_dir_str in ("/tmp", "/var/tmp"):  # noqa: S108
            tmp_p = Path(tmp_dir_str).resolve()
            if resolved == tmp_p or tmp_p in resolved.parents:
                raise PathPolicyError(
                    f"Temporary directory '{tmp_dir_str}' and its descendants "
                    "are strictly forbidden"
                )
            if path_str.startswith(tmp_dir_str):
                raise PathPolicyError(
                    f"Temporary path prefix '{tmp_dir_str}' is strictly forbidden"
                )

        # Repository root and descendants
        if resolved == self.repo_root or self.repo_root in resolved.parents:
            raise PathPolicyError(
                f"Repository directory '{self.repo_root}' and its descendants "
                "are strictly forbidden"
            )

        # benchmarks/ and checkpoints/ inside repo or as named paths
        benchmarks_dir = (self.repo_root / "benchmarks").resolve()
        checkpoints_dir = (self.repo_root / "checkpoints").resolve()

        if (
            resolved == benchmarks_dir
            or benchmarks_dir in resolved.parents
            or "benchmarks" in p.parts
            or "benchmarks" in resolved.parts
        ):
            raise PathPolicyError(
                "benchmarks directory and its descendants are strictly forbidden"
            )

        if (
            resolved == checkpoints_dir
            or checkpoints_dir in resolved.parents
            or "checkpoints" in p.parts
            or "checkpoints" in resolved.parts
        ):
            raise PathPolicyError(
                "checkpoints directory and its descendants are strictly forbidden"
            )

        if p.is_symlink() or (p.exists() and os.path.islink(p)):
            symlink_target = p.resolve()
            if (
                symlink_target == self.repo_root
                or self.repo_root in symlink_target.parents
            ):
                raise PathPolicyError(
                    f"Symlink target '{symlink_target}' is inside repository root"
                )
            for tmp_dir_str in ("/tmp", "/var/tmp"):  # noqa: S108
                tmp_p = Path(tmp_dir_str).resolve()
                if symlink_target == tmp_p or tmp_p in symlink_target.parents:
                    raise PathPolicyError(
                        f"Symlink target '{symlink_target}' is inside "
                        f"temporary directory '{tmp_dir_str}'"
                    )

        if (
            not is_allowed_root
            and self.allowed_roots is not None
            and not any(
                allowed == resolved or allowed in resolved.parents
                for allowed in self.allowed_roots
            )
        ):
            raise PathPolicyError(
                f"Artifact root '{resolved}' is not within allowed_roots list"
            )

    def validate_target_directory(
        self, target_directory: Path | str, must_be_empty: bool = False
    ) -> Path:
        """Validate target_directory fail-closed without modifying filesystem."""
        if target_directory is None or (
            isinstance(target_directory, str) and target_directory == ""
        ):
            raise PathPolicyError("Target directory path cannot be empty")

        path_str = str(target_directory)
        self._check_forbidden_candidate(path_str)

        resolved = Path(path_str).resolve()

        if must_be_empty and resolved.exists():
            if resolved.is_file():
                raise PathPolicyError(
                    f"Target path is an existing file, not a directory: {resolved}"
                )
            if resolved.is_dir() and any(resolved.iterdir()):
                raise PathPolicyError(
                    f"Target directory is not empty: {resolved}"
                )

        return resolved

    def derive_run_directory(
        self, artifact_root: Path | str, slice_id: str, run_id: str
    ) -> Path:
        """Derive deterministic run directory path <root>/<slice_id>/<run_id>."""
        if not _is_safe_identifier(slice_id):
            raise PathPolicyError(f"Invalid or unsafe slice_id: '{slice_id}'")
        if not _is_safe_identifier(run_id):
            raise PathPolicyError(f"Invalid or unsafe run_id: '{run_id}'")

        validated_root = self.validate_target_directory(artifact_root)
        run_dir = (validated_root / slice_id / run_id).resolve()

        try:
            run_dir.relative_to(validated_root)
        except ValueError:
            raise PathPolicyError(
                f"Run directory '{run_dir}' escapes artifact_root '{validated_root}'"
            ) from None

        return run_dir
