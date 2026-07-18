"""Dataset loaders with hidden provenance.

The full notebooks devote ~200 lines to download-once / SHA-verify / release-lock caching. This
module hides that plumbing behind ``load_*`` functions. Each returns tidy data plus a
``Provenance`` record you can inspect (``.sha256``, ``.source``, ``.bytes``).

Discovery order for the immutable classroom bytes:
1. ``SEAS8414_CACHE_BASE`` user cache (already-verified copy),
2. the release cache (``phase09-phase10-release-cache-lock.json``) — a lock reachable from CWD
   (course repo / pre-warmed tree, walking up) or, failing that, the copy shipped as package data
   (``seas8414_teach/_data``), so a bare ``pip install`` resolves the pinned assets offline,
3. the declared public URL (only when not ``SEAS8414_OFFLINE`` and no cache hit).

The pinned classroom assets ship in the wheel as of 0.1.4, so ``load_*`` runs offline out of the
box; the CWD lock still takes precedence for course-repo work.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

SEED = 8414
RELEASE_LOCK_NAME = "phase09-phase10-release-cache-lock.json"
RELEASE_LOCK_EXPECTED_SHA256 = (
    "2cc2ffb39bb5b6d93ce5b927e4b01e1e2a0e51f6b8fb26f127cb5471bb5960e2"
)


@dataclass(frozen=True)
class Provenance:
    """Inspectable provenance for one loaded asset."""

    filename: str
    sha256: str
    bytes: int
    source: str
    url: str | None = None
    acquired_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source": self.source,
            "url": self.url,
            "acquired_at_utc": self.acquired_at,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _offline() -> bool:
    return os.environ.get("SEAS8414_OFFLINE", "0") == "1"


def _cache_base() -> Path:
    return Path(
        os.environ.get(
            "SEAS8414_CACHE_BASE",
            Path.home() / ".cache" / "seas8414-phase-projects",
        )
    )


def _bundled_data_root() -> Any:
    """The release-cache assets shipped as package data (``seas8414_teach/_data``), or ``None``.

    Present in a normal wheel/sdist install; absent only if the package was installed without its
    data (e.g. a hand-trimmed build). Returns an ``importlib.resources`` traversable.
    """
    try:
        root = resources.files("seas8414_teach") / "_data"
    except (ModuleNotFoundError, AttributeError, TypeError):
        return None
    try:
        return root if root.is_dir() else None
    except (FileNotFoundError, OSError):
        return None


def _is_bundled_root(root: Any) -> bool:
    bundled = _bundled_data_root()
    return bundled is not None and str(root) == str(bundled)


def _read_and_index_lock(lock_path: Any, root: Any):
    raw = lock_path.read_bytes()
    if _sha256(raw) != RELEASE_LOCK_EXPECTED_SHA256:
        raise ValueError(f"Release-cache lock SHA-256 mismatch at {lock_path}")
    lock = json.loads(raw)
    index = {(entry["project"], entry["filename"]): entry for entry in lock["entries"]}
    return root, index


def _find_release_lock() -> tuple[Any, dict[tuple[str, str], dict[str, Any]]] | None:
    """Find and verify the release lock; return (root, entry index).

    Prefers a lock reachable from CWD (the course repo or a pre-warmed tree, walking up); otherwise
    falls back to the copy shipped as package data, so a bare ``pip install`` still resolves the
    pinned classroom assets offline.
    """
    roots = [Path.cwd(), *Path.cwd().parents]
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / RELEASE_LOCK_NAME)
        candidates.append(root / "docs" / "course" / "notebooks" / RELEASE_LOCK_NAME)
    for lock_path in candidates:
        if lock_path.is_file():
            return _read_and_index_lock(lock_path, lock_path.parent)

    bundled_root = _bundled_data_root()
    if bundled_root is not None:
        lock_path = bundled_root / RELEASE_LOCK_NAME
        if lock_path.is_file():
            return _read_and_index_lock(lock_path, bundled_root)
    return None


def fetch(
    project: str,
    filename: str,
    *,
    expected_sha256: str,
    max_bytes: int = 20_000_000,
    url: str | None = None,
) -> tuple[bytes, Provenance]:
    """Return verified bytes for ``filename`` and its provenance.

    Prefers the user cache, then the release cache (repo tree or bundled package data), then the
    public URL (online only). Every path enforces the size ceiling and SHA-256.
    """
    user_path = _cache_base() / project / filename
    if user_path.exists():
        raw = user_path.read_bytes()
        if _sha256(raw) != expected_sha256:
            raise ValueError(f"Cached SHA-256 mismatch for {filename}")
        return raw, Provenance(filename, expected_sha256, len(raw), "user-cache", url)

    located = _find_release_lock()
    if located is not None:
        root, index = located
        entry = index.get((project, filename))
        if entry is not None:
            # joinpath segment-by-segment so both a filesystem Path and an importlib traversable
            # (bundled package data) resolve the relative path identically.
            asset = root.joinpath(*entry["relative_path"].split("/"))
            raw = asset.read_bytes()
            if len(raw) != entry["bytes"] or len(raw) > max_bytes:
                raise ValueError(f"Release-cache size mismatch for {filename}")
            if _sha256(raw) != expected_sha256 or _sha256(raw) != entry["sha256"]:
                raise ValueError(f"Release-cache SHA-256 mismatch for {filename}")
            source = "package-data" if _is_bundled_root(root) else "release-cache"
            user_path.parent.mkdir(parents=True, exist_ok=True)
            user_path.write_bytes(raw)
            return raw, Provenance(
                filename, expected_sha256, len(raw), source, url,
                entry.get("acquired_at_utc"),
            )

    if _offline():
        raise RuntimeError(
            f"Offline and no cached copy of {filename}. Run inside the SEAS-8414 repo (release "
            f"cache present) or set SEAS8414_OFFLINE=0 with network access."
        )
    if url is None:
        raise RuntimeError(f"No cached copy of {filename} and no URL to fetch it from.")
    import requests  # lazy: online path only

    response = requests.get(url, timeout=(10, 90))
    response.raise_for_status()
    raw = response.content
    if len(raw) > max_bytes:
        raise ValueError(f"{filename} exceeds the {max_bytes:,}-byte ceiling")
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"Downloaded SHA-256 mismatch for {filename}")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_bytes(raw)
    return raw, Provenance(filename, expected_sha256, len(raw), "network", url)


# --------------------------------------------------------------------------- loaders

_KEV_SHA = "337e9a41ccd4"  # short; full verified via release lock entry
_KEV_FULL_SHA = None


def load_kev():
    """CISA Known-Exploited-Vulnerabilities catalog as a tidy DataFrame (+ provenance)."""
    import pandas as pd

    # SHA taken from the release lock entry so this stays a single source of truth.
    located = _find_release_lock()
    sha = None
    if located is not None:
        _, index = located
        entry = index.get(("phase09-01-kev-triage", "known_exploited_vulnerabilities.json"))
        sha = entry["sha256"] if entry else None
    raw, prov = fetch(
        "phase09-01-kev-triage", "known_exploited_vulnerabilities.json",
        expected_sha256=sha, max_bytes=3_000_000,
        url="https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    )
    catalog = json.loads(raw.decode("utf-8"))
    kev = pd.DataFrame(catalog["vulnerabilities"])
    kev["dateAdded"] = pd.to_datetime(kev["dateAdded"], utc=True)
    kev.attrs["provenance"] = prov
    return kev


@dataclass
class CowrieDay:
    """One fixed Cowrie observation day.

    ``sessions`` is the per-session feature frame (indexed by session id). ``sequences`` and
    ``command_tokens`` hold the per-session event-state lists and sanitized first command tokens
    used by :func:`seas8414_teach.methods.markov_surprise`.
    """

    sessions: Any
    sequences: dict[str, list[str]]
    command_tokens: dict[str, list[str]]
    provenance: Provenance


def _event_state(event_id: str) -> str:
    if "session.connect" in event_id:
        return "connect"
    if "client." in event_id:
        return "fingerprint"
    if "login.failed" in event_id:
        return "auth-fail"
    if "login.success" in event_id:
        return "auth-success"
    if "command." in event_id:
        return "command"
    if "file_download" in event_id:
        return "download"
    if "session.closed" in event_id:
        return "close"
    return "other"


def _command_token(message: object) -> str:
    text = str(message or "")
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    token = re.split(r"[\s;&|><]+", text)[0].strip()
    token = re.sub(r"[^A-Za-z0-9_.-]", "", token)
    return token[:32] or "unknown"


def load_cowrie() -> CowrieDay:
    """Parse one fixed CyberLab Cowrie day into privacy-preserving session features.

    Reproduces the full notebook's parse exactly, including the held-out ``pivot_escalation_outcome``
    (a ``direct-tcpip.request`` jump-host session) that is independent of the baseline's scoring
    signals. No raw identifiers, passwords, or full commands are surfaced.
    """
    import numpy as np
    import pandas as pd

    located = _find_release_lock()
    sha = None
    if located is not None:
        _, index = located
        entry = index.get(("phase10-02-honeypot-profiling", "cyberlab_2019-05-13.json.gz"))
        sha = entry["sha256"] if entry else None
    raw, prov = fetch(
        "phase10-02-honeypot-profiling", "cyberlab_2019-05-13.json.gz",
        expected_sha256=sha, max_bytes=3_000_000,
        url="https://zenodo.org/records/3687527/files/cyberlab_2019-05-13.json.gz?download=1",
    )
    raw_sessions = json.loads(gzip.decompress(raw))

    rows: list[dict[str, Any]] = []
    sequences: dict[str, list[str]] = {}
    tokens: dict[str, list[str]] = {}
    for group in raw_sessions:
        session_id, events = next(iter(group.items()))
        event_ids = [e.get("eventid", "") for e in events]
        states = [_event_state(x) for x in event_ids]
        commands = [
            _command_token(e.get("message")) for e in events
            if "command." in e.get("eventid", "")
        ]
        timestamps = pd.to_datetime(
            [e.get("timestamp") for e in events], utc=True, errors="coerce"
        )
        duration = (
            (timestamps.max() - timestamps.min()).total_seconds()
            if len(timestamps) and not timestamps.isna().all() else 0.0
        )
        protocols = [str(e.get("protocol") or "unknown") for e in events]
        rows.append({
            "session_id": session_id,
            "event_count": len(events),
            "duration_seconds": max(0.0, float(duration)),
            "unique_event_types": len(set(event_ids)),
            "login_failed": sum("login.failed" in x for x in event_ids),
            "login_success": sum("login.success" in x for x in event_ids),
            "command_success": sum("command.success" in x for x in event_ids),
            "command_failed": sum("command.failed" in x for x in event_ids),
            "file_downloads": sum("file_download" in x for x in event_ids),
            "unique_command_tokens": len(set(commands)),
            "telnet_fraction": float(np.mean([p == "telnet" for p in protocols])),
            "has_client_fingerprint": int(any("fingerprint" in x for x in event_ids)),
            "direct_tcpip_requests": sum("direct-tcpip.request" in x for x in event_ids),
        })
        sequences[session_id] = states
        tokens[session_id] = commands

    sessions = pd.DataFrame(rows).set_index("session_id")
    sessions["pivot_escalation_outcome"] = (
        sessions["direct_tcpip_requests"].gt(0).astype(int)
    )
    return CowrieDay(sessions, sequences, tokens, prov)
