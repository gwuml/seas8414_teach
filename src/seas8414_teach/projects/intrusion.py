"""Loader for Phase 10 Project 1 — Intrusion Detection Under Dataset Shift (NSL-KDD).

Hides the download / SHA-verify / cache plumbing behind :func:`load_nsl_kdd`; the freeze
discipline, model choices, calibration, threshold, and audits stay VISIBLE in the notebook. The
parse (43 NSL-KDD columns, the DOS/PROBE/R2L/U2R family mapping, the label rule) is ported
line-for-line from the transparent full edition so the teach edition produces the same
load-bearing decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..provenance import Provenance, _find_release_lock, fetch

PROJECT_ID = "phase10-01-intrusion-shift"

# NSL-KDD is redistributed under the official UNB citation terms; cite Tavallaee et al. (2009).
NSL_COMMIT = "9d544d0eb9b87d7e2f43ff65733bdb644631d12f"
TRAIN_URL = (
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/"
    f"{NSL_COMMIT}/KDDTrain%2B.txt"
)
TEST_URL = (
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/"
    f"{NSL_COMMIT}/KDDTest%2B.txt"
)

# The 43 published NSL-KDD columns (41 features + attack_name + difficulty).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_name", "difficulty",
]

# The 41 flow features used for fitting (everything but the two label/meta columns).
FEATURE_COLUMNS = [c for c in COLUMNS if c not in {"attack_name", "difficulty"}]

# Attack-family mapping (benchmark taxonomy, not a current operational taxonomy).
DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
       "apache2", "processtable", "udpstorm"}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {"guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
       "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
       "httptunnel", "sendmail", "named"}
U2R = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm",
       "ps"}


def attack_family(name: str) -> str:
    """Map a raw NSL-KDD attack name to its benchmark family."""
    name = name.strip().lower()
    if name == "normal":
        return "normal"
    if name in DOS:
        return "dos"
    if name in PROBE:
        return "probe"
    if name in R2L:
        return "r2l"
    if name in U2R:
        return "u2r"
    return "other-attack"


@dataclass
class NslKdd:
    """The two fixed NSL-KDD partitions with feature/label/family columns.

    ``train`` (125,973 rows) and ``test`` (22,544 rows) are the published partitions, unmodified —
    no row is removed or deduplicated. ``feature_columns`` is the 41-feature fitting list;
    ``categorical`` names the three categorical features. ``provenance`` holds one
    :class:`~seas8414_teach.provenance.Provenance` record per loaded file.
    """

    train: Any
    test: Any
    feature_columns: list[str]
    categorical: list[str]
    provenance: list[Provenance]


def _release_sha(index: dict, filename: str) -> str | None:
    entry = index.get((PROJECT_ID, filename)) if index is not None else None
    return entry["sha256"] if entry else None


def _parse_partition(path, columns: list[str]):
    """Read one NSL-KDD partition and attach label + attack_family, exactly as the full edition."""
    import pandas as pd

    frame = pd.read_csv(path, names=columns)
    frame["attack_name"] = frame["attack_name"].str.strip().str.lower()
    frame["label"] = frame["attack_name"].ne("normal").astype(int)
    frame["attack_family"] = frame["attack_name"].map(attack_family)
    return frame.reset_index(drop=True)


def load_nsl_kdd() -> NslKdd:
    """Load both NSL-KDD partitions (train + test) with 41 features, label, and family mapping.

    Hides the download / SHA-verify / release-cache plumbing (`fetch`). The SHA-256 for each file
    comes from the release lock so this stays a single source of truth. Returns tidy train/test
    DataFrames plus inspectable :class:`~seas8414_teach.provenance.Provenance` records.
    """
    import pandas as pd  # noqa: F401 — imported for the class-run kernel; parse uses it lazily

    located = _find_release_lock()
    index = located[1] if located is not None else None

    train_raw, train_prov = fetch(
        PROJECT_ID, "KDDTrain+-original.txt",
        expected_sha256=_release_sha(index, "KDDTrain+-original.txt"),
        max_bytes=20_000_000, url=TRAIN_URL,
    )
    test_raw, test_prov = fetch(
        PROJECT_ID, "KDDTest+-original.txt",
        expected_sha256=_release_sha(index, "KDDTest+-original.txt"),
        max_bytes=20_000_000, url=TEST_URL,
    )

    from ..provenance import _cache_base

    train_path = _cache_base() / PROJECT_ID / "KDDTrain+-original.txt"
    test_path = _cache_base() / PROJECT_ID / "KDDTest+-original.txt"

    train = _parse_partition(train_path, COLUMNS)
    test = _parse_partition(test_path, COLUMNS)
    # 43 published columns + our label + attack_family = 45.
    assert train.shape[1] == 45 and len(train) == 125_973, "unexpected train shape"
    assert test.shape[1] == 45 and len(test) == 22_544, "unexpected test shape"

    return NslKdd(
        train=train,
        test=test,
        feature_columns=list(FEATURE_COLUMNS),
        categorical=["protocol_type", "service", "flag"],
        provenance=[train_prov, test_prov],
    )
