"""The pinned classroom assets ship as package data, so loaders run offline on a bare install.

These tests simulate a bare ``pip install`` outside the course repo: they run from a temp CWD
(no release lock to walk up to) with an empty user cache and ``SEAS8414_OFFLINE=1`` (no network),
and assert the loaders still resolve their bytes — from the bundled ``seas8414_teach/_data`` — and
report ``package-data`` provenance.
"""
import os

import pytest

import seas8414_teach as st
from seas8414_teach import provenance
from seas8414_teach.projects import sbom, intrusion, attack


@pytest.fixture()
def bare_install(tmp_path, monkeypatch):
    """CWD with no reachable release lock, an empty cache, and no network."""
    monkeypatch.chdir(tmp_path)                                   # nothing to walk up to
    monkeypatch.setenv("SEAS8414_CACHE_BASE", str(tmp_path / "cache"))
    monkeypatch.setenv("SEAS8414_OFFLINE", "1")                   # force a cache/bundle hit
    # Sanity: from here, the only lock discoverable is the bundled one.
    root, _ = provenance._find_release_lock()
    assert provenance._is_bundled_root(root)
    return tmp_path


def test_sbom_loads_from_bundled_data(bare_install):
    # The user's exact snippet — previously RuntimeError on bare pip (OSV had no URL).
    snapshot = sbom.load_sbom()
    assert snapshot.graph.number_of_nodes() == 168
    assert snapshot.provenance.source == "package-data"
    assert "transitive_dependents" in snapshot.nodes.columns


def test_kev_loads_offline_from_bundle(bare_install):
    kev = st.load_kev()
    assert len(kev) > 1000
    assert kev.attrs["provenance"].source == "package-data"


def test_nsl_kdd_loads_offline_from_bundle(bare_install):
    nsl = intrusion.load_nsl_kdd()
    assert nsl.train.shape[0] == 125973
    # one Provenance record per loaded file (train + test), both from the bundle
    assert [p.source for p in nsl.provenance] == ["package-data", "package-data"]


def test_attack_loads_offline_from_bundle(bare_install):
    # attack already downloaded cold (pinned SHA); confirm it also resolves offline from the bundle.
    ics = attack.load_attack_ics()
    assert len(ics.technique_ids) == 97
    assert ics.provenance.source == "package-data"
