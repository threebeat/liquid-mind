"""Provenance / artifact-safety gate (Priority 0)."""
import json
import os

import pytest
import torch

from provenance import (SEMANTICS_VERSION, checkpoint_ref, file_checksum,
                        gather_provenance, import_legacy_checkpoint,
                        load_checkpoint, save_checkpoint, write_results)


def _state():
    return {"w": torch.arange(4, dtype=torch.float32)}


def _meta():
    return gather_provenance({"a": 1}, experiment_name="unit_test",
                             seeds={"seed": 0})


COMPAT = {"semantics_version": SEMANTICS_VERSION, "obs_dim": 22,
          "policy_input": "spikes_obs", "mask_direct_dt": True}


def test_roundtrip_and_metadata(tmp_path):
    path = str(tmp_path / "ck.pt")
    sha = save_checkpoint(path, _state(), _meta(), COMPAT)
    assert sha == file_checksum(path)
    state, meta = load_checkpoint(path, expected_compat=COMPAT)
    assert torch.equal(state["w"], _state()["w"])
    assert meta["experiment"] == "unit_test"
    assert meta["semantics_version"] == SEMANTICS_VERSION
    assert meta["compat"]["mask_direct_dt"] is True
    assert "git" in meta and "packages" in meta and "config" in meta
    assert meta["state_checksum"]


def test_incompatible_checkpoint_fails_actionably(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, _state(), _meta(), COMPAT)
    bad = dict(COMPAT, mask_direct_dt=False, policy_input="obs_only")
    with pytest.raises(ValueError) as e:
        load_checkpoint(path, expected_compat=bad)
    msg = str(e.value)
    assert "mask_direct_dt" in msg and "policy_input" in msg
    assert "incompatible" in msg


def test_no_silent_overwrite(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, _state(), _meta(), COMPAT)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, _state(), _meta(), COMPAT)
    # explicit force is allowed
    save_checkpoint(path, _state(), _meta(), COMPAT, force=True)


def test_bare_state_dict_rejected(tmp_path):
    path = str(tmp_path / "bare.pt")
    torch.save(_state(), path)
    with pytest.raises(ValueError) as e:
        load_checkpoint(path)
    assert "import-legacy" in str(e.value)


def test_legacy_import_and_gating(tmp_path):
    src = str(tmp_path / "old.pt")
    dst = str(tmp_path / "old_legacy.pt")
    torch.save(_state(), src)
    import_legacy_checkpoint(src, dst, note="unit test")
    # without allow_legacy: refuse loudly
    with pytest.raises(ValueError) as e:
        load_checkpoint(dst)
    assert "LEGACY" in str(e.value)
    # explicit legacy load works and is marked
    state, meta = load_checkpoint(dst, allow_legacy=True)
    assert torch.equal(state["w"], _state()["w"])
    assert meta["legacy"] is True
    # a provenance-carrying file cannot be re-imported as legacy
    with pytest.raises(ValueError):
        import_legacy_checkpoint(dst, str(tmp_path / "x.pt"))


def test_results_never_overwrite(tmp_path):
    p1 = write_results("unit", {"x": 1}, results_dir=str(tmp_path))
    p2 = write_results("unit", {"x": 2}, results_dir=str(tmp_path))
    assert p1 != p2
    assert os.path.exists(p1) and os.path.exists(p2)
    with open(p1, encoding="utf-8") as f:
        assert json.load(f)["x"] == 1


def test_checkpoint_ref(tmp_path):
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, _state(), _meta(), COMPAT)
    ref = checkpoint_ref(path)
    assert ref["sha256"] == file_checksum(path)
    assert ref["experiment"] == "unit_test"
    assert ref["legacy"] is False


def test_state_checksum_verified_on_load(tmp_path):
    from provenance import state_checksum
    path = str(tmp_path / "ck.pt")
    state = _state()
    save_checkpoint(path, state, _meta(), COMPAT)
    # Tamper with weights while keeping recorded checksum
    payload = torch.load(path, weights_only=True)
    payload["state"]["w"] = payload["state"]["w"] + 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError) as e:
        load_checkpoint(path, expected_compat=COMPAT)
    assert "state_checksum" in str(e.value)


def test_state_checksum_canonical_stable():
    from provenance import state_checksum
    s = {"a": {"w": torch.arange(4.0)}, "b": torch.ones(2, 3)}
    assert state_checksum(s) == state_checksum(s)
    s2 = {"b": torch.ones(2, 3), "a": {"w": torch.arange(4.0)}}
    assert state_checksum(s) == state_checksum(s2)


def test_missing_checksum_requires_legacy(tmp_path):
    path = str(tmp_path / "old.pt")
    payload = {"state": _state(), "meta": {**_meta(), "compat": COMPAT}}
    # deliberately omit state_checksum
    payload["meta"].pop("state_checksum", None)
    torch.save(payload, path)
    with pytest.raises(ValueError) as e:
        load_checkpoint(path, expected_compat=COMPAT)
    assert "state_checksum" in str(e.value)
    state, meta = load_checkpoint(path, expected_compat=COMPAT,
                                  allow_legacy=True)
    assert torch.equal(state["w"], _state()["w"])
