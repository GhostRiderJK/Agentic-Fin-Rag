"""Gold-set attribution: what the loader accepts, derives, and refuses.

The versioned file states its own question IDs and expected source filenames.
The legacy file states neither, so the loader has to invent them from line
order and a hardcoded table — which is fine only while it is recorded. These
tests pin the difference, the validation failures that must stop a run, and the
backward compatibility the existing checker tests depend on.

All fixtures live under ``tmp_path``; ``data/`` is gitignored and a test must
never depend on it being present.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from app.evaluation import pif_chat_check as checker


def _v2_item(index: int, **overrides) -> dict:
    item = {
        "id": f"pif-{index:03d}",
        "question": f"Question {index}?",
        "expected": "SAR 1,000 billion",
        "expected_numbers": ["1,000"],
        "expected_keywords": ["aum"],
        "source_page": 10 + index,
        "expected_source_filenames": ["2024-en.pdf"],
        "attribution": {"derived_from": "test fixture",
                        "invented_by_this_script": False},
    }
    item.update(overrides)
    return item


def _write(path, items) -> None:
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) +
                    "\n", encoding="utf-8")


@pytest.fixture
def pif_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# the versioned file is taken at its word
# ---------------------------------------------------------------------------


def test_versioned_gold_reads_stable_ids_and_explicit_attribution(pif_dir):
    items = [_v2_item(i) for i in range(1, 4)]
    # A deliberate filename disagreement proves nothing is inherited from the
    # legacy table.
    items[1]["expected_source_filenames"] = ["2023-en.pdf", "2024-en.pdf"]
    _write(pif_dir / checker.GOLD_V2_FILENAME, items)

    loaded, provenance = checker.load_gold(pif_dir)

    assert [i["id"] for i in loaded] == ["pif-001", "pif-002", "pif-003"]
    assert loaded[1]["expected_source_filenames"] == ["2023-en.pdf", "2024-en.pdf"]
    assert provenance["gold_version"] == "v2"
    assert provenance["attribution"] == "explicit in gold file"
    assert provenance["derived_fields"] == []
    assert provenance["questions"] == 3


def test_provenance_hashes_the_exact_bytes_of_the_file_it_read(pif_dir):
    path = pif_dir / checker.GOLD_V2_FILENAME
    _write(path, [_v2_item(1)])
    _loaded, provenance = checker.load_gold(pif_dir)
    assert provenance["gold_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["gold_file"].endswith(checker.GOLD_V2_FILENAME)


def test_reordering_the_versioned_file_reorders_ids_not_meaning(pif_dir):
    _write(pif_dir / checker.GOLD_V2_FILENAME, [_v2_item(3), _v2_item(1)])
    loaded, _provenance = checker.load_gold(pif_dir)
    assert [i["id"] for i in loaded] == ["pif-003", "pif-001"]
    assert [i["source_page"] for i in loaded] == [13, 11]


# ---------------------------------------------------------------------------
# the legacy file works, and says what it invented
# ---------------------------------------------------------------------------


def test_legacy_gold_is_accepted_with_every_derivation_recorded(pif_dir):
    legacy = [{"question": f"Q{i}?", "expected": "SAR 1,000 billion",
               "expected_numbers": [], "expected_keywords": [],
               "source_page": 9} for i in range(1, 4)]
    legacy[1]["id"] = "pif-002"
    _write(pif_dir / checker.GOLD_V1_FILENAME, legacy)

    loaded, provenance = checker.load_gold(pif_dir)

    assert provenance["gold_version"] == "v1"
    assert provenance["attribution"].startswith("derived")
    assert [i["id"] for i in loaded] == ["pif-001", "pif-002", "pif-003"]
    assert all(i["expected_source_filenames"]
               == checker.EXPECTED_SOURCE_FILES[i["id"]] for i in loaded)
    assert any("id from line number" in note for note in provenance["derived_fields"])
    assert any("filenames from EXPECTED_SOURCE_FILES" in note
               for note in provenance["derived_fields"])


def test_versioned_file_is_preferred_when_both_exist(pif_dir):
    _write(pif_dir / checker.GOLD_V1_FILENAME,
           [{"question": "old", "expected": "x", "source_page": 1}])
    _write(pif_dir / checker.GOLD_V2_FILENAME, [_v2_item(1)])
    loaded, provenance = checker.load_gold(pif_dir)
    assert provenance["gold_version"] == "v2"
    assert loaded[0]["question"] == "Question 1?"


def test_an_explicit_filename_selects_that_file(pif_dir):
    _write(pif_dir / checker.GOLD_V2_FILENAME, [_v2_item(1)])
    _write(pif_dir / "pif_gold_small.jsonl",
           [_v2_item(7, id="pif-007", source_page=6)])
    loaded, provenance = checker.load_gold(pif_dir, filename="pif_gold_small.jsonl")
    assert [i["id"] for i in loaded] == ["pif-007"]
    # a non-v2 name is treated as legacy, so its attribution is derived
    assert provenance["gold_version"] == "v1"


def test_limit_caps_the_loaded_set_and_the_reported_denominator(pif_dir):
    _write(pif_dir / checker.GOLD_V2_FILENAME, [_v2_item(i) for i in range(1, 6)])
    loaded, provenance = checker.load_gold(pif_dir, limit=2)
    assert [i["id"] for i in loaded] == ["pif-001", "pif-002"]
    assert provenance["questions"] == 2


# ---------------------------------------------------------------------------
# validation failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["id", "question", "expected", "source_page"])
def test_a_versioned_item_missing_a_required_field_is_refused(pif_dir, field):
    item = _v2_item(1)
    item.pop(field)
    _write(pif_dir / checker.GOLD_V2_FILENAME, [item])
    with pytest.raises(checker.GoldFormatError, match=field):
        checker.load_gold(pif_dir)


@pytest.mark.parametrize("page", [0, -1, 1.5, "6", True, None])
def test_a_physical_page_must_be_a_positive_int(pif_dir, page):
    _write(pif_dir / checker.GOLD_V2_FILENAME, [_v2_item(1, source_page=page)])
    with pytest.raises(checker.GoldFormatError):
        checker.load_gold(pif_dir)


@pytest.mark.parametrize("names", [[], ["  "], "2024-en.pdf", [None], [7]])
def test_source_filenames_must_be_a_list_of_non_blank_strings(pif_dir, names):
    item = _v2_item(1)
    if names == []:
        # an empty list is a missing field, reported as such
        item["expected_source_filenames"] = []
    else:
        item["expected_source_filenames"] = names
    _write(pif_dir / checker.GOLD_V2_FILENAME, [item])
    with pytest.raises(checker.GoldFormatError):
        checker.load_gold(pif_dir)


def test_a_versioned_item_cannot_silently_lose_its_filename(pif_dir):
    """A v2 file with no filenames would otherwise fall back to the hardcoded
    table, which is exactly the attribution the versioned file replaced."""
    item = _v2_item(1)
    del item["expected_source_filenames"]
    _write(pif_dir / checker.GOLD_V2_FILENAME, [item])
    with pytest.raises(checker.GoldFormatError, match="declares no"):
        checker.load_gold(pif_dir)


def test_a_versioned_item_cannot_silently_lose_its_id(pif_dir):
    item = _v2_item(1)
    del item["id"]
    _write(pif_dir / checker.GOLD_V2_FILENAME, [item])
    with pytest.raises(checker.GoldFormatError, match="stable id"):
        checker.load_gold(pif_dir)


def test_a_legacy_item_the_table_cannot_attribute_is_refused(pif_dir):
    """Line 11 has no EXPECTED_SOURCE_FILES entry: guessing would invent an
    expected document, so the run stops instead."""
    _write(pif_dir / checker.GOLD_V1_FILENAME,
           [{"question": f"Q{i}?", "expected": "x", "source_page": 9}
            for i in range(1, 12)])
    with pytest.raises(checker.GoldFormatError, match="pif-011"):
        checker.load_gold(pif_dir)


def test_a_missing_gold_file_is_reported_not_silently_empty(pif_dir):
    with pytest.raises(checker.GoldFormatError, match="not found"):
        checker.load_gold(pif_dir)


def test_numeric_and_keyword_fields_stay_optional(pif_dir):
    item = _v2_item(1)
    del item["expected_numbers"], item["expected_keywords"]
    _write(pif_dir / checker.GOLD_V2_FILENAME, [item])
    loaded, _provenance = checker.load_gold(pif_dir)
    assert loaded[0]["expected_numbers"] == []
    assert loaded[0]["expected_keywords"] == []


# ---------------------------------------------------------------------------
# the scorer consumes the attribution it is handed
# ---------------------------------------------------------------------------


def _response(filename: str) -> dict:
    return {
        "answer": "Offline test stub: no answer.",
        "generation": {"provider": "stub", "is_stub": True},
        "retrieval": {"n_chunks": 3, "candidates": [
            {"source_filename": filename, "physical_page": 6} for _ in range(3)]},
        "citations": [],
    }


def test_scorer_uses_the_items_own_filenames_over_the_legacy_table():
    """pif-001's table entry is 2024-en.pdf; this item says otherwise and the
    table must not rescue it."""
    item = dict(_v2_item(1, id="pif-001", source_page=6),
                expected_source_filenames=["2023-en.pdf"])
    result = checker._evaluate_item(item, _response("2024-en.pdf"), index=1)
    assert result["criteria"]["correct_source_document_retrieved"] is False


def test_scorer_matches_the_duplicate_stem_normalization_of_gold_filenames():
    item = dict(_v2_item(1, id="pif-001", source_page=6),
                expected_source_filenames=["2024-en.pdf"])
    result = checker._evaluate_item(item, _response("2024-en_2024-en.pdf"), index=1)
    assert result["criteria"]["correct_source_document_retrieved"] is True


def test_a_raw_legacy_item_still_scores_through_the_table():
    """Backward compatibility: callers that hand over an un-versioned dict keep
    working; only ``load_gold`` refuses to invent attribution."""
    item = {"id": "pif-001", "question": "Q?", "expected": "x", "source_page": 6}
    result = checker._evaluate_item(item, _response("2024-en.pdf"), index=1)
    assert result["criteria"]["correct_source_document_retrieved"] is True
