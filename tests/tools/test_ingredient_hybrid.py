import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools/ingredient_experiment/hybrid_region.py"
_SPEC = importlib.util.spec_from_file_location("ingredient_hybrid", _PATH)
hybrid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hybrid)
choose_region, refine, area = hybrid.choose_region, hybrid.refine, hybrid.area


def test_no_semantic_support_falls_back():
    baseline = [0, 0, 100, 100]
    result, source = choose_region(
        baseline,
        [{"bbox": [5, 5, 50, 50], "conf": 0.99}],
        [[5, 5, 30, 20, "제품 광고"]],
        (100, 100, 3),
    )
    assert result == baseline and source == "fallback_rules"


def test_high_confidence_nutrition_is_not_ingredient_evidence():
    result, source = choose_region(
        [0, 0, 100, 100],
        [{"bbox": [0, 0, 50, 30], "conf": 0.99}],
        [[0, 0, 50, 30, "영양정보 나트륨 100mg"]],
        (100, 100, 3),
    )
    assert source == "fallback_rules"


def test_semantic_candidate_preserves_positive_text_and_stays_in_bounds():
    words = [
        [10, 20, 90, 35, "원재료명 밀가루 설탕"],
        [10, 40, 90, 55, "팜유 코코아"],
        [10, 60, 90, 75, "품목보고번호 123"],
    ]
    b = refine([0, 0, 100, 100], words, (100, 100, 3), 0.02)
    assert b[1] <= 20 and b[3] >= 55 and b[3] < 75
    assert 0 <= b[0] < b[2] <= 100 and 0 <= b[1] < b[3] <= 100 and area(b) > 0


def test_contains_notice_is_not_a_stop_heading():
    assert not hybrid.STOP.search("우유, 대두 함유")
    assert hybrid.NOTICE.search("우유, 대두 함 유")
    assert not hybrid.NOTICE.search("우유 함유량 10%")
    assert not hybrid.NOTICE.search("함유될 수 있음")


def test_separate_contains_notice_across_blank_space_is_preserved():
    words = [[10, 20, 90, 35, "원재료명 밀가루 설탕"], [10, 80, 90, 95, "우유, 대두 함유"]]
    box, source = choose_region(
        [0, 0, 100, 100], [{"bbox": [10, 18, 90, 40], "conf": 0.9}], words, (100, 100, 3)
    )
    assert box[1] <= 20 and box[3] >= 95
    assert source.endswith("with_notice")


def test_unreadable_notice_does_not_allow_narrowing():
    words = [[10, 20, 90, 35, "원재료명 밀가루 설탕"]]
    box, source = choose_region(
        [0, 0, 100, 100], [{"bbox": [10, 18, 90, 40], "conf": 0.99}], words, (100, 100, 3)
    )
    assert box == [0, 0, 100, 100]
    assert source == "fallback_notice_unreadable"


def test_split_ocr_notice_keeps_the_allergen_names():
    words = [[10, 70, 65, 80, "우유, 대두"], [70, 70, 90, 80, "함유"]]
    box, _ = hybrid.preserve_notice(
        [10, 20, 90, 40], [0, 0, 100, 100], words, (100, 100, 3), "model"
    )
    assert box[0] <= 10 and box[2] >= 90 and box[3] >= 80


def test_different_column_notice_falls_back():
    box, source = hybrid.preserve_notice(
        [0, 0, 40, 40], [0, 0, 100, 100], [[80, 70, 100, 80, "우유 함유"]], (100, 100, 3), "model"
    )
    assert box == [0, 0, 100, 100] and source == "fallback_notice_unaligned"
