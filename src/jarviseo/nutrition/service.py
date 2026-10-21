"""성분 판정 전체 흐름.  담당: 권용현

크롭 이미지 한 장 → 바코드 조회 → (못 찾으면) 성분표 OCR → 판정 → 안내 문장.
그래프 노드는 ``IngredientChecker.check`` 하나만 부르면 된다. LLM 을 쓰지 않는다.

다른 모듈에서 붙일 때 (크롭 하나 + 성분 분석이 필요한지)
------------------------------------------------------
    checker = make_checker(store)          # 앱 시작 때 한 번 (OCR 모델·푸드QR 연결)
    text = checker.answer(crop, needs_ingredient, user_allergens)
    if text is not None:
        tts.speak(text)                    # "해당 제품에는 밀, 대두가 들어 있고, ..."
    checker.close()                        # 앱 끝날 때
``needs_ingredient`` 가 False 면 아무것도 하지 않고 None 을 돌려준다(바코드·OCR 안 돎).

재처리·재촬영 (기획 05-workflow 5.4, 2026-10-02 반영)
------------------------------------------------------
0. OCR 전에 사진 품질을 본다(quality.py). 흐리거나 반사가 심하면 OCR 을 돌리지 않고
   바로 재촬영을 안내한다. 무리하게 읽어서 '못 읽은 것'을 미검출로 말하지 않기 위해서다.
1. OCR 결과가 검증(신뢰도 기준 + "원재료명" 머리말)을 통과하지 못하면
   같은 사진을 대비 보정·더 크게 확대해서 한 번 더 읽는다 (사진당 1회).
2. 그래도 실패하면 재촬영 안내를 돌려준다. 호출한 쪽이 새 사진을 찍어
   ``retake_count`` 를 하나 올려 다시 부른다 (요청당 최대 2회).
3. 재촬영 상한을 넘으면 확인 불가로 끝낸다.
단, 판독이 불완전해도 함유를 찾았으면 그 자리에서 경고한다. 경고는 확실하기 때문이다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from jarviseo import config
from jarviseo.nutrition.allergen import (
    judge_allergens,
    judge_product,
    load_synonyms,
    user_labels,
)
from jarviseo.nutrition.barcode import (
    BarcodeLookup,
    FoodQrClient,
    lookup_product,
    make_food_qr_client,
)
from jarviseo.nutrition.occlusion import Borders, find_occlusion
from jarviseo.nutrition.ocr import (
    OcrReader,
    PanelSections,
    crop_and_upscale,
    join_lines,
    mean_confidence,
    parse_ingredient_text,
    read_upright,
    split_sections,
)
from jarviseo.nutrition.quality import ImageQuality, assess_quality, best_frame
from jarviseo.nutrition.region import locate_panel
from jarviseo.nutrition.response import (
    OCCLUDED_NOTE,
    RETAKE_CLOSER,
    RETAKE_GLARE,
    RETAKE_STEADY,
    RETAKE_UNCOVER,
    RETAKE_WHOLE,
    SMALL_TEXT_NOTE,
    build_message,
)
from jarviseo.types import (
    AllergenJudgement,
    AllergenVerdict,
    BBox,
    IngredientSource,
    OCRLine,
    ProductInfo,
)

__all__ = ["IngredientCheck", "IngredientChecker"]

# 성분표를 읽었지만 알레르기 표시가 없던 제품을 캐시에 남길 때 쓴다. 판정 쪽은 비었는지만 본다.
OCR_NO_NOTICE = "알레르기 유발물질 표시 없음 (성분표 판독)"

# 원본 사진에서 성분표 글줄 높이(px, 중앙값)가 이보다 작으면 '글씨가 작다'고 본다.
# 키워 읽어도 없는 정보는 생기지 않아서 '밀'을 '일'로 읽는 식의 오독이 남는다.
# 실측(평가 세트 21장, PaddleOCR): 22px 미만 4장(label_05·09·14·19)이 모두 알레르기를 하나씩 놓쳤다.
MIN_TEXT_PX = 22.0
# 글줄 높이를 잴 때 쓰는 줄의 최소 글자 수. 짧은 조각(숫자·기호)은 높이가 들쭉날쭉하다.
_TEXT_PX_MIN_CHARS = 8


@dataclass
class IngredientCheck:
    """한 번 확인한 결과. ``message`` 를 그대로 TTS 로 보낸다."""

    judgement: AllergenJudgement
    message: str
    # 재촬영이 필요하면 안내 문구. None 이면 이번 결과로 끝이다.
    retake: str | None = None
    barcode: BarcodeLookup | None = None
    ingredients: list[str] = field(default_factory=list)  # 정규화한 성분 목록 (기록용)
    ocr_text: str = ""  # OCR 원문 (틀렸을 때 어디가 틀렸는지 보려고 남긴다)
    ocr_conf: float | None = None
    reprocessed: bool = False
    retake_count: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)  # barcode / quality / ocr / match
    quality: ImageQuality | None = None  # OCR 전 품질 검사 결과 (바코드로 찾으면 None)
    rotation: int = 0  # 성분표가 돌아가 있어 돌려 읽은 각도 (0 / 90 / 270)
    # 성분표가 잘린 사유 (occlusion.CUT_HEADER / CUT_EDGE). 빈 문자열이면 다 보였다.
    occlusion: str = ""
    panel_region: BBox | None = None  # 자동으로 찾은 성분표 영역 (못 찾았거나 끄면 None)
    # 원본 사진 기준 글줄 높이(px). MIN_TEXT_PX 보다 작으면 오독으로 놓쳤을 수 있다고 알린다.
    text_px: float = 0.0


@dataclass
class _OcrPass:
    lines: list[OCRLine]
    sections: PanelSections
    conf: float
    angle: int = 0  # 회전 보정으로 돌린 각도 (시계 방향)
    size: tuple[int, int] = (0, 0)  # 읽은 이미지 (너비, 높이). 돌려 읽었으면 돌린 뒤 크기
    borders: Borders = field(default_factory=Borders)  # 읽은 이미지의 변이 사진 가장자리인가
    scale: float = 1.0  # 원본 대비 키운 배율

    @property
    def text_px(self) -> float:
        """원본 사진 기준 글줄 높이(px)의 중앙값. 잴 줄이 없으면 0."""
        hs = sorted(
            ln.bbox.height / self.scale for ln in self.lines if len(ln.text) >= _TEXT_PX_MIN_CHARS
        )
        return hs[len(hs) // 2] if hs else 0.0

    def occlusion(self) -> str:
        s = self.sections
        panel = s.ingredients + s.allergen_notice + s.cross_contamination
        return find_occlusion(self.lines, *self.size, panel, self.borders)

    @property
    def text(self) -> str:
        return join_lines(self.lines)


def _enhance(image: np.ndarray) -> np.ndarray:
    """재처리용 보정. 흑백 → 국소 대비 강화(CLAHE). 반사·그늘진 포장에서 글자를 살린다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class IngredientChecker:
    """성분 판정 흐름. 앱에서 하나만 만들어 돌려 쓴다(동의어 사전·OCR 모델을 한 번만 읽으려고).

    Args:
        store: ``MemoryStore``. 제품 캐시를 쓴다. 없으면 캐시 없이 동작한다.
        food_qr: ``FoodQrClient``. 캐시 다음 가장 먼저 묻는다(포장 알레르기 표시 그대로).
            앱에서는 ``barcode.make_food_qr_client()`` 로 만든다(인증키가 없으면 None).
        synonyms: 동의어 사전. 없으면 ``config.ALLERGEN_SYNONYMS_PATH`` 에서 읽는다.
        reader: OCR 엔진. 없으면 EasyOCR. 테스트에서 가짜를 끼운다.
        quality_check: OCR 전 품질 검사. None 이면 검사하지 않는다(테스트용 단색 이미지 등).
        try_rotation: 원재료명을 못 찾으면 90°·270° 로 돌려서도 읽는다(ocr.read_upright).
        auto_region: 성분표 영역을 먼저 찾아 그 부분만 크게 읽는다(region.locate_panel).
        min_text_px: 글씨가 작다고 볼 글줄 높이(원본 px). 0 이면 보지 않는다.
    """

    def __init__(
        self,
        store=None,
        synonyms: dict[str, list[str]] | None = None,
        reader: OcrReader | None = None,
        conf_threshold: float = config.OCR_CONF_THRESHOLD,
        max_reprocess: int = config.OCR_REPROCESS_PER_PHOTO,
        max_retake: int = config.OCR_MAX_RETAKE,
        quality_check: Callable[[np.ndarray], ImageQuality] | None = assess_quality,
        try_rotation: bool = config.OCR_TRY_ROTATION,
        food_qr: FoodQrClient | None = None,
        auto_region: bool = config.OCR_AUTO_REGION,
        min_text_px: float = MIN_TEXT_PX,
    ) -> None:
        self.min_text_px = min_text_px
        self.try_rotation = try_rotation
        self.auto_region = auto_region
        self.store = store
        self.food_qr = food_qr
        self.synonyms = synonyms or load_synonyms(config.ALLERGEN_SYNONYMS_PATH)
        self.reader = reader
        self.quality_check = quality_check
        self.conf_threshold = conf_threshold
        self.max_reprocess = max_reprocess
        self.max_retake = max_retake

    # -- 진입점 ------------------------------------------------------------

    def answer(
        self, crop: np.ndarray | None, needs_ingredient: bool, user_allergens: list[str]
    ) -> str | None:
        """다른 모듈이 부르는 입구. 성분 분석이 필요할 때만 돌고, 읽어 줄 문장 하나를 돌려준다.

        Args:
            crop: 가리킨 물건 하나를 자른 이미지 (BGR).
            needs_ingredient: 성분 분석이 필요한가. False 면 바로 None (아무것도 안 돈다).
            user_allergens: 사용자가 등록한 알레르기 (예: ["계란", "땅콩"]).

        Returns:
            "해당 제품에는 밀이 들어 있고, 새우는 ... 섞여 들어갔을 수 있어요." 같은 문장.
            사진으로 못 읽으면 재촬영 안내 문장. 분석이 필요 없거나 크롭이 없으면 None.
        """
        if not needs_ingredient or crop is None or crop.size == 0:
            return None
        result = self.check(crop, user_allergens)
        return result.retake or result.message

    def close(self) -> None:
        """푸드QR 연결을 닫는다. 앱이 끝날 때 한 번 부른다."""
        if self.food_qr is not None:
            self.food_qr.close()

    def check(
        self,
        crop: np.ndarray,
        user_allergens: list[str],
        retake_count: int = 0,
        product_name: str = "",
    ) -> IngredientCheck:
        """크롭 이미지 한 장을 확인한다.

        Args:
            crop: 가리킴 모델이 넘겨준 타깃 크롭 (BGR).
            user_allergens: ``MemoryStore.get_user_allergens`` 결과.
            retake_count: 이 요청에서 재촬영한 횟수. 첫 사진은 0.
            product_name: 제품명을 알면 문장에 넣는다. 바코드로 찾으면 그 이름을 쓴다.
        """
        timings: dict[str, int] = {}

        t0 = time.perf_counter()
        lookup = lookup_product(
            crop,
            store=self.store,
            food_qr=self.food_qr,
        )
        timings["barcode"] = _ms(t0)

        if lookup.hit:
            result = self._from_product(lookup, user_allergens, timings)
            if not self._needs_label(result, lookup.product, user_allergens):
                return result
            # 조회 정보에 알레르기 표시가 없어서 '못 찾음'을 믿을 수 없다. 성분표를 직접 읽는다.
            product_name = product_name or lookup.product.product_name
        return self._from_ocr(crop, lookup, user_allergens, retake_count, product_name, timings)

    @staticmethod
    def _needs_label(result: IngredientCheck, info: ProductInfo, user_allergens: list[str]) -> bool:
        """조회한 정보만으로 '미검출'이라고 말하면 안 되는 경우인가.

        조회 원재료가 '면류, 분말스프'처럼 상위 분류 이름만 오는 제품이 있다.
        그 안의 밀·대두는 보이지 않아서, 알레르기 표시 없이 원재료만 보고 '못 찾았다'고 하면
        먹어도 된다는 말로 들린다. 그래서 포장의 알레르기 표시를 받지 못한 정보로는
        '미검출'을 말하지 않고 성분표 OCR 로 넘어간다. 함유·혼입은 그대로 바로 경고한다.
        """
        return (
            bool(user_allergens)
            and result.judgement.verdict is AllergenVerdict.NOT_DETECTED
            and not info.allergen_notice.strip()
        )

    def check_frames(
        self,
        frames: list[np.ndarray],
        user_allergens: list[str],
        retake_count: int = 0,
        product_name: str = "",
    ) -> IngredientCheck:
        """여러 프레임 중 흔들림·반사가 가장 적은 한 장으로 확인한다."""
        i, _ = best_frame(frames)
        return self.check(frames[i], user_allergens, retake_count, product_name)

    # -- 바코드로 찾은 경우 ------------------------------------------------

    def _from_product(
        self, lookup: BarcodeLookup, user_allergens: list[str], timings: dict[str, int]
    ) -> IngredientCheck:
        info = lookup.product
        t0 = time.perf_counter()
        judgement = judge_product(info, user_allergens, self.synonyms)
        timings["match"] = _ms(t0)
        judgement.response_text = build_message(
            judgement, info.product_name, user_labels(user_allergens, self.synonyms)
        )
        return IngredientCheck(
            judgement=judgement,
            message=judgement.response_text,
            barcode=lookup,
            ingredients=parse_ingredient_text(info.raw_ingredients),
            timings_ms=timings,
        )

    # -- 성분표 OCR -------------------------------------------------------

    def _read(self, image: np.ndarray, scale: float, region: BBox | None = None) -> _OcrPass:
        h, w = image.shape[:2]
        panel = crop_and_upscale(image, region or BBox(0, 0, w, h), scale=scale)
        lines, angle = read_upright(panel, reader=self.reader, try_rotation=self.try_rotation)
        sections = split_sections(join_lines(lines))
        ph, pw = panel.shape[:2]
        size = (ph, pw) if angle in (90, 270) else (pw, ph)
        # 영역만 잘라 읽었으면, 그 영역의 변이 사진 가장자리에 닿은 쪽만 '잘림'을 본다.
        r = region or BBox(0, 0, w, h)
        borders = Borders(r.y1 <= 1, r.x2 >= w - 1, r.y2 >= h - 1, r.x1 <= 1).rotated(angle)
        scale_used = panel.shape[1] / max(1.0, r.width)  # 최대 크기 제한으로 줄었을 수 있다
        return _OcrPass(lines, sections, mean_confidence(lines), angle, size, borders, scale_used)

    def _passes(self, p: _OcrPass) -> bool:
        return p.conf >= self.conf_threshold and p.sections.has_ingredient_header

    def _from_ocr(
        self,
        crop: np.ndarray,
        lookup: BarcodeLookup,
        user_allergens: list[str],
        retake_count: int,
        product_name: str,
        timings: dict[str, int],
    ) -> IngredientCheck:
        quality = None
        if self.quality_check is not None:
            t0 = time.perf_counter()
            quality = self.quality_check(crop)
            timings["quality"] = _ms(t0)
            if not quality.ok:
                return self._poor_photo(quality, lookup, user_allergens, retake_count, timings)

        region = None
        if self.auto_region:
            # 성분표 영역을 먼저 찾는다. 못 찾으면(단서 없음) 지금처럼 사진 전체를 읽는다.
            t0 = time.perf_counter()
            found, anchors = locate_panel(crop, self.reader)
            timings["region"] = _ms(t0)
            if anchors:
                region = found

        t0 = time.perf_counter()
        if region is not None:
            # 영역은 긴 변이 1600px 이 되게 키운다(평가 때와 같은 배율, 최대 3배).
            scale = min(3.0, max(1.0, 1600 / max(region.width, region.height)))
            best = self._read(crop, scale=scale, region=region)
        else:
            best = self._read(crop, scale=2.0)
        reprocessed = False
        if not self._passes(best) and self.max_reprocess > 0:
            # 같은 사진을 다시 읽는다. 단순 반복은 같은 결과라 보정·배율을 바꾼다.
            # 영역을 잘못 잡았을 수도 있어서 재처리는 사진 전체로 한다.
            again = self._read(_enhance(crop), scale=3.0)
            reprocessed = True
            if self._passes(again) or again.conf > best.conf:
                best = again
        timings["ocr"] = _ms(t0)

        readable = self._passes(best)
        reason = "" if readable else (
            "no_ingredients" if not best.sections.has_ingredient_header else "low_ocr_conf"
        )  # fmt: skip
        sec = best.sections
        occlusion = best.occlusion() if readable else ""
        t0 = time.perf_counter()
        judgement = judge_allergens(
            user_allergens,
            self.synonyms,
            ingredients=sec.ingredients,
            notice=sec.allergen_notice,
            cross=sec.cross_contamination,
            readable=readable,
            reason=reason,
        )
        timings["match"] = _ms(t0)

        barcode = lookup.barcode.code if lookup.barcode else None
        judgement.source = IngredientSource.OCR
        judgement.product = ProductInfo(
            barcode=barcode,
            product_name=product_name,
            raw_ingredients=sec.ingredients,
            allergen_notice=sec.allergen_notice,
            cross_contamination=sec.cross_contamination,
            source=IngredientSource.OCR,
        )
        result = IngredientCheck(
            judgement=judgement,
            message="",
            barcode=lookup,
            ingredients=parse_ingredient_text(sec.ingredients),
            ocr_text=best.text,
            ocr_conf=best.conf,
            reprocessed=reprocessed,
            retake_count=retake_count,
            timings_ms=timings,
            quality=quality,
            rotation=best.angle,
            panel_region=region,
            occlusion=occlusion,
            text_px=best.text_px,
        )
        small = readable and 0 < result.text_px < self.min_text_px

        found = judgement.verdict in (AllergenVerdict.CONTAINS, AllergenVerdict.MAY_CONTAIN)
        if occlusion and not found:
            # 보이는 데서 못 찾았어도 가려진 자리에 있을 수 있다. '못 찾았어요'라고 하지 않는다.
            judgement.verdict, judgement.reason = AllergenVerdict.UNDETERMINED, "occluded"
            if retake_count < self.max_retake:
                result.retake = result.message = RETAKE_UNCOVER
                return result

        if small and not found and judgement.verdict is AllergenVerdict.NOT_DETECTED:
            # 글씨가 작으면 오독으로 놓쳤을 수 있다(실측: '밀'→'일'). '못 찾았어요'라고 하지 않는다.
            judgement.verdict, judgement.reason = AllergenVerdict.UNDETERMINED, "small_text"
            if retake_count < self.max_retake:
                result.retake = result.message = RETAKE_CLOSER
                return result

        needs_retake = (
            judgement.verdict is AllergenVerdict.UNDETERMINED
            and reason
            and retake_count < self.max_retake
        )
        if needs_retake:
            # 원재료명이 안 보이면 잘린 것이다. 신뢰도만 낮으면 멀거나 흐린 것이다.
            result.retake = RETAKE_WHOLE if reason == "no_ingredients" else RETAKE_CLOSER
            result.message = result.retake
            return result

        judgement.response_text = build_message(
            judgement, product_name, user_labels(user_allergens, self.synonyms)
        )
        if occlusion and found:
            # 찾은 경고는 바로 전하고, 가려진 데가 있었다는 것도 함께 말한다.
            judgement.response_text += " " + OCCLUDED_NOTE
        elif small and found:
            # 찾은 경고는 바로 전하고, 글씨가 작아 빠진 게 있을 수 있다는 것도 함께 말한다.
            judgement.response_text += " " + SMALL_TEXT_NOTE
        result.message = judgement.response_text
        # 가려진 사진에서 읽은 성분은 캐시에 남기지 않는다. 일부만 읽은 정보가 굳으면 안 된다.
        if readable and not occlusion and not small and self.store is not None and barcode:
            # OCR 로 처음 읽은 제품은 확인 전(pending)으로 캐시한다. 틀린 판독이 굳지 않게.
            # 성분표를 직접 읽었는데 알레르기 표시가 없었다는 것도 정보다. 칸을 비워 두면
            # 다음에 캐시에서 꺼냈을 때 '표시를 못 받은 정보'로 보고 또 OCR 을 돌린다.
            saved = judgement.product
            if not saved.allergen_notice:
                saved = replace(saved, allergen_notice=OCR_NO_NOTICE)
            self.store.save_product(saved, status="pending", ingredients=result.ingredients)
        return result

    def _poor_photo(
        self,
        quality: ImageQuality,
        lookup: BarcodeLookup,
        user_allergens: list[str],
        retake_count: int,
        timings: dict[str, int],
    ) -> IngredientCheck:
        """흐리거나 반사가 심한 사진. OCR 없이 재촬영을 안내하고, 상한을 넘으면 확인 불가."""
        judgement = judge_allergens(
            user_allergens, self.synonyms, readable=False, reason=quality.reason
        )
        judgement.source = IngredientSource.OCR
        result = IngredientCheck(
            judgement=judgement,
            message="",
            barcode=lookup,
            retake_count=retake_count,
            timings_ms=timings,
            quality=quality,
        )
        if judgement.reason == quality.reason and retake_count < self.max_retake:
            result.retake = RETAKE_GLARE if quality.reason == "glare" else RETAKE_STEADY
            result.message = result.retake
            return result
        judgement.response_text = build_message(judgement)
        result.message = judgement.response_text
        return result


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def make_checker(store=None, **kwargs) -> IngredientChecker:
    """앱에서 쓸 판정기. 푸드QR 인증키가 있으면 푸드QR 까지 붙인다(없으면 캐시 → OCR).

    OCR 모델과 동의어 사전을 한 번만 읽도록 앱 시작 때 하나만 만들어 돌려 쓴다.
    """
    return IngredientChecker(store=store, food_qr=make_food_qr_client(), **kwargs)
