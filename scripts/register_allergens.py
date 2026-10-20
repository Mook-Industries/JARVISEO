"""내 알레르기를 등록·확인하는 스크립트.  담당: 권용현

성분 판정은 등록한 알레르기와 대조한다. 등록이 비어 있으면 경고 문장이 나오지 않는다.
assistant.py 와 같은 DB·같은 사용자(owner)에 넣으므로, 등록하면 앱이 바로 읽는다.

등록한 이름 그대로 말한다("계란"으로 등록하면 "알류"가 아니라 "계란이 들어 있어요").
동의어 사전에 없는 이름(예: 키위)도 등록할 수 있다. 그 글자 그대로 찾는다.

사용:
    python scripts/register_allergens.py                  # 지금 등록된 목록 보기
    python scripts/register_allergens.py 계란 땅콩         # 이 목록으로 바꾸기
    python scripts/register_allergens.py --add 새우        # 지금 목록에 더하기
    python scripts/register_allergens.py --remove 땅콩     # 지금 목록에서 빼기
    python scripts/register_allergens.py --clear           # 모두 지우기
    python scripts/register_allergens.py 계란 --db postgresql+psycopg://...   # 다른 DB
"""

from __future__ import annotations

import argparse

from jarviseo import config
from jarviseo.memory import MemoryStore
from jarviseo.nutrition.allergen import load_synonyms, normalize_term

# assistant.py 의 기본 DB 와 같아야 등록한 알레르기를 앱이 읽는다.
DEFAULT_DB = "sqlite:///data/assistant.db"


def describe(name: str, synonyms: dict[str, list[str]]) -> str:
    """등록한 이름이 무엇으로 판정되는지 한 줄. 사전에 없으면 그 글자 그대로 찾는다고 알린다."""
    term = normalize_term(name)
    for canonical, terms in synonyms.items():
        if term in terms:
            if canonical == term:
                return f"{name}"
            return f"{name} (= {canonical})"
    return f"{name} (사전에 없음 → '{name}' 글자 그대로 찾음)"


def apply(
    current: list[str], names: list[str], *, add: bool, remove: bool, clear: bool
) -> list[str]:
    """새 목록. 순서는 등록한 순서를 지키고 같은 이름은 한 번만 둔다."""
    if clear:
        return []
    names = [n.strip() for n in names if n.strip()]
    if remove:
        return [a for a in current if a not in names]
    base = list(current) if add else []
    for n in names:
        if n not in base:
            base.append(n)
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="내 알레르기 등록·확인")
    ap.add_argument("names", nargs="*", help="알레르기 이름 (예: 계란 땅콩)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--add", action="store_true", help="지금 목록에 더한다")
    mode.add_argument("--remove", action="store_true", help="지금 목록에서 뺀다")
    mode.add_argument("--clear", action="store_true", help="모두 지운다")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"DB 주소 (기본 {DEFAULT_DB})")
    args = ap.parse_args()
    if (args.add or args.remove) and not args.names:
        ap.error("--add / --remove 에는 알레르기 이름이 필요합니다")

    store = MemoryStore(database_url=args.db)
    store.init_schema()
    user_id = store.ensure_user()
    synonyms = load_synonyms(config.ALLERGEN_SYNONYMS_PATH)

    current = store.get_user_allergens(user_id)
    if args.names or args.clear:
        new = apply(current, args.names, add=args.add, remove=args.remove, clear=args.clear)
        store.set_user_allergens(user_id, new)
        current = store.get_user_allergens(user_id)

    if not current:
        print("등록된 알레르기가 없습니다. 예: python scripts/register_allergens.py 계란 땅콩")
        return
    print(f"등록된 알레르기 {len(current)}개:")
    for name in current:
        print(f"  - {describe(name, synonyms)}")


if __name__ == "__main__":
    main()
