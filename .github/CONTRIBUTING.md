# 작업 컨벤션

무크 인더스트리 · 자비서 (JARVISEO)

**트렁크 기반(Trunk-Based Development)으로 간다.** `main` 하나만 두고,
브랜치는 하루이틀 살다 죽고, PR 은 작게 올리고, 검사는 CI 가 한다.

> Git Flow (`develop` + `release/*` + `hotfix/*`)는 쓰지 않는다.
> 릴리스 주기가 길면 그동안 브랜치가 쌓이는데, 7주 동안 매일 합쳐야 하는
> 팀에는 맞지 않는다.

---

## 기본 원칙 6가지

### 1. `main` 은 보호한다

직접 push 금지 · PR 필수 · 승인 1명 이상 · CI 통과 필수.
GitHub 설정에서 거는 것이고, 설정 방법은 5번에 있다.

### 2. 브랜치는 짧게 산다

**목표는 1~3일.** 일주일 넘어가면 잘못 쪼갠 것이다.
브랜치 나이가 곧 충돌 비용이다. 오래 들고 있을수록 그 사이 남이 같은 파일을
고쳐놓고, 합칠 때 그걸 다 풀어야 한다.

### 3. PR 은 작게

**300줄 넘어가면 리뷰어가 제대로 못 본다.** 큰 기능은 쪼개서 올린다.

```
❌  "성분표 기능 구현"                     900줄 한 방
✅  "성분표 타입 정의"        → 80줄
    "성분표 검출기"           → 200줄
    "OCR 파이프라인 연결"     → 150줄
```

쪼갠 각각이 그 자체로 말이 되어야 한다. 반쯤 만든 걸 올리는 게 아니라,
"여기까지는 완결된 단위"로 끊는다.

### 4. 안 끝난 기능은 플래그로 감싸서 머지한다

브랜치를 오래 들고 있는 대신, **코드는 `main` 에 넣되 꺼둔다.**

```python
from jarviseo import config

if config.ENABLE_GAZE_CUE:
    scores += score_gaze(...)
```

플래그는 `src/jarviseo/config.py` 에 모아 둔다. 기본값은 `False` 고,
내 브랜치에서 작업할 때만 `.env` 에서 켠다.

**기능이 완성되면 플래그와 `if` 분기를 같이 지운다.** 다 만든 기능의 플래그를
남겨두면 경우의 수만 늘어나고 아무도 안 지운다.

### 5. Squash merge 로 히스토리를 평평하게

`main` 의 커밋 하나 = PR 하나. `git log` 가 읽히고, 문제가 생기면 revert
하나로 되돌아간다.

### 6. CI 가 게이트를 지킨다

PR 을 올리면 macOS · Windows · Ubuntu 세 군데서 린트와 테스트가 자동으로 돈다.
빨간불이면 머지 버튼이 잠긴다.

**사람 리뷰는 "설계가 맞나"를 보고, "깨졌나"는 기계가 본다.**

---

## 전체 흐름

```
① 이슈를 연다        템플릿이 type 라벨을 붙임 → area 라벨은 직접 고름
② 브랜치를 판다      main 에서. feat/7-panel-detector
③ 커밋한다           feat(nutrition): 성분표 검출기 추가
④ PR 을 연다         제목은 커밋과 같은 형식, 본문에 Closes #7
⑤ CI 통과 + 승인 1명  CODEOWNERS 가 리뷰어를 자동 지정
⑥ Squash and merge   브랜치 삭제
```

---

## 1. 이슈

템플릿 3종이 있다.

| 템플릿 | `type:` 라벨 | 쓸 때 |
|---|---|---|
| 🐛 Bug Report | `type: bug` **자동** | 안 되는 것 |
| ♻️ Refactor | `type: refactor` **자동** | 동작 그대로, 코드만 개선 |
| 📋 Task | `type: feature` **기본값** | 그 외 전부 |

Task 는 여러 종류를 받으므로, 가장 흔한 `type: feature` 를 기본으로 깔아두고
다른 종류면 사이드바에서 바꾼다.

| 기능 개발이 아니면 이걸로 | 이런 작업 |
|---|---|
| `type: experiment` | 모델 학습 · baseline 측정 · ablation |
| `type: docs` | 문서 · 발표자료 · 회의록 |
| `type: chore` | 촬영 · 라벨링 · 환경 설정 · 그 외 |

그리고 **모든 이슈에 `area:` 라벨을 하나 단다.** 누가 볼 이슈인지가 여기서 정해진다.

`area: pointing` (A) · `area: nutrition` (B) · `area: voice` (C) · `area: data` · `area: infra`

---

## 2. 브랜치

```
<타입>/<이슈번호>-<영문-소문자-설명>
```

```
feat/7-panel-detector
fix/12-windows-hangul-path
exp/23-fingertip-baseline
docs/5-design-deliverables
```

- **항상 `main` 에서 판다.** 남의 브랜치에서 파지 않는다.
- 설명은 **영문 소문자 + 하이픈.** 한글·공백은 맥과 윈도우에서 인코딩이 달라 깨진다.
- 이슈 페이지 오른쪽 **Development → Create a branch** 를 쓰면 GitHub 이
  브랜치를 이슈에 자동으로 연결해준다. 이름만 위 형식으로 고쳐서 만든다.
- 머지되면 삭제한다.

### 작업 중에 `main` 이 앞서갔다면

```bash
git fetch origin
git rebase origin/main
```

브랜치가 짧게 살면 이 명령이 거의 항상 조용히 끝난다.
충돌이 크게 났다면 브랜치를 너무 오래 들고 있었다는 신호다.

---

## 3. 커밋 — Conventional Commits

```
<타입>(<범위>): <제목>

<본문 — 필요할 때만>
```

### 타입

| 타입 | 쓸 때 |
|---|---|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `refactor` | 동작은 그대로, 코드 구조 개선 |
| `docs` | 문서 |
| `test` | 테스트 코드 |
| `perf` | 성능 개선 (지연시간 최적화) |
| `style` | 포맷팅 등 동작에 영향 없는 변경 |
| `chore` | 설정, 잡일 |
| `ci` | GitHub Actions |
| `exp` | 모델 학습 · 측정 · ablation *(우리가 추가한 타입)* |

> **라벨은 `type: bug` 인데 커밋은 왜 `fix` 인가?**
> 이슈는 **상태**를 가리키고("이건 버그다"), 커밋은 **행위**를 가리킨다
> ("이걸 고쳤다"). 둘이 다른 게 맞고, 실무에서도 이렇게 쓴다.

### 범위(scope)

건드린 모듈 이름. 여러 모듈이면 생략한다.

`types` `config` `capture` `pointing` `nutrition` `memory` `voice` `vlm` `graph` `api` `github`

### 제목

**한글로, 50자 이내, 마침표 없이.** 무엇을 어떻게 했는지 쓴다.

| ❌ | ✅ |
|---|---|
| `fix: 버그 수정` | `fix(capture): 영상 파일에서 timestamp가 0으로 나오던 문제 수정` |
| `feat: 기능 추가` | `feat(voice): 웨이크워드 감지 후 발화 시작 시각 반환` |

### 본문

**왜** 그렇게 했는지가 필요할 때만 쓴다. 코드를 보면 아는 것은 안 쓴다.

```
fix(capture): 윈도우에서 한글 경로 이미지가 열리지 않던 문제 수정

cv2.imread 는 경로를 ASCII 로 처리해서 한글 파일명을 못 읽는다.
np.fromfile 로 바이트를 읽고 cv2.imdecode 로 디코딩하도록 바꿨다.
```

### ⚠️ 계약 파일을 바꿀 때

`src/jarviseo/types.py` 와 `src/jarviseo/graph/state.py` 는 세 사람이 전부
의존한다. 필드 하나만 바뀌어도 남의 코드가 조용히 깨진다.

이 두 파일의 **필드를 추가 · 삭제 · 개명**할 때는 타입 뒤에 `!` 를 붙이고,
PR 에 `breaking` 라벨을 단다.

```
feat(types)!: Frame 에 sharpness 필드 추가
```

`!` 가 붙은 PR 은 **머지 직후 팀 채널에 공지한다.** 남이 pull 받고 나서
자기 코드가 깨진 이유를 못 찾는 상황을 막는 표시다.

---

## 4. Pull Request

### 제목

커밋 제목과 같은 형식.

```
feat(nutrition): 성분표 검출기 추가
```

**Squash and merge 를 쓰기 때문에 PR 제목이 그대로 `main` 의 커밋이 된다.**
브랜치 안에서 `wip`, `오타 수정` 같은 커밋을 해도 상관없는 이유가 이것이다.

### 본문

템플릿이 자동으로 뜬다. 이슈 번호를 반드시 연결한다.

```
Closes #12
```

### 리뷰

- **CI 통과 + 승인 1명**이 있어야 머지된다.
- 리뷰어는 CODEOWNERS 가 자동으로 지정한다. 건드린 파일의 담당자에게 간다.
- 리뷰가 2시간 넘게 안 붙으면 팀 채널에서 직접 요청한다 (2시간 룰).
- 리뷰어는 "확인했습니다"만 남기지 않는다. 최소 한 군데는 질문하거나 짚는다.

### 머지

**Squash and merge** 로 통일한다. 머지 후 브랜치는 삭제한다.

---

## 5. CI

PR 을 올리면 `.github/workflows/ci.yml` 이 자동으로 돈다.

| 검사 | 명령 | 무엇을 잡나 |
|---|---|---|
| 린트 | `ruff check .` | 안 쓰는 import, 문법 냄새 |
| 포맷 | `ruff format --check .` | 들여쓰기·줄바꿈 스타일 |
| 테스트 | `pytest -q` | 계약 파일이 깨졌는지 |

**macOS · Windows · Ubuntu 세 군데서 전부 돈다.** 맥 1대 + 윈도우 2대로
개발하기 때문이다. `encoding` 누락이나 경로 문자열 결합처럼 한쪽에서만 터지는
문제를 머지 전에 잡는다.

### 올리기 전에 로컬에서 돌려보기

```bash
python -m pip install -r requirements-dev.txt
ruff check . && ruff format . && pytest -q
```

`ruff format .` 은 `--check` 없이 쓰면 알아서 고쳐준다.

### 브랜치 보호 설정 (한 번만)

레포 **Settings → Branches → Add branch ruleset** 에서 `main` 에 건다.

- ☑ Require a pull request before merging — Required approvals **1**
- ☑ Require status checks to pass — `ubuntu-latest`, `windows-latest`, `macos-latest` 추가
- ☑ Require branches to be up to date before merging
- ☑ Block force pushes

> 상태 체크 목록은 **CI 가 한 번 돌아간 뒤에야** 선택지에 나타난다.
> 첫 PR 을 올려 CI 를 한 번 돌린 다음에 설정하는 것이 순서다.

---

## 6. 절대 커밋하지 않는 것

| 대상 | 이유 |
|---|---|
| `.env` | API 키가 히스토리에 **영구히** 남는다. 사고 시 파일 삭제가 아니라 **키 폐기 후 재발급** |
| `*.pt`, `*.onnx` | 가중치는 공용 Drive `models/` 에. Git 에 넣으면 clone 이 느려진다 |
| `data/datasets/*`, `data/videos/*` | 공용 Drive 에서 받는다 |
| `*.db` | SQLite 는 각자 로컬에서 생성된다 |

`.gitignore` 에 다 들어 있다. `git add .` 전에 `git status` 로 확인하는 습관.

---

## 7. 코드 규칙 (맥 1대 · 윈도우 2대)

- 파일 입출력에 **`encoding="utf-8"` 을 반드시 명시한다.** 윈도우 기본값은
  cp949 라서 생략하면 맥에서만 되고 팀원 노트북에서 한글이 깨진다.
- 경로는 **`pathlib.Path`.** 문자열 결합 금지.
- 카메라 열기의 OS 분기는 `capture/frame_source.py` 안에서만 한다.
- 모델 배포 포맷은 **ONNX.** CoreML 은 맥 전용이라 팀원이 못 쓴다.
- 오디오는 **sounddevice.** PyAudio 는 윈도우 설치가 자주 실패한다.
