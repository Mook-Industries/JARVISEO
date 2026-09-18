# 작업 컨벤션

무크 인더스트리 · 자비서 (JARVISEO)

---

## 전체 흐름

```
① 이슈를 연다        템플릿이 type 라벨을 자동으로 붙임 → area 라벨은 직접 추가
② 브랜치를 판다      fix/12-windows-hangul-path
③ 커밋한다           fix(capture): 윈도우에서 한글 경로 이미지가 열리지 않던 문제 수정
④ PR 을 연다         제목은 커밋과 같은 형식, 본문에 Closes #12
⑤ 리뷰 1명 승인      CODEOWNERS 가 리뷰어를 자동 지정
⑥ Squash and merge   브랜치 삭제
```

---

## 1. 이슈

템플릿 3종이 있다.

| 템플릿 | `type:` 라벨 | 쓸 때 |
|---|---|---|
| 🐛 Bug Report | `type: bug` **자동** | 안 되는 것 |
| ♻️ Refactor | `type: refactor` **자동** | 동작 그대로, 코드만 개선 |
| 📋 Task | `type: feature` **기본값** | 그 외 전부 — 기능 개발, 모델 학습, 촬영, 문서 |

Bug 와 Refactor 는 종류가 하나뿐이라 라벨이 고정이다.
**Task 는 여러 종류를 받으므로, 가장 흔한 `type: feature` 를 기본으로 깔아두고
다른 종류면 사이드바에서 바꾼다.** 바꾸는 것을 잊어도 라벨 없는 이슈는 생기지 않는다.

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

- 설명은 **영문 소문자 + 하이픈.** 한글·공백은 쓰지 않는다 — 맥과 윈도우에서
  인코딩이 달라 깨진다.
- 이슈 페이지 오른쪽 **Development → Create a branch** 를 쓰면 GitHub 이
  브랜치를 이슈에 자동으로 연결해준다. 이름만 위 형식으로 고쳐서 만든다.
- **`main` 에 직접 push 하지 않는다.** 반드시 PR 로 올린다.
- 머지되면 브랜치는 삭제한다.

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
> 억지로 같은 단어로 맞추지 않는다.

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

- **승인 1명**이 있어야 머지된다.
- 리뷰어는 CODEOWNERS 가 자동으로 지정한다. 건드린 파일의 담당자에게 간다.
- 리뷰가 2시간 넘게 안 붙으면 팀 채널에서 직접 요청한다 (2시간 룰).
- 리뷰어는 "확인했습니다"만 남기지 않는다. 최소 한 군데는 질문하거나 짚는다.

### 머지

**Squash and merge** 로 통일한다. 머지 후 브랜치는 삭제한다.

---

## 5. 절대 커밋하지 않는 것

| 대상 | 이유 |
|---|---|
| `.env` | API 키가 히스토리에 **영구히** 남는다. 사고 시 파일 삭제가 아니라 **키 폐기 후 재발급** |
| `*.pt`, `*.onnx` | 가중치는 공용 Drive `models/` 에. Git 에 넣으면 clone 이 느려진다 |
| `data/datasets/*`, `data/videos/*` | 공용 Drive 에서 받는다 |
| `*.db` | SQLite 는 각자 로컬에서 생성된다 |

`.gitignore` 에 다 들어 있다. `git add .` 전에 `git status` 로 확인하는 습관.

---

## 6. 코드 규칙 (맥 1대 · 윈도우 2대)

- 파일 입출력에 **`encoding="utf-8"` 을 반드시 명시한다.** 윈도우 기본값은
  cp949 라서 생략하면 맥에서만 되고 팀원 노트북에서 한글이 깨진다.
- 경로는 **`pathlib.Path`.** 문자열 결합 금지.
- 카메라 열기의 OS 분기는 `capture/frame_source.py` 안에서만 한다.
- 모델 배포 포맷은 **ONNX.** CoreML 은 맥 전용이라 팀원이 못 쓴다.
- 오디오는 **sounddevice.** PyAudio 는 윈도우 설치가 자주 실패한다.
