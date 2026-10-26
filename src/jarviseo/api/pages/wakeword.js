/* 브라우저 호출어 감지 ("자비서", #218).  담당: 문태현

   노트북 voice/wakeword.py 와 같은 모델 3개를 onnxruntime-web 으로 브라우저에서 돌린다.
   상시 듣는 것은 이 브라우저 안에서만 돌고, 서버로 가는 것은 호출된 뒤의 질문뿐이다.

     마이크 16kHz → 80ms(1280샘플)씩
       → melspectrogram.onnx  (openWakeWord 공용) 소리 → 멜 스펙트로그램(32칸), x/10 + 2
       → embedding_model.onnx (openWakeWord 공용) 멜 76줄 → 숫자 96개
       → jarviseo.onnx        (팀 학습)           최근 16개(약 1.3초) → 0~1 점수

   버퍼 크기 · 자르는 위치는 openwakeword/utils.py AudioFeatures._streaming_features 와
   model.py Model.predict 를 한 줄씩 따랐다. 같은 wav 를 넣으면 파이썬과 같은 점수가 나와야 한다
   (scripts/wakeword/compare_browser.py 로 잰다).

   WakeWordScorer 는 브라우저 · Node 둘 다에서 돈다(비교 스크립트는 Node). listenForWakeWord 는 브라우저 전용.
*/

export const SAMPLE_RATE = 16000;
export const FRAME = 1280; // 80ms
const MEL_CONTEXT = 160 * 3; // 새 80ms 앞에 붙여 멜을 다시 구하는 샘플(openWakeWord 와 같다)
const MEL_WINDOW = 76; // 임베딩 하나가 보는 멜 줄 수
const MEL_STEP = 8; // 80ms 마다 멜이 8줄 는다
const MEL_MAX = 10 * 97; // 멜은 10초어치만 둔다
const FEATURE_MAX = 120; // 임베딩은 약 10초어치만 둔다
const MODEL_FRAMES = 16; // jarviseo.onnx 가 보는 임베딩 수
const WARMUP = 5; // 처음 5번은 점수를 0 으로 낸다(openWakeWord 와 같다)
const NOISE_SAMPLES = SAMPLE_RATE * 4; // 켤 때 임베딩 버퍼를 채우는 난수 소리
export const MODEL_FILES = ["melspectrogram.onnx", "embedding_model.onnx", "jarviseo.onnx"];

const randomNoise = () => Int16Array.from({ length: NOISE_SAMPLES }, () => Math.floor(Math.random() * 2000) - 1000);

export class WakeWordScorer {
  /** ort 는 onnxruntime-web 모듈, sessions 는 MODEL_FILES 순서의 InferenceSession 3개. */
  constructor(ort, [mel, embed, model]) {
    Object.assign(this, { ort, mel, embed, model });
    this.inputs = [mel, embed, model].map((s) => s.inputNames[0]);
    this.outputs = [mel, embed, model].map((s) => s.outputNames[0]);
  }

  /** base 에서 모델 3개를 받아 연다. 브라우저는 "/models/wakeword/", Node 는 파일 경로나 버퍼. */
  static async load(ort, base, opts = { executionProviders: ["wasm"] }) {
    const sessions = [];
    for (const f of MODEL_FILES) {
      const src = typeof base === "function" ? await base(f) : base + f;
      sessions.push(await ort.InferenceSession.create(src, opts));
    }
    const scorer = new WakeWordScorer(ort, sessions);
    await scorer.reset();
    return scorer;
  }

  /** 버퍼를 비운다. noise 는 비교 스크립트가 파이썬과 같은 난수를 넣을 때만 준다. */
  async reset(noise = randomNoise()) {
    this.raw = new Int16Array(0);
    this.melRows = Array.from({ length: MEL_WINDOW }, () => new Float32Array(32).fill(1));
    const rows = await this._melspec(noise);
    const windows = [];
    for (let i = 0; i + MEL_WINDOW <= rows.length; i += MEL_STEP) windows.push(rows.slice(i, i + MEL_WINDOW));
    this.features = await this._embed(windows);
    this.scored = 0;
  }

  /** 80ms(1280샘플, Int16Array) 하나를 넣고 0~1 점수를 돌려준다. 순서대로 하나씩 불러야 한다. */
  async score(frame) {
    if (frame.length !== FRAME) throw new Error(`프레임은 ${FRAME}샘플이어야 한다: ${frame.length}`);
    const raw = new Int16Array(this.raw.length + FRAME);
    raw.set(this.raw);
    raw.set(frame, this.raw.length);
    this.raw = raw.slice(-(FRAME + MEL_CONTEXT));
    this.melRows.push(...(await this._melspec(this.raw)));
    if (this.melRows.length > MEL_MAX) this.melRows.splice(0, this.melRows.length - MEL_MAX);
    if (this.melRows.length >= MEL_WINDOW) {
      this.features.push(...(await this._embed([this.melRows.slice(-MEL_WINDOW)])));
      if (this.features.length > FEATURE_MAX) this.features.splice(0, this.features.length - FEATURE_MAX);
    }
    const last = this.features.slice(-MODEL_FRAMES);
    const x = new Float32Array(MODEL_FRAMES * 96);
    last.forEach((row, i) => x.set(row, i * 96));
    const out = await this.model.run({ [this.inputs[2]]: new this.ort.Tensor("float32", x, [1, MODEL_FRAMES, 96]) });
    const score = out[this.outputs[2]].data[0];
    return this.scored++ < WARMUP ? 0 : score;
  }

  async _melspec(samples) {
    const x = Float32Array.from(samples);
    const out = await this.mel.run({ [this.inputs[0]]: new this.ort.Tensor("float32", x, [1, x.length]) });
    const d = out[this.outputs[0]].data;
    const rows = [];
    for (let i = 0; i < d.length; i += 32) rows.push(d.slice(i, i + 32).map((v) => v / 10 + 2));
    return rows;
  }

  async _embed(windows) {
    const x = new Float32Array(windows.length * MEL_WINDOW * 32);
    windows.forEach((w, b) => w.forEach((row, i) => x.set(row, (b * MEL_WINDOW + i) * 32)));
    const t = new this.ort.Tensor("float32", x, [windows.length, MEL_WINDOW, 32, 1]);
    const d = (await this.embed.run({ [this.inputs[1]]: t }))[this.outputs[1]].data;
    return windows.map((_, b) => d.slice(b * 96, b * 96 + 96));
  }
}

// 마이크 소리를 128샘플씩 받아 메인 스레드로 넘기는 작은 오디오 워클릿
const WORKLET = `
registerProcessor("jarviseo-tap", class extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
});`;

/** 16kHz 가 아닌 소리를 16kHz 로 줄인다(구간 평균). AudioContext 가 16kHz 를 못 열 때만 쓴다. */
function downsampler(rate) {
  const ratio = rate / SAMPLE_RATE;
  let acc = 0, n = 0, pos = 0;
  return (block) => {
    const out = [];
    for (const v of block) {
      acc += v;
      n += 1;
      pos += 1;
      if (pos >= ratio) {
        out.push(acc / n);
        pos -= ratio;
        acc = 0;
        n = 0;
      }
    }
    return Float32Array.from(out);
  };
}

/**
 * 마이크를 열고 "자비서"가 들릴 때마다 onDetect(score) 를 부른다. 돌려준 stop() 으로 멈춘다.
 *
 * 대답을 읽는 동안(isPlaying() 이 참이거나 끝난 뒤 0.3초)은 playbackThreshold 를 쓴다(gate.py 와 같은 뜻).
 * 읽는 중에도 계속 들어 "자비서"로 끼어들 수 있다. 한 번 깨어나면 refractory 초는 다시 깨지 않는다.
 * 잡음 제거 · 자동 음량은 끈다(학습한 소리와 달라진다). 에코 제거는 켠다 — 이 페이지가 읽는
 * 대답 소리를 마이크에서 빼 줘서 대답 위에서도 "자비서"를 듣기 쉽다. ?aec=0 이면 끈다.
 */
export async function listenForWakeWord({
  ort,
  base = "/models/wakeword/",
  threshold = 0.97,
  playbackThreshold = 0.97,
  refractory = 2.0,
  isPlaying = () => false,
  onDetect,
  onScore = () => {},
}) {
  const scorer = await WakeWordScorer.load(ort, base);
  const aec = new URLSearchParams(location.search).get("aec") !== "0";
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: aec, noiseSuppression: false, autoGainControl: false },
  });
  let ctx, source;
  try {
    ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    source = ctx.createMediaStreamSource(stream);
  } catch {
    ctx?.close();
    ctx = new AudioContext(); // 16kHz 로 못 열면 기본 속도로 열고 직접 줄인다
    source = ctx.createMediaStreamSource(stream);
  }
  const shrink = ctx.sampleRate === SAMPLE_RATE ? (b) => b : downsampler(ctx.sampleRate);
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const tap = new AudioWorkletNode(ctx, "jarviseo-tap");
  source.connect(tap);

  let pending = new Float32Array(0);
  let busy = Promise.resolve();
  let quietUntil = 0; // 이 시각(초)까지는 다시 깨지 않는다
  let playedAt = -Infinity; // 마지막으로 대답을 읽고 있던 시각
  let stopped = false;
  tap.port.onmessage = ({ data }) => {
    const block = shrink(data);
    const joined = new Float32Array(pending.length + block.length);
    joined.set(pending);
    joined.set(block, pending.length);
    pending = joined;
    while (pending.length >= FRAME) {
      const f = pending.slice(0, FRAME);
      pending = pending.slice(FRAME);
      const frame = Int16Array.from(f, (v) => Math.max(-32768, Math.min(32767, Math.round(v * 32767))));
      const now = performance.now() / 1000;
      if (isPlaying()) playedAt = now;
      const limit = now - playedAt < 0.3 ? playbackThreshold : threshold;
      // 점수는 앞 프레임이 끝난 뒤 순서대로 낸다(버퍼가 섞이지 않게)
      busy = busy.then(async () => {
        if (stopped) return;
        const s = await scorer.score(frame);
        onScore(s);
        if (s >= limit && now >= quietUntil) {
          quietUntil = now + refractory;
          onDetect(s);
        }
      });
    }
  };

  const track = stream.getAudioTracks()[0];
  return {
    label: track?.label || "기본 마이크",
    aec,
    // 페이지를 한 번도 누르지 않았으면 브라우저가 오디오를 멈춰 둔다. 누를 때 resume() 한다
    get suspended() {
      return ctx.state === "suspended";
    },
    resume: () => ctx.resume(),
    stop() {
      stopped = true;
      tap.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
    },
  };
}
