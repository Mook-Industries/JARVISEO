/* 브라우저 마이크 · 스피커를 서버의 팀 음성 부품에 잇는다 (#218).  담당: 문태현

   서버에 마이크가 없을 때(클라우드 · 휴대폰) 대시보드가 쓴다. 받아쓰기 · 읽기 · "잠깐만요" ·
   되묻기 · 끼어들기는 서버가 노트북과 같은 코드로 한다(api/browser_voice.py). Chrome 받아쓰기 ·
   브라우저 목소리는 쓰지 않는다. 이 파일이 하는 일은 셋뿐이다.

     1. 소리 보내기: 마이크 80ms 를 늘 최근 4초만 들고 있다가, 호출되면 그 4초를 한꺼번에 보내고
        턴이 끝날 때(서버의 idle)까지 이어서 보낸다. 평소 소리는 서버로 가지 않는다.
     2. 장면 보내기: 턴 동안 카메라 장면을 초당 4장쯤(앞의 것이 가야 다음 것을) 보낸다.
        서버가 말을 시작한 순간 근처에서 가장 선명한 장면을 고른다(안경과 같다).
     3. 소리 내기: 서버가 보낸 PCM(24kHz)을 받은 순서대로 이어서 낸다. 채널 0 = 대답,
        1 = 시작음 · 종료음. 서버가 flush 를 보내면(끼어들기) 남은 소리를 버린다.

   보내는 모양은 api/browser_voice.py 머리 주석과 같다. 바이트 = [종류 1][pos float64][내용].
*/

const AUDIO = 1;
const FRAME = 2;
const HEADER = 9;
const SAMPLE_RATE = 16000;
const PREROLL_SEC = 4; // 호출 직전 이만큼 같이 보낸다(1.5초 앞 읽기 + 2단계 확인 1.6초 + 여유)
const FRAME_MS = 250; // 장면 간격
const FRAME_WIDTH = 1920;
const MAX_BUFFERED = 512 * 1024; // 아직 못 보낸 것이 이보다 많으면 장면은 건너뛴다(소리가 먼저)
const PLAY_RATE = 24000;

const packet = (kind, pos, body) => {
  const out = new Uint8Array(HEADER + body.byteLength);
  const view = new DataView(out.buffer);
  view.setUint8(0, kind);
  view.setFloat64(1, pos, true);
  out.set(new Uint8Array(body.buffer ?? body, body.byteOffset ?? 0, body.byteLength), HEADER);
  return out;
};

export class VoiceLink {
  /**
   * onEvent(msg): 서버 알림(ready · wake · thinking · result · rejected · no_question · error · idle).
   * camera(): 지금 장면을 보낼 수 있으면 "live"(카메라) · "photo"(올린 사진), 아니면 null.
   * grab(): 지금 장면 JPEG Blob(카메라) 또는 올린 사진 Blob. 없으면 null.
   */
  constructor({ onEvent = () => {}, camera = () => null, grab = async () => null } = {}) {
    Object.assign(this, { onEvent, camera, grab });
    this.ring = []; // 최근 80ms 들 [{pos, data}]
    this.lastPos = 0; // 받은 소리의 끝 샘플 번호
    this.streaming = false;
    this.sentUpTo = 0; // 서버에 보낸 소리의 끝 번호
    this.wakeId = 0;
    this.level = 0; // 마지막 80ms 소리 크기(RMS, 0~1). 파형용
    this.volume = 0.7;
    this.out = null; // 소리 내는 AudioContext
    this.channels = [0, 1].map(() => ({ next: 0, sources: new Set() }));
  }

  /** 서버에 붙는다. ready 를 받으면 끝나고, 못 켜면(off · 끊김) 이유와 함께 실패한다. */
  connect() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/voice/ws`;
    const ws = (this.ws = new WebSocket(url));
    ws.binaryType = "arraybuffer";
    return new Promise((ok, fail) => {
      let ready = false;
      ws.onmessage = ({ data }) => {
        if (data instanceof ArrayBuffer) return this._play(data);
        const msg = JSON.parse(data);
        if (msg.type === "ready") {
          ready = true;
          ok(msg);
        } else if (msg.type === "off") {
          fail(new Error(msg.reason || "꺼짐"));
        } else if (msg.type === "flush") {
          this._flush(msg.ch);
        } else if (msg.type === "idle" && msg.upto >= this.wakeId) {
          this.streaming = false; // 이 턴이 끝났다. 다음 호출까지 소리 · 장면을 보내지 않는다
        }
        if (ready) this.onEvent(msg);
      };
      ws.onclose = () => {
        this.streaming = false;
        if (!ready) fail(new Error("서버에 붙지 못했어요"));
        else this.onEvent({ type: "closed" });
      };
    });
  }

  get open() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /** 마이크 80ms 하나(wakeword.js onAudio). 턴 중이면 바로 보낸다. */
  feed(frame, pos) {
    let sum = 0;
    for (const v of frame) sum += v * v;
    this.level = Math.sqrt(sum / frame.length) / 32768;
    this.ring.push({ pos, data: frame });
    const keep = Math.ceil((PREROLL_SEC * SAMPLE_RATE) / frame.length);
    if (this.ring.length > keep) this.ring.shift();
    this.lastPos = pos + frame.length;
    if (this.streaming && this.open) this._sendAudio([{ pos, data: frame }]);
  }

  /** "자비서"(호출어 · 버튼). 직전 소리를 보내고 턴 동안 소리 · 장면을 이어 보낸다. */
  wake({ pos = this.lastPos, score = 1, button = false } = {}) {
    if (!this.open) return;
    this._unlock();
    this._sendAudio(this.ring); // 이미 보낸 앞부분은 서버가 버린다
    const camera = this.camera();
    const first = !this.streaming;
    this.streaming = true;
    this.wakeId += 1;
    this.ws.send(JSON.stringify({ type: "wake", id: this.wakeId, pos, score, button, camera: !!camera }));
    if (camera === "photo") this._sendFrame();
    else if (camera === "live" && first) this._frameLoop();
  }

  /** 입력칸 질문. 대답은 서버 목소리로 읽는다. image 는 data URL(없으면 사진 없이). */
  ask(question, image = null) {
    if (!this.open) return false;
    this._unlock();
    this.ws.send(JSON.stringify({ type: "ask", question, image }));
    return true;
  }

  /** 읽던 대답을 멈춘다(화면을 떠날 때). */
  stop() {
    if (this.open) this.ws.send(JSON.stringify({ type: "stop" }));
    this._flush(0);
  }

  /** 환경설정 읽기 속도(0.5~2배) → 노트북 TTS 속도 SLOW · NORMAL · FAST. */
  setRate(rate) {
    const value = rate < 0.9 ? "SLOW" : rate > 1.1 ? "FAST" : "NORMAL";
    if (this.open) this.ws.send(JSON.stringify({ type: "speed", value }));
  }

  setVolume(v) {
    this.volume = v;
    if (this.gain) this.gain.gain.value = v;
  }

  /** 대답을 읽는 중인지(호출어 감지가 끼어들기 임계값을 쓴다). */
  get playing() {
    return !!this.out && this.channels[0].next > this.out.currentTime;
  }

  close() {
    this.ws?.close();
    this.out?.close().catch(() => {});
  }

  // ---- 안쪽 ----

  _sendAudio(frames) {
    const fresh = frames.filter((f) => f.pos + f.data.length > this.sentUpTo);
    if (!fresh.length) return;
    const n = fresh.reduce((a, f) => a + f.data.length, 0);
    const pcm = new Int16Array(n);
    let at = 0;
    for (const f of fresh) {
      pcm.set(f.data, at);
      at += f.data.length;
    }
    // 이어지지 않는 조각(쉬었다 다시 보냄)도 첫 번호 하나로 보낸다. 링은 늘 이어져 있다
    this.ws.send(packet(AUDIO, fresh[0].pos, pcm));
    this.sentUpTo = fresh[0].pos + n;
  }

  async _sendFrame() {
    const pos = this.lastPos; // 찍은 순간의 소리 번호
    const blob = await this.grab(FRAME_WIDTH);
    if (!blob || !this.open) return;
    const body = new Uint8Array(await blob.arrayBuffer());
    this.ws.send(packet(FRAME, pos, body));
  }

  async _frameLoop() {
    while (this.streaming && this.open && this.camera() === "live") {
      const started = performance.now();
      if (this.ws.bufferedAmount < MAX_BUFFERED) await this._sendFrame();
      await new Promise((ok) => setTimeout(ok, Math.max(0, FRAME_MS - (performance.now() - started))));
    }
  }

  // 휴대폰 · 사파리는 사용자가 누르기 전에는 소리를 못 낸다. 호출 · 질문 때 연다
  _unlock() {
    if (!this.out) {
      this.out = new (window.AudioContext || window.webkitAudioContext)();
      this.gain = this.out.createGain();
      this.gain.gain.value = this.volume;
      this.gain.connect(this.out.destination);
    }
    if (this.out.state === "suspended") this.out.resume().catch(() => {});
  }

  _play(buf) {
    this._unlock();
    const ch = this.channels[new Uint8Array(buf, 0, 1)[0]] ?? this.channels[0];
    const pcm = new Int16Array(buf.slice(1));
    if (!pcm.length) return;
    const audio = this.out.createBuffer(1, pcm.length, PLAY_RATE);
    const data = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const src = this.out.createBufferSource();
    src.buffer = audio;
    src.connect(this.gain);
    const at = Math.max(ch.next, this.out.currentTime + 0.05);
    src.start(at);
    ch.next = at + audio.duration;
    ch.sources.add(src);
    src.onended = () => ch.sources.delete(src);
  }

  _flush(i) {
    const ch = this.channels[i];
    if (!ch) return;
    ch.sources.forEach((s) => {
      try {
        s.stop();
      } catch {
        /* 이미 끝났다 */
      }
    });
    ch.sources.clear();
    ch.next = 0;
  }
}
