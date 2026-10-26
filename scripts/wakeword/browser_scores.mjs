// 브라우저 호출어(src/jarviseo/api/pages/wakeword.js)를 Node 에서 돌려 wav 마다 80ms 점수를 낸다.  담당: 문태현
//
// compare_browser.py 가 부른다. 대시보드와 같은 WakeWordScorer · 같은 onnxruntime-web(wasm)이다.
//   node scripts/wakeword/browser_scores.mjs <manifest.json>
// manifest: {"ort": "<onnxruntime-web 의 dist/ort.wasm.min.mjs>", "models": {"<파일 이름>": "<경로>"},
//            "files": [{"wav": "...", "noise": "<난수 int16 .bin>", "out": "<점수 float32 .bin>"}]}
// wav 는 16kHz 모노 16비트. evaluate.py(predict_clip, padding=1)처럼 앞뒤에 무음 1초를 붙여 흘린다.

import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { FRAME, SAMPLE_RATE, WakeWordScorer } from "../../src/jarviseo/api/pages/wakeword.js";

const manifest = JSON.parse(readFileSync(process.argv[2], "utf8"));
const ort = await import(pathToFileURL(manifest.ort).href);
ort.env.wasm.numThreads = 1;
const scorer = await WakeWordScorer.load(ort, (name) => readFileSync(manifest.models[name]));

function readWav(path) {
  const b = readFileSync(path);
  let pos = 12;
  let fmt = null;
  while (pos < b.length) {
    const id = b.toString("ascii", pos, pos + 4);
    const size = b.readUInt32LE(pos + 4);
    if (id === "fmt ") fmt = { channels: b.readUInt16LE(pos + 10), rate: b.readUInt32LE(pos + 12), bits: b.readUInt16LE(pos + 22) };
    if (id === "data") {
      if (!fmt || fmt.channels !== 1 || fmt.rate !== SAMPLE_RATE || fmt.bits !== 16) throw new Error(`16kHz 모노 16비트가 아니다: ${path}`);
      const data = b.subarray(pos + 8, pos + 8 + size);
      return new Int16Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length));
    }
    pos += 8 + size + (size % 2);
  }
  throw new Error(`data 조각이 없다: ${path}`);
}

for (const [i, f] of manifest.files.entries()) {
  const noise = readFileSync(f.noise);
  await scorer.reset(new Int16Array(noise.buffer.slice(noise.byteOffset, noise.byteOffset + noise.length)));
  const audio = readWav(f.wav);
  const padded = new Int16Array(audio.length + 2 * SAMPLE_RATE);
  padded.set(audio, SAMPLE_RATE);
  const scores = [];
  // predict_clip 과 같은 범위: range(0, len - 1280, 1280). 80ms 가 다 찬 조각만, 끝 조각은 버린다
  for (let s = 0; s < padded.length - FRAME; s += FRAME) scores.push(await scorer.score(padded.slice(s, s + FRAME)));
  writeFileSync(f.out, Buffer.from(Float32Array.from(scores).buffer));
  process.stdout.write(`\r  브라우저 ${i + 1}/${manifest.files.length}`);
}
process.stdout.write("\n");
