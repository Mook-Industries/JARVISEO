/* ==========================================================================
   JARVISEO 대시보드 동작
   - 화면 전환(#monitor / #admin / #settings)
   - 모니터 시연 상태(대기 → 호출어 감지 → 처리 중 → 응답)
   - 성능 리포트 그래프, 환경설정 스위치
   원본: api/static/jarviseo.js (사용자 목업). 이 파일은 그 복사본에 실제 API 를 붙였다.
   - 모니터: 호출 → 받아쓰기(브라우저) · 말 시작 순간의 장면 → POST /api/ask(팀 그래프)
   - 환경설정 알러지 → GET · PUT /api/allergens
   - 04 대화 내역 → GET /api/turns(노트북 assistant.py 가 보낸 턴 · 이 서버에서 물은 턴, 3초마다)
   - 성능 리포트는 아직 예시 데이터다
   ========================================================================== */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  // 저장소는 막혀 있을 수 있다(사생활 보호 창 등). 실패해도 화면은 그대로 돈다.
  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v === null ? fallback : JSON.parse(v);
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch {
        /* 저장 못 해도 동작에는 지장 없음 */
      }
    },
  };

  const app = $("#app");
  const pad2 = (n) => String(n).padStart(2, "0");

  /* ------------------------------------------------------------------------
     예시 데이터
     ------------------------------------------------------------------------ */

  const SAMPLE_LOG = []; // 실제 대화는 addLog 가 쌓는다

  // 파이프라인 단계. 순서가 곧 처리 순서다.
  // 파이프라인 단계. sec 은 응답이 올 때 실제 값(latency_ms)으로 채운다.
  // 받아쓰기 · 읽기는 브라우저가 하므로 받아쓰기는 듣는 시간을 재고, 읽기는 비워 둔다.
  const STAGES = [
    { name: "음성 인식", code: "STT", sec: 0 },
    { name: "대상 검출", code: "DETECT", sec: 0 },
    { name: "성분 판정", code: "OCR", sec: 0 },
    { name: "장면 이해", code: "VLM", sec: 0 },
    { name: "음성 합성", code: "TTS", sec: 0 },
  ];
  const stageTotal = () => STAGES.reduce((s, st) => s + st.sec, 0) || 1;

  // 상태별 화면 문구. stages 는 각 단계의 모습: done(완료) · run(진행) · off(아직)
  const STATES = {
    idle: {
      badge: "STANDBY",
      core: "READY",
      title: "부르시면, 함께 볼게요.",
      sub: "“자비서”라고 부르면 대화를 시작합니다.",
      mic: "마이크 대기 중",
      reasonBadge: "대기 중",
      q: "무엇이 궁금하신가요?",
      a: "대상을 가리킨 뒤 자비서를 불러보세요.",
      conf: "—",
      feedTag: "DEMO FEED",
      feedMsg: "시선과 손끝을 따라 대상을 기다립니다.",
      targetLabel: "TARGET LOCKED",
      stages: ["off", "off", "off", "off", "off"],
    },
    wake: {
      badge: "LISTENING",
      core: "LISTENING",
      title: "네, 듣고 있어요.",
      sub: "“저거 뭐야?”",
      mic: "음성 수신 중",
      reasonBadge: "듣는 중",
      q: "“저거 뭐야?”",
      a: "말을 시작한 순간의 장면을 고정했어요.",
      conf: "—",
      feedTag: "FRAME HOLD",
      feedMsg: "호출어를 감지했습니다. 말을 시작한 순간의 프레임을 고정합니다.",
      targetLabel: "TARGET LOCKED",
      stages: ["run", "off", "off", "off", "off"],
    },
    processing: {
      badge: "PROCESSING",
      core: "ANALYZING",
      title: "함께 살펴보고 있어요.",
      sub: "가리킨 대상과 주변 장면을 분석하는 중입니다.",
      mic: "분석 중",
      reasonBadge: "처리 중",
      q: "“저거 뭐야?”",
      a: "손끝 방향과 시선으로 후보 3개 중 대상을 좁히고 있어요.",
      conf: "…",
      feedTag: "ANALYZING",
      feedMsg: "후보 3개 중 가리킨 대상을 특정하고 있습니다.",
      targetLabel: "ANALYZING",
      stages: ["done", "done", "done", "run", "off"],
    },
    response: {
      badge: "RESPONDING",
      core: "RESPONSE",
      title: "청록색 스낵 봉지예요.",
      sub: "성분표를 비춰 주시면 알레르기 성분도 확인해 드릴게요.",
      mic: "음성 응답 중",
      reasonBadge: "응답 완료",
      q: "가운데 청록색 스낵 봉지",
      a: "손끝 방향과 시선이 모두 가운데 봉지를 가리킵니다. 2순위 후보와 점수 차 0.12",
      conf: "96%",
      feedTag: "TARGET CONFIRMED",
      feedMsg: "선택한 대상: 스낵 봉지 · 확신도 96%",
      targetLabel: "TARGET LOCKED",
      stages: ["done", "done", "done", "done", "run"],
    },
  };


  /* ------------------------------------------------------------------------
     시계
     ------------------------------------------------------------------------ */

  const clock = $("#clock");
  const tick = () => {
    const d = new Date();
    clock.textContent = `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  };
  tick();
  setInterval(tick, 1000);

  /* ------------------------------------------------------------------------
     사이드바 · 프로필
     ------------------------------------------------------------------------ */

  const sbToggle = $("#sb-toggle");
  const setCollapsed = (on) => {
    app.classList.toggle("is-collapsed", on);
    sbToggle.setAttribute("aria-expanded", String(!on));
    sbToggle.setAttribute("aria-label", on ? "사이드바 펼치기" : "사이드바 접기");
    store.set("jarviseo.sidebarCollapsed", on);
  };
  setCollapsed(store.get("jarviseo.sidebarCollapsed", false));

  // 테마: 블랙(기본) · 화이트. 위쪽 막대의 버튼으로 바꾸고 이 브라우저에 저장한다
  // (head 의 짧은 스크립트가 그리기 전에 붙여서 블랙으로 번쩍였다 바뀌지 않는다)
  const setTheme = (theme) => {
    if (theme === "light") document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    store.set("jarviseo.theme", theme);
    $$("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === theme)));
  };
  setTheme(store.get("jarviseo.theme", "dark") === "light" ? "light" : "dark");
  $$("[data-theme-set]").forEach((b) => b.addEventListener("click", () => setTheme(b.dataset.themeSet)));
  sbToggle.addEventListener("click", () => setCollapsed(!app.classList.contains("is-collapsed")));

  // 좁은 화면에서는 사이드바가 서랍처럼 열린다
  $("#menu-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    app.classList.toggle("is-drawer-open");
  });

  // 로그인은 쓰지 않는다. 아래 칸(#profile-btn)은 환경설정으로 가는 링크다.

  document.addEventListener("click", (e) => {
    if (app.classList.contains("is-drawer-open") && !$("#sidebar").contains(e.target)) {
      app.classList.remove("is-drawer-open");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      app.classList.remove("is-drawer-open");
    }
  });

  /* ------------------------------------------------------------------------
     화면 전환
     ------------------------------------------------------------------------ */

  const VIEWS = ["monitor", "admin", "settings"];

  const route = () => {
    const name = location.hash.replace("#", "");
    const view = VIEWS.includes(name) ? name : "monitor";
    $$("[data-view]").forEach((el) => (el.hidden = el.dataset.view !== view));
    $$("[data-nav]").forEach((el) => {
      const on = el.dataset.nav === view;
      el.classList.toggle("is-active", on);
      if (on) el.setAttribute("aria-current", "page");
      else el.removeAttribute("aria-current");
    });
    app.classList.remove("is-drawer-open");
    if (view !== "monitor") stopDemo();
    if (view === "admin") drawChart();
  };

  window.addEventListener("hashchange", () => {
    route();
    window.scrollTo(0, 0);
  });

  /* ------------------------------------------------------------------------
     대화 내역 (사이드바 · 04 패널)
     ------------------------------------------------------------------------ */

  const logList = $("#log-list");
  const sbHistory = $("#sb-history");

  const logItem = (item) => {
    const li = document.createElement("li");
    li.className = "log-item";
    li.dataset.id = item.id;
    li.innerHTML = `
      <div class="log-meta"><span class="num">${item.time}</span><span class="tag"></span></div>
      <p class="log-q"></p>
      <p class="log-a"></p>`;
    $(".tag", li).textContent = item.tag;
    $(".log-q", li).textContent = item.q;
    $(".log-a", li).textContent = item.a;
    return li;
  };

  SAMPLE_LOG.forEach((item) => {
    logList.append(logItem(item));

    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.q;
    btn.addEventListener("click", () => {
      if (location.hash !== "#monitor") location.hash = "monitor";
      const li = $(`[data-id="${item.id}"]`, logList);
      if (!li) return;
      li.classList.remove("is-new");
      void li.offsetWidth; // 애니메이션을 다시 걸기 위해 리플로우
      li.classList.add("is-new");
      logList.scrollTo({ top: li.offsetTop - logList.offsetTop, behavior: "smooth" });
    });
    sbHistory.append(btn);
  });

  /* ------------------------------------------------------------------------
     모니터: 음성 파형 · 단계 막대
     ------------------------------------------------------------------------ */

  const wave = $("#wave");
  const waveBars = [];
  for (let i = 0; i < 64; i++) {
    const bar = document.createElement("i");
    bar.style.height = "2px"; // 조용하면 납작하다. 높이는 마이크 소리 크기로만 바뀐다(startMeter)
    wave.append(bar);
    waveBars.push(bar);
  }

  // 듣는 동안 마이크 소리 크기로 파형을 그린다. 목소리가 없으면(문턱 아래) 납작하게 둔다.
  // 브라우저 음성은 이미 연 마이크(서버로 보내는 소리)의 크기를 쓴다. 팀 음성 모드(서버 마이크)만
  // 파형용 마이크를 따로 연다. 휴대폰에서는 열지 않는다. 안드로이드 · 아이폰은 마이크를 한 곳만
  // 쓰게 해서, 한 번 더 열면 다른 쪽이 소리를 못 받는다(10/26).
  const VOICE_FLOOR = 0.02; // RMS 문턱. 방 소음은 이 아래, 말소리는 0.05~0.3
  const isPhone = window.matchMedia("(pointer: coarse)").matches;
  const meter = { ctx: null, stream: null, raf: 0 };
  const flatWave = () => waveBars.forEach((b) => (b.style.height = "2px"));
  const drawLevel = (rms) => {
    const level = rms < VOICE_FLOOR ? 0 : Math.min(1, (rms - VOICE_FLOOR) / 0.2);
    const t = performance.now() / 120;
    waveBars.forEach((b, i) => {
      const shape = 0.55 + 0.45 * Math.abs(Math.sin(i * 0.7 + t)); // 막대마다 조금씩 다르게
      b.style.height = `${(2 + level * 22 * shape).toFixed(1)}px`;
    });
  };
  const startMeter = async () => {
    stopMeter();
    flatWave();
    if (link && mic) {
      // 브라우저 음성: 서버로 보내는 그 마이크의 소리 크기(voicelink.js level)로 그린다
      const tick = () => {
        drawLevel(link?.level || 0);
        meter.raf = requestAnimationFrame(tick);
      };
      tick();
      return;
    }
    if (isPhone || !navigator.mediaDevices?.getUserMedia) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const an = ctx.createAnalyser();
      an.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(an);
      Object.assign(meter, { ctx, stream });
      const buf = new Float32Array(an.fftSize);
      const tick = () => {
        an.getFloatTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += v * v;
        drawLevel(Math.sqrt(sum / buf.length));
        meter.raf = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      flatWave(); // 마이크를 못 열면 파형 없이 받아쓰기만 한다
    }
  };
  const stopMeter = () => {
    cancelAnimationFrame(meter.raf);
    meter.stream?.getTracks().forEach((t) => t.stop());
    meter.ctx?.close().catch(() => {});
    Object.assign(meter, { ctx: null, stream: null, raf: 0 });
    flatWave();
  };

  const stagesEl = $("#stages");
  STAGES.forEach((st) => {
    const div = document.createElement("div");
    div.className = "stage";
    div.innerHTML = `
      <div class="stage-name">${st.name}</div>
      <div class="stage-val"><b>${st.code}</b><span class="num"></span></div>
      <div class="stage-bar"><i></i></div>`;
    stagesEl.append(div);
  });
  const stageEls = $$(".stage", stagesEl);

  const setStages = (statuses) => {
    let total = 0;
    statuses.forEach((status, i) => {
      const st = STAGES[i];
      const el = stageEls[i];
      // 시간을 잰 단계만 켠다(이번 질문에서 안 거친 단계는 '—')
      const on = status === "run" || (status !== "off" && st.sec > 0);
      // 아주 짧은 단계도 눈에 보이도록 최소 폭을 준다
      const w = on ? Math.max(st.sec / stageTotal(), 0.06) * 100 : 0;
      el.classList.toggle("is-running", status === "run");
      $(".stage-bar i", el).style.width = `${w.toFixed(1)}%`;
      $(".stage-val span", el).textContent = on && st.sec > 0 ? `${st.sec.toFixed(1)}s` : "—";
      if (on) total += st.sec;
    });
    $("#total-time").textContent = total ? total.toFixed(1) : "—";
  };

  /* ------------------------------------------------------------------------
     모니터: 상태 전환
     ------------------------------------------------------------------------ */

  // id 를 "monitor" 로 두면 주소의 #monitor 가 그 위치로 스크롤해 버린다
  const monitor = $("#monitor-grid");
  const core = $("#core");
  let demoLogged = true; // 시연용 자동 기록은 끈다. 실제 대화는 addLog 가 남긴다

  const setText = (sel, text) => {
    $(sel).textContent = text;
  };

  const setState = (name, { stages } = {}) => {
    const s = STATES[name];
    monitor.dataset.state = name;
    core.classList.toggle("is-listening", name === "wake");
    core.classList.toggle("is-processing", name === "processing");
    core.classList.toggle("is-responding", name === "response");

    setText("#core-badge", s.badge);
    setText("#core-state", s.core);
    setText("#core-title", s.title);
    setText("#core-sub", s.sub);
    setText("#mic-state", s.mic);
    setText("#reason-badge", s.reasonBadge);
    setText("#reason-q", s.q);
    setText("#reason-a", s.a);
    setText("#confidence", s.conf);
    setText("#feed-msg", s.feedMsg);

    // 처리 중 스캔선은 CSS 가 data-state 를 보고 켠다
    setStages(stages || s.stages);

    if (name === "response" && !demoLogged) {
      demoLogged = true;
      const d = new Date();
      const li = logItem({ id: "demo", time: `${pad2(d.getHours())}:${pad2(d.getMinutes())}`, ...DEMO_ANSWER });
      li.classList.add("is-new");
      logList.prepend(li);
      logList.scrollTo({ top: 0 });
    }
  };

  /* ------------------------------------------------------------------------
     AI 코어 (원본 neural-core 애니메이션)
     그리기 코드는 원본 그대로다. 이 프로젝트에 맞게 바꾼 건 두 가지뿐:
     - 상태: body[data-state] 대신 #core 의 상태 클래스에서 읽는다
       (대기 0 · 듣기 1 · 처리 2 · 응답 3)
     - 정지: 모니터 화면이 아님 / 탭 숨김 / HUD 애니메이션 끔(.no-motion) /
       OS 동작 줄이기 일 때 멈춘다
     ------------------------------------------------------------------------ */

  (() => {
  const canvas=document.querySelector('#neural-core');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  if(!ctx)return;
  const coreEl=document.querySelector('#core');
  const monitorView=document.querySelector('#view-monitor');
  const appEl=document.querySelector('#app');
  const motionQuery=matchMedia('(prefers-reduced-motion: reduce)');
  let size=280,dpr=1,time=0,previous=0,frame=0;
  const TAU=Math.PI*2;
  const getPhase=()=>coreEl.classList.contains('is-listening')?1:coreEl.classList.contains('is-processing')?2:coreEl.classList.contains('is-responding')?3:0;
  const particles=Array.from({length:180},(_,i)=>{const y=1-2*(i+.5)/180,r=Math.sqrt(1-y*y),a=i*2.39996323;return [Math.cos(a)*r,y,Math.sin(a)*r];});
  function resize(){size=canvas.getBoundingClientRect().width;dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(size*dpr);canvas.height=Math.round(size*dpr);draw();}
  function project(x,y,z,angle){const c=Math.cos(angle),s=Math.sin(angle),xx=x*c+z*s,zz=z*c-x*s;const tilt=.35,yy=y*Math.cos(tilt)-zz*Math.sin(tilt),depth=y*Math.sin(tilt)+zz*Math.cos(tilt);const scale=1/(1-depth*.16);return [xx*scale,yy*scale,depth];}
  function line(points,color,width=1){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
  function arc(r,start,length,color,width=1){ctx.beginPath();ctx.arc(0,0,r,start,start+length);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
  function draw(){
    if(!size)return;
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,size,size);ctx.translate(size/2,size/2);ctx.scale(size/320,size/320);
    const phase=getPhase(),power=[.65,.9,1,.8][phase];
    const breathe=1+Math.sin(time*1.8)*.016;
    const halo=ctx.createRadialGradient(0,0,7,0,0,149);halo.addColorStop(0,`rgba(70,230,245,${.12*power})`);halo.addColorStop(.5,'rgba(29,173,207,.055)');halo.addColorStop(1,'rgba(4,22,30,0)');ctx.fillStyle=halo;ctx.fillRect(-160,-160,320,320);
    ctx.globalCompositeOperation='lighter';
    // Circular spectrum follows the current assistant state.
    for(let i=0;i<160;i++){const a=i/160*TAU;const wave=Math.sin(a*7+time*2.5)*Math.sin(a*3-time*1.7);const length=3+(wave+1)*([3,8,5,6][phase]);const radius=144;line([[Math.cos(a)*radius,Math.sin(a)*radius],[Math.cos(a)*(radius+length),Math.sin(a)*(radius+length)]],`rgba(91,231,244,${.23+Math.abs(wave)*.45})`,.85);}
    arc(137,0,TAU,'rgba(84,228,241,.21)',.7);
    for(let i=0;i<4;i++){const angle=time*.15+i*TAU/4;ctx.shadowColor='#39e8ee';ctx.shadowBlur=6;arc(131,angle,.98,'rgba(110,251,255,.85)',1.8);ctx.shadowBlur=0;}
    arc(126,0,TAU,'rgba(83,228,236,.34)',.6);
    for(let i=0;i<96;i++){const a=i/96*TAU-time*.11,r=i%8===0?113:118;line([[Math.cos(a)*r,Math.sin(a)*r],[Math.cos(a)*122,Math.sin(a)*122]],`rgba(120,234,244,${i%8===0?.65:.26})`,i%8===0?1.1:.65);}
    for(let i=0;i<3;i++){arc(108,-time*.24+i*TAU/3,1.32,'rgba(61,199,220,.55)',1);arc(103,time*.12+i*TAU/3,.5,'rgba(146,246,255,.55)',.65);}
    ctx.save();ctx.scale(breathe,breathe);
    const angle=time*.24;
    // A rotating wireframe sphere with depth-sensitive meridians.
    for(let m=0;m<12;m++){const longitude=m*Math.PI/6;let last;
      for(let k=0;k<=72;k++){const a=k/72*TAU,p=project(Math.cos(a)*Math.cos(longitude),Math.sin(a),Math.cos(a)*Math.sin(longitude),angle);const point=[p[0]*66,p[1]*66];if(last)line([last,point],`rgba(65,219,241,${.09+(p[2]+1)*.16})`,.65);last=point;}}
    for(let j=-3;j<=3;j++){const y=j/4,r=Math.sqrt(1-y*y);const pts=[];for(let k=0;k<=96;k++){const a=k/96*TAU,p=project(r*Math.cos(a),y,r*Math.sin(a),angle);pts.push([p[0]*66,p[1]*66]);}line(pts,'rgba(83,224,244,.24)',.6);}
    for(const [x,y,z] of particles){const p=project(x,y,z,angle),radius=.55+(p[2]+1)*.35;ctx.fillStyle=`rgba(150,250,255,${.2+(p[2]+1)*.28})`;ctx.beginPath();ctx.arc(p[0]*66,p[1]*66,radius,0,TAU);ctx.fill();}
    // Independently precessing ellipses and moving light particles.
    for(let orbit=0;orbit<5;orbit++){const inclination=orbit*Math.PI/5+time*.06;const rotation=time*(orbit%2?.19:-.14)+orbit;const pts=[];
      const orbitPoint=a=>{const x=Math.cos(a)*91,y=Math.sin(a)*31;const xx=x*Math.cos(inclination)-y*Math.sin(inclination),yy=x*Math.sin(inclination)+y*Math.cos(inclination);return [xx,yy];};
      for(let k=0;k<=120;k++)pts.push(orbitPoint(k/120*TAU));line(pts,`rgba(74,224,248,${.23+orbit*.025})`,.8);
      const a=rotation+time*.7;const segment=[];for(let k=0;k<=22;k++)segment.push(orbitPoint(a-k*.025));ctx.shadowColor='#5af3ff';ctx.shadowBlur=7;line(segment,'rgba(116,248,255,.65)',1.3);const p=orbitPoint(a);ctx.fillStyle='#c4ffff';ctx.beginPath();ctx.arc(p[0],p[1],2,0,TAU);ctx.fill();ctx.shadowBlur=0;
    }
    const nucleus=ctx.createRadialGradient(0,0,0,0,0,20);nucleus.addColorStop(0,'rgba(163,253,255,.48)');nucleus.addColorStop(.25,'rgba(66,227,248,.2)');nucleus.addColorStop(1,'rgba(44,204,244,0)');ctx.fillStyle=nucleus;ctx.fillRect(-20,-20,40,40);
    ctx.restore();ctx.globalCompositeOperation='source-over';
  }
  function active(){return !document.hidden&&!monitorView.hidden&&!appEl.classList.contains('no-motion')&&!motionQuery.matches;}
  function animate(now){frame=0;const delta=previous?Math.min((now-previous)/1000,.05):0;previous=now;time+=delta*([1,1.35,2.1,1.15][getPhase()]);draw();if(active())frame=requestAnimationFrame(animate);}
  function sync(){if(frame)cancelAnimationFrame(frame);frame=0;previous=0;draw();if(active())frame=requestAnimationFrame(animate);}
  new ResizeObserver(resize).observe(canvas);
  // 상태 클래스 · 화면 전환(hidden) · HUD 설정이 바뀌면 다시 맞춘다
  const mo=new MutationObserver(sync);
  mo.observe(coreEl,{attributes:true,attributeFilter:['class']});
  mo.observe(monitorView,{attributes:true,attributeFilter:['hidden']});
  mo.observe(appEl,{attributes:true,attributeFilter:['class']});
  document.addEventListener('visibilitychange',sync);motionQuery.addEventListener('change',sync);resize();sync();
  })();

  /* ------------------------------------------------------------------------
     모니터: 자비서에게 묻기 (실제 API)
     - 호출("자비서" · 버튼 · 스페이스바) → 서버가 노트북과 같은 부품으로 듣고 읽는다(아래 #218,
       voicelink.js · api/browser_voice.py). Chrome 받아쓰기 · 브라우저 목소리는 쓰지 않는다
     - 장면은 서버가 말을 시작한 순간 근처에서 고른다(안경과 같다). 카메라가 꺼져 있으면 올린 사진
     - 팀 그래프 한 턴(의도 · 가리킴 · VLM · 기억 · 성분)이 끝나면 결과를 받아 보여 준다
     - 서버에 OpenAI 키가 없으면 음성은 꺼지고 입력칸 질문(POST /api/ask)만 된다
     ------------------------------------------------------------------------ */

  let busy = false;
  let voiceMode = false; // 팀 음성 모드: 서버가 이 PC 마이크로 듣고 읽는다(api/voice_loop.py)
  let link = null; // 브라우저 음성: 이 브라우저 마이크 · 스피커로 서버가 듣고 읽는다(voicelink.js)
  let mic = null; // 브라우저 마이크(wakeword.js listenForWakeWord). "자비서"도 여기서 듣는다
  let photo = null; // 카메라가 꺼져 있을 때 올린 사진(data URL)
  const shot = $("#feed-shot");
  const askForm = $("#ask-form");
  const askInput = $("#ask-input");
  const INTENT_TAG = { general: "일반 질문", pointing: "가리킴", ingredient: "성분 판정", recall: "기억 검색" };
  const STATUS = { contains: "들어 있음", may: "섞였을 수 있음" };

  // 화면을 떠나면 읽던 대답을 멈춘다(원본의 시연 정지 자리)
  function stopDemo() {
    link?.stop();
    stopMeter();
  }

  const showShot = (url) => {
    shot.hidden = !url;
    shot.classList.remove("is-big");
    if (url) shot.src = url;
    else shot.removeAttribute("src");
    if (typeof placeShotClose === "function") placeShotClose();
  };
  // 작은 결과 창은 누르면 커지고, 다시 누르면 작아진다
  shot.addEventListener("click", () => shot.classList.toggle("is-big"));

  // 결과 창 닫기(×). 보고 싶지 않으면 끈다. 다음 질문의 결과는 다시 뜬다.
  // 카메라가 켜져 있어 작은 창일 때만 보인다(올린 사진은 화면 전체라 닫으면 볼 것이 없다).
  const shotClose = document.createElement("button");
  shotClose.type = "button";
  shotClose.className = "shot-close";
  shotClose.setAttribute("aria-label", "결과 그림 닫기");
  shotClose.textContent = "×";
  shotClose.hidden = true;
  shot.after(shotClose);
  const placeShotClose = () => {
    const box = shot.parentElement;
    shotClose.hidden = shot.hidden || box.dataset.cam !== "on";
    if (shotClose.hidden) return;
    const f = box.getBoundingClientRect();
    const b = shot.getBoundingClientRect();
    shotClose.style.left = `${b.right - f.left - 32}px`;
    shotClose.style.top = `${b.top - f.top + 6}px`;
  };
  new ResizeObserver(placeShotClose).observe(shot);
  shot.addEventListener("load", placeShotClose);
  window.addEventListener("resize", placeShotClose);
  shotClose.addEventListener("click", (e) => {
    e.stopPropagation();
    showShot(null);
  });

  function clearPhoto() {
    photo = null;
    if (feed.dataset.cam === "photo") setNoSignal("카메라가 꺼져 있습니다.");
    showShot(null);
  }

  // 지금 장면 한 장. 카메라가 켜져 있으면 그 프레임, 아니면 올린 사진
  const grabFrame = () => {
    if (feed.dataset.cam === "on" && video.videoWidth) {
      const c = document.createElement("canvas");
      c.width = video.videoWidth;
      c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      return c.toDataURL("image/jpeg", 0.9);
    }
    return photo;
  };

  $("#photo-btn").addEventListener("click", () => $("#photo-file").click());
  $("#photo-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = () => {
      photo = r.result;
      stopCamera();
      feed.dataset.cam = "photo";
      showShot(photo);
      setText("#feed-tag", "PHOTO");
      setText("#cam-state", "사진");
      setText("#cam-name", file.name);
      syncCamToggle();
      setText("#feed-msg", "올린 사진으로 물어볼게요. 자비서를 불러 보세요.");
    };
    r.readAsDataURL(file);
    e.target.value = "";
  });

  let logNo = 0;
  const shownTurns = new Set(); // 이미 보인 서버 turn_id. 물은 턴과 불러온 턴이 두 번 뜨지 않게
  // turnId: 서버 기록 번호(있으면). at: 물은 시각. quiet: 켤 때 불러온 지난 기록이라 깜빡이지 않는다
  const addLog = (tag, q, a, { turnId = null, at = new Date(), quiet = false, followUp = false } = {}) => {
    if (turnId != null) {
      if (shownTurns.has(turnId)) return;
      shownTurns.add(turnId);
    }
    // 되물은 뒤 호출어 없이 이어 말한 대답은 새 줄이 아니라 바로 앞 대화 밑에 붙인다(한 대화로 보인다)
    const parent = followUp ? $(".log-item[data-id^='turn-']", logList) : null;
    if (parent) {
      const more = document.createElement("div");
      more.className = "log-follow";
      more.innerHTML = `<p class="log-q"></p><p class="log-a"></p>`;
      $(".log-q", more).textContent = q;
      $(".log-a", more).textContent = a;
      parent.append(more);
      if (!quiet) {
        parent.classList.remove("is-new");
        void parent.offsetWidth;
        parent.classList.add("is-new");
        logList.scrollTo({ top: 0 });
      }
      return;
    }
    const id = turnId != null ? `turn-${turnId}` : `t${++logNo}`;
    const item = { id, time: `${pad2(at.getHours())}:${pad2(at.getMinutes())}`, tag, q, a };
    const li = logItem(item);
    logList.prepend(li);
    if (!quiet) {
      li.classList.add("is-new");
      logList.scrollTo({ top: 0 });
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      if (location.hash !== "#monitor") location.hash = "monitor";
      li.classList.remove("is-new");
      void li.offsetWidth;
      li.classList.add("is-new");
      logList.scrollTo({ top: li.offsetTop - logList.offsetTop, behavior: "smooth" });
    });
    sbHistory.prepend(btn);
  };

  /* ------------------------------------------------------------------------
     대화 기록 불러오기 (#144) — 이 서버 DB 의 턴. 노트북(assistant.py)이 보낸 턴과 웹에서 물은 턴.
     켤 때 최근 30개, 그 뒤 3초마다 새 턴만 묻는다(?after=마지막 번호).
     ------------------------------------------------------------------------ */

  const TURN_POLL_MS = 3000;
  let lastTurnId = null; // null = 아직 한 번도 못 불러옴

  async function loadTurns() {
    const first = lastTurnId === null;
    try {
      const r = await fetch(first ? "/api/turns?limit=30" : `/api/turns?after=${lastTurnId}`);
      if (!r.ok) return;
      const { turns } = await r.json();
      // 처음에는 새것부터 오고, 그 뒤에는 들어온 순서로 온다. 오래된 것부터 위에 쌓는다.
      (first ? turns.reverse() : turns).forEach((t) => {
        addLog(t.where, t.question, t.answer || "(대답 없음)", {
          turnId: t.id,
          at: new Date(t.asked_at),
          quiet: first,
          followUp: t.follow_up,
        });
        lastTurnId = Math.max(lastTurnId ?? 0, t.id);
      });
      lastTurnId ??= 0;
    } catch {
      /* 서버가 잠깐 안 되면 다음 차례에 다시 묻는다 */
    }
  }
  loadTurns();
  setInterval(loadTurns, TURN_POLL_MS);

  // 03 판단 근거의 한 줄 설명
  const reasonOf = (d) => {
    const ing = d.ingredient;
    if (ing) {
      const found = ing.evidence.filter((e) => e.status !== "not_found");
      const parts = found.map((e) => `${e.allergen}(${STATUS[e.status]} · ${e.where.map((w) => w.section).join("·")})`);
      const steps = ing.barcode_steps.join(" → ");
      return { text: (parts.length ? parts.join(", ") : "등록한 알레르기를 찾지 못했어요") + ` — ${steps}`, warn: parts.length > 0 };
    }
    if (d.target) return { text: `가리킨 대상: ${d.target} · 후보 ${d.candidates}개 중 선택`, warn: false };
    if (d.memories && d.memories.length) return { text: `기억 ${d.memories.length}개를 찾아 답했어요.`, warn: false };
    return { text: d.vlm_real ? "사진과 질문을 장면 이해 모델(VLM)에 보내 답했어요." : "VLM 키가 없어 가짜 답이에요.", warn: !d.vlm_real };
  };

  function showResult(question, d, sttSec) {
    const lat = d.latency_ms || {};
    STAGES[0].sec = sttSec || 0;
    STAGES[1].sec = (lat.detect || 0) / 1000;
    STAGES[2].sec = (lat.ingredient || 0) / 1000;
    STAGES[3].sec = (lat.vlm || 0) / 1000;
    STAGES[4].sec = 0;
    const answer = d.answer || (d.spoken || []).join(" ") || "대답을 만들지 못했어요.";
    const tag = INTENT_TAG[String(d.intent).toLowerCase()] || "질문";
    setState("response", { stages: ["done", "done", "done", "done", "done"] });
    setText("#core-title", answer);
    setText("#core-sub", `${tag}${d.target ? ` · ${d.target}` : ""}`);
    setText("#reason-q", `“${question}”`);
    const why = reasonOf(d);
    setText("#reason-a", why.text);
    $("#reason-a").classList.toggle("is-warn", why.warn);
    setText("#confidence", d.target_score != null ? `${Math.round(d.target_score * 100)}%` : "—");
    setText("#total-time", (d.elapsed_ms / 1000).toFixed(1));
    setText("#feed-tag", d.ingredient ? "INGREDIENT" : d.target ? "TARGET CONFIRMED" : "ANSWERED");
    setText("#feed-msg", d.target ? `선택한 대상: ${d.target}` : answer);
    // 찍은 사진 전체가 아니라 판정 · 대답에 쓴 크롭만 보여 준다.
    // 성분 판정이 크롭을 읽었으면 찾은 글자 표시가 그려진 그 크롭(where), 아니면 크롭 그대로.
    // 크롭이 없으면(가리키지 않은 질문) 사진을 덮지 않고 카메라 화면을 그대로 둔다.
    const whereIsCrop = d.ingredient && d.ingredient.where && d.crop && !d.retried_whole;
    const overlay = d.crop ? (whereIsCrop ? d.ingredient.where : d.crop) : null;
    if (overlay && !app.classList.contains("no-overlay")) showShot(overlay);
    addLog(tag, question, answer, { turnId: d.turn_id, followUp: d.follow_up });
    // 대답 소리는 서버가 읽었다(팀 음성 모드는 PC 스피커, 브라우저 음성은 이 브라우저로 보냄)
  }

  // 서버에 OpenAI 키가 없을 때만 쓴다(입력칸 질문, 소리 없이 화면에만)
  async function ask(question, frame, sttSec, more = [], trigger = "WAKEWORD") {
    busy = true;
    STAGES.forEach((st) => (st.sec = 0));
    STAGES[0].sec = sttSec || 0;
    setState("processing", { stages: ["done", "run", "off", "off", "off"] });
    setText("#reason-q", `“${question}”`);
    setText("#reason-a", frame ? "말을 시작한 순간의 장면으로 살펴보고 있어요." : "사진 없이 물어보고 있어요.");
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          image: frame || null,
          more: frame ? more.map((m) => m.shot) : [],
          more_ms: frame ? more.map((m) => m.ms) : [],
          trigger,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      showResult(question, d, sttSec);
    } catch (err) {
      setState("idle");
      setText("#core-title", "대답하지 못했어요.");
      setText("#core-sub", String(err.message || err));
    } finally {
      busy = false;
    }
  }

  // 호출 버튼 · 스페이스바 = "자비서"라고 부른 것(읽는 중이면 끼어들기)
  async function listen() {
    camWake(); // 부른 순간 바로 고해상도로(카메라가 바뀌는 데 잠깐 걸린다)
    if (voiceMode) {
      fetch("/api/voice/call", { method: "POST" }); // 듣기 · 읽기는 서버가 PC 마이크로 한다
      return;
    }
    if (!link?.open) {
      setText("#core-sub", "음성은 서버에 OpenAI 키가 있어야 해요. 아래에 입력해 주세요.");
      askInput.focus();
      return;
    }
    try {
      await ensureMic(); // 휴대폰은 처음 누를 때 마이크를 연다
    } catch (err) {
      setText("#core-sub", "마이크를 열지 못했어요. 권한을 확인하거나 아래에 입력해 주세요.");
      return;
    }
    link.wake({ pos: mic.pos, button: true });
  }

  $("#call-btn").addEventListener("click", listen);

  // 스페이스바 = 자비서 호출 (입력칸·버튼에 초점이 있을 때는 건드리지 않는다)
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat) return;
    if ($("#view-monitor").hidden) return;
    if (e.target.closest("input, textarea, select, button, a, [contenteditable]")) return;
    e.preventDefault();
    listen();
  });

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = askInput.value.trim();
    if (!q || busy) return;
    askInput.value = "";
    stopDemo();
    // 대기(저해상도) 중이면 호출 설정으로 바꾸고 새 장면이 들어온 뒤에 찍는다.
    // 저해상도 장면으로는 성분표 글자 · 바코드를 못 읽는다
    if (camMode !== "active" && feed.dataset.cam === "on") {
      await setCamMode("active");
      await new Promise((ok) => setTimeout(ok, CAM_SETTLE_MS));
    }
    if (link?.ask(q, grabFrame())) {
      setState("processing", { stages: ["done", "run", "off", "off", "off"] });
      setText("#reason-q", `“${q}”`);
      return; // 대기로 돌아가는 것은 서버 알림(result)이 한다
    }
    await ask(q, grabFrame(), 0);
    camRest();
  });

  /* ------------------------------------------------------------------------
     모니터: 실시간 카메라
     - 켜기 전에 확인 창을 먼저 띄우고, 동의하면 브라우저 권한을 요청한다
     - 카메라를 못 쓰는 환경(미지원·권한 거부·장치 없음)은 '신호 없음'으로 표시
     ------------------------------------------------------------------------ */

  const feed = $("#feed");
  const video = $("#feed-video");
  const camModal = $("#cam-modal");
  const camRetry = $("#cam-retry");
  const camToggle = $("#cam-toggle");
  let camStream = null;

  /* 카메라 모드 — "자비서"를 부르기 전에는 낮은 해상도 · 낮은 Hz 로 아껴 두었다가(대기),
     부르면 높은 해상도 · 15Hz 로 바꾼다(호출). 대답이 끝나고 CAM_IDLE_HOLD_MS 뒤 대기로 돌아간다.
     이어 묻기 · 되묻기는 그 사이에 다시 wake 가 와서 호출 그대로다.
     카메라 · 해상도 · Hz 는 환경설정 '카메라'에서 고르고 이 브라우저에 저장한다.
     Hz 는 카메라가 내는 초당 장면 수이자 팀 음성 모드에서 서버로 보내는 간격이다(feedFrames).
     카메라가 그 값을 못 내면 브라우저가 가장 가까운 값으로 연다(ideal). */
  const CAM_DEFAULT = {
    deviceId: "",
    saver: true,
    idle: { res: "640x360", hz: 5 },
    active: { res: "1920x1080", hz: 15 },
  };
  const savedCam = store.get("jarviseo.camera", {});
  const camCfg = {
    ...CAM_DEFAULT,
    ...savedCam,
    idle: { ...CAM_DEFAULT.idle, ...(savedCam.idle || {}) },
    active: { ...CAM_DEFAULT.active, ...(savedCam.active || {}) },
  };
  const CAM_IDLE_HOLD_MS = 4000;
  // 해상도를 바꾼 뒤 새 크기의 장면이 들어오기까지 기다리는 시간(입력창 질문에서 찍기 전)
  const CAM_SETTLE_MS = 400;
  let camMode = "idle";
  let camIdleTimer = null;

  // 절전을 끄면 늘 호출 설정으로 연다
  const camProfile = (mode = camMode) => (camCfg.saver ? camCfg[mode] : camCfg.active);
  const camSize = (res, caps = {}) => {
    if (res === "max") return [Math.min(caps.width?.max || 1920, 3840), Math.min(caps.height?.max || 1080, 2160)];
    return res.split("x").map(Number);
  };
  const camConstraints = (p, caps) => {
    const [w, h] = camSize(p.res, caps);
    // crop-and-scale: 카메라가 30Hz 로만 내도 브라우저가 장면을 덜어 낮은 Hz 로 맞춘다
    return { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: p.hz, max: p.hz }, resizeMode: "crop-and-scale" };
  };
  // 팀 음성 모드에서 장면을 서버로 보내는 간격(ms)
  const frameGapMs = () => Math.round(1000 / Math.max(1, camProfile().hz));

  const showCamSpec = (track) => {
    const s = track.getSettings();
    const label = camCfg.saver ? (camMode === "active" ? "호출" : "대기") : "고정";
    $("#feed-spec").innerHTML =
      `<span>${s.width || "—"} × ${s.height || "—"}</span><span>${s.frameRate ? Math.round(s.frameRate) : "—"} FPS · ${label}</span>`;
  };

  async function setCamMode(mode) {
    clearTimeout(camIdleTimer);
    camIdleTimer = null;
    camMode = mode;
    const track = camStream?.getVideoTracks()[0];
    if (!track) return;
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    try {
      await track.applyConstraints(camConstraints(camProfile(mode), caps));
    } catch (err) {
      console.warn("[camera] 해상도 · Hz 를 바꾸지 못함", mode, err);
    }
    console.info("[camera]", mode, track.getSettings());
    showCamSpec(track);
  }
  // "자비서"를 들었다(호출어 · 버튼 · 서버 알림). 이미 호출이면 대기로 돌아가는 것만 멈춘다
  const camWake = () => {
    if (camMode !== "active") setCamMode("active");
    else clearTimeout(camIdleTimer);
  };
  // 턴이 끝났다. 잠시 뒤 대기로(바로 이어 묻거나 되물으면 그대로 호출)
  const camRest = () => {
    if (!camCfg.saver || camMode === "idle") return;
    clearTimeout(camIdleTimer);
    camIdleTimer = setTimeout(() => setCamMode("idle"), CAM_IDLE_HOLD_MS);
  };

  // 헤더 버튼: 켜져 있으면 '끄기', 꺼져 있으면 '켜기'
  const syncCamToggle = () => {
    const on = feed.dataset.cam === "on";
    camToggle.classList.toggle("is-on", on);
    camToggle.disabled = feed.dataset.cam === "connecting";
    setText("#cam-toggle-label", on ? "카메라 끄기" : "카메라 켜기");
  };

  // getUserMedia 는 https 또는 localhost 에서만 동작한다
  const camSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && window.isSecureContext;

  const setNoSignal = (msg, { canRetry = true } = {}) => {
    feed.dataset.cam = "off";
    video.srcObject = null;
    setText("#no-signal-msg", msg);
    camRetry.hidden = !canRetry;
    setText("#feed-tag", "NO SIGNAL");
    $("#feed-spec").innerHTML = "<span>— × —</span><span>— FPS</span>";
    setText("#cam-state", "꺼짐");
    setText("#cam-name", "—");
    syncCamToggle();
  };

  const stopCamera = () => {
    if (camStream) camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  };

  const startCamera = async () => {
    stopCamera();
    feed.dataset.cam = "connecting";
    setText("#no-signal-msg", "카메라를 연결하는 중…");
    camRetry.hidden = true;
    setText("#feed-tag", "CONNECTING");
    syncCamToggle();

    // 휴대폰은 뒤 카메라(가리키는 물건 쪽). 노트북 웹캠은 하나라 facingMode 를 무시한다.
    // 환경설정에서 카메라를 골랐으면 그 카메라로 연다. 해상도 · Hz 는 지금 모드(대기 · 호출) 값이다
    camMode = camCfg.saver ? "idle" : "active";
    const size = camConstraints(camProfile());
    try {
      try {
        if (camCfg.deviceId) {
          camStream = await navigator.mediaDevices.getUserMedia({
            video: { ...size, deviceId: { exact: camCfg.deviceId } },
            audio: false,
          });
        } else {
          // ideal 로만 주면 해상도 조건과 겨뤄 휴대폰이 앞 카메라를 고르기도 했다(10/26). 휴대폰은 꼭 뒤로
          if (!isPhone) throw new Error("not phone");
          camStream = await navigator.mediaDevices.getUserMedia({
            video: { ...size, facingMode: { exact: "environment" } },
            audio: false,
          });
        }
      } catch (err) {
        if (err.name === "NotAllowedError") throw err;
        // 고른 카메라가 빠졌거나 휴대폰이 아니면 기본 카메라로
        camStream = await navigator.mediaDevices.getUserMedia({
          video: { ...size, facingMode: { ideal: "environment" } },
          audio: false,
        });
      }
    } catch (err) {
      camStream = null;
      const reasons = {
        NotAllowedError: "카메라 권한이 거부되었습니다. 주소창의 카메라 아이콘에서 허용해 주세요.",
        NotFoundError: "연결된 카메라를 찾지 못했습니다.",
        NotReadableError: "다른 프로그램이 카메라를 사용 중입니다.",
        OverconstrainedError: "카메라가 요청한 해상도를 지원하지 않습니다.",
      };
      setNoSignal(reasons[err.name] || "카메라를 켜지 못했습니다.");
      return;
    }

    const track = camStream.getVideoTracks()[0];
    clearPhoto();
    video.srcObject = camStream;
    feed.dataset.cam = "on";

    // 열고 나서 카메라가 낼 수 있는 범위(최대 해상도 등)를 알게 되면 지금 모드 값으로 다시 맞춘다.
    // 'ideal 1920×1080' 만 주면 1080p 를 지원하는 웹캠도 720p 로 열리는 일이 있다.
    await setCamMode(camMode);
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    console.info("카메라", track.label, "설정", track.getSettings(), "최대", caps.width, caps.height);
    listCameras();
    setText("#feed-tag", "LIVE");
    setText("#cam-state", "켜짐");
    setText("#cam-name", track.label || "CAMERA");
    syncCamToggle();

    // USB 케이블이 빠지는 등 연결이 끊기면 다시 '신호 없음'
    track.addEventListener("ended", () => {
      camStream = null;
      setNoSignal("카메라 연결이 끊겼습니다.");
    });
  };

  const openCamModal = () => {
    camModal.hidden = false;
    $("#cam-allow").focus();
  };
  const closeCamModal = () => {
    camModal.hidden = true;
  };

  $("#cam-allow").addEventListener("click", () => {
    closeCamModal();
    startCamera();
  });
  $("#cam-deny").addEventListener("click", () => {
    closeCamModal();
    setNoSignal("카메라를 켜지 않았습니다.");
  });
  camModal.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#cam-deny").click();
  });
  camRetry.addEventListener("click", openCamModal);
  camToggle.addEventListener("click", () => {
    if (feed.dataset.cam === "on") {
      stopCamera();
      setNoSignal("카메라를 껐습니다.");
    } else {
      openCamModal();
    }
  });
  window.addEventListener("pagehide", stopCamera);

  /* 환경설정 '카메라' — 카메라 고르기 · 호출 전 절전 · 대기/호출 해상도와 Hz */
  const camDevice = $("#cam-device");
  const camSaver = $("#cam-saver");
  const camPick = {
    idle: { res: $("#cam-idle-res"), hz: $("#cam-idle-hz") },
    active: { res: $("#cam-active-res"), hz: $("#cam-active-hz") },
  };
  // 카메라 이름은 권한을 받은 뒤에야 보인다. 카메라를 켠 뒤 · 장치가 바뀔 때 다시 읽는다
  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const keep = camCfg.deviceId;
    camDevice.replaceChildren(new Option("기본 카메라", ""));
    cams.forEach((d, i) => camDevice.append(new Option(d.label || `카메라 ${i + 1}`, d.deviceId)));
    camDevice.value = cams.some((d) => d.deviceId === keep) ? keep : "";
  }
  navigator.mediaDevices?.addEventListener?.("devicechange", listCameras);

  const syncCamPrefs = () => {
    camSaver.checked = camCfg.saver;
    $("#cam-saver-state").textContent = camCfg.saver ? "켜짐" : "꺼짐";
    $("#cam-idle-field").hidden = !camCfg.saver;
    for (const mode of ["idle", "active"]) {
      camPick[mode].res.value = camCfg[mode].res;
      camPick[mode].hz.value = String(camCfg[mode].hz);
    }
  };
  const saveCamPrefs = () => {
    store.set("jarviseo.camera", camCfg);
    syncCamPrefs();
  };
  camDevice.addEventListener("change", () => {
    camCfg.deviceId = camDevice.value;
    saveCamPrefs();
    if (feed.dataset.cam === "on") startCamera(); // 다른 카메라는 다시 열어야 한다
  });
  camSaver.addEventListener("change", () => {
    camCfg.saver = camSaver.checked;
    saveCamPrefs();
    setCamMode(camCfg.saver ? "idle" : "active");
  });
  for (const mode of ["idle", "active"]) {
    const apply = () => {
      camCfg[mode] = { res: camPick[mode].res.value, hz: Number(camPick[mode].hz.value) };
      saveCamPrefs();
      if (camMode === mode || !camCfg.saver) setCamMode(camMode); // 지금 쓰는 설정이면 바로 바꾼다
    };
    camPick[mode].res.addEventListener("change", apply);
    camPick[mode].hz.addEventListener("change", apply);
  }
  syncCamPrefs();
  listCameras();

  const browserCamera = () => {
    if (camSupported) {
      camToggle.hidden = false;
      setNoSignal("카메라가 꺼져 있습니다.");
      openCamModal();
    } else {
      setNoSignal("이 환경에서는 카메라를 사용할 수 없습니다.", { canRetry: false });
    }
  };

  /* 팀 음성 모드 — 안경 실행(scripts/assistant.py)과 같은 팀 부품을 서버가 그대로 돌린다.
     호출어 모델 · 시작음 · OpenAI 받아쓰기 · "잠깐만요" · OpenAI 목소리 · 끼어들기 · 되묻기 · 기억.
     서버가 이 PC 의 마이크 · 스피커 · 웹캠을 열고, 페이지는 결과(/api/voice/events)와
     카메라 미리 보기(/video)만 보여 준다. 못 켜면(키 · 장치 · 모델 없음) 브라우저 방식으로 돈다. */
  // 카메라 장면을 서버의 프레임 버퍼로 보낸다(지금 카메라 모드의 Hz, 앞의 것이 가야 다음 것을 보낸다).
  // 서버는 "자비서" 뒤 말을 시작한 순간 근처에서 가장 선명한 장면을 고른다(안경과 같다).
  const FRAME_WIDTH = 1920;
  async function feedFrames() {
    const c = document.createElement("canvas");
    while (voiceMode) {
      const started = performance.now();
      if (feed.dataset.cam === "on" && video.videoWidth) {
        const scale = Math.min(1, FRAME_WIDTH / video.videoWidth);
        c.width = Math.round(video.videoWidth * scale);
        c.height = Math.round(video.videoHeight * scale);
        c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
        const blob = await new Promise((ok) => c.toBlob(ok, "image/jpeg", 0.85));
        if (blob) await fetch("/api/voice/frame", { method: "POST", body: blob }).catch(() => {});
      }
      await new Promise((ok) => setTimeout(ok, Math.max(0, frameGapMs() - (performance.now() - started))));
    }
  }

  const onVoiceEvent = (e) => {
    if (e.type === "wake") camWake();
    else if (["result", "no_question", "error"].includes(e.type)) camRest();
    if (e.type === "wake" && e.follow_up) {
      // 자비서가 되물었다(성분표 다시 보여 주기 등). 호출어 없이 이어 듣는다. 되물은 말은 그대로 둔다
      const asked = $("#core-title").textContent;
      startMeter();
      showShot(null);
      setState("wake", { stages: ["run", "off", "off", "off", "off"] });
      setText("#core-title", asked);
      setText("#core-sub", "이어서 말씀하세요… (자비서를 다시 부르지 않아도 돼요)");
      setText("#reason-q", "이어서 듣고 있어요");
    } else if (e.type === "wake") {
      startMeter(); // 질문을 듣는 동안 목소리 크기로 파형을 그린다
      showShot(null);
      setState("wake", { stages: ["run", "off", "off", "off", "off"] });
      setText("#core-sub", e.barge_in ? "끼어들었어요. 말씀하세요…" : "말씀하세요…");
      setText("#reason-q", "듣고 있어요");
      setText("#reason-a", "말을 시작한 순간 가장 선명한 장면을 고를게요.");
    } else if (e.type === "thinking") {
      stopMeter();
      setState("processing", { stages: ["done", "run", "off", "off", "off"] });
      setText("#core-sub", "생각하고 있어요…");
    } else if (e.type === "result") {
      stopMeter();
      showResult(e.data.question, e.data, 0);
    } else if (e.type === "no_question") {
      stopMeter();
      setState("idle");
      // 이어 듣기에서 말이 없었으면 사용자가 그냥 끝낸 것이다
      setText("#core-sub", e.follow_up ? "더 물어보려면 자비서를 불러 주세요." : "말씀을 듣지 못했어요. 다시 불러 주세요.");
    } else if (e.type === "error") {
      setState("idle");
      setText("#core-title", "대답하지 못했어요.");
      setText("#core-sub", e.message || "");
    }
  };

  let voiceAfter = 0;
  async function pollVoice() {
    while (voiceMode) {
      try {
        const r = await (await fetch(`/api/voice/events?after=${voiceAfter}`)).json();
        voiceAfter = r.last;
        r.events.forEach(onVoiceEvent);
      } catch (err) {
        // 서버를 다시 켜는 중일 수 있다. 계속 묻는다
      }
      await new Promise((ok) => setTimeout(ok, 300));
    }
  }

  async function startVoice() {
    // ?voice=browser 면 서버에 마이크가 있어도 브라우저 음성으로 돈다(노트북에서 클라우드와 같게 시험)
    if (new URLSearchParams(location.search).get("voice") === "browser") {
      startBrowserVoice();
      browserCamera();
      return;
    }
    setText("#core-sub", "팀 음성 모드를 준비하고 있어요… (처음엔 20초쯤 걸려요)");
    try {
      const r = await (await fetch("/api/voice/start", { method: "POST" })).json();
      if (!r.on) throw new Error(r.reason || "꺼짐");
      voiceMode = true;
      sendVoiceSettings();
      setText("#core-sub", "“자비서”라고 부르면 대화를 시작합니다.");
      setText("#reason-a", `팀 음성 모드 · 가리킴 ${r.pointing} · 대답은 PC 스피커로 읽어요.`);
      console.info("[voice] 팀 음성 모드 켜짐 · 가리킴", r.pointing);
      // 페이지를 새로 열었으면 지난 알림은 건너뛴다
      voiceAfter = (await (await fetch("/api/voice/events")).json()).last;
      pollVoice();
      feedFrames();
    } catch (err) {
      console.info("[voice] 팀 음성 모드 꺼짐 — 브라우저 음성으로 돈다:", err.message);
      startBrowserVoice();
    }
    browserCamera();
  }
  startVoice();

  /* ------------------------------------------------------------------------
     브라우저 음성 (#218) — 서버에 마이크가 없을 때(클라우드 · 휴대폰) 이 브라우저가 마이크 · 스피커다
     - "자비서"는 노트북과 같은 모델 3개를 onnxruntime-web 으로 이 브라우저에서 듣는다(wakeword.js)
     - 호출되면 직전 4초와 그 턴 동안의 소리 · 장면만 서버로 간다(voicelink.js). 서버가 노트북과
       같은 부품으로 2단계 확인 · 받아쓰기(OpenAI) · "잠깐만요" · 읽기(OpenAI 목소리) · 되묻기 ·
       끼어들기를 하고, 소리를 이 브라우저로 보낸다(api/browser_voice.py)
     - 휴대폰은 호출어를 켜지 않고(배터리 · 상시 마이크) 호출 버튼을 누를 때 마이크를 연다
     ------------------------------------------------------------------------ */

  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";

  // 대기 화면 문구. 호출어가 켜졌는지에 따라 바뀐다(대답 뒤 대기로 돌아갈 때도 이 문구)
  const idleSub = (text) => {
    STATES.idle.sub = text;
    setText("#core-sub", text);
  };

  // 지금 장면을 서버로 보낼 수 있는지: 카메라("live") · 올린 사진("photo") · 없음(null)
  const cameraState = () => (feed.dataset.cam === "on" && video.videoWidth ? "live" : photo ? "photo" : null);

  // 지금 장면 JPEG. 카메라는 width 까지 줄이고, 올린 사진은 그대로
  async function grabBlob(width) {
    if (cameraState() === "live") {
      const c = document.createElement("canvas");
      const scale = Math.min(1, width / video.videoWidth);
      c.width = Math.round(video.videoWidth * scale);
      c.height = Math.round(video.videoHeight * scale);
      c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
      return new Promise((ok) => c.toBlob(ok, "image/jpeg", 0.85));
    }
    return photo ? (await fetch(photo)).blob() : null;
  }

  // 브라우저 마이크를 연다(한 번만). 데스크톱은 "자비서"도 듣고, 휴대폰 · 모델 없는 서버는 소리만 받는다
  let micOpening = null;
  function ensureMic() {
    if (mic) return Promise.resolve(mic);
    micOpening ??= (async () => {
      const [info, { listenForWakeWord }] = await Promise.all([
        fetch("/api/wakeword").then((r) => r.json()),
        import("/wakeword.js"),
      ]);
      const detect = info.available && !isPhone;
      const ort = detect ? await import(ORT_URL) : null;
      if (ort) ort.env.wasm.numThreads = 1;
      mic = await listenForWakeWord({
        ort,
        detect,
        threshold: info.threshold,
        playbackThreshold: info.playback_threshold,
        isPlaying: () => !!link?.playing,
        onAudio: (frame, pos) => link?.feed(frame, pos),
        onDetect: (score, clip, pos) => {
          console.info("[wake] 자비서", score.toFixed(3));
          camWake(); // 서버 알림을 기다리지 않고 바로 고해상도로
          link?.wake({ pos, score }); // 읽는 중 · 생각 중이어도 보낸다. 끼어들기는 서버가 정한다
        },
      });
      mic.detect = detect;
      console.info("[wake] 브라우저 마이크 ·", mic.label, "· 에코 제거", mic.aec, "· 호출어", detect);
      return mic;
    })().finally(() => (micOpening = null));
    return micOpening;
  }

  // 서버 알림(api/browser_voice.py). 팀 음성 모드 알림과 같은 모양이라 같은 함수로 그린다
  const onLinkEvent = (e) => {
    if (e.type === "rejected") {
      camRest();
      stopMeter(); // "자비서"가 아니었다(TV 소리 등). 아무 말 없이 다시 기다린다
      setState("idle");
    } else if (e.type === "closed") {
      link = null;
      idleSub("음성 서버와 연결이 끊겼어요. 새로고침해 주세요.");
    } else {
      onVoiceEvent(e);
    }
  };

  async function startBrowserVoice() {
    const off = (why) => {
      console.info("[voice] 브라우저 음성 꺼짐:", why);
      idleSub("음성을 쓸 수 없어요. 아래에 입력해 주세요.");
    };
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return off("https 가 아님");
    let l = null;
    try {
      const { VoiceLink } = await import("/voicelink.js");
      l = new VoiceLink({ onEvent: onLinkEvent, camera: cameraState, grab: grabBlob });
      const r = await l.connect();
      link = l;
      link.setVolume(Number(volume.value) / 100);
      link.setVoice(voicePick.value, Number(rate.value));
      console.info("[voice] 브라우저 음성 켜짐 · 가리킴", r.pointing, "· 기억", r.recall);
    } catch (err) {
      l?.close();
      return off(err.message || err.name);
    }
    if (isPhone) return idleSub("호출 버튼을 누르면 대화를 시작합니다.");
    try {
      await ensureMic();
    } catch (err) {
      console.info("[wake] 마이크 못 엶:", err.message || err.name);
      return idleSub("호출 버튼을 누르면 대화를 시작합니다.");
    }
    if (!mic.detect) return idleSub("호출 버튼이나 스페이스바로 불러 주세요.");
    const ready = () => idleSub("“자비서”라고 부르면 대화를 시작합니다.");
    if (!mic.suspended) return ready();
    idleSub("화면을 한 번 누르면 “자비서”를 듣기 시작해요.");
    const resume = () => mic.resume().then(ready);
    ["pointerdown", "keydown"].forEach((ev) => document.addEventListener(ev, resume, { once: true }));
  }

  /* ------------------------------------------------------------------------
     성능 리포트 — 목업 숫자 대신 팀 기록(/api/report, api/report.py)
     - 단계별 지연: 실제 턴에서 MemoryStore.latency_percentiles (P50 · P95)
     - 평가 실행: eval_run (scripts/run_ablation.py --db)
     - 팀 실측 결과 · 위 카드: docs/experiments.md 의 최종 숫자
     ------------------------------------------------------------------------ */

  let REPORT = null;
  const latRows = $("#lat-rows");

  const setAgg = (agg) => {
    if (!REPORT) return;
    const rows = REPORT.latency.filter((r) => r.stage !== "total");
    const total = REPORT.latency.find((r) => r.stage === "total");
    const vals = rows.map((r) => (r[agg] || 0) / 1000);
    const scaleMax = Math.max(2.5, ...vals);
    const peak = vals.indexOf(Math.max(...vals));
    latRows.replaceChildren();
    rows.forEach((r, i) => {
      const row = document.createElement("div");
      row.className = "lat-row" + (i === peak && vals[i] > 0 ? " is-peak" : "");
      row.innerHTML = `
        <div class="name">${r.name}<span>${r.en} · ${r.count}건</span></div>
        <div class="lat-track"><i style="width:${(vals[i] / scaleMax) * 100}%"></i></div>
        <div class="val num"><span>${r.count ? vals[i].toFixed(2) : "—"}</span><small>s</small></div>`;
      latRows.append(row);
    });
    setText("#lat-total", total && total.count ? ((total[agg] || 0) / 1000).toFixed(2) : "—");
    setText("#lat-total-label", `말 끝 → 대답 첫 소리 · ${agg.toUpperCase()} · ${total ? total.count : 0}건`);
    setText("#lat-goal", "");
    if (vals[peak] > 0) {
      setText("#lat-peak", `${rows[peak].name} · ${rows[peak].en}`);
      setText("#lat-peak-note", `${agg.toUpperCase()} ${vals[peak].toFixed(2)}초로 가장 오래 걸려요.`);
    }
    $$("[data-agg]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.agg === agg)));
  };
  $$("[data-agg]").forEach((b) => b.addEventListener("click", () => setAgg(b.dataset.agg)));

  const cell = (tr, text, cls) => {
    const td = document.createElement("td");
    if (cls) td.className = cls;
    td.textContent = text;
    tr.append(td);
    return td;
  };

  function renderReport(r) {
    REPORT = r;
    setText("#range-label", `턴 기록 ${r.turns}개 · 이 PC 의 DB`);
    setText("#report-turns", r.turns.toLocaleString());

    const kpis = $("#kpis");
    kpis.replaceChildren();
    r.kpis.forEach((k, i) => {
      const div = document.createElement("div");
      div.className = "kpi";
      div.innerHTML = `
        <div class="kpi-head"></div>
        <div class="kpi-val num"></div>
        <div class="kpi-desc"></div>
        <div class="kpi-bar${i === 3 ? " is-lavender" : ""}"><i style="width:${Math.min(k.bar, 100)}%"></i></div>
        <div class="kpi-goal"></div>`;
      $(".kpi-head", div).textContent = k.name;
      const num = document.createElement("span");
      num.textContent = String(i + 1).padStart(2, "0");
      $(".kpi-head", div).append(num);
      $(".kpi-val", div).textContent = k.value;
      const unit = document.createElement("small");
      unit.textContent = k.unit;
      $(".kpi-val", div).append(unit);
      $(".kpi-desc", div).textContent = k.desc;
      $(".kpi-goal", div).textContent = k.goal;
      kpis.append(div);
    });

    setAgg("p50");

    const runRows = $("#run-rows");
    runRows.replaceChildren();
    r.eval_runs.forEach((run) => {
      const tr = document.createElement("tr");
      cell(tr, run.run_at ? run.run_at.slice(0, 10) : "—", "num");
      cell(tr, run.variant);
      cell(tr, String(run.samples), "num");
      cell(tr, Object.entries(run.metrics).map(([k, v]) => `${k} ${typeof v === "number" ? +v.toFixed(3) : v}`).join(" · "));
      runRows.append(tr);
    });
    setText(
      "#run-caption",
      r.eval_runs.length
        ? `평가 실행 ${r.eval_runs.length}번`
        : "이 DB 에는 아직 평가 실행이 없어요. scripts/run_ablation.py 를 이 서버 DB(--db)로 돌리면 쌓여요."
    );

    const evalBody = $("#eval-rows");
    evalBody.replaceChildren();
    r.results.forEach((row) => {
      const tr = document.createElement("tr");
      const model = cell(tr, row.model, "model");
      const small = document.createElement("small");
      small.textContent = row.owner;
      model.append(small);
      cell(tr, row.metric);
      cell(tr, row.now, "num");
      cell(tr, row.goal, "num");
      const st = document.createElement("td");
      st.innerHTML = `<span class="pill ${row.ok ? "is-ok" : "is-warn"}">${row.ok ? "통과" : "개선 중"}</span>`;
      tr.append(st);
      cell(tr, row.source);
      evalBody.append(tr);
    });
  }

  async function drawChart() {
    // 이름은 화면 전환(route)이 부르던 그대로 둔다. 성능 리포트를 열 때마다 새로 읽는다.
    try {
      renderReport(await (await fetch("/api/report")).json());
    } catch (err) {
      setText("#range-label", "리포트를 불러오지 못했어요.");
    }
  }

  /* ------------------------------------------------------------------------
     환경설정
     ------------------------------------------------------------------------ */

  const fillRange = (input) => {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--p", `${p}%`);
  };

  // AI 음성(소리 크기 · 말하기 속도 · 보이스)은 이 브라우저에 저장하고 지금 도는 음성에 바로 쓴다.
  //   브라우저 음성: 소리 크기는 받은 소리에 곱하고(voicelink.js), 목소리 · 속도는 서버 합성이 바꾼다
  //   팀 음성 모드: 셋 다 서버에 보내 PC 스피커 대답에 쓴다(/api/voice/settings)
  const volume = $("#volume");
  const rate = $("#rate");
  const voicePick = $("#voice");
  volume.value = store.get("jarviseo.volume", 70);
  rate.value = store.get("jarviseo.rate", 1);
  voicePick.value = store.get("jarviseo.voice", "coral");
  if (!voicePick.value) voicePick.selectedIndex = 0; // 저장된 값이 목록에 없다
  const sendVoiceSettings = () => {
    if (!voiceMode) return;
    const body = { voice: voicePick.value, speed: Number(rate.value), volume: Number(volume.value) / 100 };
    fetch("/api/voice/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => {});
  };
  const syncVolume = () => {
    fillRange(volume);
    $("#volume-out").textContent = Number(volume.value) === 0 ? "음소거" : `${volume.value}%`;
  };
  const syncRate = () => {
    fillRange(rate);
    $("#rate-out").textContent = `${Number(rate.value).toFixed(1)}배`;
  };
  volume.addEventListener("input", syncVolume);
  rate.addEventListener("input", syncRate);
  volume.addEventListener("change", () => {
    store.set("jarviseo.volume", Number(volume.value));
    link?.setVolume(Number(volume.value) / 100);
    sendVoiceSettings();
  });
  const changeVoice = () => {
    store.set("jarviseo.rate", Number(rate.value));
    store.set("jarviseo.voice", voicePick.value);
    link?.setVoice(voicePick.value, Number(rate.value));
    sendVoiceSettings();
  };
  rate.addEventListener("change", changeVoice);
  voicePick.addEventListener("change", changeVoice);
  syncVolume();
  syncRate();

  const trace = $("#trace");
  const syncTrace = () => {
    $("#trace-label").textContent = trace.checked ? "받기" : "받지 않기";
    $("#trace-warn").hidden = trace.checked;
  };
  // 서버 설정(GET · PUT /api/trace)과 잇는다. 끄면 성분 판정이 함유만 말한다(api/trace.py)
  trace.addEventListener("change", async () => {
    syncTrace();
    try {
      await fetch("/api/trace", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ on: trace.checked }),
      });
    } catch (err) {
      console.warn("[trace] 설정을 저장하지 못했어요", err);
    }
  });
  syncTrace();
  fetch("/api/trace")
    .then((r) => r.json())
    .then((d) => { trace.checked = !!d.on; syncTrace(); })
    .catch(() => {});

  // HUD 애니메이션 · 대상 검출 표시는 이 브라우저에 저장한다
  const prefHud = $("#pref-hud");
  const prefOverlay = $("#pref-overlay");
  const applyPrefs = () => {
    app.classList.toggle("no-motion", !prefHud.checked);
    app.classList.toggle("no-overlay", !prefOverlay.checked);
  };
  prefHud.checked = store.get("jarviseo.hud", true);
  prefOverlay.checked = store.get("jarviseo.overlay", true);
  prefHud.addEventListener("change", () => {
    store.set("jarviseo.hud", prefHud.checked);
    applyPrefs();
  });
  prefOverlay.addEventListener("change", () => {
    store.set("jarviseo.overlay", prefOverlay.checked);
    applyPrefs();
  });
  applyPrefs();

  /* ------------------------------------------------------------------------
     환경설정: 알러지 → 실제 등록(GET · PUT /api/allergens)
     ------------------------------------------------------------------------ */

  const allergenInput = $("#allergens");
  const allergenNote = $("#allergen-saved");
  const loadAllergens = async () => {
    try {
      const list = (await (await fetch("/api/allergens")).json()).allergens;
      allergenInput.value = list.join(", ");
      allergenNote.textContent = list.length ? `등록됨: ${list.join(", ")}` : "아직 등록된 알레르기가 없어요.";
    } catch {
      allergenNote.textContent = "알레르기를 불러오지 못했어요.";
    }
  };
  const saveAllergens = async () => {
    const names = allergenInput.value.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
    try {
      const r = await fetch("/api/allergens", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allergens: names }),
      });
      const list = (await r.json()).allergens;
      allergenNote.textContent = list.length ? `저장했어요: ${list.join(", ")}` : "모두 지웠어요.";
    } catch {
      allergenNote.textContent = "저장하지 못했어요.";
    }
  };
  allergenInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      saveAllergens();
    }
  });
  allergenInput.addEventListener("change", saveAllergens);
  loadAllergens();


  /* ------------------------------------------------------------------------
     시작
     ------------------------------------------------------------------------ */

  setState("idle");
  route();
})();
