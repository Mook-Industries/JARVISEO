/* ==========================================================================
   JARVISEO 대시보드 동작
   - 화면 전환(#monitor / #admin / #settings)
   - 모니터 시연 상태(대기 → 호출어 감지 → 처리 중 → 응답)
   - 성능 리포트 그래프, 환경설정 스위치
   지금은 전부 시연용 예시 데이터다. 백엔드가 붙으면 아래 SAMPLE_* 자리를
   FastAPI 응답(AssistantResponse.latency_ms 등)으로 바꾼다.
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

  const SAMPLE_LOG = [
    {
      id: "c1",
      time: "14:26",
      tag: "가리킴 + 기억",
      q: "저거 뭐야?",
      a: "검정색 백팩입니다. 이전에 보셨던 모델과 비슷해요.",
    },
    {
      id: "c2",
      time: "14:19",
      tag: "사물 인식",
      q: "이 케이블 무슨 단자야?",
      a: "USB-C 단자로 보입니다. 양쪽 방향으로 꽂을 수 있어요.",
    },
    {
      id: "c3",
      time: "13:52",
      tag: "기억 검색",
      q: "아까 본 카페 이름이 뭐였지?",
      a: "13시 40분쯤 지나친 ‘카페 온기’로 기록돼 있어요.",
    },
  ];

  // 파이프라인 단계. 순서가 곧 처리 순서다.
  const STAGES = [
    { name: "음성 인식", code: "STT", sec: 0.8 },
    { name: "대상 검출", code: "DETECT", sec: 0.1 },
    { name: "문자 판독", code: "OCR", sec: 0.4 },
    { name: "장면 이해", code: "VLM", sec: 1.4 },
    { name: "음성 합성", code: "TTS", sec: 0.3 },
  ];
  const STAGE_TOTAL = STAGES.reduce((s, st) => s + st.sec, 0);

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
      stages: ["done", "done", "done", "done", "done"],
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

  const DEMO_ANSWER = {
    tag: "가리킴",
    q: "저거 뭐야?",
    a: "청록색 스낵 봉지예요. 성분표를 비춰 주시면 알레르기 성분도 확인해 드릴게요.",
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
  sbToggle.addEventListener("click", () => setCollapsed(!app.classList.contains("is-collapsed")));

  // 좁은 화면에서는 사이드바가 서랍처럼 열린다
  $("#menu-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    app.classList.toggle("is-drawer-open");
  });

  const profileBtn = $("#profile-btn");
  const profilePop = $("#profile-pop");
  const setPop = (open) => {
    profilePop.hidden = !open;
    profileBtn.setAttribute("aria-expanded", String(open));
  };
  profileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    setPop(profilePop.hidden);
  });

  document.addEventListener("click", (e) => {
    if (!profilePop.hidden && !profilePop.contains(e.target)) setPop(false);
    if (app.classList.contains("is-drawer-open") && !$("#sidebar").contains(e.target)) {
      app.classList.remove("is-drawer-open");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setPop(false);
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
    setPop(false);
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
  for (let i = 0; i < 64; i++) {
    const bar = document.createElement("i");
    const h = i % 7 === 3 ? 6 : i % 4 === 1 ? 4 : 2;
    bar.style.height = `${h}px`;
    bar.style.animationDelay = `${((i * 37) % 90) / 100}s`;
    wave.append(bar);
  }

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
      const on = status !== "off";
      // 아주 짧은 단계도 눈에 보이도록 최소 폭을 준다
      const w = on ? Math.max(st.sec / STAGE_TOTAL, 0.06) * 100 : 0;
      el.classList.toggle("is-running", status === "run");
      $(".stage-bar i", el).style.width = `${w.toFixed(1)}%`;
      $(".stage-val span", el).textContent = on ? `${st.sec.toFixed(1)}s` : "—";
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
  let demoLogged = false;

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
     모니터: 시연 재생
     ------------------------------------------------------------------------ */

  let timers = [];
  let playing = false;

  const later = (ms, fn) => timers.push(setTimeout(fn, ms));

  function stopDemo() {
    timers.forEach(clearTimeout);
    timers = [];
    playing = false;
  }

  const playDemo = () => {
    stopDemo();
    playing = true;

    // 실제 단계 시간(초)을 그대로 쓰면 너무 빨라서 1.5배로 늘려 보여준다
    const slow = 1500;
    let t = 0;

    setState("wake", { stages: ["run", "off", "off", "off", "off"] });
    t += STAGES[0].sec * slow + 600;

    later(t, () => setState("processing", { stages: ["done", "run", "off", "off", "off"] }));
    t += STAGES[1].sec * slow + 300;
    later(t, () => setStages(["done", "done", "run", "off", "off"]));
    t += STAGES[2].sec * slow;
    later(t, () => setStages(["done", "done", "done", "run", "off"]));
    t += STAGES[3].sec * slow;

    later(t, () => setState("response"));
    t += 5000;
    later(t, () => {
      setState("idle");
      stopDemo();
    });
  };

  $("#call-btn").addEventListener("click", () => {
    if (!playing) playDemo();
  });

  // 스페이스바 = 자비서 호출 (입력칸·버튼에 초점이 있을 때는 건드리지 않는다)
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat) return;
    if ($("#view-monitor").hidden) return;
    if (e.target.closest("input, textarea, select, button, a, [contenteditable]")) return;
    e.preventDefault();
    if (!playing) playDemo();
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

    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
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
    video.srcObject = camStream;
    feed.dataset.cam = "on";

    const s = track.getSettings();
    $("#feed-spec").innerHTML =
      `<span>${s.width || "—"} × ${s.height || "—"}</span><span>${s.frameRate ? Math.round(s.frameRate) : "—"} FPS</span>`;
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

  if (camSupported) {
    camToggle.hidden = false;
    setNoSignal("카메라가 꺼져 있습니다.");
    openCamModal();
  } else {
    setNoSignal("이 환경에서는 카메라를 사용할 수 없습니다.", { canRetry: false });
  }

  /* ------------------------------------------------------------------------
     관리자: 단계별 지연
     ------------------------------------------------------------------------ */

  const LATENCY = {
    names: [
      ["음성 인식", "STT"],
      ["대상 검출", "YOLO"],
      ["문자 판독", "OCR"],
      ["장면 이해", "VLM"],
      ["음성 합성", "TTS"],
    ],
    avg: [0.8, 0.1, 0.4, 1.4, 0.3],
    p95: [1.12, 0.16, 0.63, 2.21, 0.44],
    scaleMax: 2.5, // 막대 전체 폭 = 2.5초
  };

  const latRows = $("#lat-rows");
  LATENCY.names.forEach(([ko, en]) => {
    const row = document.createElement("div");
    row.className = "lat-row";
    row.innerHTML = `
      <div class="name">${ko}<span>${en}</span></div>
      <div class="lat-track"><i></i></div>
      <div class="val num"><span></span><small>s</small></div>`;
    latRows.append(row);
  });

  const setAgg = (agg) => {
    const vals = LATENCY[agg];
    const peak = vals.indexOf(Math.max(...vals));
    $$(".lat-row", latRows).forEach((row, i) => {
      row.classList.toggle("is-peak", i === peak);
      $(".lat-track i", row).style.width = `${Math.min(vals[i] / LATENCY.scaleMax, 1) * 100}%`;
      $(".val span", row).textContent = vals[i].toFixed(2);
    });
    const total = vals.reduce((a, b) => a + b, 0);
    setText("#lat-total", total.toFixed(2));
    setText("#lat-total-label", agg === "avg" ? "평균 응답 시간" : "단계별 P95 합계");
    setText("#lat-goal", agg === "avg" ? "예시 목표 ≤ 3.0초" : "예시 목표 ≤ 4.5초");
    $$("[data-agg]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.agg === agg)));
  };
  $$("[data-agg]").forEach((b) => b.addEventListener("click", () => setAgg(b.dataset.agg)));
  setAgg("avg");

  /* ------------------------------------------------------------------------
     관리자: 인식 성능 추이 (SVG 선 그래프)
     ------------------------------------------------------------------------ */

  const END_DATE = new Date(2026, 8, 22); // 2026-09-22
  const LAST7 = {
    pointing: [91.2, 92.4, 91.9, 93.5, 93.0, 93.9, 94.2],
    allergy: [95.9, 96.4, 96.0, 97.2, 97.1, 97.5, 97.8],
  };

  // 7일보다 앞선 날은 흐름이 이어지도록 만든 예시 값이다
  const series = (days) => {
    const pad = days - 7;
    const pointing = [];
    const allergy = [];
    for (let k = 0; k < pad; k++) {
      const r = (23 - pad + k) / 23; // 30일 전체에서의 위치 (0 ~ 1)
      pointing.push(+(88.6 + r * 2.4 + 0.55 * Math.sin(k * 1.9)).toFixed(1));
      allergy.push(+(94.2 + r * 1.5 + 0.4 * Math.sin(k * 1.3 + 1)).toFixed(1));
    }
    const dates = [];
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(END_DATE);
      d.setDate(d.getDate() - i);
      dates.push(d);
    }
    return {
      dates,
      pointing: pointing.concat(LAST7.pointing),
      allergy: allergy.concat(LAST7.allergy),
    };
  };

  const fmtDate = (d) => `${pad2(d.getMonth() + 1)}.${pad2(d.getDate())}`;
  const chart = $("#trend-chart");
  const tip = $("#chart-tip");
  const G = { x0: 62, x1: 512, y0: 16, y1: 196, min: 85, max: 100 };
  const SVGNS = "http://www.w3.org/2000/svg";
  let current = null;

  const el = (tag, attrs, parent) => {
    const node = document.createElementNS(SVGNS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.append(node);
    return node;
  };

  function drawChart() {
    const days = Number($("#period").value);
    const data = series(days);
    current = data;
    const n = data.dates.length;
    const x = (i) => G.x0 + (i * (G.x1 - G.x0)) / (n - 1);
    const y = (v) => G.y1 - ((v - G.min) / (G.max - G.min)) * (G.y1 - G.y0);

    chart.replaceChildren();

    [85, 90, 95, 100].forEach((v) => {
      el("line", { class: "grid-line", x1: G.x0, x2: G.x1 + 18, y1: y(v), y2: y(v) }, chart);
      const t = el("text", { class: "axis-text num", x: G.x0 - 22, y: y(v) + 4, "text-anchor": "end" }, chart);
      t.textContent = v;
    });

    const labelIdx = n <= 7 ? [0, 3, 6] : [0, Math.round((n - 1) / 3), Math.round(((n - 1) * 2) / 3), n - 1];
    labelIdx.forEach((i) => {
      const t = el("text", { class: "axis-text num", x: x(i), y: G.y1 + 30, "text-anchor": "middle" }, chart);
      t.textContent = fmtDate(data.dates[i]);
    });

    const cross = el("line", { class: "cross", y1: G.y0 - 6, y2: G.y1, x1: -10, x2: -10 }, chart);

    const lines = [
      ["allergy", "var(--lavender)", "#a3a1f0"],
      ["pointing", "var(--accent)", "#5fd9cb"],
    ];
    lines.forEach(([key, , hex]) => {
      const pts = data[key].map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      el("polyline", { class: "series", points: pts, stroke: hex }, chart);
      // 30일 보기에서는 점이 너무 빽빽해서 마지막 점만 강조한다
      data[key].forEach((v, i) => {
        if (n > 14 && i !== n - 1) return;
        el("circle", { cx: x(i), cy: y(v), r: 4, fill: hex, stroke: "#0d191e", "stroke-width": 2 }, chart);
      });
    });

    const hit = el("rect", { x: G.x0 - 10, y: 0, width: G.x1 - G.x0 + 20, height: G.y1 + 10, fill: "transparent" }, chart);

    const show = (evt) => {
      const box = chart.getBoundingClientRect();
      const sx = ((evt.clientX - box.left) / box.width) * 600;
      const i = Math.max(0, Math.min(n - 1, Math.round(((sx - G.x0) / (G.x1 - G.x0)) * (n - 1))));
      cross.setAttribute("x1", x(i));
      cross.setAttribute("x2", x(i));
      tip.innerHTML = `<b>${fmtDate(current.dates[i])}</b>
        <span><i class="is-c-accent"></i>지시 대상 특정 <strong class="num">${current.pointing[i].toFixed(1)}%</strong></span>
        <span><i class="is-c-lavender"></i>알레르기 인식 <strong class="num">${current.allergy[i].toFixed(1)}%</strong></span>`;
      tip.hidden = false;
      const px = (x(i) / 600) * box.width;
      const left = px + 14 + tip.offsetWidth > box.width ? px - tip.offsetWidth - 14 : px + 14;
      tip.style.left = `${left}px`;
    };
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerdown", show);
    hit.addEventListener("pointerleave", () => {
      tip.hidden = true;
      cross.setAttribute("x1", -10);
      cross.setAttribute("x2", -10);
    });

    const first = data.dates[0];
    const last = data.dates[n - 1];
    setText("#range-label", `${first.getFullYear()}.${fmtDate(first)} — ${fmtDate(last)}`);
    setText("#chart-caption", `일별 평가 결과 · ${n}일`);
  }

  $("#period").addEventListener("change", drawChart);

  /* ------------------------------------------------------------------------
     관리자: 모델별 평가 상세
     ------------------------------------------------------------------------ */

  const EVAL_ROWS = [
    ["손끝 검출", "YOLOv8n", "mAP@0.5", "0.91", "≥ 0.90", true, "역광 장면 샘플 보강"],
    ["성분표 영역 검출", "YOLOv8n", "mAP@0.5:0.95", "0.864", "≥ 0.85", true, "곡면 포장 촬영 각도 추가"],
    ["성분표 글자 읽기", "EasyOCR", "글자 인식률", "92.1%", "≥ 95%", false, "작은 글씨 확대 전처리"],
    ["알레르기 판정", "동의어 사전 + 규칙", "Recall", "97.8%", "≥ 99%", false, "미탐지 2건 동의어 추가"],
    ["웨이크워드 “자비서”", "커스텀 오디오 모델", "FRR · FAR", "4.1% · 0.3회/시간", "≤ 5% · ≤ 0.5회", true, "소음 환경 녹음 추가"],
    ["지시 대상 특정", "다중 단서 융합", "정확도", "94.2%", "≥ 95%", false, "시선 단서 가중치 조정"],
  ];

  const evalBody = $("#eval-rows");
  EVAL_ROWS.forEach(([model, engine, metric, now, goal, ok, next]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="model"></td>
      <td></td>
      <td class="num"></td>
      <td class="num"></td>
      <td><span class="pill ${ok ? "is-ok" : "is-warn"}">${ok ? "달성" : "점검"}</span></td>
      <td></td>`;
    const tds = $$("td", tr);
    tds[0].textContent = model;
    const small = document.createElement("small");
    small.textContent = engine;
    tds[0].append(small);
    tds[1].textContent = metric;
    tds[2].textContent = now;
    tds[3].textContent = goal;
    tds[5].textContent = next;
    evalBody.append(tr);
  });

  /* ------------------------------------------------------------------------
     환경설정
     ------------------------------------------------------------------------ */

  const fillRange = (input) => {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--p", `${p}%`);
  };

  const volume = $("#volume");
  const rate = $("#rate");
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
  syncVolume();
  syncRate();

  const trace = $("#trace");
  const syncTrace = () => {
    $("#trace-label").textContent = trace.checked ? "받기" : "받지 않기";
    $("#trace-warn").hidden = trace.checked;
  };
  trace.addEventListener("change", syncTrace);
  syncTrace();

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
     시작
     ------------------------------------------------------------------------ */

  setState("idle");
  route();
})();
