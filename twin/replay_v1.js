// HETERO-WORLD replay inside the BoundlessNYC digital twin (one scene: Two Bridges, Manhattan).
// Loaded by twin/server.mjs into the BoundlessNYC page when the URL has ?twin=1. Uses the in-page API
// (window.__API) for the real animated pedestrians and the camera, and the renderer's own three.js
// (window.__THREE, window.__ENGINE) for the delivery robots and the forecast overlays.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (cond) => { while (!cond()) await sleep(300); };

await until(() => window.__API && window.__ENGINE && window.__THREE && (window.__BOOT || []).some((b) => b[1] === 'loading hidden'));
const THREE = window.__THREE, E = window.__ENGINE, API = window.__API;
const call = (m, p = {}) => Promise.resolve(API.call(m, p)).then((r) => (r && r.result !== undefined ? r.result : r));

const D = await (await fetch('twin/data/replay.json')).json();
const S = D.scene;
const O = await call('geo.to_location', { lat: S.lat, lon: S.lon });
const G = S.ground;
const AX = S.axis_en, NR = S.normal_en;
const toApi = (x, y, z = 0) => ({ x: O.x + x * AX[0] + y * NR[0], y: O.y + x * AX[1] + y * NR[1], z: G + z });
const toThree = (x, y, z = 0) => { const a = toApi(x, y, z); return new THREE.Vector3(a.x, a.z, -a.y); };
const yawDeg = (th) => {
  const e = Math.cos(th) * AX[0] + Math.sin(th) * NR[0], n = Math.cos(th) * AX[1] + Math.sin(th) * NR[1];
  return Math.atan2(n, e) * 180 / Math.PI;
};
const NAMES = { union_select: 'Union of rollouts', hw_robust: 'HETERO-WORLD (ours)' };

await call('world.set_ambient_traffic', { vehicles: 0, walkers: 0 });
await call('world.apply_settings', { synchronous_mode: true, fixed_delta_seconds: D.dt });

// ---------------------------------------------------------------- scene layer (robots, overlays)
const layer = new THREE.Group();
E.scene.add(layer);
const std = (color, extra = {}) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.55, metalness: 0.05 }, extra));
const basic = (color, opacity = 1) => new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });

// The measured pinch, marked on the real sidewalk (the simulator narrows both edges to the measured clear width).
(function markPinch() {
  const xs = D.walls.x, ws = D.walls.w, W = D.corridor_half_width;
  for (let i = 0; i < xs.length - 1; i++) {
    const intrude = W - ws[i];
    if (intrude < 0.05) continue;
    const dx = xs[i + 1] - xs[i];
    for (const side of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(dx, 0.02, intrude), basic(0xf2b134, 0.55));
      m.position.copy(toThree(xs[i] + dx / 2, side * (W - intrude / 2), 0.03));
      m.rotation.y = Math.atan2(-AX[1], AX[0]);
      layer.add(m);
    }
  }
})();

function deliveryRobot() {
  const g = new THREE.Group();
  const shell = std(0xf4f6f8), trim = std(0x1d5fd1), dark = std(0x1c1f22, { roughness: 0.8 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.38, 0.5), shell); body.position.y = 0.36; g.add(body);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.08, 0.44), shell); lid.position.y = 0.59; g.add(lid);
  const band = new THREE.Mesh(new THREE.BoxGeometry(0.63, 0.06, 0.51), trim); band.position.y = 0.3; g.add(band);
  for (const sx of [-0.21, 0, 0.21]) for (const sz of [-0.27, 0.27]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.06, 18), dark);
    w.rotation.x = Math.PI / 2; w.position.set(sx, 0.1, sz); g.add(w);
  }
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.05, 0.3), basic(0xfff3c4)); lamp.position.set(0.315, 0.42, 0); g.add(lamp);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.9, 6), dark); mast.position.set(-0.24, 1.02, 0); g.add(mast);
  const flag = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.14, 0.22), basic(0xff6a00)); flag.position.set(-0.24, 1.4, 0.11); g.add(flag);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

function ribbon(points, color, opacity, width) {
  if (points.length < 2) return null;
  const curve = new THREE.CatmullRomCurve3(points);
  return new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 3), width, 6, false), basic(color, opacity));
}
const ring = () => new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.04, 8, 32), basic(0xe5383b));

// ---------------------------------------------------------------- HUD
const css = `
#hw { position: fixed; inset: 0; pointer-events: none; font: 14px/1.45 "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; color: #e8edf1; z-index: 50; }
#hw .card { pointer-events: auto; background: rgba(14, 19, 24, 0.82); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 14px 16px; }
#hw .claim { position: absolute; left: 16px; top: 16px; max-width: 430px; }
#hw h1 { font: 700 17px/1.25 "Archivo", "Helvetica Neue", Arial, sans-serif; margin: 0 0 6px; }
#hw p { margin: 0 0 8px; color: #c4ccd3; }
#hw .ab { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
#hw button { font: 600 13px inherit; font-family: inherit; color: #e8edf1; background: transparent; border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; padding: 7px 12px; cursor: pointer; }
#hw button[aria-pressed="true"] { background: #e8edf1; color: #11161b; border-color: #e8edf1; }
#hw .live { position: absolute; right: 16px; top: 16px; width: 300px; font: 13px/1.5 "IBM Plex Mono", ui-monospace, Menlo, monospace; }
#hw .live b { font-weight: 600; }
#hw .row { display: flex; justify-content: space-between; gap: 12px; }
#hw .bad { color: #ff6b6b; } #hw .good { color: #6fdc97; } #hw .warn { color: #ffc35a; }
#hw .legend { position: absolute; left: 16px; bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; }
#hw .sw { display: inline-block; width: 22px; height: 6px; border-radius: 3px; margin-right: 6px; vertical-align: 2px; }
#hw .banner { position: absolute; left: 50%; bottom: 70px; transform: translateX(-50%); font: 600 16px "Archivo", Arial, sans-serif; display: none; }
#hw .cam { position: absolute; right: 16px; bottom: 16px; display: flex; gap: 8px; }`;
document.head.insertAdjacentHTML('beforeend', `<style>${css}</style>`);
document.body.insertAdjacentHTML('beforeend', `
<div id="hw">
  <div class="card claim">
    <h1>Whose forecast of the pedestrians should the robots trust?</h1>
    <p>Delivery robots carry different world models (diffusion, JEPA, RSSM) and share their predicted futures. Pedestrians share nothing.
    The sidewalk at Two Bridges narrows to <b>1.0 m</b> at a street-tree pit (measured from NYC Open Data).</p>
    <p><b>Union</b> picks the plan whose <i>own</i> pedestrian forecast looks safest, so it believes the most convenient forecast.
    <b>HETERO-WORLD</b> scores every plan against the robots' <i>consensus</i> forecast (a Wasserstein barycenter).</p>
    <div class="ab" id="ab"></div>
  </div>
  <div class="card live" id="live"></div>
  <div class="card legend">
    <span><span class="sw" style="background:#3d8bff"></span>robot's planned path</span>
    <span><span class="sw" style="background:#ff8a3d"></span>pedestrian forecast the plan was scored against</span>
    <span><span class="sw" style="background:#f2b134"></span>measured pinch</span>
    <span><span class="sw" style="background:#e5383b"></span>contact</span>
  </div>
  <div class="card banner" id="banner"></div>
  <div class="cam" id="cam"></div>
</div>`);
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- actors
const state = { method: 'union_select', playing: true, token: 0, view: 'street' };
let walkers = [], robots = [], overlays = [], rings = [];
const bps = (await call('world.get_blueprints')).map((b) => b.id).filter((id) => /^walker\.pedestrian\.\d+$/.test(id));

async function clearActors() {
  for (const w of walkers) await call('world.destroy_actor', { id: w.id }).catch(() => {});
  walkers = [];
  robots.forEach((r) => layer.remove(r)); robots = [];
  overlays.forEach((o) => { layer.remove(o); o.geometry.dispose(); }); overlays = [];
  rings.forEach((r) => layer.remove(r)); rings = [];
}

async function spawn(ep) {
  const f0 = ep.frames[0];
  let k = 0;
  for (let i = 0; i < ep.is_human.length; i++) {
    const [x, y] = f0.p[i];
    if (ep.is_human[i]) {
      const bp = bps[(i * 7 + 3) % bps.length];
      const a = await call('world.spawn_actor', { blueprint: bp, transform: { location: toApi(x, y, 0.1), rotation: { pitch: 0, yaw: yawDeg(f0.h[i]), roll: 0 } } });
      walkers.push({ id: a.id, i });
    } else {
      const r = deliveryRobot(); r.userData.i = i; layer.add(r); robots.push(r);
    }
    const rg = ring(); rg.rotation.x = Math.PI / 2; rg.visible = false; rg.userData.i = i; layer.add(rg); rings.push(rg);
    k++;
  }
}

function placeFrame(ep, t) {
  const f = ep.frames[t], prev = ep.frames[Math.max(0, t - 1)];
  const actors = API.actors;
  for (const w of walkers) {
    const a = actors.get(w.id);
    if (!a) continue;
    const [x, y] = f.p[w.i];
    const gone = f.d[w.i] === 1 || Math.abs(x) > 40;
    const tgt = toApi(gone ? 999 : x, y, 0);
    const vx = (f.p[w.i][0] - prev.p[w.i][0]) / D.dt, vy = (f.p[w.i][1] - prev.p[w.i][1]) / D.dt;
    const ve = vx * AX[0] + vy * NR[0], vn = vx * AX[1] + vy * NR[1];
    const sp = gone ? 0 : Math.hypot(ve, vn);
    // placeWalker integrates position from control, so start one step behind the target to land on it.
    a.st.x = tgt.x - ve * D.dt; a.st.z = -(tgt.y - vn * D.dt);
    a.control = { direction: sp > 0.02 ? { x: ve, y: vn, z: 0 } : { x: 0, y: 0, z: 0 }, speed: sp, jump: false };
    if (sp <= 0.02) a.st.yaw = yawDeg(f.h[w.i]);
  }
  for (const r of robots) {
    const i = r.userData.i, [x, y] = f.p[i];
    r.visible = !(f.d[i] === 1 || Math.abs(x) > 40);
    r.position.copy(toThree(x, y, 0));
    r.rotation.y = yawDeg(f.h[i]) * Math.PI / 180;
  }
  // contact rings: any two agents closer than two radii
  const r2 = Math.pow(2 * D.agent_radius, 2);
  rings.forEach((rg) => (rg.visible = false));
  let contact = false;
  for (let i = 0; i < f.p.length; i++) for (let j = i + 1; j < f.p.length; j++) {
    if (f.d[i] === 1 || f.d[j] === 1) continue;
    const dx = f.p[i][0] - f.p[j][0], dy = f.p[i][1] - f.p[j][1];
    if (dx * dx + dy * dy < r2 && (ep.is_human[i] !== ep.is_human[j])) {
      for (const k of [i, j]) { const rg = rings[k]; rg.visible = true; rg.position.copy(toThree(f.p[k][0], f.p[k][1], 0.06)); }
      contact = true;
    }
  }
  return contact;
}

// Overlays for the plan made at replan time t: robots' planned paths and the pedestrian forecast it was scored against.
function drawPlan(ep, plan) {
  overlays.forEach((o) => { layer.remove(o); o.geometry.dispose(); }); overlays = [];
  const f = ep.frames[plan.t] || ep.frames[ep.frames.length - 1];
  for (let i = 0; i < ep.is_human.length; i++) {
    if (f.d[i] === 1 || Math.abs(f.p[i][0]) > 40) continue;
    if (!ep.is_human[i]) {
      const pts = [toThree(f.p[i][0], f.p[i][1], 0.25), ...plan.xy[i].map(([x, y]) => toThree(x, y, 0.25))];
      const m = ribbon(pts, 0x3d8bff, 0.9, 0.05); if (m) { layer.add(m); overlays.push(m); }
    } else {
      const atoms = plan.forecast;
      const op = atoms.length > 1 ? 0.22 : 0.85, wd = atoms.length > 1 ? 0.025 : 0.05;
      for (const fc of atoms) {
        const pts = [toThree(f.p[i][0], f.p[i][1], 0.05), ...fc[i].map(([x, y]) => toThree(x, y, 0.05))];
        const m = ribbon(pts, 0xff8a3d, op, wd); if (m) { layer.add(m); overlays.push(m); }
      }
    }
  }
}

// Forecast vs actual clearance between the robots' plan and the pedestrians over the plan horizon (2.5 s here).
function clearance(ep, plan) {
  const H = 10, step = Math.round(0.25 / D.dt);
  let pred = Infinity, act = Infinity;
  for (let i = 0; i < ep.is_human.length; i++) {
    if (ep.is_human[i]) continue;
    for (let j = 0; j < ep.is_human.length; j++) {
      if (!ep.is_human[j]) continue;
      for (let h = 0; h < Math.min(H, plan.xy[i].length); h++) {
        const [rx, ry] = plan.xy[i][h];
        let best = Infinity;
        for (const fc of plan.forecast) best = Math.min(best, Math.hypot(rx - fc[j][h][0], ry - fc[j][h][1]));
        const meanAtom = plan.forecast.length > 1
          ? plan.forecast.reduce((s, fc) => s + Math.hypot(rx - fc[j][h][0], ry - fc[j][h][1]), 0) / plan.forecast.length : best;
        pred = Math.min(pred, meanAtom);
        const fr = ep.frames[Math.min(ep.frames.length - 1, plan.t + (h + 1) * step)];
        if (fr.d[i] !== 1 && fr.d[j] !== 1) act = Math.min(act, Math.hypot(fr.p[i][0] - fr.p[j][0], fr.p[i][1] - fr.p[j][1]));
      }
    }
  }
  return { pred, act };
}

function renderLive(ep, t, cl, contacts) {
  const fmt = (v) => (isFinite(v) ? `${v.toFixed(2)} m` : 'n/a');
  const gap = isFinite(cl.pred) && isFinite(cl.act) ? cl.pred - cl.act : 0;
  $('live').innerHTML = `
    <div class="row"><span>planner</span><b>${NAMES[state.method]}</b></div>
    <div class="row"><span>time</span><b>${(t * D.dt).toFixed(1)} s</b></div>
    <div class="row"><span>forecast clearance</span><b>${fmt(cl.pred)}</b></div>
    <div class="row"><span>actual clearance</span><b class="${cl.act < 0.6 ? 'bad' : 'good'}">${fmt(cl.act)}</b></div>
    <div class="row"><span>over-optimism</span><b class="${gap > 0.4 ? 'warn' : ''}">${isFinite(gap) ? (gap >= 0 ? '+' : '') + gap.toFixed(2) + ' m' : 'n/a'}</b></div>
    <div class="row"><span>robot-pedestrian contacts</span><b class="${contacts ? 'bad' : 'good'}">${contacts}</b></div>
    <div class="row" style="margin-top:6px;color:#9aa6b0"><span>at this sidewalk, ${D.stats.n} seeds</span></div>
    <div class="row" style="color:#9aa6b0"><span>Union contact rate</span><span>${D.stats.union}</span></div>
    <div class="row" style="color:#9aa6b0"><span>HETERO-WORLD contact rate</span><span>${D.stats.hw}</span></div>`;
}

function setCamera(view) {
  state.view = view;
  const yaw = Math.atan2(AX[1], AX[0]) * 180 / Math.PI;
  const T = view === 'overhead'
    ? { location: toApi(0, -0.5, 17), rotation: { pitch: -72, yaw, roll: 0 } }
    : view === 'low'
      ? { location: toApi(-7, -2.2, 1.5), rotation: { pitch: -6, yaw: yaw + 8, roll: 0 } }
      : { location: toApi(-11, -3.0, 3.4), rotation: { pitch: -12, yaw: yaw + 6, roll: 0 } };
  call('actor.set_transform', { id: 1, transform: T });
  [...$('cam').children].forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === view)));
}

function buttons() {
  $('ab').innerHTML = '';
  for (const m of ['union_select', 'hw_robust']) {
    const b = document.createElement('button');
    b.textContent = (m === 'union_select' ? 'A · ' : 'B · ') + NAMES[m];
    b.setAttribute('aria-pressed', String(state.method === m));
    b.onclick = () => { state.method = m; buttons(); run(); };
    $('ab').appendChild(b);
  }
  const p = document.createElement('button');
  p.textContent = state.playing ? 'Pause' : 'Play';
  p.onclick = () => { state.playing = !state.playing; buttons(); };
  $('ab').appendChild(p);
  const r = document.createElement('button');
  r.textContent = 'Restart'; r.onclick = () => run();
  $('ab').appendChild(r);
}
$('cam').innerHTML = ['street', 'low', 'overhead'].map((v) => `<button data-v="${v}">${v[0].toUpperCase() + v.slice(1)} view</button>`).join('');
[...$('cam').children].forEach((b) => (b.onclick = () => setCamera(b.dataset.v)));

async function run() {
  const token = ++state.token;
  const ep = D.episodes[state.method];
  $('banner').style.display = 'none';
  await clearActors();
  await spawn(ep);
  setCamera(state.view);
  let contacts = 0, inContact = false, planIdx = -1, cl = { pred: Infinity, act: Infinity };
  for (let t = 0; t < ep.frames.length; t++) {
    while (!state.playing) { if (token !== state.token) return; await sleep(100); }
    if (token !== state.token) return;
    while (planIdx + 1 < ep.plans.length && ep.plans[planIdx + 1].t <= t) {
      planIdx++;
      drawPlan(ep, ep.plans[planIdx]);
      cl = clearance(ep, ep.plans[planIdx]);
    }
    const c = placeFrame(ep, t);
    if (c && !inContact) contacts++;
    inContact = c;
    renderLive(ep, t, cl, contacts);
    const t0 = performance.now();
    await call('world.tick', { dt: D.dt });
    await sleep(Math.max(0, D.dt * 1000 / (window.__HW_SPEED || 1) - (performance.now() - t0)));
  }
  const ok = contacts === 0;
  $('banner').innerHTML = ok
    ? `<span class="good">${NAMES[state.method]}: every robot passed the pinch with no pedestrian contact.</span>`
    : `<span class="bad">${NAMES[state.method]}: ${contacts} robot–pedestrian contact${contacts > 1 ? 's' : ''}. The plan trusted a forecast the pedestrian did not follow.</span>`;
  $('banner').style.display = 'block';
  window.__HW_DONE = { method: state.method, contacts };
}

buttons();
window.__HW = { run, setCamera, state };
run();
