// HETERO-WORLD replay inside the BoundlessNYC digital twin (one scene: Two Bridges, Manhattan).
// Loaded by twin/server.mjs into the BoundlessNYC page when the URL has ?twin=1. The city keeps running live (ambient
// cars and walkers from BoundlessNYC's own simulation). Every person is drawn as an NVIDIA SOMA-X body and every robot
// as a Unitree G1 humanoid, both baked by twin/build_assets.py. The experiment's own agents (the recorded episode) are
// driven from twin/data/replay.json; the ambient walkers are BoundlessNYC's, re-skinned, and kept out of the
// experiment's sidewalk strip so they never touch the recorded agents.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (cond) => { while (!cond()) await sleep(300); };

await until(() => window.__API && window.__ENGINE && window.__THREE && (window.__BOOT || []).some((b) => b[1] === 'loading hidden'));
const THREE = window.__THREE, E = window.__ENGINE, API = window.__API;
const call = (m, p = {}) => Promise.resolve(API.call(m, p)).then((r) => (r && r.result !== undefined ? r.result : r));
const Q = new URLSearchParams(location.search);

const D = await (await fetch('twin/data/replay.json')).json();
const S = D.scene;
const O = await call('geo.to_location', { lat: S.lat, lon: S.lon });
const G = S.ground;
const AX = S.axis_en, NR = S.normal_en;
const toApi = (x, y, z = 0) => ({ x: O.x + x * AX[0] + y * NR[0], y: O.y + x * AX[1] + y * NR[1], z: G + z });
const toThree = (x, y, z = 0) => { const a = toApi(x, y, z); return new THREE.Vector3(a.x, a.z, -a.y); };
const toLocal = (tx, tz) => { const dx = tx - O.x, dy = -tz - O.y; return [dx * AX[0] + dy * AX[1], dx * NR[0] + dy * NR[1]]; };
const yawRad = (th) => {
  const e = Math.cos(th) * AX[0] + Math.sin(th) * NR[0], n = Math.cos(th) * AX[1] + Math.sin(th) * NR[1];
  return Math.atan2(n, e);
};
const NAMES = { union_select: 'Union of rollouts', hw_robust: 'HETERO-WORLD (ours)' };

// Live city: real-time clock, ambient traffic and walkers (BoundlessNYC's simulation; not part of the experiment).
const N_CARS = +(Q.get('cars') || 260), N_WALKERS = +(Q.get('walkers') || 520);
// BoundlessNYC's free-run mode steps at most 0.05 s per rendered frame, so at a low frame rate the city crawls. Instead the
// world is stepped here, once per animation frame, by the real elapsed time (capped at 0.25 s per step).
await call('world.apply_settings', { synchronous_mode: true, idle_nap_ms: 0 });
const CITY_MAX_STEP = 0.25;
await call('world.set_ambient_traffic', { vehicles: N_CARS, walkers: N_WALKERS });

// ---------------------------------------------------------------- baked actors
async function loadBaked(name) {
  const [meta, bin] = await Promise.all([fetch(`twin/assets/${name}.json`).then((r) => r.json()), fetch(`twin/assets/${name}.bin`).then((r) => r.arrayBuffer())]);
  const view = (sl, T) => new T(bin, sl[0], sl[1] / T.BYTES_PER_ELEMENT);
  return { meta, view };
}

// SOMA bodies: per body, one position + normal attribute per frame (frame 0 = standing, 1..F = walk cycle).
const SOMA = await (async () => {
  const { meta, view } = await loadBaked('soma');
  const index = new THREE.BufferAttribute(view(meta.faces, Uint32Array), 1);
  const bodies = meta.bodies.map((b) => {
    const q = view(b.positions, Int16Array), n = meta.vertex_count * 3;
    // baked colours are sRGB; vertex colours are read as linear, so convert once here
    const c8 = view(b.colors, Uint8Array), cl = new Float32Array(c8.length);
    for (let i = 0; i < c8.length; i++) { const v = c8[i] / 255; cl[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }
    const col = new THREE.BufferAttribute(cl, 3);
    const frames = [];
    for (let k = 0; k <= meta.frames; k++) {
      const pos = new Float32Array(n);
      for (let i = 0; i < n; i++) pos[i] = q[k * n + i] * meta.position_scale;
      const g = new THREE.BufferGeometry();
      g.setIndex(index); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.computeVertexNormals();
      frames.push({ position: g.attributes.position, normal: g.attributes.normal });
    }
    return { frames, color: col, stride: b.stride, height: b.height };
  });
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.0 });
  return { meta, index, bodies, material };
})();

// Unitree G1: shared link geometries, one matrix per visual geom per frame.
const G1 = await (async () => {
  const { meta, view } = await loadBaked('g1');
  const geos = {};
  for (const [id, m] of Object.entries(meta.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setIndex(new THREE.BufferAttribute(view(m.faces, Uint32Array), 1));
    g.setAttribute('position', new THREE.BufferAttribute(view(m.positions, Float32Array), 3));
    g.computeVertexNormals();
    geos[id] = g;
  }
  const mats = new Map();
  const mat = (rgba) => {
    const key = rgba.join(',');
    if (!mats.has(key)) mats.set(key, new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: rgba[0] < 0.4 ? 0.55 : 0.35, metalness: rgba[0] < 0.4 ? 0.2 : 0.45 }));
    return mats.get(key);
  };
  return { meta, geos, mat, matrices: view(meta.matrices, Float32Array) };
})();

const BSPHERE = new THREE.Sphere(new THREE.Vector3(0, 0.9, 0), 1.3);

class Walker {
  // A SOMA person. `advance(dist, moving)` moves the walk phase by the distance walked (negative = backwards).
  constructor(bodyIdx, scale = 1) {
    this.body = SOMA.bodies[bodyIdx % SOMA.bodies.length];
    const g = new THREE.BufferGeometry();
    g.setIndex(SOMA.index);
    g.setAttribute('color', this.body.color);
    g.boundingSphere = BSPHERE;
    this.geo = g;
    this.obj = new THREE.Mesh(g, SOMA.material);
    this.obj.scale.setScalar(scale);
    this.obj.castShadow = true; this.obj.receiveShadow = false;
    this.phase = Math.random(); this.k = -1;
    this.setFrame(0);
  }
  setFrame(k) {
    if (k === this.k) return;
    this.k = k;
    const f = this.body.frames[k];
    this.geo.setAttribute('position', f.position);
    this.geo.setAttribute('normal', f.normal);
  }
  advance(dist, moving) {
    if (!moving) return this.setFrame(0);
    this.phase = (((this.phase + dist / (this.body.stride * this.obj.scale.x)) % 1) + 1) % 1;
    this.setFrame(1 + (Math.floor(this.phase * SOMA.meta.frames) % SOMA.meta.frames));
  }
}

class G1Robot {
  constructor() {
    this.obj = new THREE.Group();
    this.parts = G1.meta.geoms.map((g) => {
      const m = new THREE.Mesh(G1.geos[g.mesh], G1.mat(g.rgba));
      m.matrixAutoUpdate = false; m.castShadow = true;
      this.obj.add(m);
      return m;
    });
    this.phase = Math.random(); this.k = -1;
    this.setFrame(0);
  }
  setFrame(k) {
    if (k === this.k) return;
    this.k = k;
    const n = this.parts.length;
    this.parts.forEach((m, j) => { m.matrix.fromArray(G1.matrices, (k * n + j) * 16); m.matrixWorldNeedsUpdate = true; });
  }
  advance(dist, moving) {
    if (!moving) return this.setFrame(0);
    this.phase = (((this.phase + dist / G1.meta.stride) % 1) + 1) % 1;
    this.setFrame(1 + (Math.floor(this.phase * G1.meta.frames) % G1.meta.frames));
  }
}

// ---------------------------------------------------------------- scene layer (actors, overlays)
const layer = new THREE.Group();
E.scene.add(layer);
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

function ribbon(points, color, opacity, width) {
  if (points.length < 2) return null;
  const curve = new THREE.CatmullRomCurve3(points);
  return new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 3), width, 6, false), basic(color, opacity));
}
const ring = () => new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.04, 8, 32), basic(0xe5383b));
const disc = (color) => { const m = new THREE.Mesh(new THREE.CircleGeometry(0.34, 28), basic(color, 0.45)); m.rotation.x = -Math.PI / 2; return m; };

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
#hw .legend { position: absolute; left: 16px; bottom: 16px; display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; max-width: calc(100vw - 420px); }
#hw .sw { display: inline-block; width: 22px; height: 6px; border-radius: 3px; margin-right: 6px; vertical-align: 2px; }
#hw .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
#hw .banner { position: absolute; left: 50%; bottom: 70px; transform: translateX(-50%); font: 600 16px "Archivo", Arial, sans-serif; display: none; }
#hw .cam { position: absolute; right: 16px; bottom: 16px; display: flex; gap: 8px; }`;
document.head.insertAdjacentHTML('beforeend', `<style>${css}</style>`);
document.body.insertAdjacentHTML('beforeend', `
<div id="hw">
  <div class="card claim">
    <h1>Whose forecast of the pedestrians should the robots trust?</h1>
    <p>Six Unitree G1 humanoids carry different world models (diffusion, JEPA, RSSM) and share their predicted futures. Pedestrians share nothing.
    The sidewalk at Two Bridges narrows to <b>1.0 m</b> at a street-tree pit (measured from NYC Open Data).</p>
    <p><b>Union</b> picks the plan whose <i>own</i> pedestrian forecast looks safest, so it believes the most convenient forecast.
    <b>HETERO-WORLD</b> scores every plan against the robots' <i>consensus</i> forecast (a Wasserstein barycenter).</p>
    <div class="ab" id="ab"></div>
  </div>
  <div class="card live" id="live"></div>
  <div class="card legend">
    <span><span class="dot" style="background:#3d8bff"></span>robot in the experiment</span>
    <span><span class="dot" style="background:#ff8a3d"></span>pedestrian in the experiment</span>
    <span><span class="sw" style="background:#3d8bff"></span>robot's planned path</span>
    <span><span class="sw" style="background:#ff8a3d"></span>pedestrian forecast the plan was scored against</span>
    <span><span class="sw" style="background:#f2b134"></span>measured pinch</span>
    <span><span class="sw" style="background:#e5383b"></span>contact</span>
    <span style="color:#9aa6b0">Other people and cars: BoundlessNYC's live city simulation, not part of the experiment.</span>
  </div>
  <div class="card banner" id="banner"></div>
  <div class="cam" id="cam"></div>
</div>`);
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- ambient walkers -> SOMA bodies
// BoundlessNYC keeps simulating its walkers (sidewalk following, crossings, signals); only their mannequin mesh is
// hidden. Walkers within AMB_R of the scene get a SOMA body at the same pose. Walkers inside the experiment's sidewalk
// strip are not drawn, so the recorded episode stays exactly what the simulator produced.
const AMB_R = +(Q.get('ambr') || 85), AMB_MAX = +(Q.get('ambmax') || 70);
const STRIP = { x: 15.5, y: D.corridor_half_width + 0.6 };
const ambient = new Map();   // ped object -> {w, x, z}
const ambPool = [];
const center = toThree(0, 0, 0);
let ambSeq = 0, hidCrowd = 0;
function updateAmbient() {
  const peds = window.__gtRefs && window.__gtRefs.peds;
  if (!peds) return;
  peds.mesh.visible = false;
  if (!hidCrowd || (performance.now() - hidCrowd) > 2000) {   // the photoreal crowd (if loaded) draws from its own meshes
    E.scene.traverse((o) => { if (/^pedS?:/.test(o.name || '')) o.visible = false; });
    hidCrowd = performance.now();
  }
  const arr = peds.mesh.instanceMatrix.array, seen = new Set();
  const near = [];
  for (const p of peds.peds) {
    const o = p.idx * 16, x = arr[o + 12], y = arr[o + 13], z = arr[o + 14];
    const d2 = (x - center.x) ** 2 + (z - center.z) ** 2;
    if (d2 > AMB_R * AMB_R) continue;
    const [lx, ly] = toLocal(x, z);
    if (Math.abs(lx) < STRIP.x && Math.abs(ly) < STRIP.y) continue;
    near.push([d2, p, x, y, z, o]);
  }
  near.sort((a, b) => a[0] - b[0]);
  for (const [, p, x, y, z, o] of near.slice(0, AMB_MAX)) {
    let a = ambient.get(p);
    if (!a) {
      const w = ambPool.pop() || new Walker(ambSeq++ * 3 + 1, 1);
      w.obj.scale.setScalar(0.94 + ((p.idx * 37) % 13) / 100);
      w.obj.castShadow = false;   // shadows only for the experiment's agents (frame rate)
      layer.add(w.obj);
      a = { w, x, z }; ambient.set(p, a);
    }
    seen.add(p);
    const dx = x - a.x, dz = z - a.z, dist = Math.hypot(dx, dz);
    a.x = x; a.z = z;
    w_place(a.w, x, y, z, dist > 0.004 ? Math.atan2(-dz, dx) : Math.atan2(-arr[o + 10], arr[o + 8]), dist, dist > 0.004);
  }
  for (const [p, a] of ambient) if (!seen.has(p)) { layer.remove(a.w.obj); ambPool.push(a.w); ambient.delete(p); }
}
function w_place(w, x, y, z, yaw, dist, moving) {
  w.obj.position.set(x, y, z);
  if (moving) {
    // smooth the heading so a walker does not snap on a one-frame jitter
    let d = yaw - w.obj.rotation.y; d = Math.atan2(Math.sin(d), Math.cos(d));
    w.obj.rotation.y += d * 0.35;
  } else w.obj.rotation.y = yaw;
  w.advance(dist, moving);
}

// ---------------------------------------------------------------- experiment actors
const state = { method: 'union_select', playing: true, token: 0, view: 'street' };
let actors = [], overlays = [], rings = [];

function clearActors() {
  actors.forEach((a) => { layer.remove(a.obj); layer.remove(a.marker); });
  actors = [];
  overlays.forEach((o) => { layer.remove(o); o.geometry.dispose(); }); overlays = [];
  rings.forEach((r) => layer.remove(r)); rings = [];
}

function spawn(ep) {
  ep.is_human.forEach((h, i) => {
    const a = h ? new Walker(i * 7 + 2, 1) : new G1Robot();
    a.i = i; a.human = !!h;
    a.marker = disc(h ? 0xff8a3d : 0x3d8bff);
    layer.add(a.obj); layer.add(a.marker); actors.push(a);
    a.last = null; a.exit = null;
    const rg = ring(); rg.rotation.x = Math.PI / 2; rg.visible = false; layer.add(rg); rings.push(rg);
  });
}

// Pose of agent i at fractional frame t. After an agent reaches its goal (d = 1) it walks on along its last heading
// for a few seconds and then leaves the scene, instead of vanishing in mid-sidewalk.
function agentPose(ep, i, t) {
  const n = ep.frames.length, k = Math.min(n - 1, Math.floor(t)), a = Math.min(1, t - k), k1 = Math.min(n - 1, k + 1);
  const f0 = ep.frames[k], f1 = ep.frames[k1];
  if (f0.d[i] === 1) {
    let kd = k; while (kd > 0 && ep.frames[kd - 1].d[i] === 1) kd--;
    const last = ep.frames[Math.max(0, kd - 1)], th = last.h[i], s = (t - (kd - 1)) * D.dt;
    if (s > 7) return null;
    return { x: last.p[i][0] + Math.cos(th) * s * 1.0, y: last.p[i][1] + Math.sin(th) * s * 1.0, h: th, done: true };
  }
  const p1 = f1.d[i] === 1 ? f0.p[i] : f1.p[i];
  let dh = f1.h[i] - f0.h[i]; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
  return { x: f0.p[i][0] + (p1[0] - f0.p[i][0]) * a, y: f0.p[i][1] + (p1[1] - f0.p[i][1]) * a, h: f0.h[i] + dh * a, done: false };
}

function placeAgents(ep, t) {
  const pos = [];
  for (const a of actors) {
    const P = agentPose(ep, a.i, t);
    a.obj.visible = a.marker.visible = !!P;
    pos[a.i] = P;
    if (!P) continue;
    const v = toThree(P.x, P.y, 0);
    a.obj.position.copy(v); a.obj.rotation.y = yawRad(P.h);
    a.marker.position.copy(toThree(P.x, P.y, 0.02)); a.marker.visible = !P.done;
    let dist = 0;
    if (a.last) dist = (P.x - a.last.x) * Math.cos(P.h) + (P.y - a.last.y) * Math.sin(P.h);   // signed: backing off is negative
    a.last = P;
    a.advance(dist, Math.abs(dist) > 0.0015);
  }
  // contact rings: robot and pedestrian closer than two radii (from the recorded frame, not the interpolation)
  const f = ep.frames[Math.min(ep.frames.length - 1, Math.round(t))];
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

function renderLive(t, cl, contacts) {
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
    <div class="row" style="color:#9aa6b0"><span>HETERO-WORLD contact rate</span><span>${D.stats.hw}</span></div>
    <div class="row" style="margin-top:6px;color:#9aa6b0"><span>city around it</span><span>${ambient.size} people drawn</span></div>`;
}

function setCamera(view) {
  state.view = view;
  const yaw = Math.atan2(AX[1], AX[0]) * 180 / Math.PI;
  const T = view === 'overhead'
    ? { location: toApi(-5, -0.5, 16), rotation: { pitch: -70, yaw, roll: 0 } }
    : view === 'low'
      ? { location: toApi(-7, -2.2, 1.5), rotation: { pitch: -6, yaw: yaw + 8, roll: 0 } }
      : view === 'wide'
        ? { location: toApi(-26, -9, 9), rotation: { pitch: -16, yaw: yaw + 18, roll: 0 } }
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
$('cam').innerHTML = ['street', 'low', 'wide', 'overhead'].map((v) => `<button data-v="${v}">${v[0].toUpperCase() + v.slice(1)} view</button>`).join('');
[...$('cam').children].forEach((b) => (b.onclick = () => setCamera(b.dataset.v)));

// One requestAnimationFrame loop: the recorded episode plays on its own clock (paused with the Pause button; the city
// keeps moving), and the ambient walkers are re-skinned every frame.
const play = { ep: null, t: 0, planIdx: -1, cl: { pred: Infinity, act: Infinity }, contacts: 0, inContact: false, ended: 0 };
function run() {
  state.token++;
  play.ep = D.episodes[state.method];
  play.t = 0; play.planIdx = -1; play.cl = { pred: Infinity, act: Infinity }; play.contacts = 0; play.inContact = false; play.ended = 0;
  $('banner').style.display = 'none';
  clearActors();
  spawn(play.ep);
  setCamera(state.view);
}

let lastNow = performance.now();
function frame(now) {
  const rdt = Math.max(0, (now - lastNow) / 1000), dt = Math.min(0.1, rdt); lastNow = Math.max(lastNow, now);
  if (rdt > 0 && window.__advance) window.__advance(Math.min(CITY_MAX_STEP, rdt) * (window.__CITY_SPEED || 1));
  try {
    updateAmbient();
    const ep = play.ep;
    if (ep) {
      const T = ep.frames.length - 1;
      if (state.playing) play.t = Math.min(T + 70, play.t + dt / D.dt * (window.__HW_SPEED || 1));
      const tf = Math.min(T, play.t);
      while (play.planIdx + 1 < ep.plans.length && ep.plans[play.planIdx + 1].t <= tf) {
        play.planIdx++;
        drawPlan(ep, ep.plans[play.planIdx]);
        play.cl = clearance(ep, ep.plans[play.planIdx]);
      }
      const c = placeAgents(ep, play.t);
      if (play.t <= T) { if (c && !play.inContact) play.contacts++; play.inContact = c; }
      renderLive(tf, play.cl, play.contacts);
      if (play.t >= T && !play.ended) {
        play.ended = 1;
        overlays.forEach((o) => { layer.remove(o); o.geometry.dispose(); }); overlays = [];
        const ok = play.contacts === 0;
        $('banner').innerHTML = ok
          ? `<span class="good">${NAMES[state.method]}: every robot passed the pinch with no pedestrian contact.</span>`
          : `<span class="bad">${NAMES[state.method]}: ${play.contacts} robot–pedestrian contact${play.contacts > 1 ? 's' : ''}. The plan trusted a forecast the pedestrian did not follow.</span>`;
        $('banner').style.display = 'block';
        window.__HW_DONE = { method: state.method, contacts: play.contacts };
      }
      if (play.t >= T + 70 && Q.get('loop') !== '0') run();   // 7 s after the end, replay the same planner
    }
  } catch (e) { console.warn('[hw]', e); }
  requestAnimationFrame(frame);
}

buttons();
window.__HW = { run, setCamera, state, ambient };
run();
requestAnimationFrame(frame);
