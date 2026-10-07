// Check that cars and walkers move in real time: node probe_motion.mjs
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1440,900'], protocolTimeout: 900000 });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));
page.on('console', (m) => { if (/\[hw\]/.test(m.text())) console.log(m.text().slice(0, 300)); });
await page.goto('http://127.0.0.1:5220/?api=1&twin=1&x=-2258&z=7847&y=4&gi=0&crowd=0&gfx=motionBlur:0', { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 120 && !(await page.evaluate(() => !!window.__HW)); i++) await new Promise((r) => setTimeout(r, 3000));
await new Promise((r) => setTimeout(r, 15000));
const snap = () => page.evaluate(() => {
  const T = window.__gtRefs.traffic, P = window.__gtRefs.peds;
  const cars = T.cars.slice(0, 60).map((c) => { const o = c.mesh || c.obj || c.group || c; const p = o.position || c.pos || { x: c.x, z: c.z }; return [p.x, p.z]; });
  const arr = P.mesh.instanceMatrix.array;
  return { t: performance.now(), cars, peds: P.peds.slice(0, 60).map((p) => [arr[p.idx * 16 + 12], arr[p.idx * 16 + 14]]), fps: window.__ENGINE.fps, car0: Object.keys(T.cars[0] || {}).slice(0, 25) };
});
const a = await snap(); await new Promise((r) => setTimeout(r, 5000)); const b = await snap();
const sec = (b.t - a.t) / 1000;
const spd = (A, B) => { const v = A.map((p, i) => B[i] ? Math.hypot(B[i][0] - p[0], B[i][1] - p[1]) / sec : 0).filter((x) => isFinite(x)); v.sort((x, y) => x - y); return { moving: v.filter((x) => x > 0.2).length + '/' + v.length, median_mps: +(v[v.length >> 1] || 0).toFixed(2), p90_mps: +(v[Math.floor(v.length * 0.9)] || 0).toFixed(2) }; };
console.log('fps', b.fps.toFixed(1), 'cars', JSON.stringify(spd(a.cars, b.cars)), 'walkers', JSON.stringify(spd(a.peds, b.peds)));
console.log('car fields', a.car0.join(','));
await page.screenshot({ path: process.argv[2] || 'motion.png' });
await browser.close();
