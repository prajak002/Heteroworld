// Render the real BoundlessNYC city at the Two Bridges pinch point through the in-page API (hostless mode).
import puppeteer from 'puppeteer-core';

const LAT = 40.712493, LON = -73.997994;
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1280,720'],
  protocolTimeout: 900000,
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
page.on('console', (m) => { const t = m.text(); if (/\[api\]|pageerror|READY|bad tile/i.test(t)) console.log('[page]', t.slice(0, 200)); });
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));
await page.goto('http://127.0.0.1:5220/?api=1&x=-2258&z=7847&y=4&gi=0', { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 150 && !(await page.evaluate(() => !!window.__API)); i++) await new Promise((r) => setTimeout(r, 2000));
const call = (m, p = {}) => page.evaluate((m, p) => Promise.resolve(window.__API.call(m, p)), m, p);
const unwrap = (r) => (r && r.result !== undefined ? r.result : r);
console.log('info', JSON.stringify(unwrap(await call('server.info'))).slice(0, 300));
const L = unwrap(await call('geo.to_location', { lat: LAT, lon: LON }));
console.log('two bridges location', JSON.stringify(L));
for (let i = 0; i < 90; i++) {
  const stages = await page.evaluate(() => (window.__BOOT || []).map((b) => b[1]));
  if (stages.includes('loading hidden')) { console.log('boot', stages.slice(-4).join(' > ')); break; }
  await new Promise((r) => setTimeout(r, 5000));
}
const AX = [-0.189246, 0.98193], NR = [-0.98193, -0.189246];          // sidewalk axis / towards buildings (east, north)
const G = 3.52;                                                   // ground elevation at the pinch (map.get_surface)
const at = (x, y, z = 0) => ({ x: L.x + x * AX[0] + y * NR[0], y: L.y + x * AX[1] + y * NR[1], z: G + z });
for (const [x, y] of [[0, 0], [0, 1], [0, -1], [0, 2], [0, -2]]) console.log('surface', x, y, JSON.stringify(unwrap(await call('map.get_surface', { location: at(x, y) }))));
const yaw = Math.atan2(AX[1], AX[0]) * 180 / Math.PI;
await call('actor.set_transform', { id: 1, transform: { location: at(-11, -3.0, 2.6), rotation: { pitch: -9, yaw, roll: 0 } } });
for (let k = 0; k < 24; k++) {
  await new Promise((r) => setTimeout(r, 10000));
  const st = await page.evaluate(() => {
    const s = window.__STREAMER;
    return s ? { idle: typeof s.idle === 'function' ? s.idle() : null, loaded: s.loaded ? s.loaded.size : (s.tiles ? s.tiles.size : null) } : null;
  });
  console.log(`t+${(k + 1) * 10}s streamer`, JSON.stringify(st));
  if (k % 3 === 2 || (st && st.idle)) await page.screenshot({ path: `twobridges_${(k + 1) * 10}s.png` });
  if (st && st.idle && k >= 2) break;
}
await page.screenshot({ path: 'twobridges.png' });
console.log('saved twobridges.png');
await browser.close();
