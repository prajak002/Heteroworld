// Frame-rate probe: node fps_twin.mjs "<extra query>"
import puppeteer from 'puppeteer-core';
const extra = process.argv[2] || '';
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1440,900'], protocolTimeout: 900000 });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto(`http://127.0.0.1:5220/?api=1&twin=1&x=-2258&z=7847&y=4&gi=0&crowd=0&gfx=motionBlur:0${extra}`, { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 120 && !(await page.evaluate(() => !!window.__HW)); i++) await new Promise((r) => setTimeout(r, 3000));
await new Promise((r) => setTimeout(r, 20000));
const fps = await page.evaluate(() => new Promise((res) => { let n = 0; const t0 = performance.now(); const f = () => { n++; if (performance.now() - t0 < 5000) requestAnimationFrame(f); else res(n / 5); }; requestAnimationFrame(f); }));
const calls = await page.evaluate(() => window.__ENGINE.renderer.info.render);
console.log(extra || '(default)', 'rAF fps', fps, 'draw calls', calls.calls, 'tris', calls.triangles);
await browser.close();
