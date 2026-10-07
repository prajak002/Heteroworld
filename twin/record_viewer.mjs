// Deterministic 1080p60 recording of docs/hw_viewer for social posts: each frame steps the scene by exactly 1/60 s and is piped into ffmpeg.
// Usage (from the repo root): node docs/hw_viewer/serve.mjs docs/hw_viewer &
//                             (cd twin && node record_viewer.mjs ../figures/video/out.mp4)
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
const out = process.argv[2], W = 1920, H = 1080, FPS = 60;
const SEGMENTS = [ { method: 'union_select', tod: 'day', seconds: 22 }, { method: 'hw_robust', tod: 'dusk', seconds: 22 } ];
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist', `--window-size=${W},${H + 140}`], protocolTimeout: 1200000 });
const page = await browser.newPage(); await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
const errs = []; page.on('pageerror', (e) => errs.push(String(e))); page.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errs.push(m.text()); });
await page.goto('http://127.0.0.1:5231/', { waitUntil: 'domcontentloaded' });
for (let i = 0; i < 90 && !(await page.evaluate(() => !!window.__VIEW)); i++) await new Promise((r) => setTimeout(r, 1000));
await page.evaluate(() => { window.__VIEW.rec.start({ period: 52 }); window.__VIEW.cinema(true); });
await new Promise((r) => setTimeout(r, 1500));
const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
const t0 = Date.now();
for (const seg of SEGMENTS) {
  await page.evaluate(async (s) => { await window.__VIEW.setTOD(s.tod); window.__VIEW.rec.setMethod(s.method); window.__VIEW.cine.u = 0; }, seg);
  for (let k = 0; k < 20; k++) await page.evaluate(() => window.__VIEW.rec.step(1 / 60));   // settle (not recorded)
  await page.evaluate((len) => { window.__VIEW.rec.run(); window.__VIEW.rec.director(len); }, seg.seconds);
  const n = seg.seconds * FPS;
  for (let f = 0; f < n; f++) {
    await page.evaluate(() => window.__VIEW.rec.step(1 / 60));
    const buf = await page.screenshot({ type: 'jpeg', quality: 93, optimizeForSpeed: true });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (f % 300 === 0) console.log(seg.method, seg.tod, `${f}/${n}`, `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
}
ff.stdin.end();
await new Promise((r) => ff.on('close', r));
console.log('done', out, errs.slice(0, 5).join('\n') || 'no page errors');
await browser.close();
