import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1280,720'], protocolTimeout: 600000 });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
const errs = new Map();
page.on('pageerror', (e) => { const k = String(e.stack || e).slice(0, 600); errs.set(k, (errs.get(k) || 0) + 1); });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') { const k = '[' + m.type() + '] ' + m.text().slice(0, 300); errs.set(k, (errs.get(k) || 0) + 1); } });
page.on('response', (r) => { if (r.status() >= 400) { const k = `[http ${r.status()}] ${r.url().replace('http://127.0.0.1:5220/', '')}`; errs.set(k, (errs.get(k) || 0) + 1); } });
await page.goto('http://127.0.0.1:5220/?api=1', { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 36; i++) {
  await new Promise((r) => setTimeout(r, 5000));
  const st = await page.evaluate(() => (window.__BOOT || []).map((b) => `${b[1]}@${(+b[0]).toFixed(0)}s`));
  if (i % 6 === 5) console.log('boot:', st.join(' > '));
  if (st.some((s) => s.startsWith('loading hidden'))) { console.log('BOOTED'); break; }
}
for (const [k, n] of errs) console.log(n, k);
await browser.close();
