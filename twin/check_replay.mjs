// Smoke test of the HETERO-WORLD replay in the BoundlessNYC twin: screenshots of both planners.
import puppeteer from 'puppeteer-core';

const URL = 'http://127.0.0.1:5220/?api=1&twin=1&x=-2258&z=7847&y=4&gi=0&gfx=motionBlur:0';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1440,860'], protocolTimeout: 900000,
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 860 });
const errs = new Set();
page.on('pageerror', (e) => errs.add(String(e).slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error' && !/favicon|isReady/.test(m.text())) errs.add('[console] ' + m.text().slice(0, 300)); });
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 120 && !(await page.evaluate(() => !!window.__HW)); i++) await new Promise((r) => setTimeout(r, 3000));
console.log('replay started:', await page.evaluate(() => !!window.__HW));
const shot = async (name) => { await page.screenshot({ path: name }); console.log('saved', name); };
await new Promise((r) => setTimeout(r, 9000));
await shot('replay_union_a.png');
for (let i = 0; i < 60 && !(await page.evaluate(() => window.__HW_DONE)); i++) await new Promise((r) => setTimeout(r, 2000));
console.log('union done:', JSON.stringify(await page.evaluate(() => window.__HW_DONE)));
await shot('replay_union_end.png');
await page.evaluate(() => { window.__HW_DONE = null; window.__HW.state.method = 'hw_robust'; window.__HW.run(); });
await new Promise((r) => setTimeout(r, 12000));
await shot('replay_hw_a.png');
for (let i = 0; i < 90 && !(await page.evaluate(() => window.__HW_DONE)); i++) await new Promise((r) => setTimeout(r, 2000));
console.log('hw done:', JSON.stringify(await page.evaluate(() => window.__HW_DONE)));
await shot('replay_hw_end.png');
console.log('errors:', [...errs].slice(0, 8));
await browser.close();
