// Probe the public BoundlessNYC client: does ?api=1 expose window.__API, and what blueprints exist?
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] || 'https://mehmetkeremturkcan-boundless-nyc.hf.space/?api=1';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1280,720'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
page.on('console', (m) => { const t = m.text(); if (/api|error|ready|fail/i.test(t)) console.log('[page]', t.slice(0, 200)); });
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
let ok = false;
for (let i = 0; i < 90 && !ok; i++) {
  ok = await page.evaluate(() => !!(window.__API && window.__API.call));
  if (!ok) await new Promise((r) => setTimeout(r, 2000));
}
console.log('window.__API present:', ok);
if (ok) {
  const info = await page.evaluate(() => window.__API.call('server.info', {}));
  console.log('info', JSON.stringify(info).slice(0, 400));
  const bps = await page.evaluate(() => window.__API.call('world.get_blueprints', {}));
  const ids = (bps.result || bps).map ? (bps.result || bps).map((b) => b.id) : bps;
  console.log('blueprints', JSON.stringify(ids).slice(0, 1500));
  const loc = await page.evaluate(() => window.__API.call('geo.to_location', { lat: 40.712493, lon: -73.997994 }));
  console.log('two bridges location', JSON.stringify(loc));
}
await page.screenshot({ path: 'probe.png' });
await browser.close();
