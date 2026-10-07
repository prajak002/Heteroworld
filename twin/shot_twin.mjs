// Screenshot the twin replay (SOMA pedestrians, G1 robots, live city). Usage: node shot_twin.mjs [outPrefix]
import puppeteer from 'puppeteer-core';
const out = process.argv[2] || 'twin_shot';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--window-size=1440,900'], protocolTimeout: 900000,
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
page.on('console', (m) => { const t = m.text(); if (/\[hw\]|error|Error/i.test(t)) console.log('[page]', t.slice(0, 300)); });
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 400)));
await page.goto('http://127.0.0.1:5220/?api=1&twin=1&x=-2258&z=7847&y=4&gi=0&crowd=0&gfx=motionBlur:0', { waitUntil: 'domcontentloaded', timeout: 300000 });
for (let i = 0; i < 120 && !(await page.evaluate(() => !!window.__HW)); i++) await new Promise((r) => setTimeout(r, 3000));
console.log('replay up:', await page.evaluate(() => !!window.__HW));
await new Promise((r) => setTimeout(r, 25000));
const info = async () => page.evaluate(() => ({ ambient: window.__HW.ambient.size, t: document.querySelector('#live')?.innerText.split('\n').slice(0, 2).join(' '),
  cars: window.__gtRefs?.traffic?.cars?.length, peds: window.__gtRefs?.peds?.peds?.length, fps: window.__ENGINE?.fps }));
for (const [view, wait] of [['street', 0], ['wide', 6000], ['low', 6000], ['overhead', 6000]]) {
  await page.evaluate((v) => window.__HW.setCamera(v), view);
  await new Promise((r) => setTimeout(r, wait + 3000));
  console.log(view, JSON.stringify(await info()));
  await page.screenshot({ path: `${out}_${view}.png` });
}
await browser.close();
