import puppeteer from 'puppeteer';

const extraArgs = process.argv.slice(2).join(' ').split(' ').filter(Boolean);
const flags = ['--no-sandbox', '--headless=new', ...extraArgs];

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  args: flags,
  defaultViewport: { width: 320, height: 240 },
});

const page = await browser.newPage();
await page.goto('about:blank');

const info = await page.evaluate(() => {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) return { error: 'no webgl context' };
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return {
    renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
  };
});

console.log(JSON.stringify(info));
await browser.close();
