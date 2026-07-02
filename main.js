const panoCanvas = document.getElementById('panoCanvas');
const panoCtx = panoCanvas.getContext('2d');
const projectionCanvas = document.getElementById('projectionCanvas');
const projectionCtx = projectionCanvas.getContext('2d', { willReadFrequently: false });
const projectionPanel = document.getElementById('projectionPanel');
const cursorLabel = document.getElementById('cursorLabel');
const azReadout = document.getElementById('azReadout');
const elReadout = document.getElementById('elReadout');
const fovReadout = document.getElementById('fovReadout');
const inspectState = document.getElementById('inspectState');
const demoKicker = document.getElementById('demoKicker');
const demoTitle = document.getElementById('demoTitle');
const demoBody = document.getElementById('demoBody');
const targetLabel = document.getElementById('targetLabel');
const prevDemo = document.getElementById('prevDemo');
const nextDemo = document.getElementById('nextDemo');

const GUIDE_BODY = 'Move across the panorama to read azimuth and elevation. Hold the mouse button to call the 100° perspective projection tool; keep holding and drag to inspect other directions. Scroll while holding to zoom the FOV.';

const demos = [
  {
    image: 'assets/demo-panorama.webp',
    target: 'pink stool',
    title: 'Find the pink stool in the panorama.',
    body: GUIDE_BODY,
    azimuth: 137.65391583657777,
    elevation: -74.76117453970583,
    fov: 100,
  },
  {
    image: 'assets/demo-play-boat.webp',
    target: "colorful wooden children's play boat",
    title: "Find the colorful wooden children's play boat in the panorama.",
    body: GUIDE_BODY,
    azimuth: 177.32868871463714,
    elevation: 3.4507171963325236,
    fov: 100,
  },
  {
    image: 'assets/demo-122.webp',
    target: 'a round wall clock',
    title: 'Find a round wall clock in the panorama.',
    body: GUIDE_BODY,
    azimuth: -148.61800729761313,
    elevation: 23.0079856130375,
    fov: 100,
  },
  {
    image: 'assets/demo-155.webp',
    target: 'a brown plush couch with two cushions',
    title: 'Find a brown plush couch with two cushions in the panorama.',
    body: GUIDE_BODY,
    azimuth: 178.5558851001805,
    elevation: -14.670953815303278,
    fov: 100,
  },
  {
    image: 'assets/demo-376.webp',
    target: 'a black refrigerator with a shiny handle',
    title: 'Find a black refrigerator with a shiny handle in the panorama.',
    body: GUIDE_BODY,
    azimuth: -146.2529901330173,
    elevation: -16.729846941704974,
    fov: 100,
  },
  {
    image: 'assets/demo-447.webp',
    target: 'a blue mountain bike',
    title: 'Find a blue mountain bike in the panorama.',
    body: GUIDE_BODY,
    azimuth: 95.8901445420024,
    elevation: -15.458380894487242,
    fov: 100,
  },
  {
    image: 'assets/demo-544.webp',
    target: 'a flat-screen television mounted on the wall',
    title: 'Find a flat-screen television mounted on the wall in the panorama.',
    body: GUIDE_BODY,
    azimuth: -2.7387258563525596,
    elevation: 1.1771339939954135,
    fov: 100,
  },
];

const panoImg = new Image();
panoImg.decoding = 'async';
const sourceCanvas = document.createElement('canvas');
const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
let sourceData;
let activeDemoIndex = 0;
let drawRect = { x: 0, y: 0, w: 1, h: 1 };
let currentAz = demos[0].azimuth;
let currentEl = demos[0].elevation;
let cursorX = 0;
let cursorY = 0;
let fov = demos[0].fov;
let isInspecting = false;
let isPointerInside = false;
let renderQueued = false;
let projectionQueued = false;

const DPR = Math.min(window.devicePixelRatio || 1, 2);
const PROJECTION_SIZE = 500;

function activeDemo() { return demos[activeDemoIndex]; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function degToRad(value) { return value * Math.PI / 180; }

function resizeCanvas(canvas, width, height) {
  canvas.width = Math.max(1, Math.floor(width * DPR));
  canvas.height = Math.max(1, Math.floor(height * DPR));
}

function resize() {
  const rect = panoCanvas.getBoundingClientRect();
  resizeCanvas(panoCanvas, rect.width, rect.height);
  projectionCanvas.width = PROJECTION_SIZE;
  projectionCanvas.height = PROJECTION_SIZE;
  updateCursorLabel();
  queueRender();
  if (isInspecting) queueProjection();
}

function fitPanorama() {
  const cw = panoCanvas.width;
  const ch = panoCanvas.height;
  const imgRatio = panoImg.naturalWidth / panoImg.naturalHeight;
  let w = cw;
  let h = w / imgRatio;
  if (h > ch) {
    h = ch;
    w = h * imgRatio;
  }
  drawRect = { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

function drawMarker(az, el, color, radius) {
  const x = drawRect.x + ((az + 180) / 360) * drawRect.w;
  const y = drawRect.y + ((90 - el) / 180) * drawRect.h;
  panoCtx.save();
  panoCtx.strokeStyle = color;
  panoCtx.fillStyle = color;
  panoCtx.lineWidth = 2 * DPR;
  panoCtx.beginPath();
  panoCtx.arc(x, y, radius * DPR, 0, Math.PI * 2);
  panoCtx.stroke();
  panoCtx.beginPath();
  panoCtx.moveTo(x - radius * 1.8 * DPR, y);
  panoCtx.lineTo(x + radius * 1.8 * DPR, y);
  panoCtx.moveTo(x, y - radius * 1.8 * DPR);
  panoCtx.lineTo(x, y + radius * 1.8 * DPR);
  panoCtx.stroke();
  panoCtx.beginPath();
  panoCtx.arc(x, y, 2.5 * DPR, 0, Math.PI * 2);
  panoCtx.fill();
  panoCtx.restore();
}

function renderPanorama() {
  if (!panoImg.complete || !panoImg.naturalWidth) return;
  fitPanorama();
  panoCtx.clearRect(0, 0, panoCanvas.width, panoCanvas.height);
  panoCtx.fillStyle = '#020506';
  panoCtx.fillRect(0, 0, panoCanvas.width, panoCanvas.height);
  panoCtx.drawImage(panoImg, drawRect.x, drawRect.y, drawRect.w, drawRect.h);
  panoCtx.save();
  panoCtx.fillStyle = 'rgba(2, 5, 6, 0.08)';
  panoCtx.fillRect(drawRect.x, drawRect.y, drawRect.w, drawRect.h);
  panoCtx.restore();
  const demo = activeDemo();
  drawMarker(demo.azimuth, demo.elevation, 'rgba(243, 201, 91, 0.95)', 9);
  if (isPointerInside) drawMarker(currentAz, currentEl, 'rgba(85, 214, 210, 0.95)', isInspecting ? 8 : 5);
}

function updateReadout() {
  const az = currentAz.toFixed(1);
  const el = currentEl.toFixed(1);
  const fovText = `FOV ${Math.round(fov)}°`;
  azReadout.textContent = `Az ${az}°`;
  elReadout.textContent = `El ${el}°`;
  fovReadout.textContent = fovText;
  inspectState.textContent = fovText;
}

function updateCursorLabel() {
  cursorLabel.textContent = `Az ${currentAz.toFixed(1)}° · El ${currentEl.toFixed(1)}°`;
  cursorLabel.style.left = `${cursorX}px`;
  cursorLabel.style.top = `${cursorY}px`;
}

function updateDemoText() {
  const demo = activeDemo();
  demoKicker.textContent = `Interactive task demo · ${activeDemoIndex + 1} / ${demos.length}`;
  demoTitle.textContent = demo.title;
  demoBody.textContent = demo.body;
  targetLabel.textContent = `Target query: ${demo.target}`;
}

function queueRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderPanorama();
  });
}

function queueProjection() {
  if (!isInspecting || projectionQueued) return;
  projectionQueued = true;
  requestAnimationFrame(() => {
    projectionQueued = false;
    renderProjection(currentAz, currentEl, fov);
    updateReadout();
    queueRender();
  });
}

function pointerToAngles(event) {
  const rect = panoCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * DPR;
  const y = (event.clientY - rect.top) * DPR;
  const u = clamp((x - drawRect.x) / drawRect.w, 0, 1);
  const v = clamp((y - drawRect.y) / drawRect.h, 0, 1);
  return {
    az: u * 360 - 180,
    el: 90 - v * 180,
    labelX: clamp(event.clientX - rect.left, 74, rect.width - 74),
    labelY: clamp(event.clientY - rect.top, 32, rect.height - 12),
  };
}

function setViewFromEvent(event) {
  const angles = pointerToAngles(event);
  currentAz = angles.az;
  currentEl = angles.el;
  cursorX = angles.labelX;
  cursorY = angles.labelY;
  updateCursorLabel();
  updateReadout();
  queueRender();
  if (isInspecting) queueProjection();
}

function showProjection() {
  isInspecting = true;
  projectionPanel.classList.add('visible');
  cursorLabel.classList.add('visible');
  updateReadout();
  queueProjection();
}

function hideProjection() {
  isInspecting = false;
  projectionPanel.classList.remove('visible');
  queueRender();
}

function loadDemo(index) {
  activeDemoIndex = (index + demos.length) % demos.length;
  const demo = activeDemo();
  hideProjection();
  sourceData = undefined;
  currentAz = demo.azimuth;
  currentEl = demo.elevation;
  fov = demo.fov;
  updateDemoText();
  updateCursorLabel();
  updateReadout();
  panoCtx.fillStyle = '#020506';
  panoCtx.fillRect(0, 0, panoCanvas.width, panoCanvas.height);
  panoImg.src = demo.image;
}

function renderProjection(centerAz, centerEl, fovDeg) {
  if (!sourceData) return;
  const out = projectionCtx.createImageData(PROJECTION_SIZE, PROJECTION_SIZE);
  const outData = out.data;
  const src = sourceData.data;
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const yaw = degToRad(centerAz);
  const pitch = degToRad(centerEl);
  const tanHalf = Math.tan(degToRad(fovDeg) / 2);
  const cp = Math.cos(pitch);
  const forward = [Math.sin(yaw) * cp, Math.sin(pitch), Math.cos(yaw) * cp];
  const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const up = [
    forward[1] * right[2] - forward[2] * right[1],
    forward[2] * right[0] - forward[0] * right[2],
    forward[0] * right[1] - forward[1] * right[0],
  ];
  let oi = 0;
  for (let py = 0; py < PROJECTION_SIZE; py++) {
    const cameraY = (1 - 2 * (py + 0.5) / PROJECTION_SIZE) * tanHalf;
    for (let px = 0; px < PROJECTION_SIZE; px++) {
      const cameraX = (2 * (px + 0.5) / PROJECTION_SIZE - 1) * tanHalf;
      let dx = forward[0] + cameraX * right[0] + cameraY * up[0];
      let dy = forward[1] + cameraX * right[1] + cameraY * up[1];
      let dz = forward[2] + cameraX * right[2] + cameraY * up[2];
      const invLen = 1 / Math.hypot(dx, dy, dz);
      dx *= invLen;
      dy *= invLen;
      dz *= invLen;
      const srcYaw = Math.atan2(dx, dz);
      const srcPitch = Math.asin(clamp(dy, -1, 1));
      let sx = Math.floor(((srcYaw + Math.PI) / (2 * Math.PI)) * srcW);
      const sy = clamp(Math.floor(((Math.PI / 2 - srcPitch) / Math.PI) * srcH), 0, srcH - 1);
      sx = ((sx % srcW) + srcW) % srcW;
      const si = (sy * srcW + sx) * 4;
      outData[oi++] = src[si];
      outData[oi++] = src[si + 1];
      outData[oi++] = src[si + 2];
      outData[oi++] = 255;
    }
  }
  projectionCtx.putImageData(out, 0, 0);
}

panoCanvas.addEventListener('pointerenter', event => {
  isPointerInside = true;
  cursorLabel.classList.add('visible');
  setViewFromEvent(event);
});

panoCanvas.addEventListener('pointermove', event => {
  isPointerInside = true;
  setViewFromEvent(event);
});

panoCanvas.addEventListener('pointerdown', event => {
  event.preventDefault();
  panoCanvas.setPointerCapture(event.pointerId);
  isPointerInside = true;
  setViewFromEvent(event);
  showProjection();
});

panoCanvas.addEventListener('pointerup', event => {
  if (panoCanvas.hasPointerCapture(event.pointerId)) panoCanvas.releasePointerCapture(event.pointerId);
  hideProjection();
});

panoCanvas.addEventListener('pointercancel', hideProjection);
panoCanvas.addEventListener('lostpointercapture', hideProjection);
panoCanvas.addEventListener('pointerleave', () => {
  isPointerInside = false;
  if (!isInspecting) cursorLabel.classList.remove('visible');
  queueRender();
});

panoCanvas.addEventListener('wheel', event => {
  if (!isInspecting) return;
  event.preventDefault();
  fov = clamp(fov + Math.sign(event.deltaY) * 5, 45, 125);
  updateReadout();
  queueProjection();
}, { passive: false });

function handleNav(event, direction) {
  event.preventDefault();
  event.stopPropagation();
  loadDemo(activeDemoIndex + direction);
}
prevDemo.addEventListener('click', event => handleNav(event, -1));
nextDemo.addEventListener('click', event => handleNav(event, 1));
prevDemo.addEventListener('pointerdown', event => event.stopPropagation());
nextDemo.addEventListener('pointerdown', event => event.stopPropagation());

window.addEventListener('resize', resize);

panoImg.addEventListener('load', () => {
  sourceCanvas.width = panoImg.naturalWidth;
  sourceCanvas.height = panoImg.naturalHeight;
  sourceCtx.drawImage(panoImg, 0, 0);
  sourceData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  resize();
  updateReadout();
  if (isInspecting) queueProjection();
});

resize();
loadDemo(0);
