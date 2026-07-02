const panoCanvas = document.getElementById('panoCanvas');
const panoCtx = panoCanvas.getContext('2d');
const projectionCanvas = document.getElementById('projectionCanvas');
const projectionCtx = projectionCanvas.getContext('2d', { willReadFrequently: false });
const azReadout = document.getElementById('azReadout');
const elReadout = document.getElementById('elReadout');
const fovReadout = document.getElementById('fovReadout');
const lockState = document.getElementById('lockState');

const demo = {
  azimuth: 137.65391583657777,
  elevation: -74.76117453970583,
  fov: 100,
};

const panoImg = new Image();
panoImg.src = 'assets/demo-panorama.jpg';

const sourceCanvas = document.createElement('canvas');
const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
let sourceData;
let drawRect = { x: 0, y: 0, w: 1, h: 1 };
let currentAz = demo.azimuth;
let currentEl = demo.elevation;
let fov = demo.fov;
let locked = false;
let renderQueued = false;
let projectionQueued = false;

const DPR = Math.min(window.devicePixelRatio || 1, 2);
const PROJECTION_SIZE = 500;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function degToRad(value) { return value * Math.PI / 180; }
function radToDeg(value) { return value * 180 / Math.PI; }

function resizeCanvas(canvas, width, height) {
  canvas.width = Math.max(1, Math.floor(width * DPR));
  canvas.height = Math.max(1, Math.floor(height * DPR));
}

function resize() {
  const rect = panoCanvas.getBoundingClientRect();
  resizeCanvas(panoCanvas, rect.width, rect.height);
  projectionCanvas.width = PROJECTION_SIZE;
  projectionCanvas.height = PROJECTION_SIZE;
  queueRender();
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
  drawRect = {
    x: (cw - w) / 2,
    y: (ch - h) / 2,
    w,
    h,
  };
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
  panoCtx.fillStyle = 'rgba(2, 5, 6, 0.12)';
  panoCtx.fillRect(drawRect.x, drawRect.y, drawRect.w, drawRect.h);
  panoCtx.restore();

  drawMarker(demo.azimuth, demo.elevation, 'rgba(243, 201, 91, 0.95)', 9);
  drawMarker(currentAz, currentEl, 'rgba(85, 214, 210, 0.95)', 6);
}

function updateReadout() {
  azReadout.textContent = `Az ${currentAz.toFixed(1)}°`;
  elReadout.textContent = `El ${currentEl.toFixed(1)}°`;
  fovReadout.textContent = `FOV ${Math.round(fov)}°`;
  lockState.textContent = locked ? 'locked' : 'live';
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
  if (projectionQueued) return;
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
  };
}

function setViewFromEvent(event) {
  const angles = pointerToAngles(event);
  currentAz = angles.az;
  currentEl = angles.el;
  queueProjection();
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

panoCanvas.addEventListener('pointermove', event => {
  if (!locked) setViewFromEvent(event);
});

panoCanvas.addEventListener('pointerdown', event => {
  locked = !locked;
  setViewFromEvent(event);
  updateReadout();
});

panoCanvas.addEventListener('wheel', event => {
  event.preventDefault();
  fov = clamp(fov + Math.sign(event.deltaY) * 5, 45, 125);
  queueProjection();
}, { passive: false });

projectionCanvas.addEventListener('wheel', event => {
  event.preventDefault();
  fov = clamp(fov + Math.sign(event.deltaY) * 5, 45, 125);
  queueProjection();
}, { passive: false });

window.addEventListener('resize', resize);

panoImg.addEventListener('load', () => {
  sourceCanvas.width = panoImg.naturalWidth;
  sourceCanvas.height = panoImg.naturalHeight;
  sourceCtx.drawImage(panoImg, 0, 0);
  sourceData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  resize();
  queueProjection();
});
