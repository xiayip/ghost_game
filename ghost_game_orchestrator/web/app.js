const POLL_MS = 50; // 20 Hz
const STALE_MS = 2000;
// The optimized detector publishes 15 Hz. Poll slightly faster so the browser
// sees every fresh frame without allowing overlapping fetches.
const CAMERA_POLL_MS = 60;
// Dev-container port forwarding can drop an isolated request under load even
// when the camera itself is fine - don't flip to "offline" until several
// polls in a row have failed.
const CAMERA_STALE_MS = 1000;
const CAMERA_VISIBLE_PHASES = new Set([
  'success_pose_reached',
  'face_searching',
  'face_stabilizing',
  'face_centering',
  'face_centered',
  'face_not_found',
  'dancing',
  'done',
]);

const connDot = document.getElementById('conn-dot');
const connText = document.getElementById('conn-text');
const phaseBadge = document.getElementById('phase-badge');
const foundCount = document.getElementById('found-count');
const grid = document.getElementById('joint-grid');
const armLayout = document.getElementById('arm-layout');
const cameraFeed = document.getElementById('camera-feed');
const cameraPanel = document.getElementById('camera-panel');
const cameraToggle = document.getElementById('camera-toggle');

let lastGoodAt = 0;
let cardsBuilt = false;
let cameraCollapsed = false;
let cameraRevealed = false; // shown only after the success-pose turn completes

cameraToggle.addEventListener('click', () => {
  cameraCollapsed = !cameraCollapsed;
  cameraPanel.classList.toggle('collapsed', cameraCollapsed);
  cameraToggle.setAttribute('aria-expanded', String(!cameraCollapsed));
});

function buildCards(joints) {
  grid.innerHTML = '';
  joints.forEach((name, i) => {
    const label = `J${i + 1}`;
    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card-${name}`;
    card.innerHTML = `
      <div class="card-head">
        <span class="joint-name">${label}</span>
        <span class="joint-pct" id="pct-${name}">--%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" id="bar-${name}"></div></div>
      <div class="card-foot">
        <span id="dist-${name}">dist --</span>
        <span id="lock-${name}"></span>
      </div>`;
    grid.appendChild(card);
  });
  cardsBuilt = true;
}

function render(state) {
  phaseBadge.textContent = `phase: ${state.phase ?? '--'}`;
  foundCount.innerHTML = `found <b>${state.found_count ?? 0}</b>/<b>${state.total ?? 0}</b>`;

  const total = state.total ?? 0;
  const allFound = total > 0 && (state.found_count ?? 0) >= total;
  // camera_ready is latched by the orchestrator at the exact point where the
  // measured arm pose passes success validation. Keep a fallback for an old
  // backend during rolling restarts.
  cameraRevealed = state.camera_ready === true ||
    (state.camera_ready == null && allFound && CAMERA_VISIBLE_PHASES.has(state.phase));
  armLayout.classList.toggle('camera-mode-hidden', cameraRevealed);
  cameraPanel.classList.toggle('locked', !cameraRevealed);

  const joints = state.joints;
  if (!Array.isArray(joints) || joints.length === 0) {
    grid.innerHTML = '<div class="card-foot">Enable publish_debug_distances on ghost_game_node to see per-joint progress.</div>';
    cardsBuilt = false;
    return;
  }
  if (!cardsBuilt) buildCards(joints);

  const progress = state.progress || [];
  const distances = state.distances || [];
  const locked = state.locked || [];

  joints.forEach((name, i) => {
    const pct = progress[i];
    const dist = distances[i];
    const isLocked = !!locked[i];
    const shown = pct == null ? 0 : Math.round(pct);

    document.getElementById(`bar-${name}`).style.width = `${shown}%`;
    document.getElementById(`pct-${name}`).textContent = pct == null ? '--%' : `${shown}%`;
    document.getElementById(`dist-${name}`).textContent =
      dist == null ? 'dist --' : `dist ${dist.toFixed(3)} rad`;
    document.getElementById(`lock-${name}`).textContent = isLocked ? 'LOCKED' : '';
    document.getElementById(`card-${name}`).classList.toggle('locked', isLocked);
  });

  if (window.ghostGameViewer && Array.isArray(state.positions)) {
    const angles = {};
    const lockedByName = {};
    joints.forEach((name, i) => {
      angles[name] = state.positions[i];
      lockedByName[name] = !!locked[i];
    });
    window.ghostGameViewer.setJointAngles(angles);
    if (window.ghostGameViewer.setLockedJoints) {
      window.ghostGameViewer.setLockedJoints(lockedByName);
    }
  }
}

async function poll() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    if (res.ok) {
      const state = await res.json();
      lastGoodAt = Date.now();
      render(state);
    }
  } catch (err) {
    // network hiccup - fall through to the stale check below
  }

  const stale = Date.now() - lastGoodAt > STALE_MS;
  connDot.className = `dot ${lastGoodAt === 0 ? '' : stale ? 'err' : 'live'}`;
  connText.textContent = lastGoodAt === 0 ? 'connecting…' : stale ? 'no signal' : 'live';

  setTimeout(poll, POLL_MS);
}

const cameraOfflineText = document.getElementById('camera-offline');
let cameraObjectUrl = null;
let lastCameraGoodAt = 0;

async function pollCamera() {
  // fetch()+blob() instead of a plain <img src=...> so a failure surfaces
  // *why* (HTTP status vs. network error) right in the placeholder text -
  // no devtools needed to tell "camera not publishing" from "request never
  // reached the server" (proxy/tunnel issue) from "browser can't decode it".
  // Don't fetch frames nobody can see - saves bandwidth while the camera
  // tab is either manually collapsed or the arm has not reached the
  // post-game success pose yet.
  if (!cameraCollapsed && cameraRevealed) {
    try {
      const res = await fetch(`/api/camera.jpg?t=${Date.now()}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const url = URL.createObjectURL(await res.blob());
      cameraFeed.src = url;
      cameraFeed.classList.remove('offline');
      lastCameraGoodAt = Date.now();
      if (cameraObjectUrl) URL.revokeObjectURL(cameraObjectUrl);
      cameraObjectUrl = url;
    } catch (err) {
      if (Date.now() - lastCameraGoodAt > CAMERA_STALE_MS) {
        cameraFeed.classList.add('offline');
        cameraOfflineText.textContent = `no signal (${err.message})`;
      }
      // else: isolated poll blip - keep showing the last good frame
    }
  }
  setTimeout(pollCamera, CAMERA_POLL_MS);
}

poll();
pollCamera();
