const POLL_MS = 50; // 20 Hz
const TTS_POLL_MS = 100; // stage cues are sparse; 100 ms still feels immediate
const STALE_MS = 2000;
// The optimized detector publishes 30 Hz. Render its latest processed frame
// at about 17 Hz without allowing overlapping browser fetches.
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
  'face_gesture',
  'face_not_found',
  'dancing',
  'done',
]);

const NARRATIVE = {
  idle: {
    chapter: '00', signal: 'LINK STANDBY', phase: '待机 / IDLE', step: -1,
    titleCn: '等待 Ghost 潜入 Shell',
    titleEn: 'AWAITING GHOST INFILTRATION',
    guideCn: '启动交互协议，建立意识连接。',
    guideEn: 'Initiate the protocol to open a neural link.',
  },
  setup: {
    chapter: '01', signal: 'INTRUSION DETECTED', phase: '接入 / INFILTRATION', step: 0,
    titleCn: 'Ghost 已潜入 Shell',
    titleEn: 'GHOST HAS INFILTRATED THE SHELL',
    guideCn: '壳层正在解除关节约束，请等待系统完成同步。',
    guideEn: 'Joint constraints are dissolving. Wait for shell synchronization.',
  },
  searching: {
    chapter: '02', signal: 'SHARDS SCATTERED', phase: '搜索 / SEARCHING', step: 1,
    titleCn: '寻找散落的 Ghost 碎片',
    titleEn: 'RECOVER THE SCATTERED GHOST FRAGMENTS',
    guideCn: '开始掰动机械臂关节，寻找每个碎片隐藏的位置。',
    guideEn: 'Move each arm joint by hand and locate every hidden fragment.',
  },
  all_found: {
    chapter: '03', signal: 'MEMORY COMPLETE', phase: '重组 / REASSEMBLY', step: 2,
    titleCn: '六枚意识碎片已回收',
    titleEn: 'ALL SIX CONSCIOUSNESS SHARDS RECOVERED',
    guideCn: '请松开机械臂并保持距离，Ghost 正在重组运动记忆。',
    guideEn: 'Release the arm and stand clear while motor memory is rebuilt.',
  },
  success_move: {
    chapter: '03', signal: 'SHELL TURNING', phase: '重组 / REASSEMBLY', step: 2,
    titleCn: 'Ghost 正在接管 Shell',
    titleEn: 'GHOST IS TAKING CONTROL OF THE SHELL',
    guideCn: '壳层正在转向视觉接入点，请保持安全距离。',
    guideEn: 'The shell is turning toward its optical uplink. Stand clear.',
  },
  success_pose_reached: {
    chapter: '04', signal: 'OPTICAL NERVE ONLINE', phase: '视觉 / VISION', step: 2,
    titleCn: '视觉神经已上线',
    titleEn: 'OPTICAL NERVE IS NOW ONLINE',
    guideCn: '请站在相机前方并看向 Ghost，等待目标锁定。',
    guideEn: 'Stand before the camera and face Ghost for target acquisition.',
  },
  face_searching: {
    chapter: '04', signal: 'SCANNING SECTOR', phase: '扫描 / SCANNING', step: 2,
    titleCn: 'Ghost 正在搜索访客信号',
    titleEn: 'GHOST IS SEARCHING FOR A VISITOR SIGNAL',
    guideCn: '进入相机视野并保持面部可见。',
    guideEn: 'Enter the camera field and keep your face visible.',
  },
  face_stabilizing: {
    chapter: '04', signal: 'SIGNAL DETECTED', phase: '校验 / VERIFYING', step: 2,
    titleCn: '检测到未知意识信号',
    titleEn: 'UNKNOWN CONSCIOUSNESS SIGNAL DETECTED',
    guideCn: '保持静止两秒，正在确认目标身份。',
    guideEn: 'Hold still while the target signature is verified.',
  },
  face_centering: {
    chapter: '04', signal: 'TARGET LOCKING', phase: '锁定 / LOCKING', step: 2,
    titleCn: '目标锁定，特工档案生成中',
    titleEn: 'TARGET LOCKED // BUILDING AGENT DOSSIER',
    guideCn: '保持注视，允许视觉轴跟随面部信号。',
    guideEn: 'Maintain eye contact while the optical axis tracks your signal.',
  },
  face_following: {
    chapter: '04', signal: 'TRACKING LIVE', phase: '追踪 / TRACKING', step: 2,
    titleCn: '意识特征采集中',
    titleEn: 'CAPTURING CONSCIOUSNESS SIGNATURE',
    guideCn: '视觉链路已建立，Ghost 正在实时跟随。',
    guideEn: 'Visual link established. Ghost is tracking you in real time.',
  },
  face_centered: {
    chapter: '04', signal: 'PROFILE CAPTURED', phase: '建档 / PROFILING', step: 2,
    titleCn: '访客特工档案已捕获',
    titleEn: 'VISITOR AGENT PROFILE CAPTURED',
    guideCn: '扫描完成，正在写入 Ghost 记忆网络。',
    guideEn: 'Scan complete. Writing the profile into Ghost memory.',
  },
  face_gesture: {
    chapter: '04', signal: 'BEHAVIORAL SCAN', phase: '观察 / INSPECTION', step: 2,
    titleCn: 'Ghost 正在打量你的 Shell',
    titleEn: 'GHOST IS INSPECTING YOUR SHELL',
    guideCn: '档案已捕获，Ghost 正在扫描。',
    guideEn: 'Profile captured. Ghost orbit-scans, tilts to inspect, then nods twice.',
  },
  face_not_found: {
    chapter: '04', signal: 'SIGNAL LOST', phase: '丢失 / SIGNAL LOST', step: 2, tone: 'danger',
    titleCn: '目标信号已丢失',
    titleEn: 'VISITOR SIGNAL LOST',
    guideCn: '返回相机视野，重新建立视觉连接。',
    guideEn: 'Return to the camera field to restore the visual link.',
  },
  dancing: {
    chapter: '05', signal: 'GHOST AWAKENED', phase: '苏醒 / AWAKENED', step: 3, tone: 'success',
    titleCn: 'Ghost 已完成意识重构',
    titleEn: 'GHOST CONSCIOUSNESS RECONSTRUCTION COMPLETE',
    guideCn: '运动协议解除，正在展示新生 Shell。',
    guideEn: 'Motion protocol released. Witness the awakened shell.',
  },
  done: {
    chapter: '06', signal: 'DOSSIER ARCHIVED', phase: '完成 / COMPLETE', step: 4, tone: 'success',
    titleCn: '档案写入完成：信号可信',
    titleEn: 'DOSSIER ARCHIVED // SIGNAL TRUSTED',
    guideCn: '本次深网连接完成，等待下一位访客。',
    guideEn: 'Deep-net session complete. Awaiting the next visitor.',
  },
  returning_home: {
    chapter: '07', signal: 'MEMORY PURGE', phase: '回收 / RETURNING', step: -1,
    titleCn: '意识碎片正在回收',
    titleEn: 'RECLAIMING CONSCIOUSNESS FRAGMENTS',
    guideCn: '壳层正在返回待机位置，请保持安全距离。',
    guideEn: 'The shell is returning to standby. Keep the area clear.',
  },
  stuck: {
    chapter: 'ERR', signal: 'MOTION LINK BLOCKED', phase: '受阻 / BLOCKED', step: -1, tone: 'danger',
    titleCn: '运动链路受阻',
    titleEn: 'SHELL MOTION LINK IS BLOCKED',
    guideCn: '请清除机械臂周围障碍，安全协议已释放关节。',
    guideEn: 'Clear the arm workspace. Safety protocol has released the joints.',
  },
  aborted: {
    chapter: 'ERR', signal: 'LINK TERMINATED', phase: '终止 / ABORTED', step: -1, tone: 'danger',
    titleCn: '意识连接已中断',
    titleEn: 'CONSCIOUSNESS LINK TERMINATED',
    guideCn: '当前任务已经终止，可重新启动交互协议。',
    guideEn: 'Mission aborted. The interaction protocol may be restarted.',
  },
};

const CONTROL_COPY = {
  start: {
    pendingCn: '正在建立意识连接', pendingEn: 'OPENING CONSCIOUSNESS LINK',
    successCn: '启动指令已确认', successEn: 'START COMMAND ACCEPTED',
  },
  abort: {
    pendingCn: '正在终止当前任务', pendingEn: 'TERMINATING ACTIVE MISSION',
    successCn: '中止指令已确认', successEn: 'ABORT COMMAND ACCEPTED',
  },
  return_home: {
    pendingCn: '正在请求壳层归位', pendingEn: 'REQUESTING SHELL RETURN',
    successCn: '归位指令已确认', successEn: 'RETURN-HOME COMMAND ACCEPTED',
  },
  mock_solve: {
    pendingCn: '正在注入模拟轨迹', pendingEn: 'INJECTING MOCK TRAJECTORY',
    successCn: '秘密姿态轨迹已启动', successEn: 'SECRET-POSE TRAJECTORY STARTED',
  },
};

const connDot = document.getElementById('conn-dot');
const connText = document.getElementById('conn-text');
const phaseBadge = document.getElementById('phase-badge');
const foundCount = document.getElementById('found-count');
const grid = document.getElementById('joint-grid');
const armLayout = document.getElementById('arm-layout');
const cameraFeed = document.getElementById('camera-feed');
const cameraPanel = document.getElementById('camera-panel');
const cameraToggle = document.getElementById('camera-toggle');
const meshPanel = document.getElementById('mesh-panel');
const voiceTerminal = document.getElementById('voice-terminal');
const voiceText = document.getElementById('voice-text');
const voicePacket = document.getElementById('voice-packet');
const missionPanel = document.getElementById('mission-panel');
const missionChapter = document.getElementById('mission-chapter');
const missionSignal = document.getElementById('mission-signal');
const missionTitleCn = document.getElementById('mission-title-cn');
const missionTitleEn = document.getElementById('mission-title-en');
const missionGuideCn = document.getElementById('mission-guide-cn');
const missionGuideEn = document.getElementById('mission-guide-en');
const missionSteps = [...document.querySelectorAll('#mission-route li')];
const controlButtons = [...document.querySelectorAll('[data-control]')];
const controlFeedback = document.getElementById('control-feedback');
const controlStatusCn = document.getElementById('control-status-cn');
const controlStatusEn = document.getElementById('control-status-en');

let lastGoodAt = 0;
let cardsBuilt = false;
let cameraCollapsed = false;
let cameraRevealed = false; // shown only after the success-pose turn completes
let lastTtsSequence = 0;
let voicePulseTimer = null;
let lastNarrativeKey = '';

function setControlStatus(tone, chinese, english) {
  controlFeedback.dataset.tone = tone;
  controlStatusCn.textContent = chinese;
  controlStatusEn.textContent = english;
}

async function sendControl(command) {
  const copy = CONTROL_COPY[command];
  if (!copy || controlButtons.some((button) => button.disabled)) return;

  controlButtons.forEach((button) => { button.disabled = true; });
  controlFeedback.setAttribute('aria-busy', 'true');
  setControlStatus('pending', copy.pendingCn, copy.pendingEn);

  try {
    const response = await fetch(`/api/control/${command}`, {
      method: 'POST',
      headers: { Accept: 'application/json' },
    });
    const result = await response.json();
    if (!response.ok || result.ok !== true) {
      throw new Error(result.message || `HTTP ${response.status}`);
    }
    setControlStatus('success', copy.successCn, copy.successEn);
  } catch (error) {
    setControlStatus('error', '指令未执行', error.message || 'CONTROL REQUEST FAILED');
  } finally {
    controlFeedback.removeAttribute('aria-busy');
    controlButtons.forEach((button) => { button.disabled = false; });
  }
}

controlButtons.forEach((button) => {
  button.addEventListener('click', () => sendControl(button.dataset.control));
});

cameraToggle.addEventListener('click', () => {
  cameraCollapsed = !cameraCollapsed;
  cameraPanel.classList.toggle('collapsed', cameraCollapsed);
  cameraToggle.setAttribute('aria-expanded', String(!cameraCollapsed));
});

function buildCards(joints) {
  grid.innerHTML = '';
  joints.forEach((name, i) => {
    const number = String(i + 1).padStart(2, '0');
    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card-${name}`;
    card.innerHTML = `
      <div class="card-head">
        <div class="fragment-name">
          <span>GHOST FRAGMENT ${number}</span>
          <strong>意识碎片 ${number}</strong>
          <small>JOINT NODE // J${i + 1}</small>
        </div>
        <span class="joint-pct" id="pct-${name}">--%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" id="bar-${name}"></div></div>
      <div class="card-foot">
        <span id="dist-${name}">偏差 / OFFSET --</span>
        <span id="lock-${name}">搜索中 / SEEKING</span>
      </div>`;
    grid.appendChild(card);
  });
  cardsBuilt = true;
}

function renderNarrative(state) {
  const story = NARRATIVE[state.phase] || {
    ...NARRATIVE.idle,
    signal: 'UNKNOWN PHASE',
    phase: `${state.phase ?? '--'}`,
  };

  const found = state.found_count ?? 0;
  const total = state.total ?? 0;
  const narrativeKey = `${state.phase ?? 'unknown'}:${found}:${total}`;
  if (narrativeKey === lastNarrativeKey) return;
  lastNarrativeKey = narrativeKey;

  missionChapter.textContent = story.chapter;
  missionSignal.textContent = story.signal;
  missionTitleCn.textContent = story.titleCn;
  missionTitleEn.textContent = story.titleEn;

  if (state.phase === 'searching' && found > 0 && total > 0) {
    missionGuideCn.textContent = `已回收 ${found}/${total} 枚碎片。继续掰动未锁定的关节。`;
    missionGuideEn.textContent = `${found}/${total} fragments recovered. Keep moving the unlocked joints.`;
  } else {
    missionGuideCn.textContent = story.guideCn;
    missionGuideEn.textContent = story.guideEn;
  }

  phaseBadge.textContent = `系统阶段 / PHASE: ${story.phase}`;
  missionPanel.classList.toggle('tone-danger', story.tone === 'danger');
  missionPanel.classList.toggle('tone-success', story.tone === 'success');
  missionPanel.dataset.phase = state.phase ?? 'unknown';
  missionSteps.forEach((node, index) => {
    node.classList.toggle('active', story.step === index);
    node.classList.toggle('complete', story.step === 4 || (story.step >= 0 && index < story.step));
  });
}

function render(state) {
  renderNarrative(state);
  foundCount.innerHTML = `碎片回收 / FRAGMENTS <b>${state.found_count ?? 0}</b>/<b>${state.total ?? 0}</b>`;

  const total = state.total ?? 0;
  const allFound = total > 0 && (state.found_count ?? 0) >= total;
  // camera_ready is latched by the orchestrator at the exact point where the
  // measured arm pose passes success validation. Keep a fallback for an old
  // backend during rolling restarts.
  cameraRevealed = state.camera_ready === true ||
    (state.camera_ready == null && allFound && CAMERA_VISIBLE_PHASES.has(state.phase));
  armLayout.classList.toggle('camera-mode-hidden', cameraRevealed);
  cameraPanel.classList.toggle('locked', !cameraRevealed);

  const reconstruction = state.reconstruction || {};
  const reconstructionStatus = String(reconstruction.status || 'idle');
  const reconstructionVisible = cameraRevealed &&
    reconstruction.enabled !== false &&
    !['', 'idle'].includes(reconstructionStatus);
  meshPanel.classList.toggle('stage-hidden', !reconstructionVisible);
  meshPanel.setAttribute('aria-hidden', String(!reconstructionVisible));
  window.ghostReconstructionState = reconstruction;
  window.dispatchEvent(new CustomEvent('ghost-reconstruction-status', {
    detail: reconstruction,
  }));

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
      dist == null ? '偏差 / OFFSET --' : `偏差 / OFFSET ${dist.toFixed(3)} rad`;
    document.getElementById(`lock-${name}`).textContent =
      isLocked ? '已回收 / RECOVERED' : '搜索中 / SEEKING';
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

function renderTts(caption) {
  const sequence = Number(caption.sequence) || 0;
  const text = typeof caption.text === 'string' ? caption.text.trim() : '';
  if (!text || sequence <= lastTtsSequence) return;

  lastTtsSequence = sequence;
  voiceText.textContent = text;
  voicePacket.textContent = `VOICE PACKET ${String(sequence).padStart(3, '0')}`;
  voiceTerminal.classList.remove('idle', 'receiving');

  // Force one layout read so repeated packets restart the short decode/glitch
  // animation even when two stage announcements arrive close together.
  void voiceTerminal.offsetWidth;
  voiceTerminal.classList.add('receiving');
  window.clearTimeout(voicePulseTimer);
  voicePulseTimer = window.setTimeout(() => {
    voiceTerminal.classList.remove('receiving');
    voicePacket.textContent = `PACKET ${String(sequence).padStart(3, '0')} // LATCHED`;
  }, 1200);
}

async function pollTts() {
  try {
    const res = await fetch('/api/tts', { cache: 'no-store' });
    if (res.ok) renderTts(await res.json());
  } catch (err) {
    // Main connection indicator already reports whether the web bridge is
    // reachable; keep the last spoken line visible during a brief hiccup.
  }
  setTimeout(pollTts, TTS_POLL_MS);
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
pollTts();
pollCamera();
