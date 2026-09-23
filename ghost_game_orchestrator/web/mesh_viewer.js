import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const INFO_POLL_MS = 500;

const FLUX_PROGRESS = {
  submitted: [5, '人脸档案已捕获', 'FACE PROFILE CAPTURED // QUEUING'],
  accepted: [10, '重建请求已接收', 'RECONSTRUCTION REQUEST ACCEPTED'],
  encoding: [15, '正在编码视觉档案', 'ENCODING VISUAL PROFILE'],
  running: [30, '正在生成三维输入图', 'SYNTHESIZING RECONSTRUCTION PORTRAIT'],
  success: [40, '二维档案生成完成', 'PORTRAIT READY // STARTING 3D PIPELINE'],
};

function clampPercent(value) {
  return Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
}

function disposeObject(root) {
  root.traverse((object) => {
    if (object.geometry) object.geometry.dispose();
    const materials = Array.isArray(object.material)
      ? object.material : object.material ? [object.material] : [];
    materials.forEach((material) => {
      Object.values(material).forEach((value) => {
        if (value && value.isTexture) value.dispose();
      });
      material.dispose();
    });
  });
}

function initMeshViewer() {
  const panel = document.getElementById('mesh-panel');
  const container = document.getElementById('mesh-viewport');
  const status = document.getElementById('mesh-status');
  const statusText = status && status.querySelector('b');
  const placeholder = document.getElementById('mesh-placeholder');
  const progressTitle = document.getElementById('mesh-progress-title');
  const progressStage = document.getElementById('mesh-progress-stage');
  const progressTrack = document.getElementById('mesh-progress-track');
  const progressFill = document.getElementById('mesh-progress-fill');
  const progressPercent = document.getElementById('mesh-progress-percent');
  const meta = document.getElementById('mesh-meta');
  if (!panel || !container || !status || !statusText || !placeholder ||
      !progressTitle || !progressStage || !progressTrack || !progressFill ||
      !progressPercent || !meta) return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
  camera.position.set(1.8, 1.25, 2.25);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.9;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xbfefff, 0x080b10, 1.7));
  const key = new THREE.DirectionalLight(0xffffff, 2.8);
  key.position.set(3, 4, 2);
  key.castShadow = true;
  scene.add(key);
  const cyanRim = new THREE.DirectionalLight(0x00d4ff, 3.2);
  cyanRim.position.set(-3, 1.5, -2);
  scene.add(cyanRim);
  const magentaRim = new THREE.DirectionalLight(0xff3d71, 1.2);
  magentaRim.position.set(2, 0.5, -3);
  scene.add(magentaRim);

  const platform = new THREE.Mesh(
    new THREE.CylinderGeometry(0.82, 0.9, 0.035, 64),
    new THREE.MeshStandardMaterial({
      color: 0x101821,
      metalness: 0.75,
      roughness: 0.28,
      emissive: 0x003d4d,
      emissiveIntensity: 0.8,
    }),
  );
  platform.position.y = -0.72;
  platform.receiveShadow = true;
  scene.add(platform);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.64, 0.68, 64),
    new THREE.MeshBasicMaterial({
      color: 0x00d4ff, transparent: true, opacity: 0.65,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -0.699;
  scene.add(ring);

  let currentModel = null;
  let loadedVersion = 0;
  let loadingVersion = 0;
  let failedVersion = 0;
  let lastFluxKey = '';
  let lastGenerationRequest = '';

  function setStatus(tone, text, detail) {
    status.dataset.tone = tone;
    statusText.textContent = text;
    if (detail) meta.textContent = detail;
  }

  function showProgress(percent, title, stage, detail, tone = 'loading') {
    const shown = clampPercent(percent);
    panel.classList.add('reconstructing');
    panel.classList.remove('model-ready');
    panel.classList.toggle('progress-error', tone === 'error');
    placeholder.classList.remove('hidden');
    progressTitle.textContent = title;
    progressStage.textContent = stage;
    progressFill.style.width = `${shown}%`;
    progressPercent.textContent = `${shown}%`;
    progressTrack.setAttribute('aria-valuenow', String(shown));
    setStatus(tone, tone === 'error' ? 'RECONSTRUCTION LINK ERROR' : stage,
      detail || `NEURAL RECONSTRUCTION // ${shown}%`);
  }

  function showPipelineError(percent, error) {
    showProgress(
      percent,
      '三维意识镜像生成失败',
      'RECONSTRUCTION PIPELINE INTERRUPTED',
      error || 'CHECK FLUX / TRIPO LINK',
      'error',
    );
  }

  function handleFluxStatus(reconstruction) {
    const statusName = String(reconstruction?.status || 'idle');
    const requestId = String(reconstruction?.request_id || 'pending');
    const keyName = `${requestId}:${statusName}`;
    if (keyName === lastFluxKey) return;
    lastFluxKey = keyName;
    if (statusName === 'idle' || !statusName) return;
    if (statusName === 'error') {
      showPipelineError(30, reconstruction?.error);
      return;
    }
    const stage = FLUX_PROGRESS[statusName] || FLUX_PROGRESS.submitted;
    showProgress(stage[0], stage[1], stage[2],
      `FLUX PREPROCESS // ${stage[0]}%`);
  }

  window.addEventListener('ghost-reconstruction-status', (event) => {
    handleFluxStatus(event.detail || {});
  });
  handleFluxStatus(window.ghostReconstructionState || {});

  function fitModel(model) {
    const initialBox = new THREE.Box3().setFromObject(model);
    if (initialBox.isEmpty()) throw new Error('GLB contains no visible geometry');
    const size = initialBox.getSize(new THREE.Vector3());
    const center = initialBox.getCenter(new THREE.Vector3());
    const longest = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(longest) || longest <= 0) {
      throw new Error('GLB has invalid bounds');
    }
    model.position.sub(center);
    model.scale.setScalar(1.35 / longest);
    model.updateMatrixWorld(true);

    const fittedBox = new THREE.Box3().setFromObject(model);
    const fittedSize = fittedBox.getSize(new THREE.Vector3());
    model.position.y += -0.69 - fittedBox.min.y;
    model.updateMatrixWorld(true);

    const radius = Math.max(fittedSize.length() * 0.62, 0.8);
    camera.near = Math.max(0.01, radius / 100);
    camera.far = Math.max(20, radius * 20);
    camera.position.set(radius * 1.35, radius * 0.72, radius * 1.65);
    camera.updateProjectionMatrix();
    controls.target.set(0, -0.05, 0);
    controls.minDistance = radius * 0.55;
    controls.maxDistance = radius * 5;
    controls.update();
  }

  function loadModel(info) {
    const version = Number(info.version) || 0;
    loadingVersion = version;
    failedVersion = 0;
    showProgress(99, '正在解码三维意识镜像',
      'DECODING NEURAL GEOMETRY', 'GLB STREAM RECEIVED // PARSING');

    new GLTFLoader().load(
      info.model_url,
      (gltf) => {
        if (loadingVersion !== version) {
          disposeObject(gltf.scene);
          return;
        }
        try {
          fitModel(gltf.scene);
          gltf.scene.traverse((object) => {
            if (object.isMesh) {
              object.castShadow = true;
              object.receiveShadow = true;
            }
          });
          if (currentModel) {
            scene.remove(currentModel);
            disposeObject(currentModel);
          }
          currentModel = gltf.scene;
          scene.add(currentModel);
          loadedVersion = version;
          loadingVersion = 0;
          progressFill.style.width = '100%';
          progressPercent.textContent = '100%';
          progressTrack.setAttribute('aria-valuenow', '100');
          placeholder.classList.add('hidden');
          panel.classList.remove('reconstructing', 'progress-error');
          panel.classList.add('model-ready');
          const megabytes = (Number(info.bytes || 0) / 1048576).toFixed(2);
          setStatus('ready', 'NEURAL TWIN MATERIALIZED',
            `LIVE RECONSTRUCTION // V${version} // ${megabytes} MB`);
        } catch (error) {
          loadingVersion = 0;
          failedVersion = version;
          disposeObject(gltf.scene);
          showPipelineError(99, error.message);
        }
      },
      (event) => {
        if (!event.total) return;
        const received = Math.min(100, Math.round(event.loaded / event.total * 100));
        meta.textContent = `RECEIVING GLB // ${received}%`;
      },
      (error) => {
        loadingVersion = 0;
        failedVersion = version;
        showPipelineError(99, error.message || 'GLB LOAD ERROR');
      },
    );
  }

  function renderGeneration(info) {
    const generation = info.generation || {};
    const generationStatus = String(generation.status || 'idle');
    const requestId = String(generation.request_id || '');
    const tripoProgress = clampPercent(generation.progress);
    if (requestId && requestId !== lastGenerationRequest) {
      lastGenerationRequest = requestId;
      failedVersion = 0;
    }

    if (generationStatus === 'error' || generationStatus === 'failed' ||
        generationStatus === 'cancelled') {
      showPipelineError(50 + Math.round(tripoProgress * 0.45), generation.error);
      return true;
    }
    if (generationStatus === 'preprocessing') {
      return true;
    }
    if (generationStatus === 'uploading') {
      showProgress(45, '正在上传特工视觉档案',
        'UPLOADING PROFILE TO TRIPO', 'IMAGE-TO-MESH LINK // 45%');
      return true;
    }
    if (generationStatus === 'queued') {
      showProgress(50, '三维生成任务排队中',
        'NEURAL GEOMETRY QUEUED', 'TRIPO TASK ACCEPTED // 50%');
      return true;
    }
    if (generationStatus === 'running') {
      const overall = 50 + Math.round(tripoProgress * 0.45);
      showProgress(overall, '正在重建三维意识镜像',
        `GENERATING NEURAL GEOMETRY // ${tripoProgress}%`,
        `TRIPO CORE // ${overall}%`);
      return true;
    }
    if (generationStatus === 'success') {
      if (info.status === 'downloading') {
        showProgress(98, '正在接收三维模型',
          'DOWNLOADING GENERATED GLB', 'SIGNED MODEL LINK // RECEIVING');
        return true;
      }
      if (info.source !== 'published_url') {
        showProgress(96, '三维生成完成，等待模型链路',
          'MESH COMPLETE // WAITING FOR GLB LINK', 'TRIPO CORE // 96%');
        return true;
      }
    }
    return false;
  }

  async function pollModel() {
    try {
      const response = await fetch('/api/mesh/info', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const info = await response.json();
      const version = Number(info.version) || 0;
      const pipelineHandled = renderGeneration(info);
      const liveModelReady = !pipelineHandled &&
        info.source === 'published_url' && info.model_url;

      if (liveModelReady && version !== loadedVersion &&
          version !== loadingVersion && version !== failedVersion) {
        loadModel(info);
      } else if (liveModelReady && version === loadedVersion) {
        if (panel.classList.contains('reconstructing')) {
          placeholder.classList.add('hidden');
          panel.classList.remove('reconstructing', 'progress-error');
          panel.classList.add('model-ready');
        }
      } else if (!pipelineHandled && info.status === 'error') {
        showPipelineError(95, info.message || 'MODEL LINK ERROR');
      }
    } catch (error) {
      if (!panel.classList.contains('stage-hidden')) {
        showPipelineError(0, error.message || 'MODEL BRIDGE OFFLINE');
      }
    }
    window.setTimeout(pollModel, INFO_POLL_MS);
  }

  function resize() {
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }
  new ResizeObserver(resize).observe(container);
  resize();

  function animate() {
    requestAnimationFrame(animate);
    if (document.hidden || container.offsetParent === null) return;
    controls.update();
    ring.rotation.z += 0.0018;
    renderer.render(scene, camera);
  }
  animate();
  pollModel();
}

initMeshViewer();
