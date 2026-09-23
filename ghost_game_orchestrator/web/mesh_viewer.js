import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const INFO_POLL_MS = 1200;

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
  const container = document.getElementById('mesh-viewport');
  const status = document.getElementById('mesh-status');
  const statusText = status && status.querySelector('b');
  const placeholder = document.getElementById('mesh-placeholder');
  const meta = document.getElementById('mesh-meta');
  if (!container || !status || !statusText || !placeholder || !meta) return;

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

  function setStatus(tone, text, detail) {
    status.dataset.tone = tone;
    statusText.textContent = text;
    if (detail) meta.textContent = detail;
  }

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
    placeholder.classList.remove('hidden');
    setStatus('loading', 'DECODING NEURAL GEOMETRY', 'GLB STREAM RECEIVED // PARSING');

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
          placeholder.classList.add('hidden');
          const megabytes = (Number(info.bytes || 0) / 1048576).toFixed(2);
          const source = info.source === 'local' ? 'LOCAL SAMPLE' : 'ROS URL';
          setStatus('ready', 'MODEL LINK STABLE',
            `${source} // V${version} // ${megabytes} MB`);
        } catch (error) {
          loadingVersion = 0;
          failedVersion = version;
          disposeObject(gltf.scene);
          setStatus('error', 'MODEL GEOMETRY INVALID', error.message);
        }
      },
      (event) => {
        if (!event.total) return;
        const percent = Math.min(100, Math.round(event.loaded / event.total * 100));
        meta.textContent = `RECEIVING GLB // ${percent}%`;
      },
      (error) => {
        loadingVersion = 0;
        failedVersion = version;
        setStatus('error', 'MODEL STREAM FAILED', error.message || 'GLB LOAD ERROR');
      },
    );
  }

  async function pollModel() {
    try {
      const response = await fetch('/api/mesh/info', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const info = await response.json();
      const version = Number(info.version) || 0;
      if (info.model_url && version !== loadedVersion &&
          version !== loadingVersion && version !== failedVersion) {
        loadModel(info);
      } else if (!info.model_url) {
        const tone = info.status === 'error' ? 'error' : 'loading';
        setStatus(tone,
          info.status === 'error' ? 'MODEL LINK ERROR' : 'AWAITING MODEL LINK',
          info.message || 'TOPIC STANDBY // NO GLB');
      } else if (info.status === 'downloading') {
        setStatus('loading', 'UPDATING NEURAL GEOMETRY', 'ROS URL // DOWNLOADING');
      } else if (info.status === 'error' && loadedVersion === version) {
        // Keep rendering the last good model while reporting the failed update.
        setStatus('error', 'MODEL UPDATE FAILED', info.message || 'LAST MODEL RETAINED');
      }
    } catch (error) {
      setStatus('error', 'MODEL BRIDGE OFFLINE', error.message || 'NO SIGNAL');
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
