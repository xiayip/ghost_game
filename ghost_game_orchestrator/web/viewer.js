// Ghost Game 3D arm viewer.
//
// Primary path: load the REAL robot description (meshes + kinematics) via
// URDFLoader from /robot/robot.urdf, which ghost_game_web_monitor expands
// from zephyr_arm.urdf.xacro with real xacro at startup - see
// web_monitor.py's _expand_urdf. Mesh package:// URIs are resolved to
// /robot/pkg/<package>/<relpath>, which proxies straight from that
// package's installed share directory (no meshes duplicated into this repo).
//
// Fallback path: if the URDF isn't available (xacro/package missing, or no
// internet for the three.js/urdf-loader CDN), draw a lightweight kinematic
// skeleton instead, built from the same joint origins/axes (dumped
// 2026-09-18 via xacro.process_file) so the pose is still geometrically
// correct even without the real meshes. Removed from the scene the moment
// the real model loads - the two never coexist.
//
// "Ghost" reveal: every link is translucent/dim by default; once its
// governing joint is `locked` in ~/state, that link's meshes flip to a
// solid glowing --success-400 green (matching the progress-card "locked"
// styling) instead of the default cyan-tinted ghost look.
//
// Driven by window.ghostGameViewer.{setJointAngles, setLockedJoints}, which
// app.js calls every poll with the live measured angles and locked flags.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const CHAIN = [
  { name: 'joint1', xyz: [-0.00034283, -0.00098683, 0.075], rpy: [0, 0, 0], axis: [0, 0, -1] },
  { name: 'joint2', xyz: [0.020343, 0.027237, 0.07], rpy: [-1.5708, 0, 0], axis: [0, 0, 1] },
  { name: 'joint3', xyz: [-0.236, 0, 0], rpy: [0, 0, 0], axis: [0, 0, -1] },
  { name: 'joint4', xyz: [0.228, -0.072746, 0.0045], rpy: [0, 0, 0], axis: [0, 0, -1] },
  { name: 'joint5', xyz: [0.087, -0.048, -0.03075], rpy: [-1.5708, 0, 0], axis: [0, 0, -1] },
  { name: 'joint6', xyz: [0.0365, 0, 0.048], rpy: [0, 1.5708, 0], axis: [0, 0, -1] },
  { name: null, xyz: [0.017519, 0, 0], rpy: [0, 0, 0], axis: null }, // end-effector stub
];

// joint<N>'s own link in the expanded URDF (see zephyr_arm.urdf.xacro) - used
// to find which real meshes to light up once that joint is found.
const JOINT_LINK_NAMES = {
  joint1: 'link1', joint2: 'link2', joint3: 'link3',
  joint4: 'link4', joint5: 'link5', joint6: 'link6',
};

const BONE_RADIUS = 0.018;
const JOINT_RADIUS = 0.03;
const CYAN = 0x00d4ff;
const GHOST_COLOR = 0x2a3140;
const LOCKED_COLOR = 0x00f5a0; // --success-400, matches the progress-card "locked" state
const GHOST_OPACITY = 0.16;

function ghostMaterial() {
  return new THREE.MeshStandardMaterial({
    color: GHOST_COLOR, metalness: 0.35, roughness: 0.55,
    transparent: true, opacity: GHOST_OPACITY, depthWrite: false,
    emissive: CYAN, emissiveIntensity: 0.25,
  });
}

function setMaterialLocked(mat, isLocked) {
  mat.color.set(isLocked ? LOCKED_COLOR : GHOST_COLOR);
  mat.emissive.set(isLocked ? LOCKED_COLOR : CYAN);
  mat.emissiveIntensity = isLocked ? 0.6 : 0.25;
  mat.opacity = isLocked ? 1 : GHOST_OPACITY;
  mat.transparent = !isLocked;
  mat.depthWrite = isLocked;
}

function makeBone(target) {
  const length = target.length();
  if (length < 1e-6) return null;
  const geo = new THREE.CylinderGeometry(BONE_RADIUS, BONE_RADIUS, length, 12);
  const mesh = new THREE.Mesh(geo, ghostMaterial());
  // Cylinder is built along +Y; align it from the origin to `target`.
  const mid = target.clone().multiplyScalar(0.5);
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), target.clone().normalize());
  return mesh;
}

function makeJointMarker() {
  const geo = new THREE.SphereGeometry(JOINT_RADIUS, 16, 16);
  return new THREE.Mesh(geo, ghostMaterial());
}

function buildSkeleton(parent) {
  const group = new THREE.Group();
  parent.add(group);

  const rotors = {};   // joint name -> { group, axis }
  const jointMats = {}; // joint name -> materials to light up when found
  let cur = group;
  CHAIN.forEach((joint) => {
    // Bone for the link BEFORE this joint's origin offset, drawn in the
    // previous joint's rotating frame (so it swings with that joint), and
    // credited to THIS joint's "found" reveal.
    const bone = makeBone(new THREE.Vector3(...joint.xyz));
    if (bone) cur.add(bone);

    const originGroup = new THREE.Group();
    originGroup.position.set(...joint.xyz);
    originGroup.rotation.set(joint.rpy[0], joint.rpy[1], joint.rpy[2], 'XYZ');
    cur.add(originGroup);

    const rotor = new THREE.Group();
    originGroup.add(rotor);
    if (joint.name) {
      const marker = makeJointMarker();
      rotor.add(marker);
      rotors[joint.name] = { group: rotor, axis: new THREE.Vector3(...joint.axis) };
      jointMats[joint.name] = [marker.material, ...(bone ? [bone.material] : [])];
    }
    cur = rotor;
  });

  return {
    group,
    setJointAngles(angles) {
      for (const [name, rotor] of Object.entries(rotors)) {
        const angle = angles[name];
        if (typeof angle === 'number' && Number.isFinite(angle)) {
          rotor.group.quaternion.setFromAxisAngle(rotor.axis, angle);
        }
      }
    },
    setLockedJoints(locked) {
      for (const [name, mats] of Object.entries(jointMats)) {
        mats.forEach((mat) => setMaterialLocked(mat, !!locked[name]));
      }
    },
  };
}

function collectLinkMeshes(linkObj) {
  const meshes = [];
  (function walk(node) {
    // Stop at the next *movable* joint - that subtree belongs to it. Fixed
    // sub-joints (flanges, mount plates, ...) are rigidly welded to this
    // link, so their meshes are visually part of this link and should still
    // count toward its reveal (link1/link6 in particular are tiny on their
    // own and would otherwise barely show any highlight).
    if (node.isURDFJoint && node.jointType !== 'fixed') return;
    if (node.isMesh) meshes.push(node);
    node.children.forEach(walk);
  })(linkObj);
  return meshes;
}

function loadRealRobot(root, onLoaded, onFailed) {
  // STL meshes are fetched asynchronously per-visual, AFTER URDFLoader's own
  // onComplete callback already fired with the (still mesh-less) joint tree
  // - traversing for meshes right there finds nothing. A LoadingManager
  // tracks every sub-fetch (URDFLoader passes it into each STLLoader) and
  // its onLoad only fires once they've *all* actually finished.
  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  loader.packages = (pkg) => `/robot/pkg/${pkg}`;

  let robot = null;

  manager.onLoad = () => {
    if (!robot) return;
    robot.traverse((child) => {
      if (child.isMesh) child.material = ghostMaterial();
    });

    const jointMats = {};
    for (const [jointName, linkName] of Object.entries(JOINT_LINK_NAMES)) {
      const linkObj = robot.links && robot.links[linkName];
      if (linkObj) {
        jointMats[jointName] = collectLinkMeshes(linkObj).map((m) => m.material);
      } else {
        console.warn(`[ghost] link "${linkName}" for ${jointName} not found in robot.links`, Object.keys(robot.links || {}));
      }
    }
    console.info('[ghost] mesh count per joint:', Object.fromEntries(
      Object.entries(jointMats).map(([k, v]) => [k, v.length])
    ));

    onLoaded({
      setJointAngles(angles) {
        for (const [name, angle] of Object.entries(angles)) {
          if (typeof angle === 'number' && Number.isFinite(angle)) {
            robot.setJointValue(name, angle);
          }
        }
      },
      setLockedJoints(locked) {
        for (const [name, mats] of Object.entries(jointMats)) {
          mats.forEach((mat) => setMaterialLocked(mat, !!locked[name]));
        }
      },
    });
  };

  loader.load(
    '/robot/robot.urdf',
    (loadedRobot) => {
      robot = loadedRobot;
      root.add(robot);
    },
    undefined,
    onFailed,
  );
}

function disposeGroup(group) {
  group.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  });
}

function init() {
  const container = document.getElementById('viewport');
  if (!container) return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 10);
  camera.position.set(0.7, 0.55, 0.7);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.2, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0x8899aa, 0.7));
  const rim = new THREE.DirectionalLight(0x00d4ff, 1.1);
  rim.position.set(-1, 1.5, -1);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(1, 1, 1);
  scene.add(fill);

  scene.add(new THREE.GridHelper(1.4, 14, 0x006b87, 0x1b2029));

  // Fixed base pedestal at the world origin (joint1's parent, base_link).
  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.08, 0.03, 24),
    new THREE.MeshStandardMaterial({ color: 0x1b2029, emissive: CYAN, emissiveIntensity: 0.15 }));
  base.position.y = 0.015;
  scene.add(base);

  const armRoot = new THREE.Group();
  armRoot.rotation.x = -Math.PI / 2; // URDF Z-up -> three.js Y-up
  scene.add(armRoot);

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    // The whole arm layout is display:none while the post-turn camera is
    // visible. Avoid rendering a hidden WebGL scene at 60 Hz during tracking.
    if (document.hidden || container.offsetParent === null) return;
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Skeleton first so there's always something on screen immediately;
  // torn down and replaced by the real mesh model once/if it loads - the
  // two are never shown at the same time.
  const skeleton = buildSkeleton(armRoot);
  window.ghostGameViewer = skeleton;

  loadRealRobot(
    armRoot,
    (real) => {
      console.info('Ghost game viewer: real robot mesh loaded');
      armRoot.remove(skeleton.group);
      disposeGroup(skeleton.group);
      window.ghostGameViewer = real;
    },
    (err) => {
      console.warn('Ghost game viewer: falling back to schematic skeleton', err);
    },
  );
}

init();
