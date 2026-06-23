"use client";
import { useEffect, useRef } from "react";
import * as THREE from "three";
// @ts-ignore
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
// @ts-ignore
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
// @ts-ignore
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
// @ts-ignore
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type Scenario = "empty" | "stand" | "walk" | "fall";

/**
 * RuView-style WiFi observatory. Auto-animated background by default, or fully
 * controllable (orbit + scenario + bloom + wave speed) when `interactive`.
 */
export default function Observatory3D({
  className = "",
  interactive = false,
  scenario = "walk",
  bloom = 1.15,
  waveSpeed = 1,
  autoRotate = false,
}: {
  className?: string;
  interactive?: boolean;
  scenario?: Scenario;
  bloom?: number;
  waveSpeed?: number;
  autoRotate?: boolean;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  // live params the render loop reads
  const params = useRef({ scenario, bloom, waveSpeed, autoRotate, interactive });
  const api = useRef<any>(null);

  // push prop changes into the running scene
  useEffect(() => {
    params.current = { scenario, bloom, waveSpeed, autoRotate, interactive };
    if (api.current) {
      api.current.bloom.strength = bloom;
      if (api.current.controls) api.current.controls.autoRotate = autoRotate;
    }
  }, [scenario, bloom, waveSpeed, autoRotate, interactive]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = mount.clientWidth || 800, h = mount.clientHeight || 600;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, w / h, 0.1, 200);
    camera.position.set(0.5, 8.5, 15);
    camera.lookAt(0, 1.6, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(46, 46), new THREE.MeshBasicMaterial({ color: 0x05101a }));
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // green presence heatmap
    const N = 150;
    const tileGeo = new THREE.PlaneGeometry(0.5, 0.5);
    const tileMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.92 });
    const tiles = new THREE.InstancedMesh(tileGeo, tileMat, N);
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();
    const figureXZ = new THREE.Vector2(3.4, 2.2);
    const tileInfo: { x: number; z: number }[] = [];
    for (let i = 0; i < N; i++) {
      const gx = (Math.random() - 0.5) * 26;
      const gz = (Math.random() - 0.5) * 22 + 2;
      tileInfo.push({ x: gx, z: gz });
      dummy.position.set(gx, 0.02, gz);
      dummy.rotation.x = -Math.PI / 2;
      const s = 0.5 + Math.random() * 0.9;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      tiles.setMatrixAt(i, dummy.matrix);
    }
    scene.add(tiles);

    const paintHeat = (fx: number, fz: number, gain: number) => {
      for (let i = 0; i < N; i++) {
        const d = Math.hypot(tileInfo[i].x - fx, tileInfo[i].z - fz);
        const bright = Math.max(0.08, 1 - d / 12) * gain;
        col.setRGB(0.05 * bright, 0.9 * bright, 0.32 * bright);
        tiles.setColorAt(i, col);
      }
      if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;
    };
    paintHeat(figureXZ.x, figureXZ.y, 0.9);

    const emitter = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.MeshBasicMaterial({ color: 0xff8a2c }));
    emitter.position.set(0, 0.45, 0);
    scene.add(emitter);

    const domes: { mesh: THREE.Mesh; phase: number }[] = [];
    const domeGeo = new THREE.IcosahedronGeometry(1, 3);
    for (let i = 0; i < 4; i++) {
      const m = new THREE.Mesh(domeGeo, new THREE.MeshBasicMaterial({ color: 0x3a6bff, wireframe: true, transparent: true, opacity: 0.5 }));
      scene.add(m);
      domes.push({ mesh: m, phase: i / 4 });
    }

    const figure = new THREE.Group();
    const bodyMat = new THREE.MeshBasicMaterial({ color: 0x2effa0 });
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.62, 1.7, 8, 18), bodyMat);
    body.position.y = 1.5;
    figure.add(body);
    const kpGeo = new THREE.SphereGeometry(0.085, 10, 10);
    const kpMat = new THREE.MeshBasicMaterial({ color: 0xffe27a });
    [[0, 2.5, 0.45], [0, 2.0, 0.5], [0.28, 1.7, 0.5], [-0.28, 1.7, 0.5], [0, 1.4, 0.55], [0.22, 1.0, 0.5], [-0.22, 1.0, 0.5], [0, 0.6, 0.5]].forEach((p) => {
      const s = new THREE.Mesh(kpGeo, kpMat);
      s.position.set(p[0], p[1], p[2]);
      figure.add(s);
    });
    figure.position.set(figureXZ.x, 0, figureXZ.y);
    scene.add(figure);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), params.current.bloom, 0.55, 0.02);
    composer.addPass(bloomPass);

    let controls: any = null;
    if (params.current.interactive) {
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enablePan = false;
      controls.minDistance = 8;
      controls.maxDistance = 30;
      controls.maxPolarAngle = Math.PI / 2.05;
      controls.target.set(0, 1.4, 0);
      controls.autoRotate = params.current.autoRotate;
      controls.autoRotateSpeed = 0.8;
    }

    api.current = { bloom: bloomPass, controls };

    const resize = () => {
      w = mount.clientWidth || w; h = mount.clientHeight || h;
      camera.aspect = w / h; camera.updateProjectionMatrix();
      renderer.setSize(w, h); composer.setSize(w, h);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    const clock = new THREE.Clock();
    let raf = 0;
    let lastScenario = "";
    const render = () => {
      const t = clock.getElapsedTime();
      const p = params.current;
      // domes
      const speed = 0.0028 * p.waveSpeed;
      for (const d of domes) {
        d.phase += speed;
        if (d.phase > 1) d.phase -= 1;
        d.mesh.scale.setScalar(1 + d.phase * 9.5);
        (d.mesh.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.55 * (1 - d.phase));
      }
      emitter.scale.setScalar(1 + Math.sin(t * 3) * 0.06);

      // scenario
      const sc = p.scenario;
      if (sc !== lastScenario) {
        lastScenario = sc;
        figure.visible = sc !== "empty";
        bodyMat.color.set(sc === "fall" ? 0xff4d6d : 0x2effa0);
        body.rotation.z = sc === "fall" ? Math.PI / 2 : 0;
        body.position.y = sc === "fall" ? 0.7 : 1.5;
        paintHeat(figureXZ.x, figureXZ.y, sc === "empty" ? 0.12 : sc === "fall" ? 1.1 : 0.9);
      }
      if (sc === "walk") {
        figure.position.x = figureXZ.x + Math.sin(t * 1.1) * 2.2;
        figure.position.y = Math.abs(Math.sin(t * 3)) * 0.12;
        paintHeat(figure.position.x, figureXZ.y, 0.95);
      } else if (sc === "stand") {
        figure.position.x = figureXZ.x;
        figure.position.y = Math.sin(t * 1.4) * 0.06;
      } else if (sc === "fall") {
        figure.position.x = figureXZ.x;
        figure.position.y = 0;
      }

      if (controls) controls.update();
      else if (!reduce) {
        camera.position.x = 0.5 + Math.sin(t * 0.18) * 1.6;
        camera.position.z = 15 + Math.cos(t * 0.18) * 0.8;
        camera.lookAt(0, 1.6, 0);
      }
      composer.render();
      if (!reduce || controls) raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      if (controls) controls.dispose();
      renderer.dispose();
      domeGeo.dispose(); tileGeo.dispose(); kpGeo.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
      api.current = null;
    };
  }, []);

  return <div ref={mountRef} className={className} aria-hidden />;
}
