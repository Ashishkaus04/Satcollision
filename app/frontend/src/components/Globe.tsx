import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type GlobePoint = { lat: number; lon: number; alt_km: number; color: string; size?: number; highlight?: boolean };
export type GlobeTrack = { points: [number, number, number][]; color: string };

const R_EARTH = 6378.137;
// Altitudes are exaggerated x3 so a 550 km shell is visibly off the surface.
const ALT_SCALE = 3;

function toVec(lat: number, lon: number, altKm: number) {
  const r = 1 + (altKm * ALT_SCALE) / R_EARTH;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 30);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.55, "rgba(255,255,255,0.95)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.beginPath();
  g.arc(32, 32, 30, 0, Math.PI * 2);
  g.fill();
  return new THREE.CanvasTexture(c);
}

/**
 * Three.js globe. Points are re-uploaded whenever ``points`` changes (once
 * per world tick); the camera, rotation and Earth stay put.
 */
export default function Globe({ points, track, emptyMessage, height = 460 }: {
  points: GlobePoint[];
  track?: GlobeTrack | null;
  emptyMessage?: string;
  height?: number;
}) {
  const mount = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{ scene: THREE.Scene; layer: THREE.Group; dot: THREE.Texture } | null>(null);

  useEffect(() => {
    const el = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    camera.position.set(0, 0.9, 3.6);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.6;
    controls.maxDistance = 8;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.35;

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(5, 3, 5);
    scene.add(sun);

    const earthMat = new THREE.MeshPhongMaterial({ color: 0x15233f, emissive: 0x050a14, shininess: 8 });
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 48), earthMat);
    scene.add(earth);
    new THREE.TextureLoader().load("/earth.jpg", (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      earthMat.map = tex;
      earthMat.color.set(0xffffff);
      earthMat.needsUpdate = true;
    }, undefined, () => { /* no texture: keep the plain globe */ });

    // Graticule
    const grid = new THREE.Group();
    const gridMat = new THREE.LineBasicMaterial({ color: 0x3e5a8f, transparent: true, opacity: 0.35 });
    for (let lat = -60; lat <= 60; lat += 30) {
      const pts = [];
      for (let lon = -180; lon <= 180; lon += 5) pts.push(toVec(lat, lon, 1));
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }
    for (let lon = -180; lon < 180; lon += 30) {
      const pts = [];
      for (let lat = -90; lat <= 90; lat += 5) pts.push(toVec(lat, lon, 1));
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }
    scene.add(grid);

    const layer = new THREE.Group();
    scene.add(layer);
    sceneRef.current = { scene, layer, dot: dotTexture() };

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    for (const child of [...s.layer.children]) {
      s.layer.remove(child);
      (child as any).geometry?.dispose?.();
      (child as any).material?.dispose?.();
    }
    const groups: Record<string, GlobePoint[]> = {};
    for (const p of points) {
      const key = `${p.highlight ? 1 : 0}|${p.size ?? (points.length > 1500 ? 0.028 : 0.055)}`;
      (groups[key] ||= []).push(p);
    }
    for (const [key, pts] of Object.entries(groups)) {
      const [hl, size] = key.split("|");
      const pos = new Float32Array(pts.length * 3);
      const col = new Float32Array(pts.length * 3);
      const c = new THREE.Color();
      pts.forEach((p, i) => {
        const v = toVec(p.lat, p.lon, p.alt_km);
        pos.set([v.x, v.y, v.z], i * 3);
        c.set(p.color);
        col.set([c.r, c.g, c.b], i * 3);
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      const mat = new THREE.PointsMaterial({
        size: hl === "1" ? 0.12 : Number(size), vertexColors: true, map: s.dot, transparent: true,
        alphaTest: 0.2, depthWrite: false, sizeAttenuation: true,
      });
      s.layer.add(new THREE.Points(geo, mat));
    }
    if (track && track.points.length > 1) {
      const pts = track.points.map(([lat, lon, alt]) => toVec(lat, lon, alt));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: track.color, transparent: true, opacity: 0.9 }));
      s.layer.add(line);
    }
  }, [points, track]);

  return (
    <div className="globe" style={{ height }}>
      <div ref={mount} className="globe-canvas" />
      {emptyMessage && <div className="globe-empty">{emptyMessage}</div>}
    </div>
  );
}
