/**
 * The three.js side of the code-space view: an orbitable point cloud of a
 * fan's futures, rendered on demand. Framework-free; CodeSpace3D.tsx owns it.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { CodeSpace } from "../lib/codespace";

export interface SceneColors {
  bg: string;
  node: string;
  worn: string;
  selected: string;
  edge: string;
  axis: string;
  ink: string;
}

export interface Highlight {
  selectedSlot: number | null;
  wornSlot: number | null;
}

export function readColors(el: Element = document.documentElement): SceneColors {
  const cs = getComputedStyle(el);
  const v = (name: string, dflt: string) => cs.getPropertyValue(name).trim() || dflt;
  return {
    bg: v("--bg-sunk", "#070a0c"),
    node: v("--warp", "#5fb6c4"),
    worn: v("--weft", "#e0a94b"),
    selected: v("--ink", "#c6d0d8"),
    edge: v("--hair-strong", "#2c3740"),
    axis: v("--ink-faint", "#75828d"),
    ink: v("--ink", "#c6d0d8"),
  };
}

/** Node radius: AREA ∝ pull (so radius ∝ √pull), clamped to stay legible. */
export function nodeRadius(pull: number): number {
  return Math.max(0.03, Math.min(0.15, 0.055 * Math.sqrt(Math.max(0, pull))));
}

export class CodeSpaceScene {
  readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  private readonly controls: OrbitControls;
  private readonly root = new THREE.Group();
  private readonly raycaster = new THREE.Raycaster();
  private meshes: THREE.Mesh[] = [];
  private rings: THREE.Object3D[] = [];
  private space: CodeSpace | null = null;
  private colors: SceneColors;
  private frame: number | null = null;
  private readonly reducedMotion: boolean;
  private disposed = false;
  /** called after each render with each point's screen position (for HTML labels) */
  onRender: ((screen: { slot: number; x: number; y: number; visible: boolean }[]) => void) | null = null;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    opts: { reducedMotion: boolean; colors: SceneColors },
  ) {
    this.reducedMotion = opts.reducedMotion;
    this.colors = opts.colors;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(2, globalThis.devicePixelRatio || 1));
    this.scene.add(this.root);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(2, 3, 4);
    this.scene.add(key);
    this.camera.position.set(1.9, 1.35, 2.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = !this.reducedMotion;
    this.controls.dampingFactor = 0.12;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 8;
    this.controls.addEventListener("change", () => this.requestRender());
    // any user interaction ends the idle rotation for good
    this.controls.addEventListener("start", () => {
      this.controls.autoRotate = false;
    });
    this.applyColors();
  }

  setColors(colors: SceneColors): void {
    this.colors = colors;
    this.applyColors();
    if (this.space) this.build(this.space, this.lastHighlight);
  }

  private applyColors(): void {
    this.scene.background = new THREE.Color(this.colors.bg);
    this.requestRender();
  }

  private lastHighlight: Highlight = { selectedSlot: null, wornSlot: null };

  setData(space: CodeSpace, highlight: Highlight): void {
    this.space = space;
    this.build(space, highlight);
    if (space.dims === 2) {
      // a plane: look straight at it, and do not spin it edge-on
      this.camera.position.set(0, 0, 2.9);
      this.controls.autoRotate = false;
    }
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.requestRender();
  }

  setHighlight(h: Highlight): void {
    if (this.space) this.build(this.space, h);
  }

  private clear(): void {
    for (const child of [...this.root.children]) {
      this.root.remove(child);
      child.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) for (const x of mat) x.dispose();
        else mat?.dispose();
      });
    }
    this.meshes = [];
    this.rings = [];
  }

  private build(space: CodeSpace, h: Highlight): void {
    this.lastHighlight = h;
    this.clear();
    const pts = space.points;
    let maxR = 0;
    for (const p of pts) maxR = Math.max(maxR, Math.hypot(p.pos[0], p.pos[1], p.pos[2]));
    const s = maxR > 1e-12 ? 1 / maxR : 1;
    const at = (slot: number): THREE.Vector3 | null => {
      const p = pts.find((q) => q.slot === slot);
      return p ? new THREE.Vector3(p.pos[0] * s, p.pos[1] * s, p.pos[2] * s) : null;
    };

    // axes: the fan's own principal directions (arbitrary basis, shared scale)
    const axisMat = new THREE.LineBasicMaterial({
      color: this.colors.axis,
      transparent: true,
      opacity: 0.35,
    });
    const axisPts: THREE.Vector3[] = [];
    for (let d = 0; d < space.dims; d++) {
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      a.setComponent(d, -1.1);
      b.setComponent(d, 1.1);
      axisPts.push(a, b);
    }
    this.root.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(axisPts), axisMat));

    // threads: nearest fan-mate by FULL-SPACE cosine (deduplicated)
    const seen = new Set<string>();
    const edgePts: THREE.Vector3[] = [];
    for (const p of pts) {
      const key = p.slot < p.nn ? `${p.slot}-${p.nn}` : `${p.nn}-${p.slot}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const a = at(p.slot);
      const b = at(p.nn);
      if (a && b) edgePts.push(a, b);
    }
    const edgeMat = new THREE.LineBasicMaterial({
      color: this.colors.node,
      transparent: true,
      opacity: 0.45,
    });
    this.root.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(edgePts), edgeMat));

    // bodies
    for (const p of pts) {
      const pos = at(p.slot);
      if (!pos) continue;
      const isWorn = p.slot === h.wornSlot;
      const isSel = p.slot === h.selectedSlot;
      const r = nodeRadius(p.pull);
      const mat = new THREE.MeshStandardMaterial({
        color: isWorn ? this.colors.worn : this.colors.node,
        roughness: 0.55,
        metalness: 0.05,
        emissive: new THREE.Color(isWorn ? this.colors.worn : this.colors.node),
        emissiveIntensity: isWorn ? 0.45 : 0.12,
      });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), mat);
      mesh.position.copy(pos);
      mesh.userData.slot = p.slot;
      this.root.add(mesh);
      this.meshes.push(mesh);
      if (isWorn || isSel) {
        const ring = new THREE.Mesh(
          new THREE.SphereGeometry(r * 1.55, 16, 10),
          new THREE.MeshBasicMaterial({
            color: isWorn ? this.colors.worn : this.colors.selected,
            wireframe: true,
            transparent: true,
            opacity: isWorn ? 0.55 : 0.4,
          }),
        );
        ring.position.copy(pos);
        this.root.add(ring);
        this.rings.push(ring);
      }
    }
    this.requestRender();
  }

  resize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  /** The future under a canvas-relative point, or null. */
  pick(x: number, y: number): number | null {
    const w = this.canvas.clientWidth;
    const hgt = this.canvas.clientHeight;
    if (!w || !hgt) return null;
    this.raycaster.setFromCamera(new THREE.Vector2((x / w) * 2 - 1, -(y / hgt) * 2 + 1), this.camera);
    const hit = this.raycaster.intersectObjects(this.meshes, false)[0];
    const slot = hit?.object.userData.slot;
    return typeof slot === "number" ? slot : null;
  }

  requestRender(): void {
    if (this.frame !== null || this.disposed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.renderNow();
    });
  }

  private renderNow(): void {
    if (this.disposed) return;
    const moving = this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.onRender) {
      const w = this.canvas.clientWidth;
      const hgt = this.canvas.clientHeight;
      const v = new THREE.Vector3();
      this.onRender(
        this.meshes.map((m) => {
          v.copy(m.position).project(this.camera);
          return {
            slot: m.userData.slot as number,
            x: ((v.x + 1) / 2) * w,
            y: ((1 - v.y) / 2) * hgt,
            visible: v.z < 1,
          };
        }),
      );
    }
    // keep going while rotating or settling; otherwise render only on change
    if (moving || this.controls.autoRotate) this.requestRender();
  }

  dispose(): void {
    this.disposed = true;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.clear();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
