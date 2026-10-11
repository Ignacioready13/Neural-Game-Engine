/**
 * The Renderer: wires the scene graph, resolution pipeline, and blackout fills
 * together behind the exact public API the orchestrator calls.
 *
 * Per simulated frame the engine calls, in order:
 *   positionCamera(player) -> projectToScreenX(...) for candidates
 *   -> setEntities(...) + setBlackDoorways(...) after world changes
 *   -> updateFlicker(dt) -> render(snapshot)   (headless: + captureFrame())
 */

import * as THREE from 'three';
import type { EngineConfig } from '../../config/config';
import type { Doorway, Entity, FrameSnapshot, PlayerState } from '../core/types';
import { clamp } from '../core/math';
import { BackroomsScene } from './scene';
import type { EntityLights } from './scene';
import { ResolutionPipeline } from './resolution';
import { BlackoutManager } from './blackout';

const PULSE_VERTEX = /* glsl */ `
uniform vec2 uCenter;
uniform vec2 uScale;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(uCenter + position.xy * uScale, 0.0, 1.0);
}
`;

// Premultiplied-alpha disc. The pulse shader writes sRGB bytes directly, since
// it is composited over the already-sRGB-encoded frame on the canvas.
const PULSE_FRAGMENT = /* glsl */ `
uniform sampler2D uTex;
uniform vec3 uColor;
uniform float uAlpha;
varying vec2 vUv;
void main() {
  float a = texture2D(uTex, vUv).a * uAlpha;
  gl_FragColor = vec4(uColor * a, a);
}
`;

/** Soft radial-gradient disc used for the click pulse (spec Section 6.5). */
function makeDiscTexture(): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.4, 'rgba(255,255,255,0.7)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

export class Renderer {
  private readonly config: EngineConfig;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly sceneState: BackroomsScene;
  private readonly pipeline: ResolutionPipeline;
  private readonly blackout: BlackoutManager;

  // Live entities, tracked by descriptor id so we can add new groups and drop
  // disposed ones without ever touching the world generator's geometry.
  private readonly entityGroups = new Map<string, THREE.Group>();
  private readonly entityLights = new Map<string, EntityLights>();

  // Click-pulse overlay (screen-space, drawn after the downsample blit).
  private readonly overlayCamera: THREE.OrthographicCamera;
  private readonly pulseScene: THREE.Scene;
  private readonly pulseGeometry: THREE.PlaneGeometry;
  private readonly pulseMaterial: THREE.ShaderMaterial;
  private readonly pulseTexture: THREE.CanvasTexture;
  private readonly uCenter: THREE.Vector2;
  private readonly uScale: THREE.Vector2;
  private readonly uAlpha: { value: number };

  // Reused capture scratch buffers (no per-frame allocation).
  private readonly pixelBuffer: Uint8Array;
  private readonly captureCanvas: HTMLCanvasElement;
  private readonly captureCtx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement, config: EngineConfig) {
    this.config = config;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    this.renderer.autoClear = false; // each pass below clears explicitly
    this.renderer.setPixelRatio(1);
    // Display at OUTPUT resolution: the low-res downsampled image is blitted 1:1
    // to the drawing buffer; CSS (owned by the app shell) may scale it up.
    this.renderer.setSize(config.outputWidth, config.outputHeight, false);
    this.renderer.setClearColor(config.fogColor, 1);

    this.sceneState = new BackroomsScene(config);
    this.pipeline = new ResolutionPipeline(this.renderer, {
      renderWidth: config.renderWidth,
      renderHeight: config.renderHeight,
      outputWidth: config.outputWidth,
      outputHeight: config.outputHeight,
      renderScale: config.renderScale,
    });
    this.blackout = new BlackoutManager();
    this.sceneState.scene.add(this.blackout.group);

    // Click-pulse overlay setup.
    this.overlayCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.pulseTexture = makeDiscTexture();
    this.uCenter = new THREE.Vector2();
    this.uScale = new THREE.Vector2();
    this.uAlpha = { value: 0 };
    this.pulseMaterial = new THREE.ShaderMaterial({
      vertexShader: PULSE_VERTEX,
      fragmentShader: PULSE_FRAGMENT,
      uniforms: {
        uTex: { value: this.pulseTexture },
        uCenter: { value: this.uCenter },
        uScale: { value: this.uScale },
        uColor: { value: new THREE.Vector3(1.0, 0.82, 0.48) },
        uAlpha: this.uAlpha,
      },
      transparent: true,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    this.pulseGeometry = new THREE.PlaneGeometry(1, 1);
    const pulseMesh = new THREE.Mesh(this.pulseGeometry, this.pulseMaterial);
    pulseMesh.frustumCulled = false;
    this.pulseScene = new THREE.Scene();
    this.pulseScene.add(pulseMesh);

    // Capture scratch.
    this.pixelBuffer = new Uint8Array(config.outputWidth * config.outputHeight * 4);
    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = config.outputWidth;
    this.captureCanvas.height = config.outputHeight;
    this.captureCtx = this.captureCanvas.getContext('2d')!;
  }

  positionCamera(player: PlayerState): void {
    this.sceneState.positionCamera(player);
  }

  projectToScreenX(wx: number, wz: number, wy: number): number | null {
    return this.sceneState.projectToScreenX(wx, wz, wy);
  }

  /** The render camera's near plane (world units) — read by the viewUnsafe
   * clearance measurement so the label always tracks the real camera. */
  get cameraNear(): number {
    return this.sceneState.camera.near;
  }

  /** Sync live entities into the scene: add new groups, remove disposed ones
   * (tracked by descriptor id), and keep per-entity fixture lights in step. */
  setEntities(entities: Entity[]): void {
    const present = new Set<string>();
    for (const entity of entities) {
      const id = entity.descriptor.id;
      present.add(id);
      const group = entity.group as THREE.Group;
      const existing = this.entityGroups.get(id);
      if (existing === group) continue;

      if (existing) {
        this.sceneState.scene.remove(existing);
        const lights = this.entityLights.get(id);
        if (lights) this.sceneState.removeEntityLights(lights);
        this.entityLights.delete(id);
      }

      this.sceneState.scene.add(group);
      this.entityGroups.set(id, group);
      this.entityLights.set(id, this.sceneState.addEntityLights(entity.descriptor));
    }

    for (const [id, group] of this.entityGroups) {
      if (!present.has(id)) {
        this.sceneState.scene.remove(group);
        this.entityGroups.delete(id);
        const lights = this.entityLights.get(id);
        if (lights) this.sceneState.removeEntityLights(lights);
        this.entityLights.delete(id);
      }
    }
  }

  setBlackDoorways(doorways: Doorway[]): void {
    this.blackout.setDoorways(doorways);
  }

  /** Draw one frame: scene -> supersampled target -> box-filter downsample ->
   * screen, then the screen-space click pulses on top. */
  render(snapshot: FrameSnapshot): void {
    this.pipeline.renderScene(this.sceneState.scene, this.sceneState.camera);
    this.pipeline.downsample();
    this.pipeline.blit();
    this.renderPulses(snapshot.clickPulses);
  }

  /** Return the final OUTPUT-resolution frame as a PNG data URL. */
  captureFrame(): { dataUrl: string; width: number; height: number } {
    const w = this.config.outputWidth;
    const h = this.config.outputHeight;
    this.pipeline.readPixels(this.pixelBuffer);

    // readPixels returns rows bottom-up (GL origin); ImageData is top-down.
    const imageData = this.captureCtx.createImageData(w, h);
    const dst = imageData.data;
    const src = this.pixelBuffer;
    const rowBytes = w * 4;
    for (let row = 0; row < h; row++) {
      dst.set(src.subarray((h - 1 - row) * rowBytes, (h - row) * rowBytes), row * rowBytes);
    }
    this.captureCtx.putImageData(imageData, 0, 0);
    return { dataUrl: this.captureCanvas.toDataURL('image/png'), width: w, height: h };
  }

  updateFlicker(dt: number): void {
    this.sceneState.updateFlicker(dt);
  }

  dispose(): void {
    this.blackout.dispose();
    this.sceneState.dispose();
    this.pipeline.dispose();
    this.pulseGeometry.dispose();
    this.pulseMaterial.dispose();
    this.pulseTexture.dispose();
    this.renderer.dispose();
    this.entityGroups.clear();
    this.entityLights.clear();
  }

  private renderPulses(pulses: ReadonlyArray<{ screenX: number; ageSeconds: number }>): void {
    this.renderer.setRenderTarget(null);
    const aspect = this.config.outputWidth / this.config.outputHeight;
    for (const pulse of pulses) {
      const age = clamp(pulse.ageSeconds, 0, this.config.clickPulseSeconds);
      const life = 1 - age / this.config.clickPulseSeconds;
      if (life <= 0) continue;
      const alpha = life * 0.85;
      const rx = 0.055 * (1 + (1 - life) * 1.5); // grows slightly as it fades
      const ry = rx * aspect; // keep the disc circular at the output aspect
      this.uCenter.set(pulse.screenX, 0);
      this.uScale.set(rx * 2, ry * 2);
      this.uAlpha.value = alpha;
      this.renderer.render(this.pulseScene, this.overlayCamera);
    }
  }
}
