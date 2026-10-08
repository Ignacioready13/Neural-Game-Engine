/**
 * Supersampled render target + box-filter downsample pass (spec Section 8.1).
 *
 * The scene is rendered into a renderWidth x renderHeight target, then a
 * fullscreen quad averages a renderScale x renderScale texel box per output
 * pixel into an outputWidth x outputHeight target, and finally the low-res
 * target is blitted to the default framebuffer so the user sees the true
 * downsampled result.
 *
 * Color-space flow: both render targets use sRGB color space (SRGB8_ALPHA8),
 * so the GPU encodes sRGB on write and decodes to linear on sample. The box
 * filter therefore averages in linear light (the mathematically correct space
 * for a box filter), and `readPixels` returns sRGB-encoded bytes that are
 * exactly what a PNG wants. The blit shader re-encodes linear -> sRGB itself,
 * because the default (canvas) framebuffer is not an sRGB buffer.
 */

import * as THREE from 'three';

export interface ResolutionSpec {
  renderWidth: number;
  renderHeight: number;
  outputWidth: number;
  outputHeight: number;
  /** Supersampling factor; must be a positive integer. */
  renderScale: number;
}

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const BLIT_FRAGMENT = /* glsl */ `
uniform sampler2D tDiffuse;
varying vec2 vUv;
vec3 linearToSrgb(vec3 c) {
  c = max(c, 0.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
void main() {
  vec4 c = texture2D(tDiffuse, vUv);
  gl_FragColor = vec4(linearToSrgb(c.rgb), c.a);
}
`;

/** Averages an exact scale x scale texel box per output pixel. The scale is
 * baked in as compile-time constants so the GLSL loop bounds are constant. */
function boxFilterFragment(scale: number): string {
  return /* glsl */ `
uniform sampler2D tDiffuse;
uniform vec2 texelSize;
varying vec2 vUv;
void main() {
  vec4 sum = vec4(0.0);
  for (int y = 0; y < ${scale}; y++) {
    for (int x = 0; x < ${scale}; x++) {
      vec2 off = (vec2(float(x), float(y)) + 0.5 - float(${scale}) * 0.5) * texelSize;
      sum += texture2D(tDiffuse, vUv + off);
    }
  }
  gl_FragColor = sum / float(${scale} * ${scale});
}
`;
}

export class ResolutionPipeline {
  readonly renderTarget: THREE.WebGLRenderTarget;
  readonly outputTarget: THREE.WebGLRenderTarget;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly outputWidth: number;
  private readonly outputHeight: number;

  private readonly downsampleScene: THREE.Scene;
  private readonly blitScene: THREE.Scene;
  private readonly camera: THREE.OrthographicCamera;
  private readonly quadGeometry: THREE.PlaneGeometry;
  private readonly downsampleMaterial: THREE.ShaderMaterial;
  private readonly blitMaterial: THREE.ShaderMaterial;

  constructor(renderer: THREE.WebGLRenderer, spec: ResolutionSpec) {
    this.renderer = renderer;
    this.outputWidth = spec.outputWidth;
    this.outputHeight = spec.outputHeight;

    const scale = Math.max(1, Math.round(spec.renderScale));

    this.renderTarget = new THREE.WebGLRenderTarget(spec.renderWidth, spec.renderHeight, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      colorSpace: THREE.SRGBColorSpace,
    });

    this.outputTarget = new THREE.WebGLRenderTarget(spec.outputWidth, spec.outputHeight, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      colorSpace: THREE.SRGBColorSpace,
    });

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.camera.position.set(0, 0, 1);

    this.quadGeometry = new THREE.PlaneGeometry(2, 2);

    this.downsampleMaterial = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: boxFilterFragment(scale),
      uniforms: {
        tDiffuse: { value: this.renderTarget.texture },
        texelSize: { value: new THREE.Vector2(1 / spec.renderWidth, 1 / spec.renderHeight) },
      },
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });

    this.blitMaterial = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: BLIT_FRAGMENT,
      uniforms: { tDiffuse: { value: this.outputTarget.texture } },
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });

    const downsampleQuad = new THREE.Mesh(this.quadGeometry, this.downsampleMaterial);
    downsampleQuad.frustumCulled = false;
    this.downsampleScene = new THREE.Scene();
    this.downsampleScene.add(downsampleQuad);

    const blitQuad = new THREE.Mesh(this.quadGeometry, this.blitMaterial);
    blitQuad.frustumCulled = false;
    this.blitScene = new THREE.Scene();
    this.blitScene.add(blitQuad);
  }

  /** Render the 3D scene into the supersampled target. */
  renderScene(scene: THREE.Scene, camera: THREE.PerspectiveCamera): void {
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.clear(true, true, false);
    this.renderer.render(scene, camera);
  }

  /** Box-filter the supersampled target into the output target. */
  downsample(): void {
    this.renderer.setRenderTarget(this.outputTarget);
    this.renderer.clear(true, true, false);
    this.renderer.render(this.downsampleScene, this.camera);
  }

  /** Copy the output target onto the default framebuffer (the canvas). */
  blit(): void {
    this.renderer.setRenderTarget(null);
    this.renderer.clear(true, true, false);
    this.renderer.render(this.blitScene, this.camera);
  }

  /** Read the output target's pixels (RGBA, bottom-up row order). Caller flips
   * vertically for top-down consumers. */
  readPixels(out: Uint8Array): void {
    this.renderer.readRenderTargetPixels(
      this.outputTarget,
      0,
      0,
      this.outputWidth,
      this.outputHeight,
      out,
    );
  }

  dispose(): void {
    this.renderTarget.dispose();
    this.outputTarget.dispose();
    this.quadGeometry.dispose();
    this.downsampleMaterial.dispose();
    this.blitMaterial.dispose();
  }
}
