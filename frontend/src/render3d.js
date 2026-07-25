/**
 * Opaque WebGL viewport for the artifact sample viewer.
 *
 * The previous SVG renderer could only afford a few thousand primitives, which
 * forced the backend to stride its edge list and produced disconnected
 * confetti instead of a mesh.  Drawing through WebGL lets the viewer show the
 * real topology - tens of thousands of elements - with a depth buffer, so
 * every surface is opaque and correctly occluded without any alpha blending.
 */

const VERTEX_SOURCE = `
precision highp float;
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute float aValue;
uniform mat4 uMatrix;
uniform mat3 uRotation;
uniform vec2 uRange;
uniform vec3 uBaseColor;
uniform float uUseField;
uniform float uLighting;
uniform float uPointSize;
uniform float uDepthBias;
varying vec3 vColor;

vec3 turbo(float t) {
  const vec4 red4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 green4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 blue4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 red2 = vec2(-152.94239396, 59.28637943);
  const vec2 green2 = vec2(4.27729857, 2.82956604);
  const vec2 blue2 = vec2(-89.90310912, 27.34824973);
  float x = clamp(t, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return clamp(vec3(
    dot(v4, red4) + dot(v2, red2),
    dot(v4, green4) + dot(v2, green2),
    dot(v4, blue4) + dot(v2, blue2)
  ), 0.0, 1.0);
}

void main() {
  gl_Position = uMatrix * vec4(aPosition, 1.0);
  // Wireframes and points sit exactly on the surface they annotate; a small
  // constant bias toward the viewer keeps them from being z-fought away.
  gl_Position.z -= uDepthBias;
  gl_PointSize = uPointSize;
  float span = max(uRange.y - uRange.x, 1e-12);
  vec3 base = mix(uBaseColor, turbo((aValue - uRange.x) / span), uUseField);
  vec3 normal = normalize(uRotation * aNormal + vec3(0.0, 0.0, 1e-6));
  float lambert = 0.62 + 0.38 * abs(dot(normal, normalize(vec3(-0.32, 0.44, 0.84))));
  vColor = base * mix(1.0, lambert, uLighting);
}`;

const FRAGMENT_SOURCE = `
precision mediump float;
uniform float uRound;
varying vec3 vColor;
void main() {
  if (uRound > 0.5) {
    vec2 offset = gl_PointCoord - vec2(0.5);
    if (dot(offset, offset) > 0.25) discard;
  }
  gl_FragColor = vec4(vColor, 1.0);
}`;

const BACKGROUND = [0.055, 0.121, 0.101];
const SURFACE_COLOR = [0.663, 0.769, 0.729];
const WIRE_COLOR = [0.086, 0.203, 0.172];
const POINT_COLOR = [0.847, 0.925, 0.898];

const TURBO_RED = [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943];
const TURBO_GREEN = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604];
const TURBO_BLUE = [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973];

function turboChannel(coefficients, t) {
  const [a, b, c, d, e, f] = coefficients;
  return a + b * t + c * t * t + d * t ** 3 + e * t ** 4 + f * t ** 5;
}

/** Turbo colormap sample as a CSS colour, matching the shader exactly. */
export function turboColor(ratio) {
  const t = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
  const channel = coefficients => Math.round(255 * Math.max(0, Math.min(1, turboChannel(coefficients, t))));
  return `rgb(${channel(TURBO_RED)} ${channel(TURBO_GREEN)} ${channel(TURBO_BLUE)})`;
}

export function fieldColor(value, low, high) {
  const span = (high - low) || 1;
  return turboColor((Number(value) - low) / span);
}

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

/** Order the model axes by extent so the widest pair faces the viewer. */
function axisFrame(sample) {
  const axes = ["x", "y", "z"].map((name, index) => {
    const values = sample[name] || [];
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (!Number.isFinite(value)) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
    if (!Number.isFinite(low)) { low = 0; high = 0; }
    return { name, index, low, high, extent: high - low, center: (low + high) / 2 };
  });
  const ordered = [...axes].sort((left, right) => right.extent - left.extent);
  return {
    axes,
    order: ordered,
    planar: ordered[2].extent <= Math.max(ordered[0].extent, 1e-12) * 1e-6
  };
}

/**
 * Build the GPU-ready geometry for one sample: positions are centred and
 * axis-permuted, faces are un-indexed so each element carries one flat value.
 */
function buildGeometry(sample) {
  const frame = axisFrame(sample);
  const [horizontal, vertical, depth] = frame.order;
  const source = { x: sample.x || [], y: sample.y || [], z: sample.z || [] };
  const nodeCount = source.x.length;
  const values = sample.values || [];

  const positions = new Float32Array(nodeCount * 3);
  const nodeValues = new Float32Array(nodeCount);
  let radius = 0;
  for (let index = 0; index < nodeCount; index += 1) {
    const h = finiteOr(source[horizontal.name][index], horizontal.center) - horizontal.center;
    const v = finiteOr(source[vertical.name][index], vertical.center) - vertical.center;
    const d = finiteOr(source[depth.name][index], depth.center) - depth.center;
    positions[index * 3] = h;
    positions[index * 3 + 1] = v;
    positions[index * 3 + 2] = d;
    nodeValues[index] = finiteOr(values[index], NaN);
    const distance = Math.hypot(h, v, d);
    if (distance > radius) radius = distance;
  }

  const mesh = sample.mesh || null;
  const faceIndices = mesh?.faces || [];
  const faceCount = Math.floor(faceIndices.length / 3);
  const facePositions = new Float32Array(faceCount * 9);
  const faceNormals = new Float32Array(faceCount * 9);
  const faceValues = new Float32Array(faceCount * 3);
  for (let face = 0; face < faceCount; face += 1) {
    const a = faceIndices[face * 3] * 3;
    const b = faceIndices[face * 3 + 1] * 3;
    const c = faceIndices[face * 3 + 2] * 3;
    const ax = positions[a], ay = positions[a + 1], az = positions[a + 2];
    const bx = positions[b], by = positions[b + 1], bz = positions[b + 2];
    const cx = positions[c], cy = positions[c + 1], cz = positions[c + 2];
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const length = Math.hypot(nx, ny, nz) || 1;
    // One flat normal and one flat value per element gives element shading.
    const mean = (nodeValues[faceIndices[face * 3]]
      + nodeValues[faceIndices[face * 3 + 1]]
      + nodeValues[faceIndices[face * 3 + 2]]) / 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const offset = face * 9 + corner * 3;
      const vertex = [a, b, c][corner];
      facePositions[offset] = positions[vertex];
      facePositions[offset + 1] = positions[vertex + 1];
      facePositions[offset + 2] = positions[vertex + 2];
      faceNormals[offset] = nx / length;
      faceNormals[offset + 1] = ny / length;
      faceNormals[offset + 2] = nz / length;
      faceValues[face * 3 + corner] = mean;
    }
  }

  const edgeIndices = mesh?.edges?.length
    ? new Uint32Array(mesh.edges)
    : faceCount
      ? null
      : new Uint32Array(0);

  let low = Infinity;
  let high = -Infinity;
  for (let index = 0; index < nodeCount; index += 1) {
    const value = nodeValues[index];
    if (!Number.isFinite(value)) continue;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  const statLow = finiteOr(sample.stats?.min, Number.isFinite(low) ? low : 0);
  const statHigh = finiteOr(sample.stats?.max, Number.isFinite(high) ? high : 1);
  // A constant field has no gradient to show; centre it in the colormap so it
  // reads as uniform instead of looking like an unlit, broken surface.
  const constant = !(statHigh > statLow);

  return {
    frame,
    nodeCount,
    positions,
    nodeValues,
    facePositions,
    faceNormals,
    faceValues,
    faceCount,
    edgeIndices,
    radius: radius || 1,
    constant,
    domain: [statLow, statHigh],
    range: constant ? [statLow - 0.5, statLow + 0.5] : [statLow, statHigh]
  };
}

/** A straight-on view reads best for planar data; solids need an angle. */
export function defaultCamera(sample) {
  const planar = sample ? axisFrame(sample).planar : true;
  return planar
    ? { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 }
    : { yaw: 0.62, pitch: 0.42, zoom: 1, panX: 0, panY: 0 };
}

function rotationMatrix(yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  // Yaw about the screen-up axis, then pitch about the screen-right axis.
  return [
    cy, 0, sy,
    sp * sy, cp, -sp * cy,
    -cp * sy, sp, cp * cy
  ];
}

function projectionMatrix(rotation, geometry, camera, width, height) {
  const fit = Math.min(width, height) * 0.42 / geometry.radius;
  const scale = fit * camera.zoom;
  const sx = 2 * scale / width;
  const sy = 2 * scale / height;
  const sz = -1 / (geometry.radius * 1.02);
  return new Float32Array([
    rotation[0] * sx, rotation[3] * sy, rotation[6] * sz, 0,
    rotation[1] * sx, rotation[4] * sy, rotation[7] * sz, 0,
    rotation[2] * sx, rotation[5] * sy, rotation[8] * sz, 0,
    2 * camera.panX / width, -2 * camera.panY / height, 0, 1
  ]);
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Viewer shader failed to compile: ${log}`);
  }
  return shader;
}

function createProgram(gl) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SOURCE));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SOURCE));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Viewer program failed to link: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

export function createRenderer(canvas) {
  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: true,
    depth: true,
    preserveDrawingBuffer: true
  });
  if (!gl) return null;

  const program = createProgram(gl);
  const attributes = {
    position: gl.getAttribLocation(program, "aPosition"),
    normal: gl.getAttribLocation(program, "aNormal"),
    value: gl.getAttribLocation(program, "aValue")
  };
  const uniforms = Object.fromEntries(
    ["uMatrix", "uRotation", "uRange", "uBaseColor", "uUseField", "uLighting", "uPointSize", "uRound", "uDepthBias"]
      .map(name => [name, gl.getUniformLocation(program, name)])
  );
  const buffers = {
    nodePosition: gl.createBuffer(),
    nodeValue: gl.createBuffer(),
    facePosition: gl.createBuffer(),
    faceNormal: gl.createBuffer(),
    faceValue: gl.createBuffer(),
    edgeIndex: gl.createBuffer()
  };
  const uintIndices = gl.getExtension("OES_element_index_uint");
  let uploaded = null;
  let geometry = null;

  function upload(data) {
    geometry = buildGeometry(data);
    const write = (buffer, array) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
    };
    write(buffers.nodePosition, geometry.positions);
    write(buffers.nodeValue, geometry.nodeValues);
    write(buffers.facePosition, geometry.facePositions);
    write(buffers.faceNormal, geometry.faceNormals);
    write(buffers.faceValue, geometry.faceValues);
    if (geometry.edgeIndices?.length) {
      const indices = uintIndices ? geometry.edgeIndices : new Uint16Array(geometry.edgeIndices);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.edgeIndex);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
      geometry.edgeType = uintIndices ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    }
    uploaded = data;
    return geometry;
  }

  function bind(positionBuffer, valueBuffer, normalBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(attributes.position);
    gl.vertexAttribPointer(attributes.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, valueBuffer);
    gl.enableVertexAttribArray(attributes.value);
    gl.vertexAttribPointer(attributes.value, 1, gl.FLOAT, false, 0, 0);
    if (normalBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
      gl.enableVertexAttribArray(attributes.normal);
      gl.vertexAttribPointer(attributes.normal, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(attributes.normal);
      gl.vertexAttrib3f(attributes.normal, 0, 0, 1);
    }
  }

  return {
    kind: "webgl",
    /** Draw one sample; geometry is re-uploaded only when the sample changes. */
    draw(sample, mode, camera) {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const scene = sample === uploaded && geometry ? geometry : upload(sample);

      gl.viewport(0, 0, width, height);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);

      const rotation = rotationMatrix(camera.yaw, camera.pitch);
      // Pan arrives in CSS pixels; the drawing buffer is in device pixels.
      const framed = { ...camera, panX: camera.panX * ratio, panY: camera.panY * ratio };
      gl.uniformMatrix4fv(uniforms.uMatrix, false, projectionMatrix(rotation, scene, framed, width, height));
      gl.uniformMatrix3fv(uniforms.uRotation, false, new Float32Array(rotation));
      gl.uniform2f(uniforms.uRange, scene.range[0], scene.range[1]);
      gl.uniform1f(uniforms.uRound, 0);

      const showField = mode === "field" && Boolean(sample.supports?.field);
      const drawFaces = mode !== "points" && scene.faceCount > 0;
      const drawEdges = mode !== "points" && Boolean(scene.edgeIndices?.length);

      if (drawFaces) {
        bind(buffers.facePosition, buffers.faceValue, buffers.faceNormal);
        gl.uniform1f(uniforms.uUseField, showField ? 1 : 0);
        gl.uniform1f(uniforms.uLighting, 1);
        gl.uniform1f(uniforms.uDepthBias, 0);
        gl.uniform3fv(uniforms.uBaseColor, SURFACE_COLOR);
        gl.drawArrays(gl.TRIANGLES, 0, scene.faceCount * 3);
      }

      if (drawEdges && (mode === "mesh" || !drawFaces)) {
        bind(buffers.nodePosition, buffers.nodeValue, null);
        gl.uniform1f(uniforms.uDepthBias, drawFaces ? 0.004 : 0);
        gl.uniform1f(uniforms.uUseField, showField && !drawFaces ? 1 : 0);
        gl.uniform1f(uniforms.uLighting, 0);
        gl.uniform3fv(uniforms.uBaseColor, drawFaces ? WIRE_COLOR : POINT_COLOR);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.edgeIndex);
        gl.drawElements(gl.LINES, scene.edgeIndices.length, scene.edgeType, 0);
      }

      const showPoints = mode === "points" || (!drawFaces && !drawEdges);
      if (showPoints || sample.preview_kind === "series") {
        bind(buffers.nodePosition, buffers.nodeValue, null);
        gl.uniform1f(uniforms.uDepthBias, 0.006);
        gl.uniform1f(uniforms.uUseField, mode !== "mesh" && sample.supports?.field ? 1 : 0);
        gl.uniform1f(uniforms.uLighting, 0);
        gl.uniform3fv(uniforms.uBaseColor, POINT_COLOR);
        if (sample.preview_kind === "series") {
          gl.drawArrays(gl.LINE_STRIP, 0, scene.nodeCount);
        }
        if (showPoints) {
          gl.uniform1f(uniforms.uRound, 1);
          gl.uniform1f(uniforms.uPointSize, Math.max(1.5, Math.min(9, 2.4 * ratio * Math.sqrt(camera.zoom))));
          gl.drawArrays(gl.POINTS, 0, scene.nodeCount);
          gl.uniform1f(uniforms.uRound, 0);
        }
      }

      return {
        domain: scene.domain,
        constant: scene.constant,
        planar: scene.frame.planar,
        drewFaces: drawFaces,
        drewEdges: drawEdges && (mode === "mesh" || !drawFaces),
        drewPoints: showPoints
      };
    },
    invalidate() {
      uploaded = null;
    },
    dispose() {
      Object.values(buffers).forEach(buffer => gl.deleteBuffer(buffer));
      gl.deleteProgram(program);
      uploaded = null;
      geometry = null;
    }
  };
}

/**
 * Canvas 2D fallback for environments without WebGL.  It keeps the same
 * opaque, real-connectivity contract, just with a painter's-algorithm sort.
 */
export function createFallbackRenderer(canvas) {
  const context = canvas.getContext("2d");
  if (!context) return null;
  let uploaded = null;
  let geometry = null;

  return {
    kind: "canvas2d",
    draw(sample, mode, camera) {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      if (sample !== uploaded || !geometry) {
        geometry = buildGeometry(sample);
        uploaded = sample;
      }
      const rotation = rotationMatrix(camera.yaw, camera.pitch);
      const scale = Math.min(width, height) * 0.42 / geometry.radius * camera.zoom;
      const centreX = width / 2 + camera.panX * ratio;
      const centreY = height / 2 + camera.panY * ratio;
      const screen = new Float32Array(geometry.nodeCount * 3);
      for (let index = 0; index < geometry.nodeCount; index += 1) {
        const x = geometry.positions[index * 3];
        const y = geometry.positions[index * 3 + 1];
        const z = geometry.positions[index * 3 + 2];
        screen[index * 3] = centreX + (rotation[0] * x + rotation[1] * y + rotation[2] * z) * scale;
        screen[index * 3 + 1] = centreY - (rotation[3] * x + rotation[4] * y + rotation[5] * z) * scale;
        screen[index * 3 + 2] = rotation[6] * x + rotation[7] * y + rotation[8] * z;
      }

      context.setTransform(1, 0, 0, 1, 0, 0);
      context.globalAlpha = 1;
      context.fillStyle = "#0e1f1a";
      context.fillRect(0, 0, width, height);

      const [low, high] = geometry.range;
      const showField = mode === "field" && Boolean(sample.supports?.field);
      const faces = sample.mesh?.faces || [];
      const faceCount = mode === "points" ? 0 : Math.floor(faces.length / 3);
      if (faceCount) {
        const order = Array.from({ length: faceCount }, (unused, index) => index).sort((left, right) => (
          screen[faces[left * 3] * 3 + 2] - screen[faces[right * 3] * 3 + 2]
        ));
        for (const face of order) {
          const a = faces[face * 3], b = faces[face * 3 + 1], c = faces[face * 3 + 2];
          context.beginPath();
          context.moveTo(screen[a * 3], screen[a * 3 + 1]);
          context.lineTo(screen[b * 3], screen[b * 3 + 1]);
          context.lineTo(screen[c * 3], screen[c * 3 + 1]);
          context.closePath();
          context.fillStyle = showField
            ? fieldColor(geometry.faceValues[face * 3], low, high)
            : "#a9c4ba";
          context.fill();
          if (mode === "mesh") {
            context.strokeStyle = "#16342c";
            context.lineWidth = ratio * 0.5;
            context.stroke();
          }
        }
      }

      const edges = sample.mesh?.edges || [];
      if (mode !== "points" && edges.length && !faceCount) {
        context.strokeStyle = "#d8ece4";
        context.lineWidth = ratio;
        context.beginPath();
        for (let index = 0; index < edges.length; index += 2) {
          const a = edges[index] * 3;
          const b = edges[index + 1] * 3;
          context.moveTo(screen[a], screen[a + 1]);
          context.lineTo(screen[b], screen[b + 1]);
        }
        context.stroke();
      }

      const showPoints = mode === "points" || (!faceCount && !edges.length);
      if (showPoints) {
        const size = Math.max(1.4, 2.2 * ratio * Math.sqrt(camera.zoom));
        for (let index = 0; index < geometry.nodeCount; index += 1) {
          context.fillStyle = sample.supports?.field && mode !== "mesh"
            ? fieldColor(geometry.nodeValues[index], low, high)
            : "#d8ece4";
          context.fillRect(screen[index * 3] - size / 2, screen[index * 3 + 1] - size / 2, size, size);
        }
      }

      return {
        domain: geometry.domain,
        constant: geometry.constant,
        planar: geometry.frame.planar,
        drewFaces: faceCount > 0,
        drewEdges: mode !== "points" && edges.length > 0 && !faceCount,
        drewPoints: showPoints
      };
    },
    invalidate() {
      uploaded = null;
    },
    dispose() {
      uploaded = null;
      geometry = null;
    }
  };
}
