const statusEl = document.getElementById("status");
const cameraEl = document.getElementById("camera");
const resolutionFormEl = document.getElementById("resolution-form");
const resolutionPresetEl = document.getElementById("resolution-preset");
const resolutionXEl = document.getElementById("resolution-x");
const resolutionYEl = document.getElementById("resolution-y");
const resolutionCurrentEl = document.getElementById("resolution-current");
const frameSizeEl = document.getElementById("frame-size");
const requestedHz = Number(document.body.dataset.cameraHz || 2);
const effectiveHz = Number.isFinite(requestedHz) ? Math.min(2, Math.max(0.25, requestedHz)) : 2;
const FRAME_INTERVAL_MS = Math.floor(1000 / effectiveHz);
const initialXResize = parseResizeInt(document.body.dataset.cameraXResize);
const initialYResize = parseResizeInt(document.body.dataset.cameraYResize);
let cameraObjectUrl = null;
let frameInFlight = false;
const keyBindings = {
  w: { endpoint: "/api/move", action: "forward" },
  a: { endpoint: "/api/move", action: "left" },
  s: { endpoint: "/api/move", action: "backward" },
  d: { endpoint: "/api/move", action: "right" },
  arrowup: { endpoint: "/api/camera", action: "up" },
  arrowleft: { endpoint: "/api/camera", action: "left" },
  arrowright: { endpoint: "/api/camera", action: "right" },
  arrowdown: { endpoint: "/api/camera", action: "center" },
};

function setStatus(message) {
  const now = new Date().toLocaleTimeString();
  statusEl.textContent = `[${now}] ${message}`;
}

function parseResizeInt(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function formatResolutionLabel(xResize, yResize) {
  if (xResize === null || yResize === null) {
    return "API default";
  }
  return `${xResize} x ${yResize}`;
}

function inferPreset(xResize, yResize) {
  if (xResize === null || yResize === null) {
    return "default";
  }
  const value = `${xResize}x${yResize}`;
  for (const option of resolutionPresetEl.options) {
    if (option.value === value) {
      return value;
    }
  }
  return "custom";
}

function setResolutionUi(xResize, yResize) {
  if (!resolutionPresetEl || !resolutionXEl || !resolutionYEl || !resolutionCurrentEl) {
    return;
  }
  resolutionPresetEl.value = inferPreset(xResize, yResize);
  resolutionXEl.value = xResize === null ? "" : String(xResize);
  resolutionYEl.value = yResize === null ? "" : String(yResize);
  resolutionCurrentEl.textContent = formatResolutionLabel(xResize, yResize);
}

function setFrameSizeLabel(width, height) {
  if (!frameSizeEl) {
    return;
  }
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    frameSizeEl.textContent = "unknown";
    return;
  }
  frameSizeEl.textContent = `${width} x ${height}`;
}

async function readBlobDimensions(blob) {
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }

  return new Promise((resolve, reject) => {
    const probe = new Image();
    const probeUrl = URL.createObjectURL(blob);
    probe.onload = () => {
      const result = { width: probe.naturalWidth, height: probe.naturalHeight };
      URL.revokeObjectURL(probeUrl);
      resolve(result);
    };
    probe.onerror = () => {
      URL.revokeObjectURL(probeUrl);
      reject(new Error("failed to inspect frame dimensions"));
    };
    probe.src = probeUrl;
  });
}

function getResolutionFromForm() {
  const preset = resolutionPresetEl.value;
  if (preset === "default") {
    return { xResize: null, yResize: null };
  }
  if (preset !== "custom") {
    const [xStr, yStr] = preset.split("x");
    const xResize = parseResizeInt(xStr);
    const yResize = parseResizeInt(yStr);
    if (xResize === null || yResize === null) {
      throw new Error("invalid resolution preset");
    }
    return { xResize, yResize };
  }

  const xResize = parseResizeInt(resolutionXEl.value);
  const yResize = parseResizeInt(resolutionYEl.value);
  if (xResize === null || yResize === null) {
    throw new Error("custom width and height must be positive integers");
  }
  return { xResize, yResize };
}

async function updateResolution(xResize, yResize) {
  const response = await fetch("/api/resolution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      x_resize: xResize,
      y_resize: yResize,
    }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `request failed (${response.status})`);
  }
  setResolutionUi(data.x_resize ?? null, data.y_resize ?? null);
}

async function syncResolution() {
  try {
    const response = await fetch("/api/resolution", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `request failed (${response.status})`);
    }
    setResolutionUi(data.x_resize ?? null, data.y_resize ?? null);
  } catch {
    setResolutionUi(initialXResize, initialYResize);
  }
}

async function sendAction(endpoint, action) {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `request failed (${response.status})`);
    }
    setStatus(`OK: ${endpoint} -> ${action}`);
  } catch (err) {
    setStatus(`ERROR: ${endpoint} -> ${action}: ${err.message}`);
  }
}

async function refreshFrame() {
  if (frameInFlight) {
    return;
  }
  frameInFlight = true;
  try {
    const response = await fetch(`/api/frame.jpg?t=${Date.now()}`, {
      cache: "no-store",
    });
    const contentType = response.headers.get("content-type") || "";
    if (!response.ok || !contentType.includes("image/jpeg")) {
      throw new Error(`frame request failed (${response.status})`);
    }
    const blob = await response.blob();
    try {
      const dims = await readBlobDimensions(blob);
      setFrameSizeLabel(dims.width, dims.height);
    } catch {
      setFrameSizeLabel(null, null);
    }
    const nextUrl = URL.createObjectURL(blob);
    cameraEl.src = nextUrl;
    if (cameraObjectUrl) {
      URL.revokeObjectURL(cameraObjectUrl);
    }
    cameraObjectUrl = nextUrl;
  } catch (err) {
    // Keep showing the last successful frame to avoid black/broken image flashes.
    setFrameSizeLabel(null, null);
    setStatus(`Camera update error: ${err.message}`);
  } finally {
    frameInFlight = false;
  }
}

for (const button of document.querySelectorAll("button[data-endpoint]")) {
  button.addEventListener("click", () => {
    const endpoint = button.dataset.endpoint;
    const action = button.dataset.action;
    sendAction(endpoint, action);
  });
}

document.addEventListener("keydown", (event) => {
  if (event.repeat) {
    return;
  }
  if (event.target instanceof HTMLElement) {
    const tag = event.target.tagName;
    const isTypingTarget =
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      event.target.isContentEditable;
    if (isTypingTarget) {
      return;
    }
  }

  const binding = keyBindings[event.key.toLowerCase()];
  if (!binding) {
    return;
  }
  event.preventDefault();
  sendAction(binding.endpoint, binding.action);
});

if (resolutionPresetEl) {
  resolutionPresetEl.addEventListener("change", () => {
    const preset = resolutionPresetEl.value;
    if (preset === "default") {
      resolutionXEl.value = "";
      resolutionYEl.value = "";
      return;
    }
    if (preset !== "custom") {
      const [xStr, yStr] = preset.split("x");
      resolutionXEl.value = xStr;
      resolutionYEl.value = yStr;
    }
  });
}

if (resolutionFormEl) {
  resolutionFormEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const { xResize, yResize } = getResolutionFromForm();
      await updateResolution(xResize, yResize);
      setStatus(`OK: /api/resolution -> ${formatResolutionLabel(xResize, yResize)}`);
    } catch (err) {
      setStatus(`ERROR: /api/resolution: ${err.message}`);
    }
  });
}

setInterval(refreshFrame, FRAME_INTERVAL_MS);
refreshFrame();
setResolutionUi(initialXResize, initialYResize);
syncResolution();
setStatus("Ready. Keyboard: WASD drive, arrows control camera (down = forward).");
