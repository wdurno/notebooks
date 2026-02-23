const statusEl = document.getElementById("status");
const cameraEl = document.getElementById("camera");
const requestedHz = Number(document.body.dataset.cameraHz || 2);
const effectiveHz = Number.isFinite(requestedHz) ? Math.min(2, Math.max(0.25, requestedHz)) : 2;
const FRAME_INTERVAL_MS = Math.floor(1000 / effectiveHz);
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
    const nextUrl = URL.createObjectURL(blob);
    cameraEl.src = nextUrl;
    if (cameraObjectUrl) {
      URL.revokeObjectURL(cameraObjectUrl);
    }
    cameraObjectUrl = nextUrl;
  } catch (err) {
    // Keep showing the last successful frame to avoid black/broken image flashes.
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

setInterval(refreshFrame, FRAME_INTERVAL_MS);
refreshFrame();
setStatus("Ready. Keyboard: WASD drive, arrows control camera (down = forward).");
