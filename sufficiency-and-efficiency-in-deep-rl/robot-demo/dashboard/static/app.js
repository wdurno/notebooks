const statusEl = document.getElementById("status");
const cameraEl = document.getElementById("camera");
const MAX_HZ = 2;
const FRAME_INTERVAL_MS = Math.floor(1000 / MAX_HZ);

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
  const url = `/api/frame.jpg?t=${Date.now()}`;
  cameraEl.src = url;
}

for (const button of document.querySelectorAll("button[data-endpoint]")) {
  button.addEventListener("click", () => {
    const endpoint = button.dataset.endpoint;
    const action = button.dataset.action;
    sendAction(endpoint, action);
  });
}

setInterval(refreshFrame, FRAME_INTERVAL_MS);
setStatus("Ready.");
