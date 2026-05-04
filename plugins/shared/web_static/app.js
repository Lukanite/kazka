const log         = document.getElementById('log');
const input       = document.getElementById('input');
const sendBtn     = document.getElementById('send');
const dot         = document.getElementById('status-dot');
const statusTx    = document.getElementById('status-text');
const attachBtn   = document.getElementById('attach');
const fileInput   = document.getElementById('file-input');
const previewStrip = document.getElementById('preview-strip');

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5 MB

let ws;
let currentMsg = null;   // streaming assistant bubble
let lastUserEl = null;   // last user bubble (editable)
let reconnectDelay = 1000;
let pendingImages = [];  // {data: base64, media_type: string, name: string}

function setStatus(cls, text) {
  dot.className = cls;
  statusTx.textContent = text;
}

function appendUser(text, images) {
  // Remove editable status from previous user bubble
  if (lastUserEl) lastUserEl.classList.remove('editable');

  const el = document.createElement('div');
  el.className = 'msg user editable';

  // Render image thumbnails if present
  if (images && images.length) {
    const imgContainer = document.createElement('div');
    imgContainer.className = 'msg-images';
    for (const img of images) {
      const imgEl = document.createElement('img');
      imgEl.src = 'data:' + (img.media_type || 'image/jpeg') + ';base64,' + img.data;
      imgEl.alt = img.name || 'image';
      imgContainer.appendChild(imgEl);
    }
    el.appendChild(imgContainer);
  }

  const textNode = document.createElement('span');
  textNode.textContent = text;
  el.appendChild(textNode);
  el.addEventListener('click', () => startEdit(el));
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  lastUserEl = el;
}

function startEdit(el) {
  if (!el.classList.contains('editable')) return;
  if (el.classList.contains('editing')) return;

  // Extract text and images from the bubble before replacing with edit UI
  const textSpan = el.querySelector('span');
  const originalText = textSpan ? textSpan.textContent : el.textContent;
  const originalHTML = el.innerHTML;

  // Preserve attached images so they survive the edit round-trip
  const imgContainer = el.querySelector('.msg-images');
  const editImages = [];
  if (imgContainer) {
    for (const img of imgContainer.querySelectorAll('img')) {
      // Parse "data:<media_type>;base64,<data>" back out
      const src = img.src;
      if (src.startsWith('data:')) {
        const [header, data] = src.split(',', 2);
        const media_type = header.split(':')[1].split(';')[0];
        editImages.push({ data, media_type, name: img.alt || 'image' });
      }
    }
  }

  el.classList.remove('editable');
  el.classList.add('editing');
  el.innerHTML = '';

  // Editable image strip
  const editImgStrip = document.createElement('div');
  editImgStrip.className = 'edit-images';

  function rebuildEditImages() {
    editImgStrip.innerHTML = '';
    editImages.forEach((entry, i) => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      const img = document.createElement('img');
      img.src = 'data:' + entry.media_type + ';base64,' + entry.data;
      img.alt = entry.name || 'image';
      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove';
      removeBtn.textContent = '\u00D7';
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        editImages.splice(i, 1);
        rebuildEditImages();
      });
      item.appendChild(img);
      item.appendChild(removeBtn);
      editImgStrip.appendChild(item);
    });
    // Add-image button
    const addBtn = document.createElement('button');
    addBtn.className = 'add-img';
    addBtn.textContent = '+';
    addBtn.title = 'Add image';
    const editFileInput = document.createElement('input');
    editFileInput.type = 'file';
    editFileInput.accept = 'image/*';
    editFileInput.multiple = true;
    editFileInput.hidden = true;
    addBtn.addEventListener('click', (e) => { e.stopPropagation(); editFileInput.click(); });
    editFileInput.addEventListener('change', () => {
      for (const file of editFileInput.files) {
        if (!file.type.startsWith('image/') || file.size > MAX_IMAGE_SIZE) continue;
        const reader = new FileReader();
        reader.onload = () => {
          const [header, data] = reader.result.split(',', 2);
          const media_type = header.split(':')[1].split(';')[0];
          editImages.push({ data, media_type, name: file.name });
          rebuildEditImages();
        };
        reader.readAsDataURL(file);
      }
      editFileInput.value = '';
    });
    editImgStrip.appendChild(addBtn);
    editImgStrip.appendChild(editFileInput);
  }

  rebuildEditImages();

  const textarea = document.createElement('textarea');
  textarea.className = 'edit-area';
  textarea.value = originalText;
  textarea.rows = Math.max(1, Math.ceil(originalText.length / 50));

  // Paste images into the edit textarea
  textarea.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        const file = item.getAsFile();
        if (!file || file.size > MAX_IMAGE_SIZE) continue;
        const reader = new FileReader();
        reader.onload = () => {
          const [header, data] = reader.result.split(',', 2);
          const media_type = header.split(':')[1].split(';')[0];
          editImages.push({ data, media_type, name: file.name || 'pasted' });
          rebuildEditImages();
        };
        reader.readAsDataURL(file);
      }
    }
  });

  const buttons = document.createElement('div');
  buttons.className = 'edit-buttons';
  const saveBtn = document.createElement('button');
  saveBtn.className = 'save';
  saveBtn.textContent = 'Send';
  const discardBtn = document.createElement('button');
  discardBtn.className = 'discard';
  discardBtn.textContent = 'Cancel';
  buttons.appendChild(discardBtn);
  buttons.appendChild(saveBtn);

  el.appendChild(editImgStrip);
  el.appendChild(textarea);
  el.appendChild(buttons);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);

  function finishEdit(submit) {
    const newText = textarea.value.trim();

    if (submit && (newText || editImages.length)) {
      // Server will broadcast undo_last (removing this bubble)
      // followed by user_input (re-rendering the new text).
      const editPayload = { type: 'edit_last', text: newText };
      if (editImages.length) editPayload.images = editImages;
      ws.send(JSON.stringify(editPayload));
    } else {
      // Cancel -- restore the original bubble
      el.innerHTML = originalHTML;
      el.classList.remove('editing');
      el.classList.add('editable');
    }
  }

  saveBtn.addEventListener('click', (e) => { e.stopPropagation(); finishEdit(true); });
  discardBtn.addEventListener('click', (e) => { e.stopPropagation(); finishEdit(false); });
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); finishEdit(true); }
    if (e.key === 'Escape') finishEdit(false);
    e.stopPropagation();
  });
}

function startAssistantBubble(source) {
  currentMsg = document.createElement('div');
  currentMsg.className = 'msg assistant';
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = source ? `Kazka [${source}]` : 'Kazka';
  const body = document.createElement('span');
  body.className = 'body';
  currentMsg.appendChild(label);
  currentMsg.appendChild(body);
  log.appendChild(currentMsg);
  return body;
}

function appendError(text) {
  const el = document.createElement('div');
  el.className = 'msg error';
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------------------
// Tool call rendering
// ---------------------------------------------------------------------------

const MAX_PREVIEW_LINES = 20;

function clipLines(text) {
  // Returns {clipped, full, wasClipped} for the "show more" affordance
  const lines = String(text).split('\n');
  if (lines.length <= MAX_PREVIEW_LINES) {
    return { clipped: text, full: text, wasClipped: false };
  }
  return {
    clipped: lines.slice(0, MAX_PREVIEW_LINES).join('\n'),
    full: text,
    wasClipped: true,
  };
}

function formatInput(input) {
  // Tool args arrive as a JSON string; pretty-print if parseable, else show raw
  if (typeof input !== 'string') return JSON.stringify(input, null, 2);
  try { return JSON.stringify(JSON.parse(input), null, 2); }
  catch { return input; }
}

function attachClippedText(parent, text) {
  const { clipped, full, wasClipped } = clipLines(text);
  const pre = document.createElement('pre');
  pre.className = 'tool-text';
  pre.textContent = clipped;
  parent.appendChild(pre);
  if (wasClipped) {
    const more = document.createElement('button');
    more.className = 'tool-more';
    more.textContent = `Show ${full.split('\n').length - MAX_PREVIEW_LINES} more lines`;
    more.addEventListener('click', (e) => {
      e.stopPropagation();
      pre.textContent = full;
      more.remove();
    });
    parent.appendChild(more);
  }
}

// Close the streaming bubble so a tool pill lands between assistant chunks
// rather than inside the body span. Two-bubble layout (see design doc).
function closeAssistantBubble() {
  currentMsg = null;
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
}

function appendToolCall(msg) {
  closeAssistantBubble();

  const el = document.createElement('details');
  el.className = 'msg tool-call pending';
  el.dataset.toolId = msg.id || '';

  const summary = document.createElement('summary');
  summary.className = 'tool-summary';

  const caret = document.createElement('span');
  caret.className = 'tool-caret';
  caret.textContent = '▶'; // ▶ U+25B6 — fills its em-box; CSS rotates 90° when open
  const icon = document.createElement('span');
  icon.className = 'tool-icon';
  icon.textContent = '🔧';
  const name = document.createElement('span');
  name.className = 'tool-name';
  name.textContent = msg.name || 'tool';
  const status = document.createElement('span');
  status.className = 'tool-status';
  // Spinner placeholder; CSS animates it
  const spinner = document.createElement('span');
  spinner.className = 'tool-spinner';
  status.appendChild(spinner);

  summary.appendChild(caret);
  summary.appendChild(icon);
  summary.appendChild(name);
  summary.appendChild(status);
  el.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'tool-body';

  const inputLabel = document.createElement('div');
  inputLabel.className = 'tool-label';
  inputLabel.textContent = 'Input';
  body.appendChild(inputLabel);
  attachClippedText(body, formatInput(msg.input));

  // Result section is filled in by appendToolResult
  const resultSection = document.createElement('div');
  resultSection.className = 'tool-result-section';
  body.appendChild(resultSection);

  el.appendChild(body);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function appendToolResult(msg) {
  // Pair with the matching tool_call by id
  const id = msg.id || '';
  let el = id ? log.querySelector(`.tool-call[data-tool-id="${CSS.escape(id)}"]`) : null;

  if (!el) {
    // Orphan result (catch-up replay edge case): render as a standalone box
    closeAssistantBubble();
    el = document.createElement('details');
    el.className = 'msg tool-call';
    el.dataset.toolId = id;

    const summary = document.createElement('summary');
    summary.className = 'tool-summary';
    const caret = document.createElement('span');
    caret.className = 'tool-caret';
    caret.textContent = '▶';
    const icon = document.createElement('span');
    icon.className = 'tool-icon';
    icon.textContent = '🔧';
    const name = document.createElement('span');
    name.className = 'tool-name';
    name.textContent = msg.name || 'tool';
    summary.appendChild(caret);
    summary.appendChild(icon);
    summary.appendChild(name);
    summary.appendChild(document.createElement('span')).className = 'tool-status';
    el.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'tool-body';
    const resultSection = document.createElement('div');
    resultSection.className = 'tool-result-section';
    body.appendChild(resultSection);
    el.appendChild(body);
    log.appendChild(el);
  }

  el.classList.remove('pending');
  el.classList.add(msg.success ? 'success' : 'error');

  // Replace spinner with check / cross
  const status = el.querySelector('.tool-status');
  if (status) {
    status.innerHTML = '';
    status.textContent = msg.success ? '✓' : '✗';
  }

  const resultSection = el.querySelector('.tool-result-section');
  if (resultSection) {
    resultSection.innerHTML = '';
    const label = document.createElement('div');
    label.className = 'tool-label';
    label.textContent = msg.success ? 'Result' : 'Error';
    resultSection.appendChild(label);
    attachClippedText(resultSection, msg.result || '');
  }

  log.scrollTop = log.scrollHeight;
}

let thinkingEl = null;
let thinkingText = '';  // accumulated thinking for current response

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    // Clear the log before catch-up replay to avoid duplicates on reconnect
    log.innerHTML = '';
    currentMsg = null;
    lastUserEl = null;
    thinkingEl = null;
    thinkingText = '';

    setStatus('connected', 'Connected');
    input.disabled = false;
    sendBtn.disabled = false;
    attachBtn.disabled = false;
    input.focus();
    reconnectDelay = 1000;
  };

  ws.onclose = () => {
    setStatus('', 'Disconnected \u2014 reconnecting\u2026');
    input.disabled = true;
    sendBtn.disabled = true;
    attachBtn.disabled = true;
    currentMsg = null;
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    if (msg.type === 'user_input') {
      appendUser(msg.text, msg.images);

    } else if (msg.type === 'chunk') {
      // Remove any lingering thinking bubble
      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }

      if (!currentMsg) {
        const body = startAssistantBubble(msg.source);
        currentMsg._body = body;
        // Attach accumulated thinking as a collapsible section
        if (thinkingText) {
          const details = document.createElement('details');
          details.className = 'thinking-block';
          const summary = document.createElement('summary');
          summary.textContent = 'Thinking';
          const content = document.createElement('div');
          content.className = 'thinking-content';
          content.textContent = thinkingText;
          details.appendChild(summary);
          details.appendChild(content);
          // Insert before the body span
          currentMsg.insertBefore(details, body);
          thinkingText = '';
        }
      }
      const body = currentMsg._body || currentMsg.querySelector('.body');
      body.textContent += msg.text;
      if (msg.is_final) { currentMsg = null; }
      log.scrollTop = log.scrollHeight;

    } else if (msg.type === 'thinking') {
      thinkingText += msg.text;
      if (!thinkingEl) {
        thinkingEl = document.createElement('div');
        thinkingEl.className = 'msg thinking';
        thinkingEl.textContent = '\uD83D\uDCAD ';
        log.appendChild(thinkingEl);
      }
      thinkingEl.textContent = '\uD83D\uDCAD ' + thinkingText;
      log.scrollTop = log.scrollHeight;

    } else if (msg.type === 'state') {
      const stateMap = {
        LISTENING: ['listening', 'Listening\u2026'],
        PROCESSING_VAD: ['processing', 'Processing\u2026'],
        PROCESSING_PTT: ['processing', 'Processing\u2026'],
        VERIFYING: ['listening', 'Verifying\u2026'],
        SPEAKING: ['speaking', 'Speaking\u2026'],
        WAITING: ['connected', 'Ready'],
      };
      const [cls, text] = stateMap[msg.state] || ['connected', msg.state];
      setStatus(cls, text);
      // A new state means a new response is coming -- close any open bubble
      if (msg.state === 'PROCESSING_VAD' || msg.state === 'PROCESSING_PTT') {
        currentMsg = null;
        thinkingText = '';
      }

    } else if (msg.type === 'clear') {
      log.innerHTML = '';
      currentMsg = null;
      lastUserEl = null;
      thinkingText = '';
      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }

    } else if (msg.type === 'undo_last') {
      // Remove messages from the end: assistant bubble(s), then user bubble
      while (log.lastChild && !log.lastChild.classList.contains('user')) {
        log.removeChild(log.lastChild);
      }
      if (log.lastChild && log.lastChild.classList.contains('user')) {
        log.removeChild(log.lastChild);
      }
      currentMsg = null;
      thinkingText = '';
      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
      // Update lastUserEl to the new last user bubble
      const userMsgs = log.querySelectorAll('.msg.user');
      lastUserEl = userMsgs.length ? userMsgs[userMsgs.length - 1] : null;
      if (lastUserEl) lastUserEl.classList.add('editable');

    } else if (msg.type === 'tool_call') {
      appendToolCall(msg);

    } else if (msg.type === 'tool_result') {
      appendToolResult(msg);

    } else if (msg.type === 'error') {
      appendError('Error: ' + msg.message);
    }
  };
}

// ---------------------------------------------------------------------------
// Image attachment helpers
// ---------------------------------------------------------------------------

function addImageFile(file) {
  if (!file.type.startsWith('image/')) return;
  if (file.size > MAX_IMAGE_SIZE) {
    alert('Image too large (max 5 MB): ' + file.name);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    // reader.result is "data:<media_type>;base64,<data>"
    const [header, data] = reader.result.split(',', 2);
    const media_type = header.split(':')[1].split(';')[0];
    const entry = { data, media_type, name: file.name };
    pendingImages.push(entry);
    renderPreview(entry, pendingImages.length - 1);
  };
  reader.readAsDataURL(file);
}

function renderPreview(entry, index) {
  const item = document.createElement('div');
  item.className = 'preview-item';
  item.dataset.index = index;

  const img = document.createElement('img');
  img.src = 'data:' + entry.media_type + ';base64,' + entry.data;
  img.alt = entry.name;

  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove';
  removeBtn.textContent = '\u00D7';
  removeBtn.addEventListener('click', () => {
    const idx = parseInt(item.dataset.index, 10);
    pendingImages.splice(idx, 1);
    rebuildPreviews();
  });

  item.appendChild(img);
  item.appendChild(removeBtn);
  previewStrip.appendChild(item);
}

function rebuildPreviews() {
  previewStrip.innerHTML = '';
  pendingImages.forEach((entry, i) => renderPreview(entry, i));
}

function clearPendingImages() {
  pendingImages = [];
  previewStrip.innerHTML = '';
}

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  for (const file of fileInput.files) addImageFile(file);
  fileInput.value = '';
});

// Clipboard paste support
input.addEventListener('paste', (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      addImageFile(item.getAsFile());
    }
  }
});

// ---------------------------------------------------------------------------
// Send
// ---------------------------------------------------------------------------

function send() {
  const text = input.value.trim();
  const images = pendingImages.length ? pendingImages.slice() : null;
  if ((!text && !images) || !ws || ws.readyState !== WebSocket.OPEN) return;

  const payload = { type: 'text_input', text };
  if (images) payload.images = images;
  ws.send(JSON.stringify(payload));

  appendUser(text, images);
  input.value = '';
  clearPendingImages();
  currentMsg = null;  // next assistant message is a fresh bubble
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });

connect();
