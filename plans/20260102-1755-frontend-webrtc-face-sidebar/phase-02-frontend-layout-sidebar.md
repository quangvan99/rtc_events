# Phase 02: Frontend Layout & Sidebar

**Parent:** [plan.md](./plan.md)
**Dependencies:** [Phase 01](./phase-01-python-datachannel.md)
**Status:** Done (2026-01-02)
**Priority:** High

---

## Context Links

- [Sidebar UI Research](./research/researcher-02-sidebar-ui.md)
- [view.html](/home/mq/disk2T/quangnv/face/view.html)
- [features_arcface.json](/home/mq/disk2T/quangnv/face/extract/features_arcface.json)

---

## Overview

Modify `view.html` to:
1. Split layout: 70% video | 30% sidebar
2. Preload face data from `features_arcface.json`
3. Listen DataChannel for face_detected events
4. Display recognition list in sidebar

---

## Key Insights

From research:
- CSS Grid `7fr 3fr` for clean 70/30 split
- `prepend()` for new entries at top
- Limit list to 50 items to prevent memory issues
- Auto-scroll only if user near top

From existing code:
- `view.html:336-339` already creates DataChannel but no handler
- DataChannel is created by browser: `pc.createDataChannel('data')`
- Need to add `onmessage` handler to receive face events

---

## Requirements

**Functional:**
- Show video on left (70%), sidebar on right (30%)
- Sidebar displays: name + timestamp for each recognition
- Newest entries at top
- Max 50 entries

**Non-functional:**
- Responsive: stack on mobile
- No jank during updates
- Clean, minimal UI

---

## Architecture

```
view.html
├── <style>
│   └── Grid layout: 7fr 3fr
│   └── Sidebar styles
│   └── Recognition item styles
├── <body>
│   └── .main-container (grid)
│       ├── .video-panel (existing video)
│       └── .sidebar
│           └── .sidebar-header
│           └── #recognition-list
├── <script>
│   └── preloadFaceData() - fetch features_arcface.json
│   └── VideoInstance.createPeerConnection (modify)
│       └── data_channel.onmessage = handleFaceEvent
│   └── handleFaceEvent(msg) - parse, add to sidebar
│   └── addRecognitionItem(name, time) - prepend to list
```

---

## Related Code Files

| File | Action | Lines |
|------|--------|-------|
| `/home/mq/disk2T/quangnv/face/view.html` | Modify | 6-161 (CSS), 164-183 (HTML), 336-351 (JS) |

---

## Implementation Steps

### Step 1: Add new CSS styles

Location: Inside `<style>` tag, after existing styles (~line 160)

```css
/* Main layout */
.main-container {
    display: grid;
    grid-template-columns: 7fr 3fr;
    height: 100vh;
    gap: 0;
}

/* Sidebar */
.sidebar {
    background: #f8f9fa;
    border-left: 1px solid #dee2e6;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-header {
    padding: 15px;
    background: #343a40;
    color: white;
    font-weight: bold;
}

#recognition-list {
    flex: 1;
    overflow-y: auto;
    padding: 0;
    margin: 0;
    list-style: none;
}

.recognition-item {
    display: flex;
    align-items: center;
    padding: 12px 15px;
    border-bottom: 1px solid #e9ecef;
    animation: fadeIn 0.3s ease;
}

.recognition-item .name {
    font-weight: 500;
    flex: 1;
}

.recognition-item .time {
    font-size: 12px;
    color: #6c757d;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Responsive */
@media (max-width: 768px) {
    .main-container {
        grid-template-columns: 1fr;
        grid-template-rows: 60vh 40vh;
    }
    .sidebar {
        border-left: none;
        border-top: 1px solid #dee2e6;
    }
}
```

### Step 2: Modify HTML structure

Location: Replace body content (~line 164-183)

```html
<body>
    <div class="main-container">
        <!-- Video Panel (existing, wrapped) -->
        <div class="video-panel">
            <div class="video-grid" id="video-grid">
                <div class="video-item" id="video-1">
                    <div class="video-container">
                        <div class="video-number">1</div>
                        <div class="status-badge status-waiting">waiting</div>
                        <div class="video-placeholder">
                            <div>
                                <strong>Video 1</strong><br>
                                <small>Waiting for connection...</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sidebar (NEW) -->
        <div class="sidebar">
            <div class="sidebar-header">Face Recognition</div>
            <ul id="recognition-list"></ul>
        </div>
    </div>
</body>
```

### Step 3: Add face data preload

Location: After `const silentStreamPool` declaration (~line 238)

```javascript
// Face data cache
let faceDataCache = {};

async function preloadFaceData() {
    try {
        // Adjust path based on how features_arcface.json is served
        const response = await fetch('/extract/features_arcface.json');
        faceDataCache = await response.json();
        console.log(`Loaded ${Object.keys(faceDataCache).length} faces`);
    } catch (e) {
        console.warn('Could not load face data:', e);
    }
}
```

### Step 4: Add recognition list handler

Location: After preloadFaceData function

```javascript
const MAX_RECOGNITION_ITEMS = 50;

function addRecognitionItem(name, timestamp) {
    const list = document.getElementById('recognition-list');

    const item = document.createElement('li');
    item.className = 'recognition-item';
    item.innerHTML = `
        <span class="name">${escapeHtml(name)}</span>
        <span class="time">${timestamp}</span>
    `;

    // Prepend to top
    list.prepend(item);

    // Limit items
    while (list.children.length > MAX_RECOGNITION_ITEMS) {
        list.lastElementChild.remove();
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function handleFaceEvent(event) {
    try {
        const data = JSON.parse(event.data);
        if (data.type === 'face_detected') {
            addRecognitionItem(data.name, data.timestamp);
        }
    } catch (e) {
        console.warn('Invalid face event:', e);
    }
}
```

### Step 5: Modify VideoInstance.createPeerConnection

Location: `view.html:336-339` (inside createPeerConnection method)

Replace:
```javascript
this.data_channel = this.peer_connection.createDataChannel('data');
this.data_channel.onopen = () => {
    this.setStatus('connected');
};
```

With:
```javascript
this.data_channel = this.peer_connection.createDataChannel('data');
this.data_channel.onopen = () => {
    this.setStatus('connected');
    console.log('DataChannel open');
};
this.data_channel.onmessage = handleFaceEvent;
this.data_channel.onerror = (e) => console.error('DC error:', e);
this.data_channel.onclose = () => console.log('DataChannel closed');
```

### Step 6: Call preloadFaceData on init

Location: In `setupVideos()` function (~line 460)

```javascript
async function setupVideos() {
    await preloadFaceData();  // ADD this line
    await silentStreamPool.init();

    video_instances = [];
    const instance = new VideoInstance(1);
    video_instances.push(instance);
}
```

### Step 7: Adjust video-grid CSS (optional)

May need to remove padding/margin that conflicts with new layout:

```css
.video-grid {
    padding: 0;  /* was 20px */
    max-width: none;  /* was 1920px */
    height: 100%;
}

.video-panel {
    overflow: hidden;
    background: #000;
}
```

---

## Todo List

- [x] Add CSS styles for main-container, sidebar, recognition items
- [x] Modify HTML structure with main-container grid
- [x] Add preloadFaceData function (NOTE: unused - see review)
- [x] Add addRecognitionItem and handleFaceEvent functions
- [x] Add escapeHtml helper
- [x] Modify createPeerConnection to handle DC messages
- [x] Call preloadFaceData in setupVideos
- [x] Adjust video-grid CSS if needed
- [x] Test responsive layout on different screen sizes
- [x] Test face detection display end-to-end

## Review Findings (2026-01-02)

See: [Review Report](./reports/code-reviewer-260102-phase02-frontend-sidebar.md)

**Action Items:**
- [x] Remove or use `faceDataCache` (YAGNI violation)
- [x] Add null-check for `data.name`/`data.timestamp` in handleFaceEvent

---

## Success Criteria

1. Page shows 70/30 layout on desktop
2. Page stacks vertically on mobile (<768px)
3. Face detections appear in sidebar in real-time
4. Newest detections at top
5. List limited to 50 items
6. No visual jank during updates

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CORS block on JSON fetch | High | Medium | Serve JSON from same origin or add CORS headers |
| CSS conflicts with existing | Medium | Low | Use specific class names, test thoroughly |
| Layout break on edge sizes | Low | Low | Test responsive breakpoints |

---

## Security Considerations

- XSS: Use escapeHtml for name display
- No eval() or innerHTML with user data (except escaped)

---

## Next Steps

After both phases complete:
1. End-to-end testing
2. Consider adding avatar images (optional enhancement)
3. Consider adding face count badge
4. Consider WebSocket fallback if DataChannel fails
