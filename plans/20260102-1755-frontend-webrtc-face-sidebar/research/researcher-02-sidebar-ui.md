# Sidebar UI Patterns for Real-time Face Recognition

**Date:** 2026-01-02
**Focus:** 70/30 layout, avatar display, real-time updates, responsive design

---

## 1. Layout: 70/30 Split (Video | Sidebar)

### CSS Grid (Recommended)
```css
.container {
  display: grid;
  grid-template-columns: 7fr 3fr;
  height: 100vh;
  gap: 0;
}
.video-panel { overflow: hidden; }
.sidebar { overflow-y: auto; }
```
**Pros:** Simpler ratio control, native gap handling, better for 2D layouts
**Cons:** Slightly less browser support (negligible in 2025+)

### Flexbox Alternative
```css
.container {
  display: flex;
  height: 100vh;
}
.video-panel { flex: 7; min-width: 0; }
.sidebar { flex: 3; min-width: 280px; overflow-y: auto; }
```
**Use when:** Need more control over shrink/grow behavior

### Key Considerations
- Use `min-width: 0` on flex children to prevent overflow issues with video
- Video element: `object-fit: cover` or `contain` based on crop preference
- Sidebar needs `overflow-y: auto` for scrollable recognition list

---

## 2. Avatar Display: Base64 vs Preloaded

### Base64 Inline
```html
<img src="data:image/jpeg;base64,/9j/4AAQ..." alt="face" />
```
**Pros:** No extra HTTP requests, immediate render, works offline
**Cons:** ~33% larger payload, no caching, blocks HTML parsing

### Preloaded URLs (Recommended for cached faces)
```javascript
// Preload known faces on init
const faceCache = new Map();
faces.forEach(f => {
  const img = new Image();
  img.src = f.avatarUrl;
  faceCache.set(f.id, img);
});
```

### Hybrid Approach (Best)
- **Known faces:** Preload from server URL, cache in browser
- **New detections:** Use base64 from WebSocket, convert to blob URL
```javascript
function base64ToBlobUrl(base64, mime = 'image/jpeg') {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}
```
**Important:** Call `URL.revokeObjectURL()` when removing items to prevent memory leaks

### Avatar Sizing
```css
.avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
  flex-shrink: 0;
}
```

---

## 3. Real-time List Updates

### Prepend New Entries
```javascript
const MAX_ITEMS = 50;

function addRecognition(entry) {
  const list = document.getElementById('recognition-list');
  const item = createListItem(entry);
  list.prepend(item);  // O(1) DOM operation

  // Limit display count
  while (list.children.length > MAX_ITEMS) {
    const removed = list.lastElementChild;
    URL.revokeObjectURL(removed.querySelector('img')?.src); // cleanup
    removed.remove();
  }
}
```

### Auto-scroll Behavior
```javascript
function shouldAutoScroll(container) {
  // Only auto-scroll if user is near top (within 100px)
  return container.scrollTop < 100;
}

function addWithAutoScroll(entry) {
  const list = document.getElementById('recognition-list');
  const wasAtTop = shouldAutoScroll(list);

  list.prepend(createListItem(entry));

  if (wasAtTop) {
    list.scrollTo({ top: 0, behavior: 'smooth' });
  }
}
```

### Performance: Virtual Scrolling
For >100 items, use virtual scrolling libraries:
- **react-window** (React) - lightweight, ~6kb
- **@tanstack/virtual** (framework-agnostic)
- **Native CSS:** `content-visibility: auto` for partial virtualization

### Debounce High-frequency Updates
```javascript
let pendingUpdates = [];
let rafId = null;

function queueUpdate(entry) {
  pendingUpdates.push(entry);
  if (!rafId) {
    rafId = requestAnimationFrame(flushUpdates);
  }
}

function flushUpdates() {
  const fragment = document.createDocumentFragment();
  pendingUpdates.forEach(e => fragment.prepend(createListItem(e)));
  document.getElementById('recognition-list').prepend(fragment);
  pendingUpdates = [];
  rafId = null;
}
```

---

## 4. Responsive Design

### Breakpoint Strategy
```css
/* Desktop: 70/30 split */
.container {
  display: grid;
  grid-template-columns: 7fr 3fr;
}

/* Tablet (<1024px): 60/40 or stack */
@media (max-width: 1024px) {
  .container { grid-template-columns: 6fr 4fr; }
  .sidebar { min-width: 240px; }
}

/* Mobile (<768px): Stack vertically */
@media (max-width: 768px) {
  .container {
    grid-template-columns: 1fr;
    grid-template-rows: 60vh 40vh;
  }
}

/* Small mobile: Collapsible sidebar */
@media (max-width: 480px) {
  .sidebar {
    position: fixed;
    right: 0;
    width: 100%;
    transform: translateX(100%);
    transition: transform 0.3s;
  }
  .sidebar.open { transform: translateX(0); }
}
```

### Mobile: Floating Recognition Badge
```css
.recognition-badge {
  position: fixed;
  bottom: 20px;
  right: 20px;
  display: none;
}
@media (max-width: 480px) {
  .recognition-badge { display: flex; }
}
```

---

## 5. Implementation Recommendations

| Aspect | Recommendation |
|--------|----------------|
| Layout | CSS Grid `7fr 3fr` |
| Avatar | Blob URL from base64, revoke on remove |
| List updates | `prepend()` + DocumentFragment batch |
| Scroll | Auto-scroll only if user at top |
| Max items | 50-100 visible, virtual scroll if more |
| Responsive | Stack at 768px, collapsible at 480px |

---

## Unresolved Questions

1. **Accessibility:** Should new recognitions announce via ARIA live region? May be noisy for high-frequency updates
2. **Duplicate handling:** Suppress same face within N seconds or show all?
3. **Mobile UX:** Overlay sidebar vs bottom sheet vs toast notifications?
4. **Memory budget:** Max blob URLs before aggressive cleanup needed?
5. **Animation:** Entry animations (fade-in, slide) may cause jank at high update rates - benchmark needed
