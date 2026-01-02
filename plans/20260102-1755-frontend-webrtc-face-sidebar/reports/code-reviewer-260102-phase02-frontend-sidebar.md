# Code Review: Phase 02 - Frontend WebRTC Face Sidebar

**Date:** 2026-01-02
**Reviewer:** Code Reviewer Agent
**File:** `/home/mq/disk2T/quangnv/face/view.html`

---

## Summary

- **Critical issues:** 0
- **Warnings:** 3
- **Suggestions:** 4

---

## Scope

- Files reviewed: `view.html` (611 lines)
- Review focus: Phase 02 implementation (CSS grid, sidebar, DataChannel handlers)
- Plan file: `phase-02-frontend-layout-sidebar.md`

---

## Overall Assessment

Implementation follows plan spec well. Layout, sidebar, XSS protection, and list limiting all correctly implemented. One notable YAGNI violation: `faceDataCache` preloaded but never used.

---

## Critical Issues

None.

---

## Warnings (High Priority)

### W1: `faceDataCache` is dead code (YAGNI violation)

**Location:** Lines 321-332

```javascript
let faceDataCache = {};

async function preloadFaceData() {
    try {
        const response = await fetch('/extract/features_arcface.json');
        faceDataCache = await response.json();  // Loaded but NEVER referenced
        console.log(`Loaded ${Object.keys(faceDataCache).length} faces`);
    } catch (e) {
        console.warn('Could not load face data:', e);
    }
}
```

**Issue:** `faceDataCache` populated but never referenced anywhere. `handleFaceEvent` receives name directly from DataChannel, doesn't use cache.

**Impact:** Unnecessary network request, memory usage for unused data.

**Fix:** Remove preloadFaceData() and faceDataCache, OR implement actual usage (e.g., display avatar/metadata).

---

### W2: No null-check on DataChannel message fields

**Location:** Lines 359-368

```javascript
function handleFaceEvent(event) {
    try {
        const data = JSON.parse(event.data);
        if (data.type === 'face_detected') {
            addRecognitionItem(data.name, data.timestamp);  // No null check
        }
    } catch (e) {
        console.warn('Invalid face event:', e);
    }
}
```

**Issue:** If `data.name` or `data.timestamp` is undefined/null, will display "undefined" in sidebar.

**Fix:**
```javascript
if (data.type === 'face_detected' && data.name && data.timestamp) {
    addRecognitionItem(data.name, data.timestamp);
}
```

---

### W3: No rate limiting on face events

**Location:** Lines 359-368

**Issue:** Rapid face detection events could trigger many DOM updates/animations, causing jank. No debounce/throttle.

**Impact:** Low probability, medium impact if backend sends rapid bursts.

**Fix (optional):** Add simple throttle or batch updates with requestAnimationFrame.

---

## Medium Priority

### M1: Mixed language comments

**Location:** Lines 111, 135, 543, 560

Vietnamese comments mixed with English. Inconsistent for international collaboration.

```javascript
// Nút unmute ở góc dưới trái  (Vietnamese)
// Nút fullscreen ở góc dưới phải  (Vietnamese)
```

**Recommendation:** Standardize to English.

---

### M2: Console statements in production code

**Location:** Lines 328, 469, 472, 473

```javascript
console.log(`Loaded ${Object.keys(faceDataCache).length} faces`);
console.log('DataChannel open');
console.error('DC error:', e);
console.log('DataChannel closed');
```

**Recommendation:** Remove or wrap in `DEBUG` flag for production.

---

## Suggestions (Low Priority)

### S1: escapeHtml ordering

**Location:** Lines 336-340, 342-357

`escapeHtml` defined AFTER `addRecognitionItem` which calls it. Works due to hoisting but reduces readability.

**Recommendation:** Move `escapeHtml` before `addRecognitionItem`.

---

### S2: Could validate timestamp format

**Location:** Line 349

Currently displays raw timestamp from server. No format validation.

**Recommendation:** Consider formatting with `toLocaleTimeString()` if timestamp is epoch.

---

### S3: setupVideos call wrapped in setTimeout

**Location:** Lines 604-608

```javascript
window.onload = function() {
    setTimeout(async function() {
        await setupVideos();
    }, 100);
};
```

**Question:** Why 100ms delay? If for initialization timing, document with comment. If unnecessary, remove.

---

### S4: video-panel CSS missing margin reset

**Location:** Lines 173-176

```css
.video-panel {
    overflow: hidden;
    background: #000;
}
```

May want explicit `margin: 0; padding: 0;` to prevent Bootstrap overrides.

---

## Positive Observations

1. **XSS protection** - `escapeHtml()` correctly uses textContent/innerHTML pattern
2. **List limiting** - MAX_RECOGNITION_ITEMS = 50 prevents memory bloat
3. **Responsive design** - Mobile breakpoint stacks layout correctly
4. **Error handling** - try-catch around JSON parse
5. **Animation** - Uses transform for GPU acceleration
6. **Viewport meta** - Correctly added

---

## Task Completion Verification

| Task | Status |
|------|--------|
| CSS styles for main-container, sidebar | Done |
| HTML structure with grid | Done |
| preloadFaceData function | Done (but unused) |
| addRecognitionItem function | Done |
| escapeHtml helper | Done |
| handleFaceEvent function | Done |
| Modify createPeerConnection for DC | Done |
| Call preloadFaceData in setupVideos | Done |
| Adjust video-grid CSS | Done |
| Responsive layout | Done |

---

## Recommended Actions

1. **Remove or use `faceDataCache`** - Currently violates YAGNI
2. **Add null-check in handleFaceEvent** - Prevent "undefined" display
3. **Standardize comments to English** - Consistency
4. (Optional) Add throttle for rapid events

---

## Metrics

- **Lines changed:** ~140 (CSS: 72, HTML: 24, JS: 44)
- **Linting issues:** N/A (no linter configured)
- **Type coverage:** N/A (vanilla JS)

---

## Unresolved Questions

1. Why is `faceDataCache` preloaded but never used? Future feature or oversight?
2. What is the expected message rate from DataChannel? Needed for rate-limiting decision.
3. Is the 100ms setTimeout in window.onload intentional? Context needed.

---

**Verdict:** Ready to merge after addressing W1 (dead code) and W2 (null check). Other items can be addressed in follow-up.
