# Frontend WebRTC - Face Recognition Display

## Status: Phase 02 Complete (2026-01-02)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        view.html                             │
├─────────────────────────────┬───────────────────────────────┤
│                             │  Sidebar (30%)                │
│   Video Stream (70%)        │ ┌─────────────────────────┐  │
│   CSS Grid: 7fr 3fr         │ │ Name - 16:35:22         │  │
│                             │ │ Name - 16:34:15         │  │
│   [WebRTC Video]            │ │ Name - 16:33:01         │  │
│                             │ └─────────────────────────┘  │
│                             │  Max 50 items, LIFO order    │
└─────────────────────────────┴───────────────────────────────┘
                    ▲                         ▲
                    │ Video Track             │ DataChannel JSON
                    │                         │ {type, name, timestamp}
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴─────────┐
                    │   stream_face.py      │
                    └───────────────────────┘
```

---

## Implementation Summary

### Layout (CSS Grid)
- Main: `grid-template-columns: 7fr 3fr`
- Mobile (<768px): stacked, 60vh/40vh split
- Viewport meta + title tag added

### Sidebar
- Header: "Face Recognition" (dark bg)
- List: `#recognition-list` ul, scrollable
- Item: `.recognition-item` with name + timestamp
- Limit: 50 items max, oldest removed

### Functions Added

| Function | Purpose |
|----------|---------|
| `preloadFaceData()` | Fetch `/extract/features_arcface.json` |
| `escapeHtml(text)` | XSS prevention |
| `addRecognitionItem(name, ts)` | Prepend item, trim overflow |
| `handleFaceEvent(event)` | Parse DC message, call addRecognitionItem |

### DataChannel Integration
- Created in `createPeerConnection()`
- `onmessage` -> `handleFaceEvent`
- Expected message format:
  ```json
  {"type": "face_detected", "name": "...", "timestamp": "..."}
  ```

---

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Face metadata | `/extract/features_arcface.json` | Preloaded, non-blocking |
| Recognition events | DataChannel | Realtime from stream_face.py |

---

## Pending (Future)

- [ ] Avatar display (base64 from features_arcface.json)
- [ ] stream_face.py: send face events via DataChannel
