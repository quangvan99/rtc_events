# Frontend WebRTC Face Recognition Sidebar

**Created:** 2026-01-02
**Status:** Done (All Phases Complete)

---

## Overview

Add sidebar to `view.html` displaying recognized faces in real-time. Layout: 70% video | 30% sidebar. Data sent via WebRTC DataChannel from `stream_face.py`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        view.html                              │
├─────────────────────────────┬────────────────────────────────┤
│   Video Stream (70%)        │  Sidebar (30%)                 │
│                             │ ┌────────────────────────────┐ │
│   [WebRTC Video]            │ │ [Avatar] Quang - 16:35:22  │ │
│                             │ │ [Avatar] Minh  - 16:34:15  │ │
│                             │ └────────────────────────────┘ │
└─────────────────────────────┴────────────────────────────────┘
              ▲                            ▲
              │ Video Track                │ DataChannel JSON
              └───────────┬────────────────┘
                          │
              ┌───────────┴───────────┐
              │   stream_face.py      │
              │  (sends face events)  │
              └───────────────────────┘
```

---

## Phases

| # | Phase | Status | Progress | File |
|---|-------|--------|----------|------|
| 1 | Python: DataChannel Integration | Done | 100% | [phase-01](./phase-01-python-datachannel.md) |
| 2 | Frontend: Layout & Sidebar | Done | 100% | [phase-02](./phase-02-frontend-layout-sidebar.md) |

---

## Key Decisions

| Aspect | Decision | Reason |
|--------|----------|--------|
| Data transport | DataChannel (not WS) | Already exists, simpler |
| Layout | CSS Grid 7fr/3fr | Cleaner ratio control |
| Avatar source | Preload from JSON | `features_arcface.json` has face data |
| List limit | 50 items max | Prevent memory issues |

---

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `stream_face.py` | Modify | Add DataChannel handler, send face events |
| `view.html` | Modify | Add 70/30 layout, sidebar, DC message handler |

---

## Dependencies

- `features_arcface.json` accessible from browser (need CORS or static serve)
- Existing WebRTC connection must be stable

---

## Research

- [DataChannel Research](./research/researcher-01-datachannel.md)
- [Sidebar UI Research](./research/researcher-02-sidebar-ui.md)

---

## Open Questions

1. How to serve `features_arcface.json` to browser? (CORS / same origin?)
2. Include avatar image in JSON or just name reference?
3. Max history items to display?
4. Duplicate face handling within short time window?
