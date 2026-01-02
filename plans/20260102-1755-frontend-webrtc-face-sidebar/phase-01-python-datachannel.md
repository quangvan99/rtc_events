# Phase 01: Python DataChannel Integration

**Parent:** [plan.md](./plan.md)
**Status:** Complete
**Priority:** High

---

## Context Links

- [DataChannel Research](./research/researcher-01-datachannel.md)
- [stream_face.py](/home/mq/disk2T/quangnv/face/stream_face.py)

---

## Overview

Modify `stream_face.py` to:
1. Accept incoming DataChannel from browser
2. Send face detection events as JSON when face recognized

---

## Key Insights

From research:
- GStreamer webrtcbin: connect `on-data-channel` signal to receive channel
- Use `channel.emit("send-string", json.dumps(obj))` to send
- Signals: `on-open`, `on-message-string`, `on-close`
- Thread safety: emit from GStreamer thread is OK

From existing code:
- `stream_face.py:336` creates DataChannel `createDataChannel('data')` but only on browser side
- Python needs to listen for incoming channel, not create one
- Face label assigned at line 341: `trk.label = self.person_names[...]`

---

## Requirements

**Functional:**
- Send JSON when face successfully recognized (has label)
- Message format: `{"type": "face_detected", "name": "...", "timestamp": "..."}`
- Only send once per object_id (not every frame)

**Non-functional:**
- No performance impact on video pipeline
- Message < 16KB (trivial for this use case)

---

## Architecture

```
WebRTCFaceClient
├── __init__
│   └── self.data_channel = None
│   └── self.sent_faces = set()  # track already-sent object_ids
├── _create_pipeline (existing)
│   └── webrtc.connect('on-data-channel', self._on_data_channel)  # NEW
├── _on_data_channel (NEW)
│   └── Store channel reference
│   └── Connect on-open, on-close handlers
├── _probe_recognize (modify)
│   └── After setting trk.label, call _send_face_event(oid, label)
├── _send_face_event (NEW)
│   └── Check if already sent for this oid
│   └── Send JSON via DataChannel
```

---

## Related Code Files

| File | Action | Lines |
|------|--------|-------|
| `/home/mq/disk2T/quangnv/face/stream_face.py` | Modify | 107-118, 229-231, 340-341 |

---

## Implementation Steps

### Step 1: Add DataChannel state to __init__

Location: `stream_face.py:107-119`

```python
def __init__(self, loop):
    self.loop = loop
    self.conn = None
    self.pipe = None
    self.webrtc = None
    self.data_channel = None      # NEW
    self.sent_faces = set()       # NEW: track sent object_ids
    # ... rest unchanged
```

### Step 2: Connect on-data-channel signal

Location: After line 230 (after existing webrtc signal connections)

```python
self.webrtc.connect('on-data-channel', self._on_data_channel)
```

### Step 3: Add _on_data_channel handler

Location: After `_on_ice` method (~line 395)

```python
def _on_data_channel(self, _, channel):
    print('DataChannel received')
    self.data_channel = channel
    channel.connect('on-open', lambda c: print('DataChannel open'))
    channel.connect('on-close', lambda c: self._on_channel_close())
    channel.connect('on-message-string', self._on_channel_message)

def _on_channel_close(self):
    print('DataChannel closed')
    self.data_channel = None

def _on_channel_message(self, channel, msg):
    # For now, just log - browser might send commands later
    print(f'DC message: {msg}')
```

### Step 4: Add _send_face_event method

Location: After DataChannel handlers

```python
def _send_face_event(self, object_id, name):
    if object_id in self.sent_faces:
        return
    if not self.data_channel:
        return

    self.sent_faces.add(object_id)
    msg = {
        'type': 'face_detected',
        'name': name,
        'timestamp': time.strftime('%H:%M:%S'),
        'object_id': object_id
    }
    try:
        self.data_channel.emit('send-string', json.dumps(msg))
        print(f'Sent face event: {name}')
    except Exception as e:
        print(f'Failed to send face event: {e}')
```

### Step 5: Modify _probe_recognize to send events

Location: `stream_face.py:340-341`

After:
```python
if len(trk.votes) >= VOTE_THRESHOLD:
    trk.label = self.person_names[np.bincount(trk.votes).argmax()]
```

Add:
```python
    self._send_face_event(trk.object_id, trk.label)
```

### Step 6: Clean up sent_faces on tracker cleanup

Location: In `TrackerManager.cleanup()` or via callback

Option A: Add to existing cleanup:
```python
def cleanup(self):
    for trk in self.trackers:
        trk.age += 1
    removed = [t.object_id for t in self.trackers if t.age > self.max_age]
    self.trackers = [t for t in self.trackers if t.age <= self.max_age]
    return removed  # Return for sent_faces cleanup
```

Option B (simpler): Clear sent_faces periodically in _probe_fps

---

## Todo List

- [x] Add data_channel and sent_faces to __init__
- [x] Connect on-data-channel signal in _create_pipeline
- [x] Implement _on_data_channel, _on_channel_close, _on_channel_message
- [x] Implement _send_face_event
- [x] Call _send_face_event after label assignment
- [x] Modify TrackerManager.cleanup() to return removed IDs
- [x] Update _probe_fps to clean sent_faces on tracker removal
- [ ] Test DataChannel connection with browser
- [ ] Test face event transmission

---

## Success Criteria

1. DataChannel connects when browser opens view.html
2. JSON message sent when face first recognized
3. Same face not sent multiple times (per object_id)
4. No pipeline performance degradation
5. Console shows "Sent face event: {name}" for each new recognition

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Thread safety issue | Low | Medium | GStreamer signals are thread-safe |
| DataChannel not ready | Medium | Low | Check self.data_channel before send |
| Memory leak in sent_faces | Low | Low | Clear periodically or on tracker cleanup |

---

## Security Considerations

- No user input from DataChannel in this phase
- Future: validate any messages from browser

---

## Next Steps

After this phase:
1. Test with browser console
2. Proceed to Phase 02: Frontend sidebar implementation
