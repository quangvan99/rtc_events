# WebRTC DataChannel Research

**Date:** 2026-01-02
**Focus:** RTCDataChannel for real-time JSON transmission

---

## 1. RTCDataChannel API (JavaScript/Browser)

### Creation
```javascript
const pc = new RTCPeerConnection();
const dc = pc.createDataChannel("data", {
  ordered: true,        // guarantee order (default)
  maxRetransmits: 3,    // optional: reliability
});
```

### Key Methods
| Method | Description |
|--------|-------------|
| `send(data)` | Send string/ArrayBuffer/Blob to peer |
| `close()` | Close channel gracefully |

### Key Properties
| Property | Description |
|----------|-------------|
| `readyState` | `connecting` / `open` / `closing` / `closed` |
| `bufferedAmount` | Bytes queued for sending |
| `binaryType` | `arraybuffer` (default) or `blob` |
| `label` | Channel name identifier |

---

## 2. Event Handling

```javascript
dc.onopen = () => {
  console.log("Channel open, ready to send");
};

dc.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // handle parsed JSON
};

dc.onclose = () => {
  console.log("Channel closed");
};

dc.onerror = (error) => {
  console.error("DataChannel error:", error);
};
```

### Receiving Channel (answerer side)
```javascript
pc.ondatachannel = (event) => {
  const dc = event.channel;
  dc.onmessage = (e) => { /* handle */ };
  dc.onopen = () => { /* ready */ };
};
```

---

## 3. JSON Best Practices

### Sending
```javascript
function sendJSON(dc, obj) {
  if (dc.readyState !== "open") return false;
  dc.send(JSON.stringify(obj));
  return true;
}
```

### Receiving with Validation
```javascript
dc.onmessage = (event) => {
  try {
    const msg = JSON.parse(event.data);
    if (!msg.type) throw new Error("Missing type");
    handleMessage(msg);
  } catch (e) {
    console.error("Invalid JSON:", e);
  }
};
```

### Message Protocol Pattern
```javascript
// Structured message format
const message = {
  type: "face_detection",
  timestamp: Date.now(),
  payload: { faces: [...], count: 3 }
};
```

### Flow Control
```javascript
// Check buffer before sending
if (dc.bufferedAmount < 16384) {
  dc.send(JSON.stringify(data));
}
```

---

## 4. GStreamer webrtcbin DataChannel (Python)

### Creating DataChannel
```python
from gi.repository import Gst, GstWebRTC

# After webrtcbin created
webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtc")

# Create outgoing channel
channel = webrtcbin.emit("create-data-channel", "data", None)
```

### Receiving DataChannel
```python
def on_data_channel(webrtcbin, channel):
    channel.connect("on-open", on_channel_open)
    channel.connect("on-message-string", on_message_string)
    channel.connect("on-close", on_channel_close)

webrtcbin.connect("on-data-channel", on_data_channel)
```

### Event Handlers
```python
def on_channel_open(channel):
    print("DataChannel open")

def on_message_string(channel, message):
    data = json.loads(message)
    # process JSON

def on_channel_close(channel):
    print("DataChannel closed")
```

### Sending JSON
```python
import json

def send_json(channel, obj):
    message = json.dumps(obj)
    channel.emit("send-string", message)
```

### Complete Pattern
```python
class DataChannelHandler:
    def __init__(self, webrtcbin):
        self.channel = None
        webrtcbin.connect("on-data-channel", self._on_data_channel)

    def _on_data_channel(self, _, channel):
        self.channel = channel
        channel.connect("on-open", lambda c: print("Open"))
        channel.connect("on-message-string", self._on_message)
        channel.connect("on-close", lambda c: print("Closed"))

    def _on_message(self, channel, msg):
        data = json.loads(msg)
        self.handle(data)

    def send(self, obj):
        if self.channel:
            self.channel.emit("send-string", json.dumps(obj))
```

---

## 5. Key Considerations

| Aspect | Recommendation |
|--------|----------------|
| Reliability | Use `ordered: true` for JSON integrity |
| Size limit | Keep messages < 16KB per send |
| Serialization | `JSON.stringify/parse` both sides |
| Error handling | Always wrap parse in try/catch |
| State check | Verify `readyState === "open"` before send |

---

## Unresolved Questions

1. GStreamer webrtcbin binary data support via DataChannel?
2. Maximum reliable message size for webrtcbin?
3. Backpressure handling when GStreamer channel buffer full?
4. Thread safety of `emit("send-string")` from non-main thread?

---

**Sources:**
- [MDN RTCDataChannel](https://developer.mozilla.org/en-US/docs/Web/API/RTCDataChannel)
- [GStreamer webrtcbin docs](https://gstreamer.freedesktop.org/documentation/webrtc/)
