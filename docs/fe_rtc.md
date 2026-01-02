# Frontend WebRTC - Face Recognition Display

## Brainstorm: Hiển thị thông tin người nhận diện

### Yêu cầu

1. **Layout:** Stream 70% trái | Sidebar 30% phải
2. **Sidebar hiển thị:** Tên, avatar (base64), thời gian xuất hiện
3. **Data source:** `features_arcface.json` cho avatar/tên

---

## Phân tích

### Dữ liệu

| Data | Source | Type |
|------|--------|------|
| Avatar, Tên | `features_arcface.json` | Static (preload) |
| Thời gian xuất hiện | `stream_face.py` | Realtime via DataChannel |
| Ai đang xuất hiện | `stream_face.py` | Realtime via DataChannel |

### Kiến trúc đề xuất

```
┌─────────────────────────────────────────────────────────────┐
│                        view.html                             │
├─────────────────────────────┬───────────────────────────────┤
│                             │  Sidebar (30%)                │
│   Video Stream (70%)        │ ┌─────────────────────────┐  │
│                             │ │ 👤 Quang - 16:35:22     │  │
│   [WebRTC Video]            │ │ 👤 Minh  - 16:34:15     │  │
│                             │ │ 👤 Hùng  - 16:33:01     │  │
│                             │ └─────────────────────────┘  │
│                             │                               │
└─────────────────────────────┴───────────────────────────────┘
                    ▲                         ▲
                    │ Video                   │ JSON via DataChannel
                    │                         │ {name: "Quang", time: "16:35:22"}
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴─────────┐
                    │   stream_face.py      │
                    │    (recognition)      │
                    └───────────────────────┘
```

---

## Phương án so sánh

| Aspect | **A: DataChannel JSON** | **B: WebSocket riêng** | **C: Overlay trên video** |
|--------|------------------------|------------------------|--------------------------|
| Complexity | Thấp (có sẵn DC) | Cao (thêm WS) | Trung bình |
| Realtime | ✅ | ✅ | ✅ |
| Tách biệt | ✅ Sidebar riêng | ✅ | ❌ Dính video |
| Maintainability | Tốt | Phức tạp | OK |
| **Recommend** | ⭐⭐⭐ | ⭐ | ⭐⭐ |

---

## Phương án chọn: A - DataChannel JSON

### Lý do

- DataChannel đã được tạo sẵn (`createDataChannel('data')`)
- Không cần thêm kết nối mới
- Đơn giản, tuân thủ KISS principle

### Data Flow

1. `stream_face.py` detect face → gửi JSON qua DataChannel:
   ```json
   {"type": "face_detected", "name": "Quang", "timestamp": "16:35:22"}
   ```
2. `view.html` nhận message, tra cứu avatar từ `features_arcface.json` (preload)
3. Cập nhật sidebar

### Cần implement

| File | Thay đổi |
|------|----------|
| `stream_face.py` | Gửi face detection data qua DataChannel |
| `view.html` | Layout 70/30, preload JSON, xử lý DC message, render sidebar |

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Tận dụng DC có sẵn | Cần sửa cả Python & JS |
| Realtime performance | features_arcface.json cần accessible từ browser |
| Đơn giản hóa | Cần CORS/serve static file |

---

## Câu hỏi mở

1. Lịch sử hay chỉ realtime?
2. `features_arcface.json` serve như thế nào? (cùng folder với view.html?)
3. Giới hạn số người hiển thị trong sidebar?
