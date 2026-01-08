"""OSD display update logic for face recognition"""


# Display constants
class Colors:
    """OSD display color constants (RGBA)"""
    CONFIRMED = (0.0, 1.0, 0.0, 1.0)  # Green
    UNKNOWN = (1.0, 0.5, 0.0, 1.0)    # Orange
    TEXT = (1.0, 1.0, 1.0, 1.0)       # White
    TEXT_BG = (0.0, 0.0, 0.0, 0.7)    # Black transparent


BORDER_WIDTH = 3
FONT_SIZE = 14
FONT_NAME = "Serif"


class FaceDisplay:
    """Updates OSD display for detected faces"""

    def update(self, obj_meta, name: str, score: float, state: str = "confirmed") -> None:
        """Update OSD display for a detected face"""
        rect = obj_meta.rect_params
        face_w, face_h = int(rect.width), int(rect.height)

        # Border color by state
        r, g, b, a = Colors.CONFIRMED if state == "confirmed" else Colors.UNKNOWN
        rect.border_color.red = r
        rect.border_color.green = g
        rect.border_color.blue = b
        rect.border_color.alpha = a
        rect.border_width = BORDER_WIDTH

        # Display text
        if state == "confirmed":
            display_text = f"{name} ({score:.2f}) [{face_w}x{face_h}]"
        else:
            display_text = f"[{face_w}x{face_h}]"

        text = obj_meta.text_params
        text.display_text = display_text
        text.x_offset = int(rect.left)
        text.y_offset = max(0, int(rect.top) - 25)
        text.font_params.font_name = FONT_NAME
        text.font_params.font_size = FONT_SIZE

        # Text color
        r, g, b, a = Colors.TEXT
        text.font_params.font_color.red = r
        text.font_params.font_color.green = g
        text.font_params.font_color.blue = b
        text.font_params.font_color.alpha = a

        # Text background
        text.set_bg_clr = 1
        r, g, b, a = Colors.TEXT_BG
        text.text_bg_clr.red = r
        text.text_bg_clr.green = g
        text.text_bg_clr.blue = b
        text.text_bg_clr.alpha = a
