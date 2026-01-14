"""
CTC Label Decode for License Plate OCR

Adapted from PaddleOCR - removed torch dependency, numpy only.
"""

import numpy as np
import re
from typing import List, Tuple, Optional


class BaseRecLabelDecode:
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path: Optional[str] = None, use_space_char: bool = False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
            dict_character = self.character_str
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character: List[str]) -> List[str]:
        return dict_character

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: Optional[np.ndarray] = None,
        is_remove_duplicate: bool = False,
    ) -> List[Tuple[str, float]]:
        """Convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = np.ones(len(selection))

            if len(conf_list) == 0:
                conf_list = np.array([0.0])

            text = "".join(char_list)
            result_list.append((text, float(np.mean(conf_list))))

        return result_list

    def get_ignored_tokens(self) -> List[int]:
        return [0]  # CTC blank token


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC Label Decoder for OCR output."""

    def __init__(self, character_dict_path: Optional[str] = None, use_space_char: bool = False, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds: np.ndarray, label=None, **kwargs) -> List[Tuple[str, float]]:
        """Decode CTC predictions.

        Args:
            preds: Model output, shape (batch, seq_len, num_classes)

        Returns:
            List of (text, confidence) tuples
        """
        # Handle numpy array input
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
        )

        if label is None:
            return text

        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character: List[str]) -> List[str]:
        # Add blank token at index 0
        dict_character = ["blank"] + dict_character
        return dict_character
