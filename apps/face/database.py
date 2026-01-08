"""Face database - manages registered face features"""

import csv
import io
import json
import os
import re
import time
from pathlib import Path
import numpy as np


class FaceDatabase:
    """Manages registered face features for matching"""

    def __init__(self, features_path: str):
        self.names: list[str] = []
        self.features_matrix: np.ndarray | None = None
        self.avatars: dict[str, str] = {}
        self._load(features_path)

        # Mapping DB
        self.people_by_id: dict[str, dict] = {}
        self.people_by_name: dict[str, dict] = {}

        self._load_mapping_csv_same_folder()

    def _norm_key(self, s: str | None) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        # gộp nhiều khoảng trắng
        s = re.sub(r"\s+", " ", s)
        # so khớp không phân biệt hoa thường
        return s.casefold()

    def _load_mapping_csv_same_folder(self) -> None:
        """Load mapping.csv from the same directory as this db.py (auto delimiter + utf-8-sig)"""
        base_dir = Path(__file__).resolve().parent
        csv_path = base_dir / "mapping.csv"

        if not csv_path.exists():
            print(f"[FaceDB] mapping.csv not found: {csv_path}")
            return

        start_time = time.time()

        # đọc toàn bộ file 1 lần để sniff delimiter + xử lý BOM
        raw = csv_path.read_text(encoding="utf-8-sig", errors="replace")
        if not raw.strip():
            print(f"[FaceDB] mapping.csv is empty: {csv_path}")
            return

        sample = raw[:4096]
        # thử sniff delimiter
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=["\t", ",", ";", "|"])
            delim = dialect.delimiter
        except Exception:
            # fallback: ưu tiên tab nếu có
            delim = "\t" if "\t" in sample else ","

        f = io.StringIO(raw)

        # detect header
        first_line = (sample.splitlines()[0] if sample.splitlines() else "")
        has_header = ("ID_nguoi" in first_line and "Ten_nguoi" in first_line)

        if has_header:
            reader = csv.DictReader(f, delimiter=delim)
        else:
            # nếu file không có header, dùng fieldnames cố định
            reader = csv.DictReader(
                f,
                delimiter=delim,
                fieldnames=["ID_nguoi", "Ten_nguoi", "Ten_cong_ty", "Ten_chuc_danh", "Link_anh"],
            )

        count = 0
        self.people_by_id.clear()
        self.people_by_name.clear()

        for row in reader:
            pid = (row.get("ID_nguoi") or "").strip()
            pname = (row.get("Ten_nguoi") or "").strip()
            company = (row.get("Ten_cong_ty") or "").strip()
            title = (row.get("Ten_chuc_danh") or "").strip()
            avatar = (row.get("Link_anh") or "").strip()

            # bỏ qua header lẫn vào data / dòng rỗng
            if pid == "ID_nguoi" or pname == "Ten_nguoi":
                continue
            if not pid and not pname:
                continue

            info = {
                "person_id": pid or None,
                "person_name": pname or None,
                "company_name": company or None,
                "job_title": title or None,
                "avatar": avatar or None,
            }

            if pid:
                self.people_by_id[self._norm_key(pid)] = info
            if pname:
                self.people_by_name[self._norm_key(pname)] = info

            count += 1

        elapsed = time.time() - start_time
        print(
            f"[FaceDB] Loaded mapping.csv: rows={count}, delim={repr(delim)}, "
            f"by_id={len(self.people_by_id)}, by_name={len(self.people_by_name)} "
            f"in {elapsed*1000:.1f}ms"
        )

    def get_person_info(self, label: str) -> dict | None:
        """
        label can be ID_nguoi OR Ten_nguoi OR nickname (case-insensitive).
        """
        k = self._norm_key(label)
        if not k:
            return None
        return self.people_by_id.get(k) or self.people_by_name.get(k)


    def _load(self, path: str) -> None:
        """Load pre-registered face features from JSON"""
        if not os.path.exists(path):
            print(f"Warning: Features file not found: {path}")
            return

        start = time.time()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = []
        for name, info in data.items():
            if "feature" not in info:
                continue

            feat = info["feature"]
            if isinstance(feat[0], list):
                feat = feat[0]
            vec = np.array(feat, dtype=np.float32)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self.names.append(name)
            features.append(vec)
            if "avatar" in info:
                self.avatars[name] = info["avatar"]

        if features:
            self.features_matrix = np.vstack(features).astype(np.float32)

        print(f"Loaded {len(self.names)} registered faces in {(time.time() - start) * 1000:.1f}ms")
        if self.features_matrix is not None:
            print(f"Feature matrix shape: {self.features_matrix.shape}")

    def match(self, embedding: np.ndarray) -> tuple[int, float]:
        """Match embedding against database. Returns (person_idx, distance)"""
        if self.features_matrix is None:
            return -1, float("inf")

        distances = np.linalg.norm(self.features_matrix - embedding, axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])
