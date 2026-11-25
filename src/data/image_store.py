"""Case-scoped, bounded image views with source and transformation provenance."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from PIL import Image


class ImageViewLimitError(ValueError):
    """A new distinct view would exceed the examination's shared budget."""


class SessionImageStore:
    def __init__(self, session_id: str, max_views: int = 32):
        if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views < 1:
            raise ValueError("max_views must be a positive integer")
        self.session_id = session_id
        self.max_views = max_views
        self._images: dict[str, tuple[Image.Image, str]] = {}
        self._tool_chains: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}
        self._view_keys: dict[str, str] = {}
        self._counter = 0

    def save(self, image: Image.Image, description: str, tool_name: str = "", metadata: dict | None = None) -> str:
        metadata = copy.deepcopy(metadata or {})
        # Identical pixels from different slices/sequences are distinct source views.
        key_data = {"mode": image.mode, "size": image.size, "metadata": metadata}
        digest = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode() + image.tobytes()).hexdigest()
        if digest in self._view_keys:
            return self._view_keys[digest]
        if len(self._images) >= self.max_views:
            raise ImageViewLimitError(f"Examination image-view limit ({self.max_views}) reached")
        self._counter += 1
        image_id = f"IMG_{self.session_id[:8]}_{self._counter:03d}"
        self._images[image_id] = (image.copy(), description)
        parents = metadata.get("source_image_ids", [])
        if metadata.get("source_image_id"):
            parents = [metadata["source_image_id"]] + parents
        chain = []
        for parent in parents:
            for name in self.get_tool_chain(parent):
                if name not in chain:
                    chain.append(name)
        if tool_name:
            chain.append(tool_name)
        self._tool_chains[image_id] = chain
        self._metadata[image_id] = metadata
        self._view_keys[digest] = image_id
        return image_id

    def save_derived(self, image: Image.Image, description: str, tool_name: str, source_image_id: str, transform: dict) -> str:
        if source_image_id not in self._images:
            raise ValueError(f"Unknown source image: {source_image_id}")
        metadata = self.get_metadata(source_image_id)
        metadata["source_image_id"] = source_image_id
        metadata["transforms"] = list(metadata.get("transforms", [])) + [copy.deepcopy(transform)]
        if "crop_box" in transform:
            metadata["crop_box"] = transform["crop_box"]
        if transform.get("operation") == "window":
            metadata["window"] = transform["window"]
            if transform.get("quantitative_hu"):
                from src.data.chest_ct_jsonl_loader import _CT_WINDOWS
                metadata["window_center_width"] = list(_CT_WINDOWS[transform["window"]])
        return self.save(image, description, tool_name, metadata)

    def get(self, image_id: str) -> tuple[Image.Image, str] | None:
        return self._images.get(image_id)

    def get_metadata(self, image_id: str) -> dict:
        return copy.deepcopy(self._metadata.get(image_id, {}))

    def add_tool_to_chain(self, image_id: str, tool_name: str) -> None:
        if image_id in self._tool_chains and tool_name not in self._tool_chains[image_id]:
            self._tool_chains[image_id].append(tool_name)

    def get_tool_chain(self, image_id: str) -> list[str]:
        return list(self._tool_chains.get(image_id, []))

    def to_base64(self, image_id: str, fmt: str = "PNG") -> str | None:
        entry = self._images.get(image_id)
        if not entry:
            return None
        img = entry[0]
        if max(img.size) > 1024:
            scale = 1024 / max(img.size)
            img = img.resize(tuple(max(1, round(v * scale)) for v in img.size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode()

    def list_all(self) -> list[dict]:
        return [{"image_id": k, "description": v[1], "size": v[0].size,
                 "tool_chain": self.get_tool_chain(k), "metadata": self.get_metadata(k)}
                for k, v in self._images.items()]

    def count(self) -> int:
        return len(self._images)
