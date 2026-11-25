"""Select current-examination views, with coarse-to-target volumetric inspection."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.data.chest_ct_jsonl_loader import load_nifti_preview
from src.data.image_store import ImageViewLimitError

_VALID_VIEW_TYPES = {"frontal_PA", "frontal_AP", "lateral", "lateral_decubitus", "lordotic", "oblique"}


class SelectSliceTool(Tool):
    name = "select_slice_or_view"
    description = "Confirm a radiograph projection or inspect slices from sequences supplied with this examination."
    input_schema = {
        "type": "object",
        "properties": {
            "view_type": {"type": "string", "enum": sorted(_VALID_VIEW_TYPES)},
            "sequence_id": {"type": "string", "description": "ID from available_sequences; no arbitrary paths are accepted."},
            "plane": {"type": "string", "enum": ["axial", "coronal", "sagittal"], "default": "axial"},
            "strategy": {"type": "string", "enum": ["preview", "coarse", "targeted"], "default": "preview"},
            "slice_index": {"type": "integer", "minimum": 0},
            "window": {"type": "string", "enum": ["lung", "mediastinal", "soft_tissue", "bone", "mri"]},
        },
        "additionalProperties": False,
    }
    is_concurrency_safe = False
    is_read_only = False

    def prompt(self) -> str:
        return ("For radiographs, confirm the observed projection with view_type. For CT/MRI, call "
                "with strategy='coarse' to inspect scan extent before strategy='targeted' with a "
                "slice_index. Only available_sequences from this examination are accessible. "
                "The shared limit is 32 distinct views across sequences and remediation passes. "
                "Array-axis planes are designated views; use saved affine/orientation for source localization.")

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        store = context.extra.get("image_store")
        if state is None:
            return ToolResult(data=json.dumps({"error": "pipeline_state not found in context"}), is_error=True)
        paths = state.exam_info.get("image_paths") or []
        if not paths and state.exam_info.get("volume_path"):
            paths = [state.exam_info["volume_path"]]
        if not paths:
            view_type = args.get("view_type", "")
            if view_type not in _VALID_VIEW_TYPES:
                return ToolResult(data=json.dumps({"error": "A valid radiograph view_type is required; no examination volumes are available"}), is_error=True)
            state.exam_info["view"] = view_type
            size = state.original_image.size if state.original_image is not None else (0, 0)
            return ToolResult(data=json.dumps({"status": "confirmed", "view_type": view_type,
                                             "image_width": size[0], "image_height": size[1]}))
        if store is None:
            return ToolResult(data=json.dumps({"error": "image_store not found in context"}), is_error=True)
        allowed_paths = {str(Path(p).resolve()) for p in paths}
        sequences = state.exam_info.get("sequences") or [
            {"sequence_id": f"sequence_{i}", "path": path} for i, path in enumerate(paths)]
        sequences = [s for s in sequences if str(Path(s["path"]).resolve()) in allowed_paths]
        sequence_id = args.get("sequence_id") or (sequences[0]["sequence_id"] if sequences else "")
        selected = next((s for s in sequences if s["sequence_id"] == sequence_id), None)
        if selected is None:
            return ToolResult(data=json.dumps({"error": "Sequence is not supplied with this examination", "available_sequences": sequences}), is_error=True)
        strategy = args.get("strategy", "preview")
        plane = args.get("plane", "axial")
        window = args.get("window") or state.exam_info.get("preview_window") or (
            "mri" if "mri" in state.modality.lower() else "soft_tissue" if "abdom" in state.modality.lower() else "lung")
        coarse_key = (sequence_id, plane)
        inspected = getattr(store, "_coarse_inspected", set())
        if strategy == "targeted" and coarse_key not in inspected:
            return ToolResult(data=json.dumps({"error": "Inspect this sequence/plane with strategy='coarse' before targeted access"}), is_error=True)
        if strategy == "targeted" and "slice_index" not in args:
            return ToolResult(data=json.dumps({"error": "targeted access requires slice_index"}), is_error=True)
        try:
            preview, metadata = load_nifti_preview(selected["path"], plane=plane, window=window)
            n = metadata["volume_shape"][metadata["array_axis"]]
            indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}) if strategy == "coarse" else [args.get("slice_index", n // 2)]
            rendered = []
            for index in indices:
                if index == metadata["slice_index"]:
                    image, item_metadata = preview, dict(metadata)
                else:
                    image, item_metadata = load_nifti_preview(selected["path"], plane=plane, slice_index=index, window=window)
                item_metadata["sequence_id"] = sequence_id
                item_metadata["sequence_name"] = selected.get("name", "")
                image_id = store.save(image, f"{sequence_id} {plane} slice {index} ({window})", self.name, item_metadata)
                rendered.append({"image_id": image_id, **item_metadata})
                state.active_image_id = image_id
                state.exam_info["image_metadata"] = item_metadata
            if strategy == "coarse":
                inspected.add(coarse_key)
                store._coarse_inspected = inspected
            return ToolResult(data=json.dumps({"status": "selected", "views": rendered,
                                             "available_sequences": [{k:v for k,v in s.items() if k != "path"} for s in sequences],
                                             "remaining_view_budget": store.max_views-store.count()}),
                              metadata={"image_ids": [r["image_id"] for r in rendered]})
        except (ValueError, IndexError, OSError, ImportError, ImageViewLimitError) as exc:
            return ToolResult(data=json.dumps({"error": str(exc)}), is_error=True)
