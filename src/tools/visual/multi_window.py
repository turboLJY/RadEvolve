"""
MultiWindowTool — Render the image with different display windows.

Applies radiological windowing presets (bone, lung, soft_tissue,
mediastinal) to highlight different tissue types.  Each windowed
rendering is saved independently to the image store.
"""

from __future__ import annotations

import json
import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import PipelineState, ToolResult
from src.utils.dicom_utils import WINDOW_FUNCTIONS, apply_window_by_name
from src.data.chest_ct_jsonl_loader import load_nifti_preview
from src.utils.image_processing import enhance_image, zoom_image, draw_annotation

logger = logging.getLogger(__name__)

_VALID_WINDOW_TYPES = list(WINDOW_FUNCTIONS.keys())


class MultiWindowTool(Tool):
    name = "multi_window_render"
    description = (
        "Render the image with different display windows to highlight different "
        "tissue types. Available windows: bone, lung, soft_tissue, mediastinal. "
        "Each rendering is saved separately to the image store."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_id": {"type": "string", "description": "Stored source view, defaults to the selected slice."},
            "window_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": _VALID_WINDOW_TYPES,
                },
                "description": (
                    "List of window presets to apply. "
                    "Valid values: 'bone', 'lung', 'soft_tissue', 'mediastinal'."
                ),
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["window_types"],
    }
    is_concurrency_safe = True
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use multi_window_render when you need to examine different tissue densities. "
            "'bone' highlights skeletal structures and calcifications; 'lung' brings out "
            "pulmonary parenchyma detail; 'soft_tissue' improves soft tissue boundaries; "
            "'mediastinal' reveals mediastinal structures. Request multiple windows at "
            "once for comparison."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        window_types: list[str] = args.get("window_types", [])

        if not window_types:
            return ToolResult(
                data=json.dumps({"error": "window_types must be a non-empty list"}),
                is_error=True,
            )

        # Validate
        invalid = [w for w in window_types if w.strip().lower() not in WINDOW_FUNCTIONS]
        if invalid:
            return ToolResult(
                data=json.dumps({
                    "error": (
                        f"Invalid window types: {invalid}. "
                        f"Valid options: {_VALID_WINDOW_TYPES}"
                    ),
                }),
                is_error=True,
            )

        pipeline_state: PipelineState = context.extra.get("pipeline_state")
        image_store = context.extra.get("image_store")

        if pipeline_state is None or image_store is None:
            return ToolResult(
                data=json.dumps({"error": "pipeline_state or image_store not found in context"}),
                is_error=True,
            )

        source_id = args.get("image_id") or getattr(pipeline_state, "active_image_id", "")
        source_entry = image_store.get(source_id) if source_id else None
        if source_id and source_entry is None:
            return ToolResult(data=json.dumps({"error": f"Unknown source image: {source_id}"}), is_error=True)
        original = source_entry[0] if source_entry else pipeline_state.original_image
        source_metadata = image_store.get_metadata(source_id) if source_id else pipeline_state.exam_info.get("image_metadata", {})
        if original is not None and not source_id:
            source_id = image_store.save(original, "Original examination", "input", source_metadata)
        if original is None:
            return ToolResult(
                data=json.dumps({"error": "No original image available in pipeline state"}),
                is_error=True,
            )

        if source_metadata.get("volume_path") and "mri" in pipeline_state.modality.lower():
            return ToolResult(data=json.dumps({"error": "CT HU windows do not apply to MRI. Use select_slice_or_view with window='mri' for percentile-scaled sequence views."}), is_error=True)
        renderings: list[dict] = []

        for wt in window_types:
            wt = wt.strip().lower()
            try:
                if source_metadata.get("volume_path"):
                    windowed, _ = load_nifti_preview(
                        source_metadata["volume_path"], plane=source_metadata["slice_plane"],
                        slice_index=source_metadata["slice_index"], window=wt,
                    )
                    # Preserve a crop in source display coordinates, if this view is cropped.
                    for transform in source_metadata.get("transforms", []):
                        if transform.get("operation") == "crop":
                            box = transform["crop_box"]
                            windowed = windowed.crop((box["x"], box["y"], box["x"]+box["width"], box["y"]+box["height"]))
                        elif transform.get("operation") == "zoom_enhance":
                            windowed = zoom_image(windowed, 0, 0, windowed.width, windowed.height, transform["zoom_factor"])
                            windowed = enhance_image(windowed, transform["enhance_type"])
                        elif transform.get("operation") == "annotation":
                            windowed = draw_annotation(windowed, transform["annotations"])
                        elif transform.get("operation") not in {"rotate90", "window"}:
                            raise ValueError("Select a native slice before windowing an analytical probe or composite")
                else:
                    windowed = apply_window_by_name(original, wt)
            except Exception as exc:
                logger.exception("Window '%s' failed", wt)
                renderings.append({
                    "window_type": wt,
                    "error": str(exc),
                })
                continue

            desc = f"Original image — {wt} window"
            img_id = image_store.save_derived(windowed, desc, self.name, source_id, {"operation": "window", "window": wt, "quantitative_hu": bool(source_metadata.get("volume_path"))})
            renderings.append({
                "window_type": wt,
                "image_id": img_id,
                "description": desc,
                "size": {"width": windowed.width, "height": windowed.height},
            })
            logger.info("Rendered %s window -> %s", wt, img_id)

        result = {
            "renderings": renderings,
            "count": len([r for r in renderings if "image_id" in r]),
            "message": (
                f"Generated {len(renderings)} window rendering(s): "
                + ", ".join(
                    f"{r['window_type']}={r.get('image_id', 'FAILED')}"
                    for r in renderings
                )
            ),
        }

        return ToolResult(
            data=json.dumps(result),
            metadata={"image_ids": [r["image_id"] for r in renderings if "image_id" in r]},
        )