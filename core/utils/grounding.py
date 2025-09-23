import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional

from django.conf import settings

from PIL import Image, ImageDraw, ImageFont
import warnings


@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str
    score: float


def _ensure_upload_dir() -> str:
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def _try_load_font(size: int = 16) -> Optional[ImageFont.FreeTypeFont]:
    try:
        # Common fonts; fallback to default if unavailable
        for name in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(name):
                return ImageFont.truetype(name, size=size)
    except Exception:
        pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_detections(img: Image.Image, detections: List[Detection]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    # Dynamic font sizing with env override
    font_size = None
    try:
        env_fs = os.getenv("GROUNDING_LABEL_FONT_SIZE")
        if env_fs:
            font_size = max(8, int(env_fs))
    except Exception:
        font_size = None
    if font_size is None:
        long_side = max(img.size[0], img.size[1])
        # ~3% of long side, clamped
        font_size = max(14, min(64, int(long_side * 0.03)))
    font = _try_load_font(font_size)
    for det in detections:
        x1, y1, x2, y2 = det.box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
        label = f"{det.label} {det.score:.2f}"
        if font is not None and hasattr(draw, "textbbox"):
            x0, y0, x1t, y1t = draw.textbbox((0, 0), label, font=font)
            tw, th = (x1t - x0, y1t - y0)
            pad = max(2, int(font_size * 0.25))
            bg_left = x1
            bg_right = x1 + tw + pad * 2
            bg_top = y1 - th - pad * 2
            bg_bottom = y1
            text_x = x1 + pad
            text_y = y1 - th - pad
            # If too close to top, flip label below the box
            if bg_top < 0:
                bg_top = y1
                bg_bottom = y1 + th + pad * 2
                text_y = y1 + pad
            draw.rectangle([(bg_left, bg_top), (bg_right, bg_bottom)], fill=(255, 0, 0))
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1, y1), label, fill=(255, 0, 0))
    return img


def run_grounding_on_image_file(image_path: str, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25) -> Tuple[str, List[Detection]]:
    """
    Runs GroundingDINO via Hugging Face transformers to detect objects in `image_path`
    conditioned on `text_prompt`, draws bounding boxes, and saves annotated image
    under MEDIA/uploads. Returns (relative_media_path, detections).

    Requires packages: transformers, torch, Pillow, numpy.
    """
    # Silence known FutureWarning about labels->text_labels from Transformers
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"transformers\.models\.grounding_dino\.processing_grounding_dino",
    )

    # Allow thresholds to be overridden via env vars
    try:
        box_threshold = float(os.getenv("GROUNDING_BOX_THRESHOLD", box_threshold))
    except Exception:
        pass
    try:
        text_threshold = float(os.getenv("GROUNDING_TEXT_THRESHOLD", text_threshold))
    except Exception:
        pass
    # Lazy imports to avoid import cost when not used
    try:
        import torch
        try:
            # Preferred: dedicated classes available in newer transformers
            from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
            has_native_gdino = True
        except Exception:
            has_native_gdino = False
            from transformers import AutoModel, AutoProcessor
    except Exception as e:
        raise RuntimeError(
            "GroundingDINO dependencies not installed. Please install 'torch' and 'transformers'."
        ) from e

    model_id = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if has_native_gdino:
        processor = GroundingDinoProcessor.from_pretrained(model_id)
        model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    else:
        # Fallback for older transformers: use Auto classes with trust_remote_code
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

    image = Image.open(image_path).convert("RGB")

    # GroundingDINO tends to work better with dot-separated phrases
    det_text = (text_prompt or "").strip()
    if det_text and not det_text.endswith("."):
        det_text = det_text + " ."

    inputs = processor(images=image, text=det_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    if hasattr(processor, "post_process_grounded_object_detection"):
        # Try with thresholds; if unsupported, call without and filter manually
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    module=r"transformers\.models\.grounding_dino\.processing_grounding_dino",
                )
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
        except TypeError:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    module=r"transformers\.models\.grounding_dino\.processing_grounding_dino",
                )
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.input_ids,
                    target_sizes=target_sizes,
                )[0]
            # Manual filtering by box_threshold if scores available
            if "scores" in results:
                scores_t = results["scores"]
                boxes_t = results.get("boxes", None)
                labels_l = results.get("labels", None)
                if hasattr(scores_t, "detach"):
                    scores_l = scores_t.detach().cpu().tolist()
                else:
                    scores_l = [float(s) for s in scores_t]
                keep_idx = [i for i, s in enumerate(scores_l) if float(s) >= box_threshold]
                if boxes_t is not None:
                    if hasattr(boxes_t, "index_select"):
                        import torch as _torch
                        idx_t = _torch.tensor(keep_idx, dtype=_torch.long, device=boxes_t.device if hasattr(boxes_t, "device") else None)
                        results["boxes"] = boxes_t.index_select(0, idx_t)
                    else:
                        results["boxes"] = [boxes_t[i] for i in keep_idx]
                results["scores"] = [scores_l[i] for i in keep_idx]
                if labels_l is not None:
                    results["labels"] = [labels_l[i] for i in keep_idx]
    else:
        # Some older processors use a slightly different method name
        if hasattr(processor, "post_process_object_detection"):
            results = processor.post_process_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_sizes,
            )[0]
            # In this fallback, labels may be missing; set to text prompt
            if "labels" not in results:
                results["labels"] = [text_prompt] * len(results.get("scores", []))
        else:
            raise RuntimeError("The loaded processor does not support GroundingDINO post-processing.")

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    # Prefer new key 'text_labels' when available (future-proof)
    labels_raw = results.get("text_labels") if "text_labels" in results else results.get("labels", [])

    # Normalize outputs to python lists regardless of tensor/list backend
    def _to_list(x):
        try:
            import torch as _torch
            if hasattr(x, "detach"):
                return x.detach().cpu().tolist()
        except Exception:
            pass
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return x.tolist()
        except Exception:
            pass
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    boxes_l = _to_list(boxes)
    scores_l = [float(s) for s in _to_list(scores)] if scores is not None else []
    labels_l = [str(l) for l in (labels_raw or [])]

    # Align lengths just in case
    n = min(len(boxes_l), len(scores_l), len(labels_l) if labels_l else len(boxes_l))
    boxes_l = boxes_l[:n]
    scores_l = scores_l[:n]
    labels_l = (labels_l[:n] if labels_l else [text_prompt] * n)

    detections: List[Detection] = []
    for b, s, lab in zip(boxes_l, scores_l, labels_l):
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = b
            detections.append(Detection(box=(float(x1), float(y1), float(x2), float(y2)), label=lab, score=float(s)))

    # Draw and save
    annotated = image.copy()
    annotated = draw_detections(annotated, detections)

    upload_dir = _ensure_upload_dir()
    out_name = f"boxed_{uuid.uuid4().hex}.png"
    out_path = os.path.join(upload_dir, out_name)
    annotated.save(out_path)

    rel_path = f"uploads/{out_name}"
    return rel_path, detections
