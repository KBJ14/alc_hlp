import base64, imghdr, os, uuid
from django.conf import settings
from django.shortcuts import render
from .forms import VisionPromptForm
from openai import OpenAI
from .utils.grounding import run_grounding_on_image_file, draw_detections
from PIL import Image
import warnings
try:
    # Optional imports for segmentation (SAM2)
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    _SAM2_AVAILABLE = True
except Exception:
    _SAM2_AVAILABLE = False
try:
    import numpy as np
except Exception:
    np = None
try:
    import torch  # for cuda availability check
except Exception:
    torch = None

MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")

def _file_to_data_uri(django_file):
    content = django_file.read()
    kind = imghdr.what(None, h=content) or "png"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:image/{kind};base64,{b64}"

def _path_to_data_uri(abs_path: str):
    with open(abs_path, "rb") as f:
        content = f.read()
    kind = imghdr.what(None, h=content) or "png"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:image/{kind};base64,{b64}"

def index(request):
    answer, error, preview_src = None, None, None

    if request.method == "POST":
        form = VisionPromptForm(request.POST, request.FILES)
        if form.is_valid():
            base_prompt = form.cleaned_data["base_prompt"]
            query = form.cleaned_data["query"]
            image_file = form.cleaned_data.get("image")
            processed_image_path = form.cleaned_data.get("processed_image_path")
            segmented_image_path = form.cleaned_data.get("segmented_image_path")

            merged_prompt = f"## Prompt\n{base_prompt}\n\n## User Query\n{query}"

            try:
                # Prefer segmented image if available, else boxed, else raw upload
                target_rel_path = None
                if segmented_image_path:
                    target_rel_path = segmented_image_path
                elif processed_image_path:
                    target_rel_path = processed_image_path

                if target_rel_path:
                    abs_path = os.path.join(settings.MEDIA_ROOT, target_rel_path)
                    data_uri = _path_to_data_uri(abs_path)
                    preview_src = data_uri
                else:
                    if not image_file:
                        raise ValueError("No image provided. Upload an image or click Get Box first.")
                    data_uri = _file_to_data_uri(image_file)
                    preview_src = data_uri

                client = OpenAI()
                resp = client.responses.create(
                    model=MODEL,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": merged_prompt},
                            {"type": "input_image", "image_url": data_uri},
                        ],
                    }],
                )
                answer = resp.output_text
            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "Form validation error."
    else:
        form = VisionPromptForm()

    return render(
        request,
        "core/index.html",
        {"form": form, "answer": answer, "error": error, "preview_src": preview_src},
    )


def get_box(request):
    """Create bounding boxes using GroundingDINO and only return boxed preview.
    Does NOT call GPT. It returns the same index template but with
    preview set to the boxed image and a hidden processed_image_path filled.
    """
    answer, error, preview_src = None, None, None

    if request.method != "POST":
        form = VisionPromptForm()
        return render(request, "core/index.html", {"form": form})

    form = VisionPromptForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "core/index.html", {"form": form, "error": "Form validation error."})

    base_prompt = form.cleaned_data["base_prompt"]
    query = form.cleaned_data["query"]
    image_file = form.cleaned_data.get("image")
    raw_image_path = form.cleaned_data.get("raw_image_path")
    if not image_file:
        return render(request, "core/index.html", {"form": form, "error": "Please upload an image before clicking Get Box."})

    # Save uploaded image temporarily to media for processing (or reuse existing raw path)
    try:
        from django.core.files.storage import default_storage
        from django.core.files.base import ContentFile

        upload_rel_dir = "uploads"
        abs_upload_dir = os.path.join(settings.MEDIA_ROOT, upload_rel_dir)
        os.makedirs(abs_upload_dir, exist_ok=True)

        # Persist the uploaded file to disk
        if raw_image_path:
            # Reuse already saved raw image
            saved_rel_path = raw_image_path
            saved_abs_path = os.path.join(settings.MEDIA_ROOT, saved_rel_path)
        else:
            file_name = getattr(image_file, "name", "uploaded.png")
            # make unique
            root, ext = os.path.splitext(file_name)
            safe_name = f"raw_{uuid.uuid4().hex}{ext or '.png'}"
            saved_rel_path = os.path.join(upload_rel_dir, safe_name)
            saved_abs_path = os.path.join(settings.MEDIA_ROOT, saved_rel_path)
            with default_storage.open(saved_rel_path, "wb+") as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

        # Run grounding dino with the text prompt (we'll use 'query' for detection)
        boxed_rel_path, detections = run_grounding_on_image_file(saved_abs_path, text_prompt=query)

        # Prepare preview and update form hidden field
        boxed_abs_path = os.path.join(settings.MEDIA_ROOT, boxed_rel_path)
        preview_src = _path_to_data_uri(boxed_abs_path)

        # Rebuild form initial with processed_image_path retained, so next submit uses it
        initial_data = {
            "base_prompt": base_prompt,
            "query": query,
            "raw_image_path": saved_rel_path,
            "processed_image_path": boxed_rel_path,
        }
        # If no boxes were found, provide a hint in error panel (non-blocking)
        if not detections:
            error = (
                "No objects detected for the given text. Try refining the text prompt, "
                "uploading a higher-resolution image, or lowering thresholds via env vars "
                "GROUNDING_BOX_THRESHOLD/GROUNDING_TEXT_THRESHOLD."
            )
        form = VisionPromptForm(initial=initial_data)
    except Exception as e:
        error = f"Grounding error: {e}"

    return render(
        request,
        "core/index.html",
        {"form": form, "answer": answer, "error": error, "preview_src": preview_src},
    )


def get_segmentation(request):
    """Create segmentation (mask overlay) using prior boxes + SAM2 if available.
    Flow:
        1. If user has run Get Box: use processed_image_path (boxed) and raw_image_path (original)
        2. If not: require upload first (similar to get_box) then run box detection automatically
        3. Apply SAM2 masks on raw image guided by detected boxes
    """
    answer, error, preview_src = None, None, None
    if request.method != "POST":
        form = VisionPromptForm()
        return render(request, "core/index.html", {"form": form})

    form = VisionPromptForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "core/index.html", {"form": form, "error": "Form validation error."})

    base_prompt = form.cleaned_data["base_prompt"]
    query = form.cleaned_data["query"]
    raw_image_path = form.cleaned_data.get("raw_image_path")
    processed_image_path = form.cleaned_data.get("processed_image_path")
    segmented_image_path = form.cleaned_data.get("segmented_image_path")
    image_file = form.cleaned_data.get("image")

    if not raw_image_path and not image_file:
        return render(request, "core/index.html", {"form": form, "error": "Upload an image or run Get Box first."})

    # Ensure raw image exists on disk
    try:
        from django.core.files.storage import default_storage
        upload_rel_dir = "uploads"
        abs_upload_dir = os.path.join(settings.MEDIA_ROOT, upload_rel_dir)
        os.makedirs(abs_upload_dir, exist_ok=True)
        if not raw_image_path:
            # Save uploaded file now
            file_name = getattr(image_file, "name", "uploaded.png")
            root, ext = os.path.splitext(file_name)
            safe_name = f"raw_{uuid.uuid4().hex}{ext or '.png'}"
            raw_image_path = os.path.join(upload_rel_dir, safe_name)
            saved_abs_path = os.path.join(settings.MEDIA_ROOT, raw_image_path)
            with default_storage.open(raw_image_path, "wb+") as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
        else:
            saved_abs_path = os.path.join(settings.MEDIA_ROOT, raw_image_path)
    except Exception as e:
        return render(request, "core/index.html", {"form": form, "error": f"File handling error: {e}"})

    # If we have no boxes yet, generate them first (calls existing grounding function)
    detections = []
    try:
        if not processed_image_path:
            boxed_rel_path, detections = run_grounding_on_image_file(saved_abs_path, text_prompt=query)
            processed_image_path = boxed_rel_path
        else:
            # We could recompute detections if needed, but we skip to save time.
            pass
    except Exception as e:
        return render(request, "core/index.html", {"form": form, "error": f"Grounding error: {e}"})

    # Perform segmentation if SAM2 available
    seg_rel_path = None
    if not _SAM2_AVAILABLE or np is None:
        error = "SAM2 not installed. Install SAM2 to enable segmentation (pip install -e sam2)."
    else:
        try:
            # Attempt to load SAM2 (paths via env)
            sam2_checkpoint = os.getenv("SAM2_CHECKPOINT")
            sam2_model_config_raw = os.getenv("SAM2_MODEL_CONFIG")
            if not sam2_checkpoint or not os.path.exists(sam2_checkpoint):
                raise RuntimeError("SAM2_CHECKPOINT missing or invalid path.")
            if not sam2_model_config_raw:
                raise RuntimeError("SAM2_MODEL_CONFIG not set.")
            # Hydra expects a config name relative to the sam2 package's config search path.
            # If user put an absolute path, we convert it to a relative 'configs/.../file.yaml'.
            if os.path.isabs(sam2_model_config_raw) and os.path.exists(sam2_model_config_raw):
                rel_idx = sam2_model_config_raw.rfind("configs/")
                if rel_idx != -1:
                    sam2_model_config = sam2_model_config_raw[rel_idx:]
                else:
                    # Fallback: just take the basename; Hydra still might fail if not in search path.
                    sam2_model_config = os.path.basename(sam2_model_config_raw)
            else:
                # Assume user already provided a hydra-relative config string like 'configs/sam2.1/sam2.1_hiera_l.yaml'
                sam2_model_config = sam2_model_config_raw
            # Optional: warn if the path portion doesn't actually exist (helps debugging)
            possible_fs_path = None
            if 'configs/' in sam2_model_config_raw:
                # reconstruct expected absolute path to help diagnose
                possible_fs_path = sam2_model_config_raw
            if possible_fs_path and not os.path.exists(possible_fs_path):
                warnings.warn(f"Provided SAM2_MODEL_CONFIG absolute path does not exist on filesystem: {possible_fs_path}. Using Hydra name '{sam2_model_config}'.")

            from PIL import Image as _PILImage
            raw_image = _PILImage.open(saved_abs_path).convert("RGB")
            boxes_image_path = os.path.join(settings.MEDIA_ROOT, processed_image_path)
            # Derive detection boxes again for segmentation (need coordinates)
            # We re-run detection to get coordinates best aligned with raw image
            _, fresh_detections = run_grounding_on_image_file(saved_abs_path, text_prompt=query)
            if not fresh_detections:
                error = "No detections to segment."
            else:
                input_boxes = np.array([d.box for d in fresh_detections], dtype=float)
                # Decide device: prefer CUDA if available and not forced to CPU
                if os.getenv("SEG_FORCE_CPU") == "1" or torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
                    device = "cpu"
                else:
                    device = "cuda"
                sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
                predictor = SAM2ImagePredictor(sam2_model)
                predictor.set_image(np.array(raw_image))
                masks, _scores, _logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                if masks is not None and masks.ndim == 4:
                    masks = masks.squeeze(1)
                # Overlay masks
                overlay_alpha = int(os.getenv("SEG_OVERLAY_ALPHA", "110"))
                annotated = raw_image.convert("RGBA")
                base_colors = [
                    (255, 0, 0), (0, 140, 255), (0, 200, 0), (255, 165, 0), (186, 85, 211),
                    (70, 130, 180), (255, 105, 180), (46, 139, 87), (123, 104, 238), (210, 105, 30)
                ]
                import numpy as _np
                overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
                for idx, mask in enumerate(masks):
                    m = (mask > 0.5) if mask.dtype != _np.uint8 else (mask > 0)
                    color = base_colors[idx % len(base_colors)]
                    alpha = max(10, min(255, overlay_alpha))
                    mask_img = Image.new("RGBA", annotated.size, color + (0,))
                    alpha_band = Image.new("L", annotated.size, 0)
                    px_alpha = alpha_band.load()
                    ys, xs = _np.where(m)
                    for y, x in zip(ys, xs):
                        px_alpha[x, y] = alpha
                    mask_img.putalpha(alpha_band)
                    overlay = Image.alpha_composite(overlay, mask_img)
                annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
                # Reuse detection drawer for labels/boxes
                annotated = draw_detections(annotated, fresh_detections)
                seg_name = f"seg_{uuid.uuid4().hex}.png"
                seg_rel_path = os.path.join("uploads", seg_name)
                seg_abs_path = os.path.join(settings.MEDIA_ROOT, seg_rel_path)
                annotated.save(seg_abs_path)
        except Exception as e:
            error = f"Segmentation error: {e}"

    if seg_rel_path:
        preview_src = _path_to_data_uri(os.path.join(settings.MEDIA_ROOT, seg_rel_path))
        segmented_image_path = seg_rel_path
    else:
        # fallback preview to processed (boxed) image if exists
        if processed_image_path:
            preview_src = _path_to_data_uri(os.path.join(settings.MEDIA_ROOT, processed_image_path))

    initial_data = {
        "base_prompt": base_prompt,
        "query": query,
        "raw_image_path": raw_image_path,
        "processed_image_path": processed_image_path,
        "segmented_image_path": segmented_image_path,
    }
    form = VisionPromptForm(initial=initial_data)
    return render(
        request,
        "core/index.html",
        {"form": form, "answer": answer, "error": error, "preview_src": preview_src},
    )
