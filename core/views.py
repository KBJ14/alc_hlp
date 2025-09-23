import base64, imghdr, os
from django.conf import settings
from django.shortcuts import render
from .forms import VisionPromptForm
from openai import OpenAI
from .utils.grounding import run_grounding_on_image_file

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

            merged_prompt = f"## Prompt\n{base_prompt}\n\n## User Query\n{query}"

            try:
                # If we already created a boxed image via Get Box, use it for GPT
                if processed_image_path:
                    abs_path = os.path.join(settings.MEDIA_ROOT, processed_image_path)
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
    if not image_file:
        return render(request, "core/index.html", {"form": form, "error": "Please upload an image before clicking Get Box."})

    # Save uploaded image temporarily to media for processing
    try:
        from django.core.files.storage import default_storage
        from django.core.files.base import ContentFile

        upload_rel_dir = "uploads"
        abs_upload_dir = os.path.join(settings.MEDIA_ROOT, upload_rel_dir)
        os.makedirs(abs_upload_dir, exist_ok=True)

        # Persist the uploaded file to disk
        file_name = getattr(image_file, "name", "uploaded.png")
        # make unique
        import uuid
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
