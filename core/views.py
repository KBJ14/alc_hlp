import base64, imghdr, os
from django.shortcuts import render
from .forms import VisionPromptForm
from openai import OpenAI

MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")

def _file_to_data_uri(django_file):
    content = django_file.read()
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
            image_file = form.cleaned_data["image"]

            merged_prompt = f"## Prompt\n{base_prompt}\n\n## User Query\n{query}"

            try:
                data_uri = _file_to_data_uri(image_file)
                preview_src = data_uri  # ← 새로고침 후에도 프리뷰 유지

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
