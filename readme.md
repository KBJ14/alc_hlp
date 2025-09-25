# High Level Planner WebApp

Multi-step visual grounding + (optional) segmentation + vision-language reasoning built with **Django 5**, **Grounding DINO**, optional **SAM2**, and **OpenAI** vision models.

## Workflow Overview
1. Get Box – Grounding DINO text-prompted object detection (saves boxed image)
2. Get Segmentation (optional) – SAM2 segmentation using detected boxes as guidance
3. Get Action – Sends the best available image (segmented > boxed > original) plus merged prompts to OpenAI → Robot Action response

Artifacts are written to `media/uploads/` and their relative paths are persisted through hidden form fields.

---

## Features
- Multi-step UI (Get Box / Get Segmentation / Get Action)
- Text grounded detection (Grounding DINO via Hugging Face Transformers)
- Optional SAM2.1 hierarchical segmentation (colored per-object masks)
- Auto-scaling detection label font (override with env var)
- Segmented image is automatically preferred for the LLM if present
- TailwindCSS responsive layout
- All thresholds/overrides via environment variables
- Works without SAM2 (falls back to detection only)

---

## Repository Structure (Recommended Layout)
Keep the SAM2 repository outside (as a sibling) of this Django project to avoid Hydra/shadow path issues.
```
workspace_root/
  alc_hlp/                # This repo (manage.py inside alc_hlp/alc_hlp)
    manage.py
    core/
      views.py            # index / get_box / get_segmentation
      utils/grounding.py  # Grounding DINO logic
      templates/core/index.html
    media/uploads/
    readme.md
  sam2/                   # Separate SAM2 repo (editable install)
```
NOTE: Do NOT place the `sam2/` repo inside `alc_hlp/alc_hlp/`.

---

## Quick Start

### 1. Clone Project
```bash
git clone https://github.com/kbj14/alc_hlp.git
cd alc_hlp/alc_hlp   # directory containing manage.py
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3. Install Core Dependencies
```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

### 4. (Optional) Install SAM2 (Segmentation)
From inside `alc_hlp/alc_hlp`, move to the workspace root and clone SAM2 next to this repo:
```bash
# starting from: workspace_root/alc_hlp/alc_hlp
cd ../..                      # now at workspace_root
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .              # editable install
cd checkpoints
bash download_ckpts.sh        # downloads sam2.1_hiera_*.pt checkpoint files
cd ../..                      # back to workspace_root
cd alc_hlp/alc_hlp            # return to Django project
```
You should now have (example):
```
workspace_root/
  sam2/checkpoints/sam2.1_hiera_large.pt
  sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

### 5. Create .env File
Create `alc_hlp/alc_hlp/.env` with at least:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4o-mini

# SAM2 – choose one consistent variant pair
SAM2_CHECKPOINT=/abs/path/workspace_root/sam2/checkpoints/sam2.1_hiera_large.pt
SAM2_MODEL_CONFIG=/abs/path/workspace_root/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml


Variant mapping:
| Variant | Checkpoint | Config |
|---------|------------|--------|
| tiny    | sam2.1_hiera_tiny.pt       | sam2.1_hiera_t.yaml |
| small   | sam2.1_hiera_small.pt      | sam2.1_hiera_s.yaml |
| base+   | sam2.1_hiera_base_plus.pt  | sam2.1_hiera_b+.yaml |
| large   | sam2.1_hiera_large.pt      | sam2.1_hiera_l.yaml |

If segmentation variables are omitted, the app will still run (detection only).

### 6. Migrate & Run
```bash
python manage.py migrate
python manage.py runserver
```
Open: http://127.0.0.1:8000

---

## Usage
1. Upload an image and enter a Text Prompt.
2. Click Get Box to generate bounding boxes (red outlines + labels).
3. Click Get Segmentation (if SAM2 configured) to overlay per-object colored masks.
4. Click Get Action to send the (segmented > boxed > original) image plus prompts to the model.
5. Robot Action response appears on the right.

You may skip Get Box and go directly to Get Segmentation—detection will run automatically first.

---

## License
See repository LICENSE (and SAM2 upstream license inside its own repo).

---

## Acknowledgements
- Grounding DINO (IDEA Research)
- SAM 2 (Meta)
- OpenAI Vision models
- TailwindCSS

PRs / extensions welcome.
