# High Level Planner WebApp

A Django web application that combines:
- A **base prompt** (context)
- A **text prompt** (query)
- An **image upload** (PNG/JPG)

All inputs are sent to **VLM or VLA** to generate a response called **🤖 Robot Action**.  
The uploaded image is previewed both **before submission** and **after submission**.

---

## ✨ Features
- Two-column responsive layout with **TailwindCSS**
  - **Left**: Prompt + Text Prompt + Image upload + Submit button
  - **Right**: Image Preview + Robot Action output
- Client-side preview of uploaded image (before submission)
- Server-side persistence of preview (after submission, using base64 data URI)
- Styled with TailwindCSS, powered by Django 5.x

---

## 🛠️ Preparation

### 1. Clone repository
```bash
git clone https://github.com/kbj14/alc_hlp.git
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows (PowerShell)
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Add environment variables
- .env file
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4o-mini
```

### 5. Apply migrations and run the application
```bash
python manage.py migrate
python manage.py runserver
```

### 6. Usage
- Open http://127.0.0.1:8000 in your browser.
- Fill in:
- Prompt (base instructions / context)
- Upload Image (PNG or JPG)
- Text Prompt (your query)
- Click Get Action.
- The 🤖 Robot Action will appear on the right-hand panel.
- Your uploaded image remains in the preview panel even after submission.
