# High Level Planner WebApp

A Django web application that combines:
- A **base prompt** (context)
- A **text prompt** (query)
- An **image upload** (PNG/JPG)

All inputs are sent to **OpenAI GPT-4o (Vision)** to generate a response called **🤖 Robot Action**.  
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
cd gptapp


