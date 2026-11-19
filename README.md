# 🍎 Fruit Detection System

A Streamlit application for detecting apples, bananas, and oranges using a trained Ultralytics YOLO model.

## Project files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `best.pt` — your trained YOLO weights (must be supplied separately)
- `.streamlit/config.toml` — optional Streamlit appearance settings

## Requirements

Install Python 3.10 or newer, then open a terminal in this folder.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Add your trained model

Copy your trained Ultralytics model weights into this folder and name the file exactly:

```text
best.pt
```

The model must contain the classes `apple`, `banana`, and `orange` (spelling and capitalization are normalized to lowercase by the app).

## Run locally

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deploy to Streamlit Community Cloud

1. Upload this project to a GitHub repository.
2. Include `app.py`, `requirements.txt`, and `best.pt`.
3. Create an app at https://share.streamlit.io/.
4. Select the repository, branch, and `app.py` as the main file.

If `best.pt` is too large for a normal GitHub upload, use Git LFS or an appropriate model-hosting solution and update the app to download the weights securely.

## Group member

Ndayisabye Thierry — 26RP02821
