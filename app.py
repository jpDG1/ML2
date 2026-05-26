"""
app.py — Projekt 5: Interfejs demonstracyjny
  • Tab 1: Rozpoznawanie emocji z twarzy (CNN FER-2013)
  • Tab 2: Diagnostyka nowotworów piersi (klasyczny ML)
Uruchomienie: python app.py
"""

import gradio as gr
import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# ══════════════════════════════════════════════════════════════
# ŁADOWANIE MODELI
# ══════════════════════════════════════════════════════════════

# --- Model emocji (TensorFlow/Keras CNN) ---
emotion_model = None
try:
    from tensorflow.keras.models import load_model
    MODEL_PATH = 'best_model_base.keras'
    if os.path.exists(MODEL_PATH):
        emotion_model = load_model(MODEL_PATH)
        print(f"✅ Model emocji załadowany: {MODEL_PATH}")
    else:
        print(f"⚠️  Brak pliku {MODEL_PATH} — tab emocji będzie niedostępny")
except Exception as e:
    print(f"⚠️  Błąd ładowania modelu emocji: {e}")

# Klasy emocji wg FER-2013 (kolejność alfabetyczna = kolejność folderów)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_PL = {
    'angry':    'Złość',
    'disgust':  'Obrzydzenie',
    'fear':     'Strach',
    'happy':    'Radość',
    'neutral':  'Neutralny',
    'sad':      'Smutek',
    'surprise': 'Zaskoczenie',
}
EMOTION_EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲',
}
EMOTION_DESC = {
    'angry':    'Twarz wyraża złość lub frustrację. Charakteryzuje się zmrużonymi oczami i zaciśniętymi ustami.',
    'disgust':  'Widoczne oznaki obrzydzenia — uniesiona górna warga, zmarszczone czoło.',
    'fear':     'Wyraz strachu: szeroko otwarte oczy, uniesione brwi, lekko otwarte usta.',
    'happy':    'Radosny uśmiech, uniesione policzki i zmrużone oczy wskazują na pozytywny nastrój.',
    'neutral':  'Twarz spokojna, bez wyraźnych oznak emocji — typowy stan relaksu.',
    'sad':      'Opadające kąciki ust i brwi sygnalizują smutek lub przygnębienie.',
    'surprise': 'Zaskoczenie: szeroko otwarte oczy i usta, uniesione brwi.',
}

# Kolory do wykresu emocji
EMOTION_COLORS = {
    'angry': '#e74c3c', 'disgust': '#8e44ad', 'fear': '#2c3e50',
    'happy': '#f39c12', 'neutral': '#95a5a6', 'sad': '#2980b9', 'surprise': '#27ae60',
}

# Detektor twarzy OpenCV (Haar Cascade)
FACE_CASCADE = None
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    print("✅ Detektor twarzy załadowany")
except Exception as e:
    print(f"⚠️  Brak detektora twarzy: {e}")

# --- Model Breast Cancer ---
bc_model = None
bc_features = None
try:
    data = joblib.load('najlepszy_model_breast_cancer.pkl')
    bc_model = data['model']
    bc_features = data['features']
    print("✅ Model Breast Cancer załadowany")
except Exception as e:
    print(f"⚠️  Brak modelu BC: {e} — uruchom najpierw zajecia4_klasyfikacja.py z save_model=True")


# ══════════════════════════════════════════════════════════════
# FUNKCJE — TAB 1: EMOCJE
# ══════════════════════════════════════════════════════════════

def preprocess_face(img_array):
    """Konwertuje obraz do 48x48 grayscale, normalizuje."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 48, 48, 1)


def detect_and_crop_face(img_array):
    """Wykrywa twarz i zwraca wykadrowany region. Jeśli brak — zwraca cały obraz."""
    if FACE_CASCADE is None:
        return img_array, None

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return img_array, None  # Brak twarzy — użyj całego obrazu

    # Największa wykryta twarz
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Dodaj margines 20%
    margin = int(0.2 * min(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_array.shape[1], x + w + margin)
    y2 = min(img_array.shape[0], y + h + margin)
    return img_array[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def make_emotion_chart(probs):
    """Tworzy piękny wykres słupkowy z prawdopodobieństwami emocji."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0a0a14')
    ax.set_facecolor('#0a0a14')

    labels_pl = [EMOTION_PL[e] for e in EMOTION_LABELS]
    colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
    values = [p * 100 for p in probs]

    bars = ax.barh(labels_pl, values, color=colors, edgecolor='none', height=0.55)

    # Highlight najwyższy słupek
    max_idx = np.argmax(values)
    bars[max_idx].set_edgecolor('#ffffff')
    bars[max_idx].set_linewidth(1.5)

    # Etykiety wartości
    for bar, val in zip(bars, values):
        ax.text(min(val + 1.2, 97), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                color='#e0e0e0', fontsize=8.5, fontweight='600',
                fontfamily='monospace')

    ax.set_xlim(0, 108)
    ax.set_xlabel('Prawdopodobieństwo (%)', color='#555577', fontsize=9, labelpad=8)
    ax.set_title('Rozkład prawdopodobieństw', color='#9090bb',
                 fontsize=10, fontweight='600', pad=10, loc='left')
    ax.tick_params(colors='#6666aa', labelsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e1e3a')
    ax.spines['left'].set_color('#1e1e3a')
    ax.xaxis.grid(True, color='#141428', linestyle='-', linewidth=1, alpha=1)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    chart_img = Image.open(buf).copy()
    plt.close()
    return chart_img


def annotate_image(img_array, face_bbox):
    """Rysuje ramkę wokół wykrytej twarzy."""
    annotated = img_array.copy()
    if face_bbox is not None:
        x, y, w, h = face_bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (99, 102, 241), 3)
        # Narożniki akcentowe
        corner_len = max(8, min(w, h) // 6)
        thickness = 2
        for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = 1 if cx == x else -1
            dy = 1 if cy == y else -1
            cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy), (167, 139, 250), thickness + 1)
            cv2.line(annotated, (cx, cy), (cx, cy + dy * corner_len), (167, 139, 250), thickness + 1)
    return Image.fromarray(annotated)


def predict_emotion(image):
    """Główna funkcja predykcji emocji."""
    if image is None:
        return (
            "<div style='color:#6366f1;padding:40px;text-align:center;font-size:15px;"
            "font-family:\"DM Sans\",sans-serif;letter-spacing:0.5px;'>"
            "↑ Wgraj zdjęcie lub użyj kamery, aby rozpocząć analizę</div>",
            None, None
        )

    if emotion_model is None:
        return (
            "<div style='color:#ef4444;padding:20px;text-align:center;font-size:14px;"
            "font-family:\"DM Sans\",sans-serif;'>"
            "Brak pliku <code>best_model_base.keras</code> w folderze projektu.</div>",
            None, None
        )

    img_array = np.array(image)

    # Detekcja twarzy
    face_crop, face_bbox = detect_and_crop_face(img_array)

    # Preprocessing i predykcja
    processed = preprocess_face(face_crop)
    probs = emotion_model.predict(processed, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_emotion = EMOTION_LABELS[pred_idx]
    confidence = probs[pred_idx] * 100

    # HTML wynik
    color = EMOTION_COLORS[pred_emotion]
    emoji = EMOTION_EMOJI[pred_emotion]
    name_pl = EMOTION_PL[pred_emotion]
    desc = EMOTION_DESC[pred_emotion]

    face_status = ""
    if face_bbox is None and FACE_CASCADE is not None:
        face_status = (
            "<div style='display:inline-flex;align-items:center;gap:6px;"
            "background:#261a00;border:1px solid #78350f;border-radius:6px;"
            "padding:5px 12px;font-size:12px;color:#fbbf24;margin-top:12px;'>"
            "⚠ Nie wykryto twarzy — analiza całego obrazu</div>"
        )
    elif face_bbox is not None:
        face_status = (
            "<div style='display:inline-flex;align-items:center;gap:6px;"
            "background:#0a2318;border:1px solid #166534;border-radius:6px;"
            "padding:5px 12px;font-size:12px;color:#4ade80;margin-top:12px;'>"
            "✓ Twarz wykryta i wykadrowana</div>"
        )

    # Mini-tagi dla wszystkich emocji z %
    sorted_emotions = sorted(zip(EMOTION_LABELS, probs), key=lambda x: x[1], reverse=True)
    tags_html = ""
    for emo, prob in sorted_emotions:
        emo_color = EMOTION_COLORS[emo]
        is_top = emo == pred_emotion
        bg = f"{emo_color}28" if not is_top else f"{emo_color}40"
        border = f"{emo_color}55" if not is_top else f"{emo_color}cc"
        weight = "500" if not is_top else "700"
        tags_html += (
            f"<span style='background:{bg};border:1px solid {border};"
            f"color:{emo_color};padding:4px 11px;border-radius:20px;"
            f"font-size:11.5px;font-weight:{weight};white-space:nowrap;'>"
            f"{EMOTION_EMOJI[emo]} {EMOTION_PL[emo]} {prob*100:.0f}%</span>"
        )

    html = f"""
    <div style="font-family:'DM Sans',system-ui,sans-serif;max-width:580px;margin:0 auto;padding:4px 0;">

      <div style="display:flex;align-items:center;gap:18px;
                  background:linear-gradient(135deg,{color}18 0%,{color}08 100%);
                  border:1px solid {color}40;border-radius:14px;
                  padding:20px 24px;margin-bottom:14px;">
        <div style="font-size:52px;line-height:1;flex-shrink:0;">{emoji}</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;
                      color:{color}99;font-weight:600;margin-bottom:4px;">wykryta emocja</div>
          <div style="font-size:26px;font-weight:800;color:{color};letter-spacing:1px;
                      line-height:1.1;">{name_pl}</div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div style="flex:1;height:5px;background:{color}20;border-radius:3px;overflow:hidden;">
              <div style="width:{confidence:.1f}%;height:100%;background:{color};
                          border-radius:3px;transition:width 0.4s ease;"></div>
            </div>
            <span style="font-size:14px;font-weight:700;color:{color};
                         font-variant-numeric:tabular-nums;">{confidence:.1f}%</span>
          </div>
          {face_status}
        </div>
      </div>

      <div style="background:#0d0d1f;border:1px solid #1e1e3a;border-radius:10px;
                  padding:14px 18px;margin-bottom:12px;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;
                    color:#4a4a7a;font-weight:700;margin-bottom:6px;">interpretacja</div>
        <p style="color:#a0a0c8;margin:0;font-size:13.5px;line-height:1.65;">{desc}</p>
      </div>

      <div style="display:flex;flex-wrap:wrap;gap:6px;">
        {tags_html}
      </div>
    </div>
    """

    # Wykres
    chart = make_emotion_chart(probs)

    # Obraz z adnotacją twarzy
    annotated = annotate_image(img_array, face_bbox)

    return html, chart, annotated


# ══════════════════════════════════════════════════════════════
# FUNKCJE — TAB 2: BREAST CANCER
# ══════════════════════════════════════════════════════════════

def predict_breast_cancer(*args):
    if bc_model is None:
        return (
            "<div style='color:#ef4444;padding:40px;font-size:14px;text-align:center;"
            "font-family:\"DM Sans\",sans-serif;'>"
            "Brak pliku <code>najlepszy_model_breast_cancer.pkl</code>.<br>"
            "Uruchom zajecia4_klasyfikacja.py z opcją zapisu modelu.</div>"
        )

    input_df = pd.DataFrame([list(args)], columns=bc_features)
    pred = bc_model.predict(input_df)[0]
    probs = bc_model.predict_proba(input_df)[0]

    is_malignant = (pred == 4)
    result_class = "ZŁOŚLIWY" if is_malignant else "ŁAGODNY"
    confidence = probs[1] * 100 if is_malignant else probs[0] * 100
    color = "#ef4444" if is_malignant else "#22c55e"
    bg_color = "#1a0505" if is_malignant else "#051a0a"
    border_color = "#7f1d1d" if is_malignant else "#14532d"
    icon = "⚠" if is_malignant else "✓"
    label_color = "#fca5a5" if is_malignant else "#86efac"

    params_html = ""
    for col, val in zip(bc_features, args):
        bar_w = int(val * 10)
        params_html += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                    border-bottom:1px solid #0f0f20;">
          <div style="width:160px;font-size:12px;color:#6060a0;flex-shrink:0;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            {col.replace('_', ' ')}
          </div>
          <div style="flex:1;height:4px;background:#12122a;border-radius:2px;overflow:hidden;">
            <div style="width:{bar_w}%;height:100%;background:{color}80;border-radius:2px;"></div>
          </div>
          <div style="width:28px;text-align:right;font-size:12px;font-weight:700;
                      color:{color};font-variant-numeric:tabular-nums;">{int(val)}</div>
        </div>
        """

    html = f"""
    <div style="font-family:'DM Sans',system-ui,sans-serif;max-width:560px;margin:0 auto;padding:4px 0;">

      <div style="background:{bg_color};border:1px solid {border_color};
                  border-radius:14px;padding:24px;text-align:center;margin-bottom:16px;">
        <div style="width:56px;height:56px;border-radius:50%;
                    background:{color}20;border:2px solid {color}60;
                    display:inline-flex;align-items:center;justify-content:center;
                    font-size:22px;color:{color};margin-bottom:12px;">{icon}</div>
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:3px;
                    color:{color}80;font-weight:700;margin-bottom:6px;">wynik klasyfikacji</div>
        <div style="font-size:28px;font-weight:900;color:{color};letter-spacing:2px;
                    margin-bottom:10px;">{result_class}</div>
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:{color}15;border:1px solid {color}35;
                    border-radius:8px;padding:6px 16px;">
          <span style="font-size:12px;color:{label_color};">Pewność predykcji</span>
          <span style="font-size:16px;font-weight:800;color:{color};
                       font-variant-numeric:tabular-nums;">{confidence:.1f}%</span>
        </div>
      </div>

      <div style="background:#07070f;border:1px solid #141428;border-radius:10px;
                  padding:14px 18px;margin-bottom:12px;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;
                    color:#4a4a7a;font-weight:700;margin-bottom:10px;">parametry wejściowe</div>
        {params_html}
      </div>

      <div style="padding:10px 14px;background:#07070f;border-radius:8px;
                  border-left:3px solid #2a2a5a;">
        <p style="color:#4a4a6a;font-size:11.5px;margin:0;line-height:1.5;">
          System demonstracyjny — nie zastępuje profesjonalnej diagnozy medycznej.
        </p>
      </div>
    </div>
    """
    return html


# ══════════════════════════════════════════════════════════════
# BUDOWA INTERFEJSU GRADIO
# ══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: #05050f !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    min-height: 100vh;
}

/* ── Tabs ── */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1a1a2e !important;
    padding: 0 8px !important;
    gap: 4px !important;
}

.tab-nav button {
    background: transparent !important;
    color: #4a4a7a !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    color: #8080cc !important;
    background: #0d0d22 !important;
}

.tab-nav button.selected {
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
}

/* ── Panels & forms ── */
.gr-panel, .gr-box, .gr-form {
    background: #0a0a18 !important;
    border: 1px solid #14142a !important;
    border-radius: 10px !important;
}

/* ── Labels ── */
label span {
    color: #7070aa !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
.gr-button, button.primary, .btn-primary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 0 20px #4f46e520 !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: linear-gradient(135deg, #5a51f0, #8b47f7) !important;
    box-shadow: 0 0 30px #4f46e540 !important;
    transform: translateY(-1px) !important;
}

button.primary:active, .gr-button-primary:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 15px #4f46e530 !important;
}

/* ── Inputs ── */
input[type=number], input[type=text], textarea, select {
    background: #08080f !important;
    border: 1px solid #1e1e36 !important;
    color: #c0c0e0 !important;
    border-radius: 7px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

input[type=number]:focus, input[type=text]:focus, textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px #4f46e520 !important;
    outline: none !important;
}

/* ── Sliders ── */
input[type=range] {
    accent-color: #6366f1 !important;
}

.gr-slider-container .range-slider {
    accent-color: #6366f1 !important;
}

/* ── Image upload ── */
.gr-image-preview, .image-preview {
    background: #07070f !important;
    border: 1px dashed #2a2a4a !important;
    border-radius: 10px !important;
}

/* ── File upload zone ── */
.upload-box, .gr-file-upload {
    background: #07070f !important;
    border: 1.5px dashed #2a2a4a !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.upload-box:hover {
    border-color: #4f46e5 !important;
    background: #0a0a18 !important;
}

/* ── Row & column spacing ── */
.gap-4, .gr-row {
    gap: 16px !important;
}

/* ── Output HTML containers ── */
.gr-html-output, .output-html {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Accordion / details ── */
details {
    transition: all 0.2s ease;
}

details summary:hover {
    color: #8888cc !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #05050f; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #4a4a7a; }

/* ── Sub-tab (upload/camera) inner tabs ── */
.tab-nav.svelte-1uw5tnk button, .inner-tab-nav button {
    font-size: 12px !important;
    padding: 8px 14px !important;
}
"""

HEADER_HTML = """
<div style="
    font-family:'DM Sans',system-ui,sans-serif;
    padding: 40px 48px 28px;
    position: relative;
    overflow: hidden;
">
    <!-- Decorative grid -->
    <div style="
        position:absolute;top:0;left:0;right:0;bottom:0;
        background-image:
            linear-gradient(#1a1a3a22 1px, transparent 1px),
            linear-gradient(90deg, #1a1a3a22 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events:none;
    "></div>

    <!-- Decorative orbs -->
    <div style="
        position:absolute;top:-60px;right:80px;
        width:200px;height:200px;border-radius:50%;
        background:radial-gradient(circle, #4f46e530 0%, transparent 70%);
        pointer-events:none;
    "></div>
    <div style="
        position:absolute;bottom:-40px;left:40px;
        width:150px;height:150px;border-radius:50%;
        background:radial-gradient(circle, #7c3aed20 0%, transparent 70%);
        pointer-events:none;
    "></div>

    <!-- Content -->
    <div style="position:relative;z-index:1;">
        <div style="
            display:inline-flex;align-items:center;gap:8px;
            background:#0d0d22;border:1px solid #2a2a4a;
            border-radius:20px;padding:4px 14px;
            font-size:11px;font-weight:700;letter-spacing:2.5px;
            color:#6366f1;text-transform:uppercase;
            margin-bottom:16px;
        ">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#6366f1;"></span>
            Uczenie Maszynowe II &nbsp;·&nbsp; Projekt 5
        </div>

        <h1 style="
            font-size:38px;font-weight:900;margin:0 0 8px 0;
            color:#f0f0ff;letter-spacing:-0.5px;line-height:1.1;
        ">
            Akademia Tarnowska
            <span style="
                display:inline-block;
                background:linear-gradient(135deg,#6366f1,#a78bfa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;
            "> ML Demo</span>
        </h1>

        <p style="
            font-size:14px;color:#5050a0;margin:0;
            font-weight:500;letter-spacing:0.5px;
        ">
            Interfejs demonstracyjny systemów rozpoznawania emocji i diagnostyki medycznej
        </p>

        <!-- Status chips -->
        <div style="display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;">
            <div style="
                display:inline-flex;align-items:center;gap:7px;
                background:#07120d;border:1px solid #14532d;
                border-radius:8px;padding:6px 14px;
            ">
                <span style="width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0;
                             box-shadow:0 0 6px #22c55e80;"></span>
                <span style="font-size:12px;font-weight:600;color:#4ade80;">CNN FER-2013</span>
                <span style="font-size:11px;color:#166534;">35 887 obrazów · 7 klas</span>
            </div>
            <div style="
                display:inline-flex;align-items:center;gap:7px;
                background:#07120d;border:1px solid #14532d;
                border-radius:8px;padding:6px 14px;
            ">
                <span style="width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0;
                             box-shadow:0 0 6px #22c55e80;"></span>
                <span style="font-size:12px;font-weight:600;color:#4ade80;">Breast Cancer Wisconsin</span>
                <span style="font-size:11px;color:#166534;">klasyczny ML</span>
            </div>
        </div>
    </div>
</div>
"""

EMOTION_TAB_INFO = """
<div style="
    font-family:'DM Sans',system-ui,sans-serif;
    display:flex;align-items:flex-start;gap:12px;
    background:#080814;border:1px solid #141428;border-radius:10px;
    padding:14px 18px;margin-bottom:4px;
">
    <div style="
        width:32px;height:32px;border-radius:8px;
        background:#1e1b4b;border:1px solid #312e81;
        display:flex;align-items:center;justify-content:center;
        font-size:16px;flex-shrink:0;
    ">🧠</div>
    <div>
        <div style="font-size:12px;font-weight:700;color:#818cf8;
                    text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">
            Model CNN — FER-2013
        </div>
        <div style="font-size:13px;color:#5050a0;line-height:1.6;">
            Architektura: 3× [Conv2D → BatchNorm → MaxPool → Dropout] → Dense(256) → Softmax.
            Preprocessing: grayscale 48×48, normalizacja /255, augmentacja rotacja/flip/zoom.
        </div>
    </div>
</div>
"""

BC_TAB_INFO = """
<div style="
    font-family:'DM Sans',system-ui,sans-serif;
    display:flex;align-items:flex-start;gap:12px;
    background:#080814;border:1px solid #141428;border-radius:10px;
    padding:14px 18px;margin-bottom:4px;
">
    <div style="
        width:32px;height:32px;border-radius:8px;
        background:#052e16;border:1px solid #14532d;
        display:flex;align-items:center;justify-content:center;
        font-size:16px;flex-shrink:0;
    ">🔬</div>
    <div>
        <div style="font-size:12px;font-weight:700;color:#4ade80;
                    text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">
            Model ML — Breast Cancer Wisconsin
        </div>
        <div style="font-size:13px;color:#5050a0;line-height:1.6;">
            Klasyfikacja guza na podstawie parametrów histopatologicznych w skali 1–10.
            Ustaw wartości suwaków i kliknij przycisk klasyfikacji.
        </div>
    </div>
</div>
"""

FOOTER_HTML = """
<div style="
    font-family:'DM Sans',system-ui,sans-serif;
    text-align:center;padding:20px;
    border-top:1px solid #0f0f22;margin-top:8px;
    display:flex;align-items:center;justify-content:center;gap:16px;flex-wrap:wrap;
">
    <span style="color:#2a2a4a;font-size:11px;font-weight:600;letter-spacing:1px;">
        AKADEMIA TARNOWSKA
    </span>
    <span style="color:#1a1a3a;font-size:11px;">·</span>
    <span style="color:#2a2a4a;font-size:11px;">Uczenie Maszynowe II</span>
    <span style="color:#1a1a3a;font-size:11px;">·</span>
    <span style="color:#2a2a4a;font-size:11px;">Projekt 5</span>
    <span style="color:#1a1a3a;font-size:11px;">·</span>
    <span style="color:#2a2a4a;font-size:11px;">CNN + Classical ML</span>
</div>
"""

DETAILS_HTML = """
<details style="margin-top:14px;border:1px solid #141428;border-radius:10px;overflow:hidden;">
    <summary style="
        font-family:'DM Sans',sans-serif;
        color:#5050a0;cursor:pointer;font-size:12px;font-weight:700;
        text-transform:uppercase;letter-spacing:1.5px;
        padding:12px 16px;background:#07070f;
        list-style:none;display:flex;align-items:center;gap:8px;
        border-bottom:1px solid transparent;
        transition: all 0.2s ease;
    ">
        <span style="font-size:14px;">⚙</span>
        Szczegóły modelu i architektury CNN
    </summary>
    <div style="
        background:#07070f;padding:16px 20px;
        font-family:'DM Mono',monospace;font-size:12px;
        color:#5050a0;line-height:2;
    ">
        <div><span style="color:#3a3a7a;text-transform:uppercase;font-size:10px;
                          letter-spacing:1.5px;font-weight:700;">Zbiór danych</span><br>
             <span style="color:#8080c0;">FER-2013 — 28 709 treningowych / 7 178 testowych</span></div>
        <div style="margin-top:10px;"><span style="color:#3a3a7a;text-transform:uppercase;font-size:10px;
                          letter-spacing:1.5px;font-weight:700;">Klasy</span><br>
             <span style="color:#8080c0;">angry · disgust · fear · happy · neutral · sad · surprise</span></div>
        <div style="margin-top:10px;"><span style="color:#3a3a7a;text-transform:uppercase;font-size:10px;
                          letter-spacing:1.5px;font-weight:700;">Architektura</span><br>
             <span style="color:#8080c0;">3× [Conv2D → BN → Conv2D → BN → MaxPool → Dropout]<br>
             → Dense(256) → BN → Dropout → Softmax</span></div>
        <div style="margin-top:10px;"><span style="color:#3a3a7a;text-transform:uppercase;font-size:10px;
                          letter-spacing:1.5px;font-weight:700;">Optymalizacja</span><br>
             <span style="color:#8080c0;">Adam lr=0.001 · ReduceLROnPlateau · EarlyStopping</span></div>
    </div>
</details>
"""

with gr.Blocks(
    css=CUSTOM_CSS,
    title="AT · ML Demo",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif"],
    )
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── TAB 1: EMOCJE ──────────────────────────────────────────
        with gr.Tab("😊  Rozpoznawanie emocji"):
            gr.HTML(EMOTION_TAB_INFO)

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tab("📁 Upload"):
                        img_upload = gr.Image(
                            label="Wgraj zdjęcie",
                            type="pil",
                            sources=["upload"],
                            height=280,
                        )
                        btn_upload = gr.Button("Analizuj emocję →", variant="primary", size="lg")

                    with gr.Tab("📷 Kamera"):
                        img_camera = gr.Image(
                            label="Zrób zdjęcie",
                            type="pil",
                            sources=["webcam"],
                            height=280,
                            streaming=False,
                        )
                        btn_camera = gr.Button("Analizuj emocję →", variant="primary", size="lg")

                with gr.Column(scale=1):
                    emotion_html = gr.HTML(
                        value=(
                            "<div style='font-family:\"DM Sans\",sans-serif;"
                            "color:#2a2a4a;padding:48px;text-align:center;"
                            "font-size:13px;letter-spacing:0.5px;'>"
                            "← Wgraj lub zrób zdjęcie, aby zobaczyć wynik</div>"
                        )
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    annotated_img = gr.Image(label="Detekcja twarzy", height=210)
                with gr.Column(scale=2):
                    emotion_chart = gr.Image(label="Rozkład prawdopodobieństw", height=210)

            btn_upload.click(
                fn=predict_emotion,
                inputs=[img_upload],
                outputs=[emotion_html, emotion_chart, annotated_img]
            )
            btn_camera.click(
                fn=predict_emotion,
                inputs=[img_camera],
                outputs=[emotion_html, emotion_chart, annotated_img]
            )

            gr.HTML(DETAILS_HTML)

        # ── TAB 2: BREAST CANCER ───────────────────────────────────
        with gr.Tab("🔬  Diagnostyka nowotworów"):
            gr.HTML(BC_TAB_INFO)

            if bc_model is not None and bc_features is not None:
                with gr.Row():
                    with gr.Column(scale=1):
                        bc_inputs = []
                        for feat in bc_features:
                            slider = gr.Slider(
                                minimum=1, maximum=10, step=1,
                                label=feat.replace('_', ' '),
                                value=5
                            )
                            bc_inputs.append(slider)
                        btn_bc = gr.Button("Klasyfikuj guz →", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        bc_html = gr.HTML(
                            value=(
                                "<div style='font-family:\"DM Sans\",sans-serif;"
                                "color:#2a2a4a;padding:48px;text-align:center;"
                                "font-size:13px;letter-spacing:0.5px;'>"
                                "← Ustaw parametry i kliknij przycisk klasyfikacji</div>"
                            )
                        )

                btn_bc.click(
                    fn=predict_breast_cancer,
                    inputs=bc_inputs,
                    outputs=[bc_html]
                )
            else:
                gr.HTML("""
                <div style="
                    font-family:'DM Sans',sans-serif;
                    background:#0a0505;border:1px solid #7f1d1d;border-radius:12px;
                    padding:28px;text-align:center;
                ">
                    <div style="font-size:28px;margin-bottom:12px;">⚠</div>
                    <div style="font-size:16px;font-weight:700;color:#fca5a5;margin-bottom:8px;">
                        Model Breast Cancer niedostępny
                    </div>
                    <div style="font-size:13px;color:#4a1515;line-height:1.7;">
                        Uruchom <code style="background:#1a0505;padding:2px 8px;border-radius:4px;
                        color:#f87171;">zajecia4_klasyfikacja.py</code> z opcją zapisu modelu, a następnie
                        uruchom ponownie aplikację.
                    </div>
                </div>
                """)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  PROJEKT 5 — Interfejs ML")
    print("═" * 55)
    print(f"  Model emocji:  {'✅ gotowy' if emotion_model else '❌ brak best_model_base.keras'}")
    print(f"  Model BC:      {'✅ gotowy' if bc_model else '❌ brak najlepszy_model_breast_cancer.pkl'}")
    print("═" * 55 + "\n")
    demo.launch(share=True)