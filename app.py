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
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    labels_pl = [EMOTION_PL[e] for e in EMOTION_LABELS]
    colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
    values = [p * 100 for p in probs]

    bars = ax.barh(labels_pl, values, color=colors, edgecolor='none',
                   height=0.6)

    # Highlight najwyższy słupek
    max_idx = np.argmax(values)
    bars[max_idx].set_edgecolor('white')
    bars[max_idx].set_linewidth(2)

    # Etykiety wartości
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 97), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                color='white', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Prawdopodobieństwo (%)', color='#aaaaaa', fontsize=10)
    ax.set_title('Rozkład prawdopodobieństw emocji', color='white',
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#cccccc', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.xaxis.grid(True, color='#2a2a4a', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
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
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (39, 174, 96), 3)
    return Image.fromarray(annotated)


def predict_emotion(image):
    """Główna funkcja predykcji emocji."""
    if image is None:
        return (
            "<div style='color:#e74c3c;padding:20px;text-align:center;font-size:16px;'>"
            "⚠️ Brak obrazu. Wgraj zdjęcie lub użyj kamery.</div>",
            None, None
        )

    if emotion_model is None:
        return (
            "<div style='color:#e74c3c;padding:20px;text-align:center;font-size:16px;'>"
            "❌ Model <b>best_model_base.keras</b> nie został znaleziony w folderze projektu.<br>"
            "Upewnij się, że plik istnieje i uruchom ponownie.</div>",
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

    face_info = ""
    if face_bbox is None and FACE_CASCADE is not None:
        face_info = "<p style='color:#f39c12;font-size:13px;margin-top:8px;'>⚠️ Nie wykryto twarzy — analiza całego obrazu</p>"
    elif face_bbox is not None:
        face_info = "<p style='color:#2ecc71;font-size:13px;margin-top:8px;'>✅ Twarz wykryta i wykadrowana</p>"

    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, {color}22, {color}44);
                    border: 2px solid {color};
                    border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 64px; margin-bottom: 8px;">{emoji}</div>
            <div style="font-size: 28px; font-weight: 800; color: {color}; letter-spacing: 2px;">
                {name_pl.upper()}
            </div>
            <div style="font-size: 20px; color: #ffffff; margin-top: 6px; font-weight: 600;">
                Pewność: <span style="color:{color}">{confidence:.1f}%</span>
            </div>
            {face_info}
        </div>
        <div style="background: #1e1e2e; border-radius: 12px; padding: 16px;
                    border-left: 4px solid {color};">
            <h4 style="color: {color}; margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase;
                       letter-spacing: 1px;">📝 Interpretacja</h4>
            <p style="color: #cccccc; margin: 0; font-size: 14px; line-height: 1.6;">{desc}</p>
        </div>
        <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
    """

    # Mini-tagi dla wszystkich emocji z %
    sorted_emotions = sorted(zip(EMOTION_LABELS, probs), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_emotions:
        emo_color = EMOTION_COLORS[emo]
        opacity = "ff" if emo == pred_emotion else "55"
        html += f"""
            <span style="background: {emo_color}{opacity}; color: white;
                         padding: 4px 10px; border-radius: 20px; font-size: 12px;
                         font-weight: 600;">
                {EMOTION_EMOJI[emo]} {EMOTION_PL[emo]} {prob*100:.0f}%
            </span>
        """

    html += "</div></div>"

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
        return """<div style='color:#e74c3c;padding:20px;font-size:15px;text-align:center;'>
        ❌ Brak modelu <b>najlepszy_model_breast_cancer.pkl</b>.<br>
        Uruchom zajecia4_klasyfikacja.py z opcją zapisu modelu.</div>"""

    input_df = pd.DataFrame([list(args)], columns=bc_features)
    pred = bc_model.predict(input_df)[0]
    probs = bc_model.predict_proba(input_df)[0]

    result_class = "ZŁOŚLIWY" if pred == 4 else "ŁAGODNY"
    confidence = probs[1] * 100 if pred == 4 else probs[0] * 100
    color = "#e74c3c" if pred == 4 else "#2ecc71"
    icon = "⚠️" if pred == 4 else "✅"

    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: {color}22; border: 2px solid {color};
                    border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 48px;">{icon}</div>
            <div style="font-size: 26px; font-weight: 800; color: {color}; margin: 8px 0;">
                {result_class}
            </div>
            <div style="color: #ccc; font-size: 17px;">
                Pewność predykcji: <b style="color:{color}">{confidence:.1f}%</b>
            </div>
        </div>
        <div style="background: #1e1e2e; border-radius: 12px; padding: 16px;
                    border-left: 4px solid {color};">
            <h4 style="color:{color}; margin: 0 0 8px 0; font-size:13px; text-transform:uppercase;">
                📋 Wprowadzone parametry
            </h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
    """
    for col, val in zip(bc_features, args):
        html += f"""
            <div style="color:#aaa; font-size:13px;">
                {col.replace('_', ' ')}: <b style="color:white">{val}/10</b>
            </div>
        """
    html += """
            </div>
        </div>
        <div style="margin-top:12px; padding:12px; background:#111827; border-radius:8px;">
            <p style="color:#888; font-size:12px; margin:0; text-align:center;">
                ⚕️ System demonstracyjny — nie zastępuje profesjonalnej diagnozy medycznej.
            </p>
        </div>
    </div>
    """
    return html


# ══════════════════════════════════════════════════════════════
# BUDOWA INTERFEJSU GRADIO
# ══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* Tło główne */
.gradio-container {
    background: #0d0d1a !important;
    font-family: 'Segoe UI', sans-serif !important;
}
/* Nagłówek */
.main-header {
    background: linear-gradient(135deg, #0f0f23, #1a1a3e);
    border-bottom: 1px solid #2a2a5a;
    padding: 28px 40px 20px;
    text-align: center;
}
/* Karty tabów */
.tab-nav button {
    background: #1a1a2e !important;
    color: #8888aa !important;
    border: 1px solid #2a2a4a !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
}
.tab-nav button.selected {
    background: #2a2a5e !important;
    color: #ffffff !important;
    border-color: #5555aa !important;
}
/* Inputy */
.gr-box, .gr-form, .gr-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
}
"""

with gr.Blocks(
    css=CUSTOM_CSS,
    title="ML Demo — Emocje & Diagnostyka",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Syne"), "ui-sans-serif"],
    )
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 32px 20px 16px; background: linear-gradient(135deg,#0d0d1a,#1a1a3e);">
        <h1 style="font-size:32px; font-weight:900; color:white; letter-spacing:3px; margin:0;">
            🧠 AKADEMIA TARNOWSKA
        </h1>
        <p style="color:#7777cc; font-size:14px; letter-spacing:4px; margin:8px 0 0 0; font-weight:600;">
            UCZENIE MASZYNOWE II — PROJEKT 5 — INTERFEJS DEMONSTRACYJNY
        </p>
        <div style="width:60px; height:3px; background:linear-gradient(90deg,#6366f1,#8b5cf6);
                    margin:16px auto 0; border-radius:2px;"></div>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: EMOCJE ──────────────────────────────────────────
        with gr.Tab("😊 Rozpoznawanie Emocji (CNN)"):
            gr.HTML("""
            <div style="padding:16px 0 8px; color:#8888bb; font-size:13px; line-height:1.6;">
                Model CNN wytrenowany na zbiorze <b style="color:#a0a0ff">FER-2013</b>
                (35 887 obrazów 48×48, 7 klas emocji). Architektura: 3× bloki Conv2D+BN+MaxPool+Dropout,
                warstwa Dense(256), softmax. Wgraj zdjęcie twarzy lub użyj kamery.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tab("📁 Upload zdjęcia"):
                        img_upload = gr.Image(
                            label="Wgraj zdjęcie",
                            type="pil",
                            sources=["upload"],
                            height=300,
                        )
                        btn_upload = gr.Button("🔍 Analizuj emocję", variant="primary", size="lg")

                    with gr.Tab("📷 Kamera na żywo"):
                        img_camera = gr.Image(
                            label="Zrób zdjęcie kamerą",
                            type="pil",
                            sources=["webcam"],
                            height=300,
                            streaming=False,
                        )
                        btn_camera = gr.Button("🔍 Analizuj emocję", variant="primary", size="lg")

                with gr.Column(scale=1):
                    emotion_html = gr.HTML(
                        value="<div style='color:#555;padding:40px;text-align:center;font-size:14px;'>"
                              "← Wgraj lub zrób zdjęcie, aby zobaczyć wynik</div>"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    annotated_img = gr.Image(label="Obraz z detekcją twarzy", height=220)
                with gr.Column(scale=2):
                    emotion_chart = gr.Image(label="Rozkład prawdopodobieństw", height=220)

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

            gr.HTML("""
            <details style="margin-top:16px; padding:12px; background:#111827;
                            border-radius:8px; border:1px solid #1f2937;">
                <summary style="color:#7777cc; cursor:pointer; font-size:13px; font-weight:600;">
                    ℹ️ Szczegóły modelu i architektury CNN
                </summary>
                <div style="color:#888; font-size:12px; margin-top:10px; line-height:1.8;">
                    <b style="color:#aaa">Zbiór danych:</b> FER-2013 — 28 709 treningowych / 7 178 testowych<br>
                    <b style="color:#aaa">Klasy:</b> angry, disgust, fear, happy, neutral, sad, surprise<br>
                    <b style="color:#aaa">Preprocessing:</b> grayscale 48×48, normalizacja /255<br>
                    <b style="color:#aaa">Augmentacja:</b> rotacja ±15°, flip, zoom 10%, brightness 0.8–1.2<br>
                    <b style="color:#aaa">Architektura:</b> 3× [Conv2D → BN → Conv2D → BN → MaxPool → Dropout] → Dense(256) → BN → Dropout → softmax<br>
                    <b style="color:#aaa">Optymalizator:</b> Adam(lr=0.001) z ReduceLROnPlateau i EarlyStopping
                </div>
            </details>
            """)

        # ── TAB 2: BREAST CANCER ───────────────────────────────────
        with gr.Tab("🔬 Diagnostyka Nowotworów (ML)"):
            gr.HTML("""
            <div style="padding:16px 0 8px; color:#8888bb; font-size:13px; line-height:1.6;">
                Model klasyczny ML wytrenowany na zbiorze
                <b style="color:#a0a0ff">Breast Cancer Wisconsin</b>.
                Ustaw wartości parametrów histopatologicznych (skala 1–10).
            </div>
            """)

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
                        btn_bc = gr.Button("🔬 Klasyfikuj guz", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        bc_html = gr.HTML(
                            value="<div style='color:#555;padding:40px;text-align:center;font-size:14px;'>"
                                  "← Ustaw parametry i kliknij przycisk</div>"
                        )

                btn_bc.click(
                    fn=predict_breast_cancer,
                    inputs=bc_inputs,
                    outputs=[bc_html]
                )
            else:
                gr.HTML("""
                <div style="background:#1a0f0f; border:1px solid #5a2222; border-radius:12px;
                            padding:24px; text-align:center; color:#cc8888;">
                    <h3>❌ Model Breast Cancer niedostępny</h3>
                    <p style="font-size:13px; color:#888;">
                        Uruchom <b>zajecia4_klasyfikacja.py</b> z opcją zapisu modelu:<br><br>
                        <code style="background:#111; padding:4px 12px; border-radius:4px;">
                        joblib.dump({'model': best_cv_pipe, 'features': list(X.columns)},
                        'najlepszy_model_breast_cancer.pkl')
                        </code>
                    </p>
                </div>
                """)

    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#444466; font-size:11px;
                border-top:1px solid #1a1a3a; margin-top:24px;">
        Akademia Tarnowska · Uczenie Maszynowe II · Projekt 5 ·
        CNN (FER-2013) + Classical ML (Breast Cancer Wisconsin)
    </div>
    """)


if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  PROJEKT 5 — Interfejs ML")
    print("═" * 55)
    print(f"  Model emocji:  {'✅ gotowy' if emotion_model else '❌ brak best_model_base.keras'}")
    print(f"  Model BC:      {'✅ gotowy' if bc_model else '❌ brak najlepszy_model_breast_cancer.pkl'}")
    print("═" * 55 + "\n")
    demo.launch(share=True)