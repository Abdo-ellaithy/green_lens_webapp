from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "green_lens_model"
WEIGHTS_PATH = MODEL_DIR / "green_lens_mobilenetv2.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"

st.set_page_config(
    page_title="Green Lens | كشف أمراض النباتات",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 920px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(135deg, #eaf7ee, #f7fbf7); border: 1px solid #d6eadb; margin-bottom: 1rem;}
      .hero h1 {color: #14532d; margin-bottom: .35rem;}
      .muted {color: #52645a;}
      .small-note {font-size: .9rem; color: #64748b;}
      div[data-testid="stMetric"] {background: #f6faf7; border: 1px solid #dcebe0; padding: 12px; border-radius: 12px;}
    </style>
    <div class="hero" dir="rtl">
      <h1>🌿 Green Lens</h1>
      <p class="muted">ارفع صورة لورقة نبات، وسيحللها نموذج الذكاء الاصطناعي ويعرض أكثر 3 فئات متوقعة.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="جاري تحميل نموذج الذكاء الاصطناعي...")
def load_predictor():
    if not WEIGHTS_PATH.exists() or not CLASSES_PATH.exists():
        missing = []
        if not WEIGHTS_PATH.exists():
            missing.append(WEIGHTS_PATH.name)
        if not CLASSES_PATH.exists():
            missing.append(CLASSES_PATH.name)
        raise FileNotFoundError(
            "ملفات النموذج غير موجودة داخل مجلد green_lens_model: " + ", ".join(missing)
        )

    with CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("ملف class_names.json لا يحتوي على قائمة فئات صحيحة.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model, class_names, device, image_transform


def readable_label(label: str) -> str:
    # Preserve the original class name while making it easier to read.
    return label.replace("___", " — ").replace("_", " ")

try:
    model, class_names, device, image_transform = load_predictor()
    st.caption(f"النموذج جاهز · عدد الفئات: {len(class_names)} · الجهاز: {'GPU' if device.type == 'cuda' else 'CPU'}")
except (FileNotFoundError, ValueError, RuntimeError) as exc:
    st.warning("التطبيق اتجهّز، لكن ملفات النموذج المدرّب لسه مش موجودة أو غير صالحة.")
    st.markdown("**المطلوب داخل مجلد `green_lens_model`:**")
    st.code("green_lens_model/\n├── green_lens_mobilenetv2.pth\n└── class_names.json", language="text")
    st.info("افتح ملف README الموجود مع التطبيق واتبع خطوة تصدير ملفات النموذج من Google Colab.")
    with st.expander("تفاصيل الخطأ"):
        st.code(str(exc))
    st.stop()

st.subheader("جرّب فحص صورة")
uploaded_file = st.file_uploader(
    "اختار صورة ورقة نبات",
    type=["jpg", "jpeg", "png", "webp"],
    help="استخدم صورة واضحة للورقة قدر الإمكان.",
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.image(image, caption="الصورة المرفوعة", use_container_width=True)
        with right:
            st.markdown("#### نتيجة التحليل")
            with st.spinner("جاري تحليل الصورة..."):
                input_tensor = image_transform(image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    count = min(3, len(class_names))
                    top_probs, top_indices = torch.topk(probabilities, k=count)

            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
                label = class_names[int(idx.item())]
                score = float(prob.item()) * 100
                st.markdown(f"**{rank}. {readable_label(label)}**")
                st.progress(max(0.0, min(score / 100.0, 1.0)))
                st.caption(f"درجة ثقة النموذج: {score:.2f}%")
    except Exception as exc:
        st.error("لم نتمكن من قراءة الصورة أو تحليلها. جرّب صورة أخرى بصيغة JPG أو PNG.")
        with st.expander("تفاصيل الخطأ"):
            st.code(str(exc))

st.divider()
st.markdown(
    "<p class='small-note' dir='rtl'>تنبيه: النتيجة توقع من نموذج ذكاء اصطناعي وليست تشخيصًا زراعيًا مؤكدًا. قد تختلف الدقة مع الصور الواقعية أو النباتات غير الموجودة ضمن فئات التدريب.</p>",
    unsafe_allow_html=True,
)
