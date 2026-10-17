# Green Lens — Plant Disease Detection Web App

واجهة Streamlit لمشروع كشف أمراض النباتات باستخدام موديل MobileNetV2 الموجود في ملف Google Colab المرفق.

## 1) تصدير ملفات الموديل من Colab

شغّل خلايا التدريب والحفظ في الـNotebook حتى تتأكد أن هذا المجلد موجود:

`/content/green_lens_model/`

ويحتوي على:
- `green_lens_mobilenetv2.pth`
- `class_names.json`

بعدها أضف خلية جديدة في Colab والصق الكود التالي، ثم شغّلها لتنزيل الملف المضغوط:

```python
from pathlib import Path
import zipfile
from google.colab import files

model_dir = Path("/content/green_lens_model")
required = [model_dir / "green_lens_mobilenetv2.pth", model_dir / "class_names.json"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "ملفات النموذج غير موجودة. شغّل خلايا التدريب والحفظ أولًا: " + ", ".join(missing)
    )

zip_path = Path("/content/green_lens_model.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in required:
        zf.write(path, arcname=f"green_lens_model/{path.name}")

files.download(str(zip_path))
```

فك ضغط `green_lens_model.zip` وضع مجلد `green_lens_model` الناتج بجانب `app.py` داخل مشروع الويب.

يجب أن يصبح ترتيب الملفات هكذا:

```text
green_lens_webapp/
├── app.py
├── requirements.txt
├── README.md
└── green_lens_model/
    ├── green_lens_mobilenetv2.pth
    └── class_names.json
```

## 2) تشغيل التطبيق على Windows

يفضّل استخدام Python 3.11 أو 3.12 في بيئة افتراضية جديدة.

افتح Terminal / PowerShell داخل مجلد المشروع وشغّل:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

سيفتح التطبيق في المتصفح. ارفع صورة بصيغة JPG أو PNG أو WEBP لمشاهدة أعلى ثلاثة توقعات.

## 3) نشر التطبيق على الإنترنت

يمكن نشره عبر Streamlit Community Cloud أو خدمة استضافة تدعم Streamlit. ارفع ملفات المشروع إلى مستودع GitHub، وتأكد من وجود مجلد النموذج في المشروع، ثم اختر `app.py` كنقطة تشغيل. حجم ملفات الموديل وسياسة الاستضافة قد يؤثران في طريقة النشر.

## ملاحظات

- ملف `.ipynb` وحده لا يتضمن بالضرورة أوزان النموذج الناتجة عن التدريب؛ يجب تصدير ملفات النموذج بشكل منفصل كما سبق.
- التطبيق يعرض توقعات النموذج ودرجات Softmax، وهذه الدرجات ليست ضمانًا لاحتمال صحة التشخيص.
- النموذج مبني على فئات مجموعة البيانات التي استُخدمت في التدريب؛ قد تكون النتائج أضعف مع صور ميدانية مختلفة أو أنواع نباتات خارج تلك الفئات.
