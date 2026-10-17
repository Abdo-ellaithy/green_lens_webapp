# انسخ هذه الخلية وشغّلها في Google Colab بعد حفظ النموذج في /content/green_lens_model.
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
