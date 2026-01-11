from __future__ import annotations

import os
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from ultralytics import YOLO


# -----------------------
# Configuración
# -----------------------
INPUT_DIR = Path(r"C:\Users\Hugo\Desktop\repositorio\Fotos")
OUTPUT_DIR = Path(r"C:\Users\Hugo\Desktop\repositorio\bb")

MODEL_WEIGHTS = "yolo11n.pt"   # puedes cambiar a yolo11s.pt / yolo11m.pt, etc.
CONF_THRES = 0.25             # confianza mínima
IOU_THRES = 0.7               # iou NMS
PADDING_PX = 10               # padding opcional alrededor del recorte
SAVE_FORMAT = "JPEG"          # "JPEG" o "PNG"


# -----------------------
# Utilidades
# -----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def scan_image_formats(folder: Path) -> None:
    """Imprime un resumen de formatos por extensión y por PIL."""
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not paths:
        print(f"No encontré imágenes en: {folder}")
        return

    by_ext = Counter(p.suffix.lower() for p in paths)
    by_pil = Counter()

    for p in paths:
        try:
            with Image.open(p) as im:
                by_pil[im.format or "UNKNOWN"] += 1
        except Exception:
            by_pil["UNREADABLE"] += 1

    print("\n=== Formatos por extensión ===")
    for k, v in by_ext.most_common():
        print(f"{k:>6}: {v}")

    print("\n=== Formatos detectados por PIL ===")
    for k, v in by_pil.most_common():
        print(f"{k:>10}: {v}")

    print(f"\nTotal imágenes encontradas: {len(paths)}\n")


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def ensure_rgb(pil_img: Image.Image) -> Image.Image:
    # Para guardar como JPEG (no soporta alpha), pasamos a RGB
    if pil_img.mode in ("RGBA", "LA", "P"):
        return pil_img.convert("RGB")
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


# -----------------------
# Main
# -----------------------
def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Reporte de formatos
    scan_image_formats(INPUT_DIR)

    # 2) Cargar modelo YOLO11
    model = YOLO(MODEL_WEIGHTS)

    # Buscar el id de la clase "cat" desde los nombres del modelo (sin hardcodear)
    # model.names suele ser dict: {0:"person", 1:"bicycle", ...}
    cat_ids = [cls_id for cls_id, name in model.names.items() if str(name).lower() == "cat"]
    if not cat_ids:
        raise RuntimeError("No encontré la clase 'cat' en model.names. ¿Estás usando un modelo COCO?")
    cat_id = cat_ids[0]
    print(f"Clase 'cat' detectada con id = {cat_id}")

    # 3) Procesar imágenes
    img_paths = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not img_paths:
        print("No hay imágenes para procesar.")
        return

    total_crops = 0
    total_imgs_with_cat = 0

    for img_path in img_paths:
        try:
            # Inferencia
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False
            )
            r0 = results[0]

            if r0.boxes is None or len(r0.boxes) == 0:
                continue

            # Cargar imagen con PIL (para recortar y guardar)
            with Image.open(img_path) as im:
                im = ensure_rgb(im)
                w, h = im.size

                # boxes.xyxy: tensor Nx4
                xyxy = r0.boxes.xyxy.cpu().numpy()  # (N,4)
                cls = r0.boxes.cls.cpu().numpy().astype(int)  # (N,)
                conf = r0.boxes.conf.cpu().numpy()  # (N,)

                # Filtrar gatos
                idxs = np.where(cls == cat_id)[0]
                if idxs.size == 0:
                    continue

                total_imgs_with_cat += 1

                for j, i in enumerate(idxs):
                    x1, y1, x2, y2 = xyxy[i]

                    # padding + clamp a bordes
                    x1i = clamp(int(np.floor(x1)) - PADDING_PX, 0, w - 1)
                    y1i = clamp(int(np.floor(y1)) - PADDING_PX, 0, h - 1)
                    x2i = clamp(int(np.ceil(x2)) + PADDING_PX, 0, w)
                    y2i = clamp(int(np.ceil(y2)) + PADDING_PX, 0, h)

                    if x2i <= x1i or y2i <= y1i:
                        continue

                    crop = im.crop((x1i, y1i, x2i, y2i))

                    out_name = f"{img_path.stem}_cat_{j}_conf_{conf[i]:.2f}.{ 'jpg' if SAVE_FORMAT.upper()=='JPEG' else 'png' }"
                    out_path = OUTPUT_DIR / out_name
                    crop.save(out_path, format=SAVE_FORMAT)

                    total_crops += 1

        except Exception as e:
            print(f"[WARN] Error procesando {img_path.name}: {e}")

    print("\n=== Resumen ===")
    print(f"Imágenes procesadas: {len(img_paths)}")
    print(f"Imágenes con al menos 1 gato: {total_imgs_with_cat}")
    print(f"Recortes guardados: {total_crops}")
    print(f"Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()