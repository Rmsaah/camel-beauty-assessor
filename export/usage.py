# pip install ultralytics

from PIL import Image, ImageOps
from ultralytics import YOLO

MODELS = {
    "gray": {"file": "model_gray.pt", "imgsz": 640, "conf": 0.01},
    "rgb": {"file": "model_rgb.pt", "imgsz": 640, "conf": 0.01},
}

_loaded = {}


def _model(which):
    # load on first use, so a caller that wants one model does not pay for both
    if which not in _loaded:
        _loaded[which] = YOLO(MODELS[which]["file"])
    return _loaded[which]


def read_camel(image, which="gray"):
    cfg, model = MODELS[which], _model(which)

    img = Image.open(image) if isinstance(image, str) else image
    img = ImageOps.exif_transpose(img)      # phone photos store rotation here
    img = img.convert("L").convert("RGB") if which == "gray" else img.convert("RGB")

    r = model.predict(img, conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False)[0]

    top = {}
    for cls, conf, box in zip(r.boxes.cls.int().tolist(),
                              r.boxes.conf.tolist(),
                              r.boxes.xyxyn.tolist()):
        name = model.names[cls]
        if conf > top.get(name, (0.0, None))[0]:
            top[name] = (round(conf, 4), [round(v, 4) for v in box])

    traits = [c for c in model.names.values() if c != "Camel"]
    return {
        "model": cfg["file"],
        "camel": {"confidence": top.get("Camel", (0.0, None))[0]},
        "traits": {t: {"confidence": top.get(t, (0.0, None))[0],
                       "box_xyxyn": top.get(t, (0.0, None))[1]} for t in traits},
    }
