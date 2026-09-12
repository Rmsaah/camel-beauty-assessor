pip install ultralytics

from ultralytics import YOLO

model = YOLO("model.pt")

# image can be a file path, a PIL.Image, or a numpy array - ultralytics takes any of them,
# so a FastAPI UploadFile works as Image.open(file.file)
def read_camel(image):
    r = model.predict(image, conf=0.01, imgsz=640, verbose=False)[0]

    top = {}
    for cls, conf, box in zip(r.boxes.cls.int().tolist(),
                              r.boxes.conf.tolist(),
                              r.boxes.xyxyn.tolist()):
        name = model.names[cls]
        if conf > top.get(name, (0.0, None))[0]:
            top[name] = (round(conf, 4), [round(v, 4) for v in box])

    traits = [c for c in model.names.values() if c != "Camel"]
    return {
        "camel": {"confidence": top.get("Camel", (0.0, None))[0]},
        "traits": {t: {"confidence": top.get(t, (0.0, None))[0],
                        "box_xyxyn": top.get(t, (0.0, None))[1]} for t in traits},
    }
