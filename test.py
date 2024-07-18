from ultralytics import YOLO
model = YOLO(model="./FastSAM Project/Endoscapes FastSAM-s/weights/epoch87.pt")
results = model.val(data="./dataset/dataset.yaml", \
    batch=8, \
    imgsz=640, \
    split="test", \
    project='FastSAM Project', \
    name='Endoscapes FastSAM-s Test', \
    val=False,
    save_json=True, \
    conf=0.001, \
    iou=0.9, \
    max_det=100, \
)

results.confusion_matrix.plot(names=("background", "cystic\nplate", "calot\ntriangle", "cystic\nartery", "cystic\nduct", "gallbladder", "tool"), normalize=False)

print(results)