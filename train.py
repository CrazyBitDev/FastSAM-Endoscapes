from ultralytics import YOLO

model = YOLO(model="./weights/FastSAM-s.pt")

model.train(data="./dataset/dataset.yaml",
    epochs=100,
    batch=8,
    overlap_mask=False,
    save=True,
    save_period=1,
    seed=42,
    imgsz=640,
    # imgsz=800,
    project='FastSAM Project',
    name='Endoscapes FastSAM-s',
    val=False,
    plots=True,

    # data augmentation
    flipud=0.5
)