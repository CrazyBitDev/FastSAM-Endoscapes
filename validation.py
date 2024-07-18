from ultralytics import YOLO
import os
import pickle

dir_name = 'Endoscapes FastSAM-s'

file_names = {}

path = f'./FastSAM Project/{dir_name}/weights'
files = os.listdir(path)
# exclude best.pt
files = [f for f in files if f not in ['best.pt'] and f.endswith('.pt')]
file_names = sorted(files)

results = {}

# for each weight file, run validation
for file_name in file_names:
    path = f'./FastSAM Project/{dir_name}/weights/{file_name}'
    file_name = file_name if file_name != "last.pt" else "epoch99.pt"
    model = YOLO(model=path)
    result = model.val(data="./dataset/dataset.yaml", \
        epochs=20, \
        batch=8, \
        imgsz=640, \
        #device='0',\
        project='FastSAM Project Eval', \
        name=f'{dir_name} {file_name}', \
        val=False,
        save_json=True, \
        conf=0.001, \
        iou=0.9, \
        max_det=100, \
    )

    # save results
    results[file_name] = {
        "results_dict": result.results_dict,
        "names": result.names,
        "confusion_matrix": result.confusion_matrix,
        "box": result.box,
        "seg": result.seg,
        "maps": result.maps,
    }

# save results to pickle file
with open('validation.pkl', 'wb') as outp:
    pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)