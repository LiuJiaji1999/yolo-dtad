import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/powerGit/yolov8/runs/8.1.9/exp119/weights/best.pt')
    model.val(data='/home/lenovo/data/liujiaji/YOLO-DTAD/dataset/VisDrone.yaml',
              split='test',
              imgsz=640, # 默认640
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
# sim10k_to_cityscapes.yaml sourcesim10k