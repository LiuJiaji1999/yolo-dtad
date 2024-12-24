import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp76/weights/best.pt')
    model.val(data='/home/lenovo/data/liujiaji/yolov8/powerdata.yaml',
              split='test',
              imgsz=640, # 默认640
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp76',
              )
