import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/home/lenovo/data/liujiaji/powerGit/yolov8/runs/8.1.9/exp119/weights/best.pt')
    model = YOLO('/home/lenovo/data/liujiaji/YOLO-DTAD/runs/debug/mvexp4/weights/best.pt')
    model.val(data='/home/lenovo/data/liujiaji/ultralytics-yolo11-main/dataset/VisDrone.yaml',
              split='test',
              imgsz=800, # 默认640
              batch=2,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
# sim10k_to_cityscapes.yaml sourcesim10k