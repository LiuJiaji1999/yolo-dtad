import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/YOLO-DTAD/runs/train/exp/weights/best.pt')
    model.val(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/sim10k_to_cityscapes.yaml',
              split='val',
              imgsz=800, # 默认640
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='sourcesim10k',
              )
# sim10k_to_cityscapes.yaml sourcesim10k