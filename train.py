import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    model.load('yolov8m.pt') # loading pretrain weights
    # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp112/weights/best.pt') # loading pretrain weights
    # model = RTDETR('ultralytics/cfg/models/v8/yolov8m-swintransformer.yaml')

    model.train(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/sim10k_to_cityscapes.yaml',             
                cache=False,
                imgsz=640,
                epochs=2,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                resume='', # runs/train/exp/weights/last.pt   断点续训！！！！ 
                amp=False, # close amp
                # fraction=0.2,
                patience=100,
                cos_lr = True,
                project='runs/debug',
                name='exp',
                
                # conf = 0.02 , ## focal-loss 
                # cls = 1.5 
                )