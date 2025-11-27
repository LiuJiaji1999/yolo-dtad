import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m-DTADH.yaml')
    model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    # model = YOLO('ultralytics/cfg/models/yolov8n-grl.yaml')
    model.load('yolov8m.pt') # loading pretrain weights
    # model.load('/home/lenovo/data/liujiaji/YOLO-DTAD/runs/train/exp/weights/best.pt') # 合成域 
    # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp112/weights/best.pt') # loading pretrain weights
    # model = RTDETR('ultralytics/cfg/models/v8/yolov8m-swintransformer.yaml')

    model.train(data='/home/lenovo/data/liujiaji/ultralytics-yolo11-main/dataset/powerdata.yaml', # powerdata  publicallpower VisDrone     
                cache=False,
                imgsz=640,
                epochs=50,
                batch=2,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                resume='', # runs/train/exp/weights/last.pt   断点续训！！！！ 
                # amp=False, # close amp
                # fraction=0.2,
                patience=0,
                # cos_lr = True,
                project='runs/debug',
                name='mvexp',
                # conf = 0.02 , ## focal-loss 
                # cls = 1.5 
                )
