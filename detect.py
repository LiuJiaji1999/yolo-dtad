import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/YOLO-DTAD/runs/debug/mvexp3/weights/best.pt') # select your model.pt path
    # model.model.names = {0: 'pin-uninstall', 1: 'pin-rust', 2: 'pin-defect',
                        #  3: 'insulator-burn', 4: 'insulator-defect', 5: 'insulator-dirty'}  # 修改类别名称
    # print("模型修改后的类别名称:", model.model.names)
    model.predict(
                  # source='/home/lenovo/data/liujiaji/Datasets-Power/privatepower/pin/rust/img',
                  source='/home/lenovo/data/liujiaji/powerGit/yolov8/testImg',
                  imgsz=640,
                  project='runs/detect',
                  name='v8mv-prp',
                  save=True,
                  # save_txt=True,
                  # iou=0.45 # 默认
                  conf=0.3, # 框上的值，目标检测的对象置信度阈值。只有高于此阈值的对象才会被检测出来。默认值为0.25
                  line_width=5
                  # visualize=True # visualize model features maps
                )
    ##  可以对结果进行后处理！！！
#     for i in model.predict(source='dataset/images/test',
#                   imgsz=640,
#                   project='runs/detect',
#                   name='exp',
#                   save=True,):
#             print(i)
'''              
boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: None
names: {0: 'pin-defect', 1: 'pin-rust', 2: 'pin-uninstall', 3: 'Einsu-burn', 4: 'Einsu-defect', 5: 'Einsu-dirty'}
obb: None
orig_img: array([[[129, 114, 112],
        [126, 111, 109],
        [125, 109, 110],
        ...,
        [245, 225, 224],
        [244, 223, 221],
        [236, 215, 213]],
  ...,
        [ 75,  69,  62],
        [ 87,  80,  77],
        [ 85,  75,  75]]], dtype=uint8)
orig_shape: (864, 1152)
path: '/home/lenovo/data/liujiaji/yolov8/ultralytics-main/dataset/images/test/3424.jpg'
probs: None
save_dir: 'runs/detect/exp3'
speed: {'preprocess': 3.0732154846191406, 'inference': 12.158632278442383, 'postprocess': 4.642248153686523}
ultralytics.engine.results.Results object with attributes:   
'''