from ultralytics import YOLO
# model = YOLO('yolov12n.pt')
# model.predict("D:\pytorch.pycharm\yolov12-main\\ultralytics\\assets\\bus.jpg",save=True)
model = YOLO('models/yolov12n.pt')

# Train the model
results = model.train(
  data='YOLO12.yaml',
  epochs=30,
  batch=16,
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device= 'gpu',
  workers= 0,
  weight_decay=0.0005
)

# Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()
# yolo = YOLO("runs/detect/train/weights/last.pt",task="det")
# results = yolo(source="D:\pytorch.pycharm\Resources\\biaoyan.mp4",show=True,conf=0.2,iou=0.8)

