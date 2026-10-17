# This script trains a YOLOv8 model on the annotated chart dataset.

# Content of train_chart_yolo.py:

from ultralytics import YOLO

# Initialize YOLOv8 model (Nano)
model = YOLO('yolov8n.pt')

# Train on your chart dataset
results = model.train(
    data='chart_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='chart_yolo',
    name='xrp_patterns',
    exist_ok=True
)

# Uncomment below to run inference after training
preds = model.predict(
    source='chart_dataset/images_with_boxes',
    conf=0.25,
    save=True,
    project='chart_yolo',
    name='inference_results'
)
# Save the trained model
model.save('chart_yolo/best.pt')    
