You can download pytorch based weight files [HERE](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt)

you can convert to TensorRT file with command

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n-seg.pt")

# Export the model to ONNX format
model.export(format="engine")
```

