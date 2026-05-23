# Raspberry Pi Edge Application

Run on Pi:
  pip install -r requirements.txt --break-system-packages
  python application.py

Dashboard: http://<pi-ip>:5000
Calibrate: http://<pi-ip>:5000/calibrate

Model: best.pt (fine-tuned YOLOv8n, 99.4% mAP50)
       Falls back to yolov8n.pt if best.pt not found
