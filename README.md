# Diploma project by Oleksii Verkhola

## An object detection system using YOLOv11 running on a telegram bot

The project consists of three parts:
1. An ESP32-CAM web-server broadcasting an MJPEG stream as an output from the camera.
2. An interactive python script that trains a YOLOv11 model on the selected RoboFlow dataset and outputs a .pt trained model file.
3. A fully functional Telegram bot that notifies the user of detected objects, allowing the user to select the model used.
---
## Web-server guide:

1. Connect your ESP32-CAM to the programmer board and plug it into your PC
2. Open the sketch in the Arduino IDE
3. Select Board "AI Thinker ESP32-CAM"
4. Press the ESP32-CAM on-board RESET button to put your board in flashing mode
5. Upload the sketch to the board
6. Follow the link in the console readout OR find the ESP32 server in your router's connected devices list
---
## Model trainer guide:

1. Install Python 3.13
2. Inside the model-training folder, run the following commands:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ./main.py
```
3. Follow the instructions in the terminal
4. Copy the best.pt file from the folder outputted in the terminal to a convenient location, this is your model file, you should probably rename it if you plan on using several models in the Telegram bot.
---
## TODO:
* The TODO

## CREDITS:
ESP32-CAM server based on a project by [Rui Santos](https://RandomNerdTutorials.com/esp32-cam-video-streaming-web-server-camera-home-assistant/)
