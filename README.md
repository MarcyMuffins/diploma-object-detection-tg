# Diploma project by Oleksii Verkhola

## An object detection system using YOLOv11 running on a telegram bot

The project consists of three parts:
1. An ESP32-CAM web-server broadcasting an MJPEG stream as an output from the camera.
2. An interactive python script that trains a YOLOv11 model on the selected RoboFlow dataset and outputs a .pt trained model file.
3. A fully functional Telegram bot that notifies the user of detected objects, allowing the user to select the model used.
---
## Prerequisites
1. Install Python 3.13
2. Install the Arduino IDE
3. In the Arduino IDE, install the `esp32 by Espressif Systems` addon by going to Tools > Board > Boards Manager and searching for `ESP32` in the opened menu
4. In the root folder, run the following commands:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
5. Continue onto individual module instructions
---
## Web-server guide:

1. Connect your ESP32-CAM to the programmer board and plug it into your PC
2. Open the sketch in the Arduino IDE
3. Place your network SSID and Password in the following fields in the code:
```
const char* ssid = "";
const char* password = "";
```
4. Select Board "AI Thinker ESP32-CAM"
5. Press the ESP32-CAM on-board RESET button to put your board in flashing mode
6. Upload the sketch to the board
7. Follow the link in the console readout OR find the ESP32 server in your router's connected devices list
---
## Model trainer guide:

1. Go to (Roboflow Universe)[https://universe.roboflow.com/] and download your desired dataset in the YOLOv11 format and unzip it in a convenient location. There should be a data.yaml file inside the uzipped folder. 
2. Inside the model-training folder, run the following command:
```
python ./train.py
```
3. Follow the instructions in the terminal
4. Copy the best.pt file from the folder outputted in the terminal to a convenient location. This is your model file. You should probably rename it if you plan on using several models in the Telegram bot.
---
## Telegram bot guide:
1. Naviagate to the telegram-bot folder
2. Create a file named `.env` and open it in a text editor
3. Put the following inside:
```
TELEGRAM_BOT_TOKEN=[PASTE TOKEN HERE]
```
and replace `[PASTE TOKEN HERE]` with your telegram bot token
4. Run the command:
```
python tg_bot.py
```
5. Follow the instructions in the terminal
6. Once the bot is running, open it in Telegram to use it
---
## TODO:
* The TODO

## CREDITS:
ESP32-CAM server based on a project by Rui Santos and Espressif Systems
