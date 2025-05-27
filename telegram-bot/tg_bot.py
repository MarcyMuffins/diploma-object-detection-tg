import logging
from telegram.ext import (Application, ContextTypes)
from telegram import Update, Bot
from telegram.ext import CommandHandler
import os
import io
import urllib.request
import cv2
from ultralytics import YOLO
import numpy as np
from dotenv import load_dotenv
import asyncio
import nest_asyncio

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
THRESHOLD = 0.6
ESP32_IP = "http://192.168.0.104/"

running = False
processing_thread = None
user_chat_id = None

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        f"Welcome to my Diploma Project bot.\n"
        f"The default settings are:\n"
        f"URL: {ESP32_IP}\n"
        f"Threshold: {THRESHOLD}\n"
        f"To change the default settings, use /url and /threshold.\n"
        f"To start detecting, use /launch, to stop, use /stop.\n"
    )

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        f"The settings are:\n"
        f"URL: {ESP32_IP}\n"
        f"Threshold: {THRESHOLD}\n"
        f"To change the default settings, use /url and /threshold.\n"
        f"To start detecting, use /launch, to stop, use /stop.\n"
    )

async def url_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global ESP32_IP
    if context.args:
        ESP32_IP = " ".join(context.args)
        await update.message.reply_text(f"Global variable updated to: {ESP32_IP}")
    else:
        await update.message.reply_text("Usage: /url <text>")

async def threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global THRESHOLD
    if context.args:
        if is_float_try(" ".join(context.args)):
            if float(" ".join(context.args)) > 1.0 or float(" ".join(context.args)) < 0.0:
                await update.message.reply_text("Usage: /threshold <number between 0 and 1>")
            else:
                THRESHOLD = float(" ".join(context.args))
                await update.message.reply_text(f"Global variable updated to: {THRESHOLD}")
        else:
            await update.message.reply_text("Usage: /threshold <number between 0 and 1>")
    else:
        await update.message.reply_text("Usage: /threshold <number between 0 and 1>")

async def object_detection(bot: Bot):
    global running, user_chat_id
    model = YOLO("runs_100s/detect/train/weights/best.pt")
    buffer = b""
    while running:
        try:
            #print("a", end="")
            buffer = await asyncio.to_thread(lambda: urllib.request.urlopen(ESP32_IP).read(45000))
            a = buffer.find(b'\xff\xd8')  # JPEG start
            b = buffer.find(b'\xff\xd9')  # JPEG end
            #print(len(buffer))
            if a != -1 and b != -1:
                #print("b", end="")
                #e = buffer.rfind(b'\xff\xd9')
                jpg = buffer[a:b + 2]
                #buffer = b""
                if len(jpg) != 0:
                    #print("c", end="")
                    # Convert bytes to image
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # print(frame)

                    # Transform the frame
                    #frame = cv2.resize(frame, (416, 416))

                    # Classify the frame
                    results = model(frame, conf=THRESHOLD, verbose=False)[0]
                    if not results or len(results) == 0:
                        continue
                    result = results[0]
                    detection_count = result.boxes.shape[0]
                    #for i in range(detection_count):
                        #print("d", end="\n")
                    #    cls = int(result.boxes.cls[i].item())
                    #    name = result.names[cls]
                    #    confidence = float(result.boxes.conf[i].item())
                    #    bounding_box = result.boxes.xyxy[i].cpu().numpy()

                    #    x = int(bounding_box[0])
                    #    y = int(bounding_box[1])
                    #    width = int(bounding_box[2] - x)
                    #    height = int(bounding_box[3] - y)
                    #    print(cls)
                    #    print(name)
                    #    print(confidence)
                    #    print(x, y, width, height)
                    #    print("=======")
                    if detection_count != 0 and user_chat_id:
                        #print("e", end="\n")
                        annotated_frame = results.plot()
                        _, img_buffer = cv2.imencode('.jpg', annotated_frame)  # Encode image to memory
                        img_bytes = io.BytesIO(img_buffer)  # Convert to BytesIO object
                        img_bytes.seek(0)  # Move cursor to the start of the file
                        await bot.send_photo(chat_id=user_chat_id, photo=img_bytes)
                        await asyncio.sleep(2)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in object detection: {e}")

async def launch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global running, processing_thread, user_chat_id
    if not running:
        running = True
        user_chat_id = update.message.chat_id  # Store user chat ID
        bot = context.application.bot  # Get bot instance
        processing_thread = asyncio.create_task(object_detection(bot))
        await update.message.reply_text("Detection started! Sending images when an object is detected.")
    else:
        await update.message.reply_text("Already running!")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global running
    if running:
        running = False
        await update.message.reply_text("Stopping detection...")
    else:
        await update.message.reply_text("Detection is not running.")


async def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("url", url_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("launch", launch_command))
    application.add_handler(CommandHandler("stop", stop_command))

    # Run the bot until the user presses Ctrl-C
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":

    nest_asyncio.apply()

    asyncio.run(main())