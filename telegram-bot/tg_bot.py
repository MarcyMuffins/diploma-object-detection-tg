import asyncio
import io
import logging
import os
import urllib.request

import cv2
import nest_asyncio
import numpy as np
import validators
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import (Application, ContextTypes)
from telegram.ext import CommandHandler
from ultralytics import YOLO

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
THRESHOLD = 0.6
ESP32_IP = "http://192.168.0.104/"

running = False
processing_thread = None
user_chat_id = None
model_folder = ""
models = []
chosen_model = None

# Info level logging for the terminal
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Setting a higher logging level for httpx to avoid all GET and POST requests being logged
# Remove .setLevel() for a higher level logging level
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Helper func to test if a string is a float in a try-catch
# There's probably a better way of doing this. Too bad!
def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

# Send a message when the command /start is issued.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Welcome to my Diploma Project bot.\n"
        f"The default settings are:\n"
        f"URL: {ESP32_IP}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Selected Model: {chosen_model}\n"
        f"Available Models:\n"
        f"{'\n'.join(models)}\n"
        f"To change the default settings, use /url, /threshold and /model.\n"
        f"To start detecting, use /launch, to stop, use /stop.\n"
    )

# Send a message when the command /info is issued.
# Basically /start without the welcome message.
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"The settings are:\n"
        f"URL: {ESP32_IP}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Selected Model: {chosen_model}\n"
        f"Available Models:\n"
        f"{'\n'.join(models)}\n"
        f"To change the default settings, use /url, /threshold and /model.\n"
        f"To start detecting, use /launch, to stop, use /stop.\n"
    )

# Command for updating the web server url
async def url_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global ESP32_IP
    if context.args:
        # Using " ".join(context.args) to compress all the arguments into one string, it'll only work if the argument is appropriate
        if validators.url(" ".join(context.args)):
            ESP32_IP = " ".join(context.args)
            await update.message.reply_text(f"Global variable updated to: {ESP32_IP}")
        else:
            await update.message.reply_text("Usage: /url <text>")
    else:
        await update.message.reply_text("Usage: /url <text>")

# Command for updating the detection threshold, same logic as the url command
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

# Command for updating the detection threshold, same logic as the url command
async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global models, chosen_model
    if context.args:
        if models.count(" ".join(context.args)) != 0:
            chosen_model = " ".join(context.args)
            await update.message.reply_text(f"Model updated to: {chosen_model}")
        else:
            await update.message.reply_text("Usage: /model <model name>")
    else:
        await update.message.reply_text("Usage: /model <model name>")

# Launches the object detection logic
async def object_detection(bot: Bot):
    global running, user_chat_id, model_folder, chosen_model
    model = YOLO(model_folder + chosen_model)
    buffer = b""
    while running:
        try:
            # Fetching data from the URL in a separate thread to not interrupt the bot logic
            buffer = await asyncio.to_thread(lambda: urllib.request.urlopen(ESP32_IP).read(45000))

            a = buffer.find(b'\xff\xd8')  # JPEG start
            b = buffer.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1 and b > a:
                # Cutting out the jpg information from the buffer
                jpg = buffer[a:b + 2]
                if len(jpg) != 0:
                    # Converting the fetched bytes to a cv2 image format
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # Transforming the frame
                    # Not really necessary with YOLOv11, but may improve performance?
                    #frame = cv2.resize(frame, (416, 416))

                    # Classify the frame
                    results = model(frame, conf=THRESHOLD, verbose=False)[0]
                    if not results or len(results) == 0:
                        continue
                    result = results[0]
                    detection_count = result.boxes.shape[0]

                    if detection_count != 0 and user_chat_id:
                        annotated_frame = results.plot()
                        _, img_buffer = cv2.imencode('.jpg', annotated_frame)  # Encode image to memory
                        img_bytes = io.BytesIO(img_buffer)  # Convert to BytesIO object
                        img_bytes.seek(0)  # Move cursor to the start of the file
                        await bot.send_photo(chat_id=user_chat_id, photo=img_bytes)

                        # Wait a bit to not overwhelm the Telegram API
                        await asyncio.sleep(2)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in object detection: {e}")

# Starts object detection in a separate coroutine so the bot can accept commands at the same time
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

# Stopping the detection by changing the global running variable
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global running
    if running:
        running = False
        await update.message.reply_text("Stopping detection...")
    else:
        await update.message.reply_text("Detection is not running.")

async def main() -> None:
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Adding handlers for all the commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("url", url_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("launch", launch_command))
    application.add_handler(CommandHandler("stop", stop_command))

    # Run the bot until the user presses Ctrl-C
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    print("=== Telegram Bot Interface ===")

    model_folder = input("Enter path to the folder containing .pt model files: ").strip()
    while not os.path.isdir(model_folder):
        print(f"Folder '{model_folder}' not found or is not a folder. Please try again.")
        model_folder = input("Enter path to the folder containing .pt model files: ").strip()

    for filename in os.listdir(model_folder):
        if filename.endswith('.pt'):
            models.append(filename)

    if len(models) == 0:
        print("No models detected, closing program.")
        exit(0)

    chosen_model = models[0]

    nest_asyncio.apply()
    asyncio.run(main())
