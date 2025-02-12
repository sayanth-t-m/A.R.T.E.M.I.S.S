import logging
import sys
import os
import sqlite3
import time
from pathlib import Path
from typing import Union, Tuple

import cv2
import torch
from PIL import Image
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from transformers import pipeline, AutoModelForImageClassification, ViTImageProcessor

# -------------------------------
# Configuration and Initialization
# -------------------------------

# Replace with your actual admin Telegram user IDs.
ADMIN_IDS = {1272767655}  # Example: {123456789, 987654321}

# Bot token (hardcoded for now; consider using environment variables for production)
BOT_TOKEN = "7550587852:AAGImN1la_a592TzoKV5d5js6rlUPp1ozjM"

# SQLite database file for persisting violation counts.
DB_FILE = "violations.db"

# Directory for caching flagged images and videos.
FLAGGED_IMAGES_DIR = "flagged_images"
FLAGGED_VIDEOS_DIR = "flagged_videos"
if not os.path.exists(FLAGGED_IMAGES_DIR):
    os.makedirs(FLAGGED_IMAGES_DIR)
if not os.path.exists(FLAGGED_VIDEOS_DIR):
    os.makedirs(FLAGGED_VIDEOS_DIR)

# Violation threshold before banning a user.
FLAG_THRESHOLD = 3

# Set up logging.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Initialize the NSFW detection pipeline for image analysis using the Falcon AI model.
try:
    nsfw_detector = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=0,
        use_fast=True
    )
except Exception as e:
    logger.error("Error loading NSFW detection model (image pipeline): " + str(e))
    raise e

# -------------------------------
# Video Analysis Class using Falcon AI
# -------------------------------
class VideoContentAnalyzer:
    def __init__(self, model_name: str = "Falconsai/nsfw_image_detection"):
        """
        Initialize the content analyzer with the NSFW detection model.
        This class uses the same Falcon AI model for video frame analysis.
        """
        # Select device: use GPU if available, else CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the pre-trained image classification model and move it to the selected device.
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        # Load the corresponding image processor.
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Set up logging configuration.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("VideoContentAnalyzer initialized using model: %s", model_name)

    def analyze_frame(self, image: Image.Image) -> Tuple[str, float]:
        """
        Analyze a single frame for NSFW content.
        
        Parameters:
            image (PIL.Image.Image): The image to analyze.
            
        Returns:
            Tuple[str, float]: A tuple containing the prediction label and confidence score.
        """
        with torch.no_grad():
            # Process the image to the required tensor format.
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # Perform inference using the model.
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities.
            probs = torch.softmax(logits, dim=-1)
            # Find the index of the highest scoring prediction.
            pred_idx = logits.argmax(-1).item()
            confidence = probs[0][pred_idx].item()
            
            # Retrieve the human-readable label from the model's configuration.
            return self.model.config.id2label[pred_idx], confidence

    def analyze_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 6,
        stop_on_nsfw: bool = True,
        nsfw_threshold: float = 0.5
    ) -> dict:
        """
        Analyze frames from a video for NSFW content.
        
        Parameters:
            video_path (Union[str, Path]): Path to the video file.
            num_frames (int): Number of frames to analyze.
            stop_on_nsfw (bool): Whether to stop analysis when NSFW content is detected.
            nsfw_threshold (float): Confidence threshold for NSFW classification.
            
        Returns:
            dict: Analysis results including frame classifications and overall verdict.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting analysis of video: {video_path}")
        
        # Open the video file.
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total_frames // num_frames)
            
            results = {
                'frames_analyzed': 0,
                'nsfw_detected': False,
                'frame_results': [],
                'first_nsfw_frame': None
            }

            # Loop through and analyze selected frames.
            for i in range(num_frames):
                frame_position = min(i * interval, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame at position {frame_position}")
                    continue

                # Convert the frame from BGR (OpenCV default) to RGB.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Create a PIL Image from the frame.
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze the frame using the same Falcon AI NSFW detection model.
                label, confidence = self.analyze_frame(pil_image)
                
                frame_result = {
                    'frame_number': i + 1,
                    'position': frame_position,
                    'prediction': label,
                    'confidence': confidence
                }
                results['frame_results'].append(frame_result)
                results['frames_analyzed'] += 1

                self.logger.info(
                    f"Frame {i+1}/{num_frames} Analysis: "
                    f"Classification: {label}, Confidence: {confidence:.2%}"
                )

                # If NSFW content is detected above the threshold, record and optionally stop.
                if label.lower() == "nsfw" and confidence >= nsfw_threshold:
                    results['nsfw_detected'] = True
                    results['first_nsfw_frame'] = frame_result
                    if stop_on_nsfw:
                        self.logger.warning(
                            f"NSFW content detected in frame {i+1} "
                            f"with {confidence:.2%} confidence. Stopping analysis."
                        )
                        break

            return results

        finally:
            cap.release()

# -------------------------------
# Database Functions
# -------------------------------

def init_db():
    """Initializes the SQLite database and creates the violations table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            user_id INTEGER PRIMARY KEY,
            count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def get_violation_count(user_id: int) -> int:
    """Retrieves the violation count for a given user from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count FROM violations WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def add_violation(user_id: int) -> int:
    """Increments the violation count for a user and returns the updated count."""
    count = get_violation_count(user_id)
    count += 1
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO violations (user_id, count) VALUES (?, ?)", (user_id, count))
    conn.commit()
    conn.close()
    return count

def reset_violation(user_id: int):
    """Resets the violation count for a user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM violations WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_all_violations():
    """Returns all users with a violation count greater than zero."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT user_id, count FROM violations WHERE count > 0")
    rows = c.fetchall()
    conn.close()
    return rows

# -------------------------------
# Bot Command Handlers
# -------------------------------

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    if update and update.message:
        await update.message.reply_text("An error occurred while processing your request. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logger.info(f"User {user.first_name} (ID: {user.id}) started the bot.")
    await update.message.reply_text(
        "🤖 Welcome to A.R.T.E.M.I.S.S.!\n\n"
        "Send me an image or video to check for NSFW content.\n"
        "Use /violations to check your NSFW violation count.\n"
        "Use /help for more commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Use HTML formatting to avoid issues with unescaped special characters.
    help_text = (
        "🤖 <b>A.R.T.E.M.I.S.S. Help</b>\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/violations - Check your NSFW violation count\n"
        "/admin_flagged - View all flagged users (admins only)\n"
        "/admin_reset &lt;user_id&gt; - Reset violation count for a user (admins only)\n"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")

async def violations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    count = get_violation_count(user.id)
    await update.message.reply_text(f"⚠️ {user.first_name}, you have {count} NSFW violation(s).")

# Admin command: view all flagged users.
async def admin_flagged(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    if user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ You are not authorized to use this command.")
        return

    flagged = get_all_violations()
    if not flagged:
        await update.message.reply_text("No flagged users.")
    else:
        response = "Flagged Users:\n"
        for uid, count in flagged:
            response += f"User ID: {uid}, Violations: {count}\n"
        await update.message.reply_text(response)

# Admin command: reset a user's violation count.
async def admin_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    if user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ You are not authorized to use this command.")
        return

    if len(context.args) != 1:
        await update.message.reply_text("Usage: /admin_reset <user_id>")
        return

    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID. It should be a number.")
        return

    reset_violation(target_user_id)
    await update.message.reply_text(f"Reset violation count for user ID: {target_user_id}")

# -------------------------------
# Message Handlers
# -------------------------------

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    user_id = user.id
    chat_id = update.message.chat_id

    logger.info(f"Received video from {user.first_name} (ID: {user_id})")
    
    try:
        # Show a "processing" action.
        await context.bot.send_chat_action(chat_id, action=ChatAction.UPLOAD_VIDEO)

        # Retrieve the video file from the message.
        video = update.message.video
        file = await video.get_file()

        # Download the video to a temporary file.
        temp_path = f"temp_{user_id}.mp4"
        await file.download_to_drive(custom_path=temp_path)
        logger.info(f"Video downloaded to {temp_path}")

        # Initialize the video analyzer and analyze the video.
        analyzer = VideoContentAnalyzer()
        results = analyzer.analyze_video(
            video_path=temp_path,
            num_frames=6,
            stop_on_nsfw=True,
            nsfw_threshold=0.5
        )

        # Check if any frame was flagged as NSFW.
        if results.get("nsfw_detected", False):
            violation_count = add_violation(user_id)
            # Delete the original video message.
            await update.message.delete()
            logger.warning(f"🚨 NSFW detected in video from {user.first_name}, Violation: {violation_count}")
            await context.bot.send_message(
                chat_id,
                f"⚠️ NSFW content detected in your video! Violation {violation_count}/{FLAG_THRESHOLD}."
            )

            # Cache the flagged video for review.
            timestamp = int(time.time())
            cached_filename = os.path.join(FLAGGED_VIDEOS_DIR, f"user_{user_id}_{timestamp}.mp4")
            os.rename(temp_path, cached_filename)
            logger.info(f"Flagged video saved as {cached_filename}")

            # If violation count exceeds the threshold, attempt to ban the user (if in a group chat).
            if violation_count >= FLAG_THRESHOLD:
                if update.message.chat.type in ["group", "supergroup"]:
                    await context.bot.ban_chat_member(chat_id, user_id)
                    logger.warning(f"🚫 Banned {user.first_name} for repeated NSFW violations.")
                    await context.bot.send_message(
                        chat_id,
                        f"🚫 {user.first_name} has been banned for multiple NSFW violations."
                    )
                    reset_violation(user_id)
                else:
                    logger.warning("🚫 Violation threshold exceeded but cannot ban in private chats.")
                    await context.bot.send_message(
                        chat_id,
                        "🚫 You have exceeded the NSFW violation threshold, but banning is not supported in private chats."
                    )
        else:
            # If the video is safe, remove the temporary file.
            os.remove(temp_path)
            logger.info("Video passed NSFW checks; temporary file removed.")

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        await update.message.reply_text("⚠️ Error processing your video. Please try again.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    user_id = user.id
    chat_id = update.message.chat_id

    logger.info(f"Received image from {user.first_name} (ID: {user_id})")
    
    try:
        # Show a "processing" action.
        await context.bot.send_chat_action(chat_id, action=ChatAction.UPLOAD_PHOTO)

        # Retrieve the highest resolution image from the message.
        photo = update.message.photo[-1]
        file = await photo.get_file()

        # Download the image to a temporary file.
        temp_path = f"temp_{user_id}.jpg"
        await file.download_to_drive(custom_path=temp_path)

        # Analyze the image using the NSFW detection pipeline.
        result = nsfw_detector(temp_path)
        label = result[0]["label"]
        score = result[0]["score"]

        logger.info(f"Image analysis - User: {user.first_name}, Label: {label}, Confidence: {score:.2f}")

        if label.lower() == "nsfw":
            violation_count = add_violation(user_id)
            await update.message.delete()
            logger.warning(f"🚨 NSFW detected in image from {user.first_name}, Violation: {violation_count}")
            await context.bot.send_message(
                chat_id,
                f"⚠️ NSFW content detected in your image! Violation {violation_count}/{FLAG_THRESHOLD}."
            )

            # Cache the flagged image for review.
            timestamp = int(time.time())
            cached_filename = os.path.join(FLAGGED_IMAGES_DIR, f"user_{user_id}_{timestamp}.jpg")
            os.rename(temp_path, cached_filename)

            # If violation count exceeds the threshold, attempt to ban the user (if in a group chat).
            if violation_count >= FLAG_THRESHOLD:
                if update.message.chat.type in ["group", "supergroup"]:
                    await context.bot.ban_chat_member(chat_id, user_id)
                    logger.warning(f"🚫 Banned {user.first_name} for repeated NSFW violations.")
                    await context.bot.send_message(
                        chat_id,
                        f"🚫 {user.first_name} has been banned for multiple NSFW violations."
                    )
                    reset_violation(user_id)
                else:
                    logger.warning("🚫 Violation threshold exceeded but cannot ban in private chats.")
                    await context.bot.send_message(
                        chat_id,
                        "🚫 You have exceeded the NSFW violation threshold, but banning is not supported in private chats."
                    )
        else:
            # If the image is safe, remove the temporary file.
            os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await update.message.reply_text("⚠️ Error processing your image. Please try again.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Do nothing for text messages.
    pass

# -------------------------------
# Main Function
# -------------------------------

def main() -> None:
    # Initialize the database.
    init_db()

    # Create the Application and pass it your bot's token.
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register command handlers.
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("violations", violations))
    application.add_handler(CommandHandler("admin_flagged", admin_flagged))
    application.add_handler(CommandHandler("admin_reset", admin_reset))
    
    # Register message handlers.
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Register the error handler.
    application.add_error_handler(error_handler)

    logger.info("🚀 Bot is starting...")
    # Run the bot until the user presses Ctrl-C.
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
