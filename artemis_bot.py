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
import telegram.error  # Add this import

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("VideoContentAnalyzer initialized using model: %s", model_name)

    def analyze_frame(self, image: Image.Image) -> Tuple[str, float]:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = logits.argmax(-1).item()
            confidence = probs[0][pred_idx].item()
            return self.model.config.id2label[pred_idx], confidence

    def analyze_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 6,
        stop_on_nsfw: bool = True,
        nsfw_threshold: float = 0.5
    ) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Starting analysis of video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results = {
                'frames_analyzed': 0,
                'nsfw_detected': False,
                'frame_results': [],
                'first_nsfw_frame': None
            }

            # Calculate interval for the frames
            interval = max(1, total_frames // num_frames)

            for i in range(num_frames):
                frame_position = min(i * interval, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame at position {frame_position}")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                lbl, conf = self.analyze_frame(pil_image)
                frame_result = {
                    'frame_number': i + 1,
                    'position': frame_position,
                    'prediction': lbl,
                    'confidence': conf
                }
                results['frame_results'].append(frame_result)
                results['frames_analyzed'] += 1
                self.logger.info(
                    f"Frame {i+1}/{num_frames} Analysis: Classification: {lbl}, Confidence: {conf:.2%}"
                )
                if lbl.lower() == "nsfw" and conf >= nsfw_threshold:
                    results['nsfw_detected'] = True
                    results['first_nsfw_frame'] = frame_result
                    self.logger.warning(
                        f"NSFW content detected in frame {i+1} with {conf:.2%} confidence. Stopping analysis."
                    )
                    break
            return results

        finally:
            cap.release()

# -------------------------------
# Database Functions
# -------------------------------

def init_db():
    """Initializes the SQLite database and creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            user_id INTEGER PRIMARY KEY,
            count INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_action(user_id: int, action: str):
    """Logs an action with a timestamp in the database."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO actions (user_id, action, timestamp) VALUES (?, ?, ?)", (user_id, action, timestamp))
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
    increment_stat("total_violations")
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

def update_dashboard(user_id: int, violation_count: int, media_type: str):
    """Updates the dashboard with the latest violation information."""
    # Placeholder for dashboard update logic
    logger.info(f"Dashboard updated: User ID {user_id}, Violations {violation_count}, Media Type {media_type}")

def increment_stat(key: str):
    """Increments a statistic in the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO stats (key, value) VALUES (?, 0)", (key,))
    c.execute("UPDATE stats SET value = value + 1 WHERE key = ?", (key,))
    conn.commit()
    conn.close()

def get_stat(key: str) -> int:
    """Retrieves a statistic from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM stats WHERE key = ?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def get_all_stats() -> dict:
    """Retrieves all statistics from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value FROM stats")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

# -------------------------------
# Bot Command Handlers
# -------------------------------

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    if update and update.message:
        try:
            await update.message.reply_text("An error occurred while processing your request. Please try again later.")
        except telegram.error.BadRequest as e:
            if "Message to be replied not found" in str(e):
                logger.error("Failed to send error message: Message to be replied not found.")
            else:
                raise e

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
    help_text = (
        "🤖 <b>A.R.T.E.M.I.S.S. Help</b>\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/violations - Check your NSFW violation count\n"
        "/admin_flagged - View all flagged users (admins only)\n"
        "/admin_reset &lt;user_id&gt; - Reset violation count for a user (admins only)\n"
        "/admin_ban &lt;user_id&gt; - Ban a user (admins only)\n"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")

async def violations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    count = get_violation_count(user.id)
    await update.message.reply_text(f"⚠️ {user.first_name}, you have {count} NSFW violation(s).")

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

async def admin_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    if user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ You are not authorized to use this command.")
        return

    if len(context.args) != 1:
        await update.message.reply_text("Usage: /admin_ban <user_id>")
        return

    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID. It should be a number.")
        return

    chat_id = update.message.chat_id
    try:
        await context.bot.ban_chat_member(chat_id, target_user_id)
        await update.message.reply_text(f"🚫 User ID {target_user_id} has been banned.")
        logger.info(f"Admin {user.first_name} banned user ID {target_user_id}.")
    except Exception as e:
        logger.error(f"Error banning user ID {target_user_id}: {str(e)}")
        await update.message.reply_text(f"⚠️ Error banning user ID {target_user_id}. Please try again.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the bot statistics to the user."""
    stats = get_all_stats()
    stats_message = (
        f"📊 <b>Bot Statistics</b>\n\n"
        f"Total Contents Scanned: {stats.get('total_contents_scanned', 0)}\n"
        f"Total NSFW Detected: {stats.get('total_nsfw_detected', 0)}\n"
        f"Total SFW: {stats.get('total_sfw', 0)}\n"
        f"Total Users Banned: {stats.get('total_users_banned', 0)}\n"
        f"Total Violations: {stats.get('total_violations', 0)}\n"
    )
    await update.message.reply_text(stats_message, parse_mode="HTML")

# -------------------------------
# Message Handlers
# -------------------------------

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    user_id = user.id
    chat_id = update.message.chat_id

    logger.info(f"Received video or GIF from {user.first_name} (ID: {user_id})")
    
    try:
        await context.bot.send_chat_action(chat_id, action=ChatAction.UPLOAD_VIDEO)
        if update.message.video:
            file = await update.message.video.get_file()
            temp_path = f"temp_{user_id}.mp4"
        elif update.message.animation:  # GIFs are sent as animations
            file = await update.message.animation.get_file()
            temp_path = f"temp_{user_id}.gif"
        else:
            await update.message.reply_text("⚠️ Unsupported media type.")
            return

        await file.download_to_drive(custom_path=temp_path)
        logger.info(f"Media downloaded to {temp_path}")

        analyzer = VideoContentAnalyzer()
        results = analyzer.analyze_video(
            video_path=temp_path,
            num_frames=6,
            stop_on_nsfw=True,
            nsfw_threshold=0.5
        )

        increment_stat("total_contents_scanned")

        if results.get("nsfw_detected", False):
            increment_stat("total_nsfw_detected")
            violation_count = add_violation(user_id)
            update_dashboard(user_id, violation_count, "video")  # Update the dashboard
            await update.message.delete()
            log_action(user_id, "content_removed")
            logger.warning(f"🚨 NSFW detected in media from {user.first_name}, Violation: {violation_count}")
            await context.bot.send_message(
                chat_id,
                f"⚠️ NSFW content detected in your media! Violation {violation_count}/{FLAG_THRESHOLD}."
            )
            timestamp = int(time.time())
            cached_filename = os.path.join(FLAGGED_VIDEOS_DIR, f"user_{user_id}_{timestamp}.mp4" if update.message.video else f"user_{user_id}_{timestamp}.gif")
            os.rename(temp_path, cached_filename)
            logger.info(f"Flagged media saved as {cached_filename}")

            for admin_id in ADMIN_IDS:
                await context.bot.send_message(
                    admin_id,
                    f"🚨 NSFW content detected from user {user.first_name} (ID: {user_id}). Violation {violation_count}/{FLAG_THRESHOLD}."
                )
                if update.message.video:
                    await context.bot.send_video(admin_id, video=open(cached_filename, 'rb'))
                else:
                    await context.bot.send_animation(admin_id, animation=open(cached_filename, 'rb'))

            if violation_count >= FLAG_THRESHOLD:
                increment_stat("total_users_banned")
                if update.message.chat.type in ["group", "supergroup"]:
                    try:
                        await context.bot.ban_chat_member(chat_id, user_id)
                        log_action(user_id, "user_banned")
                        logger.warning(f"🚫 Banned {user.first_name} for repeated NSFW violations.")
                        await context.bot.send_message(
                            chat_id,
                            f"🚫 {user.first_name} has been banned for multiple NSFW violations."
                        )
                        reset_violation(user_id)
                    except telegram.error.BadRequest as e:
                        if "Can't remove chat owner" in str(e):
                            logger.warning(f"🚫 Cannot ban chat owner {user.first_name}.")
                            await context.bot.send_message(
                                chat_id,
                                "🚫 You have exceeded the NSFW violation threshold, but banning the chat owner is not allowed."
                            )
                        else:
                            raise e
                else:
                    logger.warning("🚫 Violation threshold exceeded but cannot ban in private chats.")
                    await context.bot.send_message(
                        chat_id,
                        "🚫 You have exceeded the NSFW violation threshold, but banning is not supported in private chats."
                    )
        else:
            increment_stat("total_sfw")
            os.remove(temp_path)
            logger.info("Media passed NSFW checks; temporary file removed.")

    except Exception as e:
        logger.error(f"Error processing media: {str(e)}")
        try:
            await update.message.reply_text("⚠️ Error processing your media. Please try again.")
        except telegram.error.BadRequest as e:
            if "Message to be replied not found" in str(e):
                logger.error("Failed to send error message: Message to be replied not found.")
            else:
                raise e

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    user_id = user.id
    chat_id = update.message.chat_id

    logger.info(f"Received image from {user.first_name} (ID: {user_id})")
    
    try:
        await context.bot.send_chat_action(chat_id, action=ChatAction.UPLOAD_PHOTO)
        photo = update.message.photo[-1]
        file = await photo.get_file()
        temp_path = f"temp_{user_id}.jpg"
        await file.download_to_drive(custom_path=temp_path)

        result = nsfw_detector(temp_path)
        label = result[0]["label"]
        score = result[0]["score"]

        logger.info(f"Image analysis - User: {user.first_name}, Label: {label}, Confidence: {score:.2f}")

        increment_stat("total_contents_scanned")

        if label.lower() == "nsfw":
            increment_stat("total_nsfw_detected")
            violation_count = add_violation(user_id)
            update_dashboard(user_id, violation_count, "image")  # Update the dashboard
            await update.message.delete()
            log_action(user_id, "content_removed")
            logger.warning(f"🚨 NSFW detected in image from {user.first_name}, Violation: {violation_count}")
            await context.bot.send_message(
                chat_id,
                f"⚠️ NSFW content detected in your image! Violation {violation_count}/{FLAG_THRESHOLD}."
            )
            timestamp = int(time.time())
            cached_filename = os.path.join(FLAGGED_IMAGES_DIR, f"user_{user_id}_{timestamp}.jpg")
            os.rename(temp_path, cached_filename)

            for admin_id in ADMIN_IDS:
                await context.bot.send_message(
                    admin_id,
                    f"🚨 NSFW content detected from user {user.first_name} (ID: {user_id}). Violation {violation_count}/{FLAG_THRESHOLD}."
                )
                await context.bot.send_photo(admin_id, photo=open(cached_filename, 'rb'))

            if violation_count >= FLAG_THRESHOLD:
                increment_stat("total_users_banned")
                if update.message.chat.type in ["group", "supergroup"]:
                    try:
                        await context.bot.ban_chat_member(chat_id, user_id)
                        log_action(user_id, "user_banned")
                        logger.warning(f"🚫 Banned {user.first_name} for repeated NSFW violations.")
                        await context.bot.send_message(
                            chat_id,
                            f"🚫 {user.first_name} has been banned for multiple NSFW violations."
                        )
                        reset_violation(user_id)
                    except telegram.error.BadRequest as e:
                        if "Can't remove chat owner" in str(e):
                            logger.warning(f"🚫 Cannot ban chat owner {user.first_name}.")
                            await context.bot.send_message(
                                chat_id,
                                "🚫 You have exceeded the NSFW violation threshold, but banning the chat owner is not allowed."
                            )
                        else:
                            raise e
                else:
                    logger.warning("🚫 Violation threshold exceeded but cannot ban in private chats.")
                    await context.bot.send_message(
                        chat_id,
                        "🚫 You have exceeded the NSFW violation threshold, but banning is not supported in private chats."
                    )
        else:
            increment_stat("total_sfw")
            os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await update.message.reply_text("⚠️ Error processing your image. Please try again.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pass

# -------------------------------
# Main Function
# -------------------------------

def main() -> None:
    init_db()
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("violations", violations))
    application.add_handler(CommandHandler("admin_flagged", admin_flagged))
    application.add_handler(CommandHandler("admin_reset", admin_reset))
    application.add_handler(CommandHandler("admin_ban", admin_ban))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)

    logger.info("🚀 Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
