import logging
import sys
import os
import sqlite3
import time
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from transformers import pipeline

# -------------------------------
# Configuration and Initialization
# -------------------------------

# Replace with your actual admin Telegram user IDs.
ADMIN_IDS = {7012902263}  # Example: {123456789, 987654321}

# Bot token (hardcoded for now; consider using environment variables for production)
BOT_TOKEN = "7550587852:AAGImN1la_a592TzoKV5d5js6rlUPp1ozjM"

# SQLite database file for persisting violation counts.
DB_FILE = "violations.db"

# Directory for caching flagged images.
FLAGGED_IMAGES_DIR = "flagged_images"
if not os.path.exists(FLAGGED_IMAGES_DIR):
    os.makedirs(FLAGGED_IMAGES_DIR)

# Violation threshold before banning a user.
FLAG_THRESHOLD = 3

# Set up logging.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Initialize the NSFW detection pipeline using the Falconsai model.
try:
    nsfw_detector = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=0,
        use_fast=True
    )
except Exception as e:
    logger.error("Error loading NSFW detection model: " + str(e))
    raise e

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
        "Send me an image to check for NSFW content.\n"
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
            # Increase the user's violation count in the database.
            violation_count = add_violation(user_id)
            # Delete the original image message.
            await update.message.delete()
            logger.warning(f"🚨 NSFW detected! {user.first_name}, Violation: {violation_count}")
            await context.bot.send_message(
                chat_id,
                f"⚠️ NSFW content detected! Violation {violation_count}/{FLAG_THRESHOLD}."
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
            # If the image is safe, remove the temporary file without responding.
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
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Register the error handler.
    application.add_error_handler(error_handler)

    logger.info("🚀 Bot is starting...")
    # Run the bot until the user presses Ctrl-C.
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()