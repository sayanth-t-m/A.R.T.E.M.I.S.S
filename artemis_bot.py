import logging
from telegram import Update, ChatPermissions
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Set up logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# NSFW detection pipeline using Hugging Face model
nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# A dictionary to track user flags {user_id: number_of_flags}
user_flags = {}

# Threshold of NSFW violations before user is banned
FLAG_THRESHOLD = 3

# Command handler for /start
async def start(update: Update, context) -> None:
    await update.message.reply_text("Welcome to A.R.T.E.M.I.S.S.! Send me an image, and I'll analyze it for NSFW content.")

# Handle image messages and apply flagging system
async def handle_image(update: Update, context) -> None:
    # Get the user and chat information
    user = update.message.from_user
    user_id = user.id
    chat_id = update.message.chat_id

    # Get the image sent by the user
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_url = file.file_path

    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Classify the image
    result = nsfw_detector(img)

    # Respond to the user based on the classification
    label = result[0]['label']
    score = result[0]['score']

    if label == 'nsfw':
        # Increase the user's flag count
        user_flags[user_id] = user_flags.get(user_id, 0) + 1

        # Remove the NSFW message from the chat
        await update.message.delete()

        # Notify the user that they have been flagged
        await update.message.reply_text(f"⚠️ NSFW content detected! This is violation {user_flags[user_id]} out of {FLAG_THRESHOLD}.")

        # Check if the user has exceeded the flag threshold
        if user_flags[user_id] >= FLAG_THRESHOLD:
            await context.bot.ban_chat_member(chat_id, user_id)
            await context.bot.send_message(chat_id, f"🚫 User {user.first_name} has been removed from the group for multiple NSFW violations.")
            del user_flags[user_id]
    else:
        await update.message.reply_text(f"✅ The image is safe (SFW) with confidence: {score:.2f}")

# Handle unknown commands/messages
async def unknown(update: Update, context) -> None:
    await update.message.reply_text("Sorry, I only process images. Send me an image to analyze.")

# Main function to start the bot
def main():
    # Create application with bot token
    application = ApplicationBuilder().token('7550587852:AAGImN1la_a592TzoKV5d5js6rlUPp1ozjM').build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
