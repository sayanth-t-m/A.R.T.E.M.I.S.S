# A.R.T.E.M.I.S.S

Automated Review for Telegram Environments Monitoring Inappropriate Submissions System

Aphrodite is a Telegram bot designed to automatically detect and remove NSFW (Not Safe for Work) content in group chats. The bot uses a machine learning model for image classification, flags users who send inappropriate content, and removes them after multiple violations.

## Features

- **NSFW Detection**: Detects NSFW images in group chats using a pre-trained image classification model.
- **Automated Warnings**: Flags users who send NSFW content. Users receive a warning each time they send inappropriate content.
- **Auto Removal**: Users are banned from the group if they reach a threshold of 3 violations.
- **Image Removal**: NSFW images are automatically deleted from the chat.
- **Persistent Flagging**: The bot persists user flags between restarts using either JSON or SQLite.

## How It Works

1. When a user sends an image, the bot checks if the image is NSFW using the `Falconsai/nsfw_image_detection` model from Hugging Face.
2. If the image is NSFW:
   - The image is deleted from the group.
   - The user is flagged and warned.
   - If a user gets 3 flags, they are automatically banned from the group.
3. Safe images are allowed, and the user is notified.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Telegram Bot Token (create a bot via [BotFather](https://t.me/BotFather))
- Install required libraries:
  
  ```bash
  pip install python-telegram-bot transformers torch pillow requests
  ```

### Run the Bot

1. **Clone this repository**:

   ```bash
   git clone https://github.com/your-username/aphrodite-nsfw-detection.git
   cd aphrodite-nsfw-detection
   ```

2. **Set up your bot token**:
   
   Replace `'YOUR_BOT_TOKEN_HERE'` in the code with the token you got from BotFather.

3. **Choose your persistent storage method**:
   
   - For **JSON** storage, no additional setup is required.
   - For **SQLite** storage, make sure SQLite is installed.

4. **Run the bot**:

   ```bash
   python bot.py
   ```

### Example Usage

- When a user sends an image, the bot automatically checks if it is NSFW.
- If the image is NSFW, the bot deletes the image, flags the user, and warns them.
- If the user accumulates 3 flags, the bot removes them from the group.

## Persistent Storage

### JSON File

The bot stores user flags in a `user_flags.json` file, allowing flag data to persist between bot restarts.

### SQLite Database

Alternatively, the bot can store flag data in an SQLite database (`user_flags.db`), which is more scalable for larger groups.

## How to Customize

- **Flag Threshold**: Change the `FLAG_THRESHOLD` variable in `bot.py` to modify the number of violations a user can have before being banned.
- **Persistent Storage**: You can switch between JSON and SQLite by modifying the respective sections in the code.

## Technologies Used

- [Python-Telegram-Bot](https://python-telegram-bot.org/) for bot interaction.
- [Transformers](https://huggingface.co/transformers/) for NSFW image classification.
- [Pillow](https://pillow.readthedocs.io/) for image handling.
- [Torch](https://pytorch.org/) for the deep learning model.
- SQLite or JSON for persistent storage.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## Disclaimer

This bot is designed to help moderate Telegram groups, but false positives and negatives may occur. Always review flagged content, and use it responsibly.

---

**Aphrodite** — Keeping your Telegram groups clean and safe!
