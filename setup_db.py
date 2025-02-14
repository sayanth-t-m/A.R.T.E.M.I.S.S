import sqlite3

DB_FILE = "violations.db"

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
    ''')

    # Create actions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            action TEXT,
            value INTEGER
        )
    ''')

    # Create contents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            is_nsfw BOOLEAN
        )
    ''')

    # Insert initial data if necessary
    cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)', ('total_contents_scanned', 0))
    cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)', ('total_nsfw_detected', 0))
    cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)', ('total_sfw', 0))
    cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)', ('total_users_banned', 0))
    cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, ?)', ('total_violations', 0))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    setup_database()
