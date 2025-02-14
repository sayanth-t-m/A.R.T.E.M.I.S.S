from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
import os
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
DB_FILE = "violations.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    conn = get_db_connection()
    stats = conn.execute('SELECT key, value FROM stats').fetchall()
    conn.close()
    return jsonify({stat['key']: stat['value'] for stat in stats})

@app.route('/api/actions')
def api_actions():
    conn = get_db_connection()
    actions = conn.execute('SELECT * FROM actions ORDER BY timestamp DESC').fetchall()
    conn.close()
    return jsonify([dict(action) for action in actions])

@app.route('/api/all_data')
def api_all_data():
    conn = get_db_connection()
    stats = conn.execute('SELECT key, value FROM stats').fetchall()
    actions = conn.execute('SELECT * FROM actions ORDER BY timestamp DESC').fetchall()
    violations = conn.execute('SELECT timestamp, COUNT(*) as count FROM actions WHERE action="violation" GROUP BY timestamp').fetchall()
    content_types = conn.execute('SELECT type, COUNT(*) as count FROM contents GROUP BY type').fetchall()
    conn.close()
    return jsonify({
        'stats': {stat['key']: stat['value'] for stat in stats},
        'violations': [dict(violation) for violation in violations],
        'actions': [dict(action) for action in actions],
        'content_types': {content['type']: content['count'] for content in content_types}
    })

@socketio.on('connect')
def handle_connect(auth):
    emit('initial_data', get_initial_data())

def get_initial_data():
    conn = get_db_connection()
    stats = conn.execute('SELECT key, value FROM stats').fetchall()
    actions = conn.execute('SELECT * FROM actions ORDER BY timestamp DESC').fetchall()
    violations = conn.execute('SELECT timestamp, COUNT(*) as count FROM actions WHERE action="violation" GROUP BY timestamp').fetchall()
    content_types = conn.execute('SELECT type, COUNT(*) as count FROM contents GROUP BY type').fetchall()
    conn.close()
    return {
        'stats': {stat['key']: stat['value'] for stat in stats},
        'violations': [dict(violation) for violation in violations],
        'actions': [dict(action) for action in actions],
        'content_types': {content['type']: content['count'] for content in content_types}
    }

def background_thread():
    while True:
        time.sleep(10)
        socketio.emit('update_data', get_initial_data())

if __name__ == '__main__':
    socketio.start_background_task(target=background_thread)
    socketio.run(app, debug=True)
