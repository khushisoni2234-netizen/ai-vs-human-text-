from flask import Flask, render_template, request, redirect, session
import pickle
import sqlite3
import hashlib
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Database
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    text TEXT,
    result TEXT
)
""")

conn.commit()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Extra features
def extract_features(text):
    words = text.split()
    sentences = text.split('.')

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_length = len(words) / len(sentences) if sentences else 0
    unique_ratio = len(set(words)) / len(words) if words else 0

    return [
        len(text),
        text.count(','),
        text.count('.'),
        avg_word_len,
        sentence_length,
        unique_ratio
    ]

# Home
@app.route('/')
def home():
    if "user" in session:
        return redirect('/index')
    return redirect('/login')

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()

        return redirect('/login')

    return render_template('signup.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()

        if user:
            session['user'] = username
            return redirect('/index')
        else:
            return "Invalid credentials"

    return render_template('login.html')

# Main Page
@app.route('/index')
def index():
    if "user" not in session:
        return redirect('/login')

    return render_template('index.html')

# Detect
@app.route('/detect', methods=['POST'])
def detect():
    if "user" not in session:
        return redirect('/login')

    text = request.form['text']

    tfidf = vectorizer.transform([text]).toarray()
    extra = np.array([extract_features(text)])

    final = np.hstack((tfidf, extra))

    prediction = model.predict(final)[0]

    result = "AI Generated 🤖" if prediction == 1 else "Human Written 🧑"

    cursor.execute("INSERT INTO history (username, text, result) VALUES (?, ?, ?)",
                   (session['user'], text, result))
    conn.commit()

    return render_template('index.html', result=result, text=text)

# History
@app.route('/history')
def history():
    if "user" not in session:
        return redirect('/login')

    cursor.execute("SELECT text, result FROM history WHERE username=?", (session['user'],))
    data = cursor.fetchall()

    return render_template('history.html', data=data)

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# Run
if __name__ == "__main__":
    app.run(debug=True)