import sqlite3

conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

class User:
    def __init__(self, email, password):
        self.email = email
        self.password = password

    @classmethod
    def create_table(cls):
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (email text PRIMARY KEY, password text)''')
        conn.commit()

    @classmethod
    def insert(cls, email, password):
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
        conn.commit()

    @classmethod
    def get(cls, email):
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        return c.fetchone() 
    
    @classmethod
    def get_all(cls):
        c.execute("SELECT * FROM users")
        return c.fetchall()


User.create_table() 

