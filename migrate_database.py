#!/usr/bin/env python3
"""
Database migration script to add last_login column to users table
"""
import sqlite3
from datetime import datetime

def migrate_database():
    """Add last_login column to users table"""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('medical_chatbot.db')
        cursor = conn.cursor()
        
        # Check if last_login column already exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'last_login' not in columns:
            # Add last_login column
            cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
            print("✅ Added last_login column to users table")
        else:
            print("ℹ️  last_login column already exists")
        
        conn.commit()
        conn.close()
        print("🎉 Database migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    migrate_database()
