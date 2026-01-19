import sqlite3
import json
import os

print("="*70)
print("NEUROSQL SYSTEM ANALYSIS")
print("="*70)

# 1. Analyze the database
print("\n📊 DATABASE ANALYSIS")
print("-"*70)

db_path = "knowledge_graph.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        table_name = table[0]
        print(f"\n  📋 Table: {table_name}")
        
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print("     Schema:")
        for col in columns:
            print(f"       - {col[1]} ({col[2]})")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"     Rows: {count:,}")
        
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            rows = cursor.fetchall()
            print(f"     Sample:")
            for row in rows[:3]:
                print(f"       {row}")
    
    conn.close()
else:
    print(f"❌ Database not found: {db_path}")

print("\n" + "="*70)
