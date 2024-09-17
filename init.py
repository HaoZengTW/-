import sqlite3
conn = sqlite3.connect('./db/lite.db')
cursor = conn.cursor()
table_query = """
CREATE TABLE qa_vector_store (
    q   TEXT,
    a   TEXT
)
"""
cursor.execute(table_query)
conn.commit()
conn.close()