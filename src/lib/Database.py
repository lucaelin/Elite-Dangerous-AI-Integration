import json
import os
import sqlean as sqlite3
from typing import Any, final, List, Dict, Tuple
import sqlite_vec
import numpy as np

from .Config import get_cn_appdata_path

def get_db_path() -> str:
    return os.path.join(get_cn_appdata_path(), 'covas.db')

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def instantiate_class_by_name(self, classes: list[Any], class_name: str, data: dict[str, Any]) -> Any:
    for cls in classes:
        if cls.__name__ == class_name:
            return cls(**data)
    return None

@final
class EventStore():
    def __init__(self, store_name: str, event_classes: list[Any]):
        self.conn = get_connection()
        self.cursor = self.conn.cursor()
        self.store_name = store_name
        self.table_name = f'{store_name}_v1'
        self.event_classes = event_classes
        
        self.cursor.execute(f'''                
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class TEXT,
                data TEXT,
                processed_at FLOAT,
                inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
    def __del__(self):
        if 'conn' in self.__dict__:
            self.conn.close()
    
    def commit(self) -> None:
        self.conn.commit()
    
    def insert_event(self, event: Any, processed_at: float, commit: bool = True) -> None:
        event_data = json.dumps(event.__dict__)
        event_class = event.__class__.__name__
        _ = self.cursor.execute(f'''
            INSERT INTO {self.table_name} (class, data, processed_at)
            VALUES (?, ?, ?)
        ''', (event_class, event_data, processed_at))
        if commit:
            self.conn.commit()
    
    def get_latest(self, limit: int = 100) -> list[Any]:
        self.cursor.execute(f'''
            SELECT class, data, processed_at
            FROM {self.table_name}
            ORDER BY processed_at DESC
            LIMIT (?)
        ''', (limit,))
        rows = self.cursor.fetchall()
        events = []
        for row in rows:
            instance = instantiate_class_by_name(self, self.event_classes, row[0], json.loads(row[1]))
            events.append(instance)
        return events
    
    def delete_all(self) -> None:
        _ = self.cursor.execute(f'''
            DELETE FROM {self.table_name}
        ''')
        self.conn.commit()
    
    
@final
class KeyValueStore():
    def __init__(self, store_name: str):
        self.conn = get_connection()
        self.cursor = self.conn.cursor()
        self.store_name = store_name
        self.table_name = f'{store_name}_v1'
        
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                key TEXT PRIMARY KEY,
                version TEXT,
                value TEXT,
                inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
                    
    def __del__(self):
        if 'conn' in self.__dict__:
            self.conn.close()
                        
    def get_version(self, key: str) -> str | None:
        self.cursor.execute(f'''
            SELECT version
            FROM {self.table_name}
            WHERE key = ?
        ''', (key,))
        row = self.cursor.fetchone()
        if row:
            return row[0]
        
    def init(self, key: str, version: str, value: Any) -> Any:
        current_version = self.get_version(key)
        if current_version == version:
            return self.get(key)
        
        _ = self.cursor.execute(f'''
            INSERT OR REPLACE INTO {self.table_name} (key, version, value)
            VALUES (?, ?, ?)
        ''', (key, version, json.dumps(value)))
        self.conn.commit()
        
        return self.get(key)
    
    def set(self, key: str, value: Any) -> None:
        _ = self.cursor.execute(f'''
            UPDATE {self.table_name}
            SET value = ?
            WHERE key = ?
        ''', (json.dumps(value), key, ))
        self.conn.commit()
    
    def get(self, key: str, default: Any = None) -> Any:
        self.cursor.execute(f'''
            SELECT value
            FROM {self.table_name}
            WHERE key = ?
        ''', (key,))
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        return default

    def get_all(self) -> dict[str, Any]:
        self.cursor.execute(f'''
            SELECT key, value
            FROM {self.table_name}
        ''')
        rows = self.cursor.fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}
    
    def delete(self, key: str) -> None:
        _ = self.cursor.execute(f'''
            DELETE FROM {self.table_name}
            WHERE key = ?
        ''', (key,))
        self.conn.commit()
    
    def delete_all(self) -> None:
        _ = self.cursor.execute(f'''
            DELETE FROM {self.table_name}
        ''')
        self.conn.commit()
    
@final
class VectorStore():
    def __init__(self, store_name: str, embedding_dim: int = 384):
        self.conn = get_connection()
        self.cursor = self.conn.cursor()
        self.store_name = store_name
        self.table_name = f'{store_name}_v1'
        self.vector_table = f'{store_name}_vec_v1'
        
        # Create table for metadata
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                metadata TEXT,
                inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create virtual table for vector storage using sqlite-vec
        self.cursor.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table} USING vec0(
                embedding FLOAT[{embedding_dim}],
            )
        ''')
        
    def __del__(self):
        if 'conn' in self.__dict__:
            self.conn.close()
    
    def store(self, id: int, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Store an embedding vector with associated id and metadata
        
        Args:
            id: Unique identifier for the embedding
            embedding: List of floats representing the embedding vector
            metadata: Dictionary of metadata to store with the embedding
        """
        # Store metadata in the main table
        self.cursor.execute(f'''
            INSERT OR REPLACE INTO {self.table_name} (id, metadata)
            VALUES (?, ?)
        ''', (id, json.dumps(metadata)))
        
        # Convert embedding to JSON format for the vec table
        embedding_json = json.dumps(embedding)
        
        # Store the embedding in the vector table, using the id as rowid
        self.cursor.execute(f'''
            INSERT OR REPLACE INTO {self.vector_table} (rowid, embedding)
            VALUES (?, ?)
        ''', (id, embedding_json))
        
        self.conn.commit()
    
    def search(self, query_embedding: List[float], n: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: The embedding vector to search for
            n: Number of results to return
            
        Returns:
            List of tuples containing (id, metadata, similarity_score)
        """
        # Convert query embedding to JSON for the match query
        query_json = json.dumps(query_embedding)
        
        # Query for nearest neighbors using vector similarity
        self.cursor.execute(f'''
            SELECT v.rowid, t.metadata, v.distance
            FROM {self.vector_table} v
            JOIN {self.table_name} t ON v.rowid = t.id
            WHERE v.embedding MATCH ? and k = ?
            ORDER BY v.distance
            LIMIT ?
        ''', (query_json, n, n))
        
        results = self.cursor.fetchall()
        
        # Convert results to the expected format
        return [(str(row[0]), json.loads(row[1]), 1.0 - row[2]) for row in results]
    
    def delete(self, id: str) -> None:
        """Delete an embedding by id"""
        # Delete from metadata table
        self.cursor.execute(f'''
            DELETE FROM {self.table_name}
            WHERE id = ?
        ''', (id,))
        
        # Delete from vector table
        self.cursor.execute(f'''
            DELETE FROM {self.vector_table}
            WHERE rowid = ?
        ''', (id,))
        
        self.conn.commit()
    
    def delete_all(self) -> None:
        """Delete all embeddings"""
        self.cursor.execute(f'''
            DELETE FROM {self.table_name}
        ''')
        
        self.cursor.execute(f'''
            DELETE FROM {self.vector_table}
        ''')
        
        self.conn.commit()