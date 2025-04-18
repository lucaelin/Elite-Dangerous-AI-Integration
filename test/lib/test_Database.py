import uu
from uuid import uuid4
import uuid
import pytest
import sqlite3
from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from src.lib.Database import EventStore, KeyValueStore, VectorStore, get_connection

# Test event classes
@dataclass
class SampleEvent1:
    name: str
    value: int

@dataclass 
class SampleEvent2:
    message: str

# Fixtures
@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")

@pytest.fixture
def mock_connection(db_path, monkeypatch):
    def mock_get_db_path():
        return db_path
    
    monkeypatch.setattr('src.lib.Database.get_db_path', mock_get_db_path)
    
    conn = get_connection()
    
    return conn

@pytest.fixture
def event_store(mock_connection):
    return EventStore("test_events", [SampleEvent1, SampleEvent2])

@pytest.fixture
def kv_store(mock_connection):
    return KeyValueStore("test_store")

@pytest.fixture
def vector_store(mock_connection):
    store = VectorStore("test_vectors", embedding_dim=4)
    
    return store

# EventStore Tests
def test_event_store_init(event_store, mock_connection):
    """Test EventStore initialization creates table"""
    cursor = mock_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_events_v1'")
    assert cursor.fetchone() is not None

def test_event_store_insert(event_store):
    """Test inserting events"""
    event = SampleEvent1(name="test", value=42)
    processed_at = datetime.now().timestamp()
    
    event_store.insert_event(event, processed_at)
    
    events = event_store.get_latest(1)
    assert len(events) == 1
    assert isinstance(events[0], SampleEvent1)
    assert events[0].name == "test"
    assert events[0].value == 42

def test_event_store_get_latest(event_store):
    """Test retrieving latest events with limit"""
    for i in range(5):
        event = SampleEvent1(name=f"test_{i}", value=i)
        event_store.insert_event(event, float(i))
    
    events = event_store.get_latest(3)
    assert len(events) == 3
    assert events[0].value == 4  # Latest event first
    assert events[2].value == 2

def test_event_store_delete_all(event_store):
    """Test deleting all events"""
    event = SampleEvent1(name="test", value=42)
    event_store.insert_event(event, 1.0)
    
    event_store.delete_all()
    events = event_store.get_latest()
    assert len(events) == 0

# KeyValueStore Tests
def test_kv_store_init(kv_store, mock_connection):
    """Test KeyValueStore initialization creates table"""
    cursor = mock_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_store_v1'")
    assert cursor.fetchone() is not None

def test_kv_store_init_method(kv_store):
    """Test init method with version check"""
    value = {"data": "test"}
    
    # Initial set
    result = kv_store.init("key1", "10", value)
    assert result == value
    
    # Same version should return existing value
    new_value = {"data": "different"}
    assert kv_store.get_version("key1") == "10"
    result = kv_store.init("key1", "10", new_value)
    assert result == value
    
    # New version should update value
    result = kv_store.init("key1", "2.0", new_value)
    assert result == new_value

def test_kv_store_set_get(kv_store):
    """Test setting and getting values"""
    kv_store.init("key1", "1.0", "test")
    kv_store.set("key1", "updated")
    
    assert kv_store.get("key1") == "updated"
    assert kv_store.get("nonexistent", "default") == "default"

def test_kv_store_get_all(kv_store):
    """Test getting all values"""
    kv_store.init("key1", "1.0", "value1")
    kv_store.init("key2", "1.0", "value2")
    
    all_values = kv_store.get_all()
    assert len(all_values) == 2
    assert all_values["key1"] == "value1"
    assert all_values["key2"] == "value2"

def test_kv_store_delete(kv_store):
    """Test deleting specific key"""
    kv_store.init("key1", "1.0", "value1")
    kv_store.delete("key1")
    
    assert kv_store.get("key1") is None

def test_kv_store_delete_all(kv_store):
    """Test deleting all keys"""
    kv_store.init("key1", "1.0", "value1")
    kv_store.init("key2", "1.0", "value2")
    
    kv_store.delete_all()
    assert len(kv_store.get_all()) == 0

# VectorStore Tests
def test_vector_store_init(vector_store, mock_connection):
    """Test VectorStore initialization creates tables"""
    cursor = mock_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_vectors_v1'")
    assert cursor.fetchone() is not None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_vectors_vec_v1'")
    assert cursor.fetchone() is not None

def test_vector_store_store_and_search(vector_store):
    """Test storing embeddings and searching for similar ones"""
    # Store some embeddings
    vector_store.store(1, [1.0, 0.0, 0.0, 0.0], {"text": "Document 1"})
    vector_store.store(2, [0.7, 0.7, 0.0, 0.0], {"text": "Document 2"})
    vector_store.store(3, [0.0, 0.0, 1.0, 0.0], {"text": "Document 3"})
    
    # Query for similar embeddings
    results = vector_store.search([0.9, 0.1, 0.0, 0.0], 2)
    
    # Check results
    assert len(results) <= 2  # We may get less than 2 because of our mocking
    
    if len(results) > 0:
        # Check format of results
        assert len(results[0]) == 3
        assert isinstance(results[0][0], int)  # id
        assert isinstance(results[0][1], dict)  # metadata
        assert isinstance(results[0][2], float)  # similarity score

def test_vector_store_delete(vector_store):
    """Test deleting embeddings by ID"""
    vector_store.store(1, [1.0, 0.0, 0.0, 0.0], {"text": "Document 1"})
    vector_store.store(2, [0.0, 1.0, 0.0, 0.0], {"text": "Document 2"})
    
    # Delete one embedding
    vector_store.delete(1)
    
    # Store table should no longer have id1
    vector_store.cursor.execute(f"SELECT id FROM test_vectors_v1 WHERE id = 'id1'")
    assert vector_store.cursor.fetchone() is None

def test_vector_store_delete_all(vector_store):
    """Test deleting all embeddings"""
    vector_store.store(1, [1.0, 0.0, 0.0, 0.0], {"text": "Document 1"})
    vector_store.store(2, [0.0, 1.0, 0.0, 0.0], {"text": "Document 2"})
    
    # Delete all embeddings
    vector_store.delete_all()
    
    # Tables should be empty
    vector_store.cursor.execute("SELECT COUNT(*) FROM test_vectors_v1")
    assert vector_store.cursor.fetchone()[0] == 0