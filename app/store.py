import json
import os
import fcntl
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

JSON_FILE = "data.json"


@contextmanager
def _file_lock():
    """Context manager for thread-safe file operations."""
    lock_file = JSON_FILE + ".lock"
    with open(lock_file, "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_data() -> List[Dict[str, Any]]:
    """Load JSON data from file, return list."""
    if not os.path.exists(JSON_FILE):
        return []

    with _file_lock():
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []


def save_data(data: List[Dict[str, Any]]) -> None:
    """Save list to JSON file with thread safety."""
    with _file_lock():
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def get_record(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific record by thread_id."""
    data = load_data()
    for item in data:
        if item.get("thread_id") == thread_id:
            return item
    return None


def add_record(record: Dict[str, Any]) -> None:
    """Add a new record (dict) to JSON file."""
    data = load_data()
    data.append(record)
    save_data(data)
    print(f"Record added: {record}")


def update_record(record_id: str, updates: Dict[str, Any]) -> bool:
    """Update record by thread_id field. Returns True if updated, False if not found."""
    data = load_data()
    for item in data:
        if item.get("thread_id") == record_id:
            item.update(updates)
            save_data(data)
            print(f"Record {record_id} updated with {updates}")
            return True
    print(f"Record {record_id} not found.")
    return False


def delete_record(record_id: str) -> bool:
    """Delete record by thread_id field. Returns True if deleted, False if not found."""
    data = load_data()
    new_data = [item for item in data if item.get("thread_id") != record_id]
    if len(new_data) != len(data):
        save_data(new_data)
        print(f"Record {record_id} deleted.")
        return True
    else:
        print(f"Record {record_id} not found.")
        return False


def get_all_records() -> List[Dict[str, Any]]:
    """Get all records."""
    return load_data()


if __name__ == "__main__":
    # Add records
    add_record({"thread_id": "1", "name": "Alice", "age": 25})
    add_record({"thread_id": "2", "name": "Bob", "age": 30})

    # Update a record
    update_record("1", {"age": 26, "city": "New York"})

    # Delete a record
    delete_record("2")

    # Show current data
    print("Final JSON:", load_data())
