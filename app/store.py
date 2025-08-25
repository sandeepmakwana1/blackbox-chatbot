import json
import os

JSON_FILE = "data.json"


def load_data():
    """Load JSON data from file, return list."""
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data):
    """Save list to JSON file."""
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def add_record(record):
    """Add a new record (dict) to JSON file."""
    data = load_data()
    data.append(record)
    save_data(data)
    print(f"✅ Record added: {record}")


def update_record(record_id, updates):
    """Update record by 'id' field."""
    data = load_data()
    for item in data:
        if item.get("thread_id") == record_id:
            item.update(updates)
            save_data(data)
            print(f"✏️ Record {record_id} updated with {updates}")
            return
    print(f"⚠️ Record {record_id} not found.")


def delete_record(record_id):
    """Delete record by 'id' field."""
    data = load_data()
    new_data = [item for item in data if item.get("thread_id") != record_id]
    if len(new_data) != len(data):
        save_data(new_data)
        print(f"🗑️ Record {record_id} deleted.")
    else:
        print(f"⚠️ Record {record_id} not found.")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Add records
    add_record({"thread_id": 1, "name": "Alice", "age": 25})
    add_record({"thread_id": 2, "name": "Bob", "age": 30})

    # Update a record
    update_record(1, {"age": 26, "city": "New York"})

    # Delete a record
    delete_record(2)

    # Show current data
    print("📂 Final JSON:", load_data())
