import json
import os
import sys

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

class VoterDatabase:
    def __init__(self, db_file="db.json"):
        self.db_file = resource_path(db_file)
        self.voters = []
        self.load_database()

    def load_database(self):
        try:
            with open(self.db_file, "r") as f:
                self.voters = json.load(f)
        except Exception as e:
            print(f"Error loading database {self.db_file}: {e}")
            self.voters = []

    def find_voter(self, voter_id):
        for voter in self.voters:
            if voter.get("voter_id") == voter_id:
                return voter
        return None
