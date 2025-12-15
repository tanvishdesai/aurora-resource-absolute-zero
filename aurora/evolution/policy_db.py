import json
import os

class PolicyDatabase:
    def __init__(self, db_path="data\policy_database.json"):
        self.db_path = db_path
        self.policies = self._load_db()

    def _load_db(self):
        """Load policies from JSON file if it exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    print(f"✅ Loaded {len(data.get('policies', []))} policies from {self.db_path}")
                    return data.get("policies", [])
            except Exception as e:
                print(f"⚠️ Error loading database: {e}. Starting fresh.")
                return []
        else:
            print(f"ℹ️ No existing policy database found. Creating new at {self.db_path}")
            return []

    def _save_db(self):
        """Save policies to JSON file."""
        try:
            with open(self.db_path, "w") as f:
                json.dump({"policies": self.policies}, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save policy database: {e}")

    def add_policy(self, policy, result, reward):
        """
        Add a successful policy to the database.
        Includes simple ELO/Score mechanism (omitted for brevity, using raw reward).
        """
        entry = {
            "id": f"policy_{len(self.policies)}",
            "reward": reward,
            "policy": policy,
            "result_summary": {
                "avg_latency": result.get('avg_latency'),
                "sla_violations": result.get('sla_violations')
            },
            # "scenario_hash": ... (Could add for retrieval)
        }
        self.policies.append(entry)
        
        # Sort by reward (descending) so best policies are first
        self.policies.sort(key=lambda x: x['reward'], reverse=True)
        
        # Keep only top 100 to save space/time (optional)
        if len(self.policies) > 100:
            self.policies = self.policies[:100]
            
        # Persist to disk
        self._save_db()

    def get_best_policy(self):
        """Return the highest rated policy."""
        if not self.policies:
            return None
        return self.policies[0]

    def retrieve_similar_policy(self, scenario):
        """
        Future Stub: Retrieve policy based on scenario similarity.
        For now, returns best policy.
        """
        return self.get_best_policy()
