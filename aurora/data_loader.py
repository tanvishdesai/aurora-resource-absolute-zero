import os
import json
import pandas as pd
import numpy as np
from aurora import config

class RealWorldDataManager:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(config.BASE_DIR, "..", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_training_scenarios(self, source='synthetic', num_scenarios=10):
        if source == 'google':
            return self._load_google_trace(num_scenarios)
        elif source == 'azure':
            return self._load_azure_trace(num_scenarios)
        else: # synthetic
            return self._generate_synthetic_google_like(num_scenarios)

    def _load_google_trace(self, num_scenarios):
        """Load from available Google sources (2019 subset -> Local Repo -> Public GCS)"""
        
        # 1. High Priority: User provided 2019 subset
        subset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cluster-data-2019", "borg_traces_data.csv")
        if os.path.exists(subset_path):
            print(f"✅ Found 2019 Borg Trace subset: {subset_path}")
            return self._parse_google_2019_subset(subset_path, num_scenarios)

        # 2. Check for local cluster-data repo (2011 traces)
        repo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cluster-data")
        repo_trace_path = os.path.join(repo_dir, "task_events", "part-00000-of-00500.csv.gz")
        
        # 3. Default download path (in data/google_trace)
        local_trace_path = os.path.join(self.data_dir, "google_trace", "part-00000-of-00500.csv.gz")
        
        target_path = local_trace_path
        if os.path.exists(repo_trace_path):
            print(f"✅ Found trace in cluster-data repo: {repo_trace_path}")
            target_path = repo_trace_path

        # Download / Parse 2011 Trace
        max_retries = 3
        backoff_factor = 2
        for attempt in range(max_retries):
            try:
                if not os.path.exists(target_path):
                    print(f"⬇️ Downloading Google Trace sample to {target_path} (Attempt {attempt+1}/{max_retries})...")
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    url = "https://storage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz"
                    import urllib.request
                    urllib.request.urlretrieve(url, target_path)
                
                scenarios = self._parse_google_trace(target_path, num_scenarios)
                if self._validate_trace(scenarios):
                    return scenarios
                else:
                    print("⚠️ Trace validation failed, retrying...")
                    if os.path.exists(target_path):
                        os.remove(target_path) 
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(backoff_factor ** attempt)
        
        print("⚠️ All real data sources failed. Falling back to synthetic.")
        return self._generate_synthetic_google_like(num_scenarios)

    def _parse_google_2019_subset(self, file_path, num_scenarios):
        """Parse the 2019 CSV subset with headers and JSON resource strings."""
        scenarios = []
        try:
            import pandas as pd
            import ast
            
            print(f"📖 Reading 2019 Trace: {file_path}")
            # Read first chunk to avoid memory explosion if file is huge
            # But specific file is ~300MB, acceptable for pandas
            df = pd.read_csv(file_path, nrows=20000) 
            
            # Filter for SUBMIT/SCHEDULE events to get task requirements
            # 'event' column: SUBMIT (not explicitly seen in head, maybe 'ENABLE'?), SCHEDULE
            # 'instance_events_type' might be better. 
            # Looking at head: 0 treated as SUBMIT/ENABLE? 
            # Actually, let's look at tasks that were SCHEDULED or ENABLED to get their specs.
            
            # Create unique job/task entries
            unique_tasks = {}
            nodes = set()
            
            for _, row in df.iterrows():
                job_id = str(row['collection_id'])
                task_idx = str(row['instance_index'])
                full_id = f"{job_id}_{task_idx}"
                
                # Parse Resources
                # Format: "{'cpus': 0.02..., 'memory': 0.01...}"
                try:
                    res = ast.literal_eval(row['resource_request'])
                    cpu = float(res.get('cpus', 0.01))
                    mem = float(res.get('memory', 0.01))
                except:
                    # Fallback if parse fails or None
                    cpu, mem = 0.01, 0.01
                
                # Scale up (traces are normalized 0-1 usually, but here they seem small)
                # Google 2019: cpus are normalized to machine capacity? 
                # Values like 0.02 suggest fractions of a machine.
                # Aurora expects absolute values (e.g. 2 cores, 4GB).
                # We will scale by a "Standard Machine" size e.g. 64 cores, 128GB
                cpu *= 64
                mem *= 128
                
                if cpu <= 0: cpu = 0.1
                if mem <= 0: mem = 0.1
                
                if full_id not in unique_tasks:
                    unique_tasks[full_id] = {
                        'id': full_id,
                        'cpu': cpu,
                        'memory': mem,
                        'arrival_time': float(row['time']) / 1_000_000, # Microm? Check unit. 
                        # 2011 was microsec. 2019 seems to be microsec (13 digits).
                        # Convert to seconds.
                        'duration': 300 # Default if no end time found
                    }
                
                # If we have start/end, update duration
                # start_time, end_time cols exist
                if not pd.isna(row['start_time']) and not pd.isna(row['end_time']):
                    duration = (row['end_time'] - row['start_time']) / 1_000_000
                    if duration > 0:
                        unique_tasks[full_id]['duration'] = duration
                
                # Track machines
                if not pd.isna(row['machine_id']):
                    nodes.add(row['machine_id'])

            # Convert to list
            all_tasks = list(unique_tasks.values())
            all_nodes = list(nodes)
            
            # Generate Scenarios
            current_tasks = []
            scenario_idx = 0
            
            # Create chunks of tasks
            chunk_size = 50
            for i in range(0, len(all_tasks), chunk_size):
                if scenario_idx >= num_scenarios:
                    break
                    
                batch = all_tasks[i:i+chunk_size]
                if len(batch) < 10: continue
                
                # Generate Nodes (Synthetic based on real count or just N standard nodes)
                # Using real machine IDs implies precise mapping, but we don't have machine specs in this file usually.
                # So we simulate a cluster.
                scenario_nodes = []
                for k in range(10): # Standard 10 nodes per scenario for now
                    scenario_nodes.append({
                        'id': f"node_{k}",
                        'cpu_capacity': 64.0,
                        'memory_capacity': 128.0
                    })
                
                # Normalize arrival times to start at 0
                min_time = min(t['arrival_time'] for t in batch)
                for t in batch:
                    t['arrival_time'] -= min_time
                
                scenarios.append({
                    'id': f'google_2019_{scenario_idx}',
                    'tasks': batch,
                    'nodes': scenario_nodes
                })
                scenario_idx += 1
                
            print(f"✅ Parsed {len(scenarios)} scenarios from 2019 traces.")
            return scenarios
            
        except ImportError:
            print("❌ Pandas not found. Please install pandas: pip install pandas")
            return self._generate_synthetic_google_like(num_scenarios)
        except Exception as e:
            print(f"❌ Error parsing 2019 trace: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_synthetic_google_like(num_scenarios)

    def _parse_google_trace(self, trace_path, num_scenarios):
        scenarios = []
        # Read a chunk to avoid memory issues
        df = pd.read_csv(trace_path, compression='gzip', header=None, nrows=10000)
        
        # Google trace specific parsing logic
        # Columns: 0=timestamp, 2=jobID, 3=taskID, 9=CPU, 10=MEM
        
        # Filter valid rows
        df = df.dropna(subset=[9, 10])
        
        # Group random rows into scenarios
        for i in range(num_scenarios):
            tasks_df = df.sample(n=np.random.randint(5, 50))
            tasks = []
            for _, row in tasks_df.iterrows():
                tasks.append({
                    "id": f"t_{row.name}",
                    # Canonical usage is normalized [0,1], scaling to realistic core counts for sim
                    "cpu": float(row[9]) * 32, 
                    "memory": float(row[10]) * 64, 
                    "duration": np.random.randint(50, 500), # Trace doesn't have easy duration in this file, using random
                    "sla_latency_ms": 100.0,
                    "priority": 1
                })
            scenarios.append({
                "scenario_id": f"google_{i}",
                "description": "Real-world trace derived scenario",
                "tasks": tasks,
                "nodes": [{"id": f"n{k}", "cpu_capacity": 32, "memory_capacity": 64} for k in range(5)]
            })
        return scenarios

    def _validate_trace(self, scenarios):
        """Validate that scenarios loaded correctly"""
        if not scenarios: return False
        for s in scenarios:
            if not s['tasks']: return False
            if 'cpu' not in s['tasks'][0]: return False
        return True

    def _load_azure_trace(self, num_scenarios):
        """
        Mock implementation for Azure Public Dataset loading.
        """
        print("🔍 Attempting to load Azure Public Dataset...")
        trace_path = os.path.join(self.data_dir, "azure_trace", "vmtable.csv")
        
        if not os.path.exists(trace_path):
             print("⚠️  Azure trace file not found. Please download it to data/azure_trace/")
             return self._generate_synthetic_azure_like(num_scenarios)

        return []

    def _generate_synthetic_google_like(self, num_scenarios):
        """Generate synthetic scenarios that statistically resemble Google traces."""
        scenarios = []
        for i in range(num_scenarios):
            scenarios.append({
                "scenario_id": f"google_synth_{i}",
                "description": "Synthetic scenario based on Google Cluster Trace stats",
                "tasks": [{"id": f"t{j}", "cpu": np.random.lognormal(0, 1), "memory": np.random.lognormal(0, 1), "duration": 100} for j in range(10)],
                "nodes": [{"id": f"n{j}", "cpu_capacity": 16, "memory_capacity": 32} for j in range(5)]
            })
        return scenarios

    def _generate_synthetic_azure_like(self, num_scenarios):
        """Generate synthetic scenarios that statistically resemble Azure traces."""
        scenarios = []
        for i in range(num_scenarios):
            scenarios.append({
                "scenario_id": f"azure_synth_{i}",
                "description": "Synthetic scenario based on Azure VM stats",
                "tasks": [{"id": f"t{j}", "cpu": np.random.uniform(1, 4), "memory": np.random.uniform(2, 8), "duration": 3600} for j in range(5)],
                "nodes": [{"id": f"n{j}", "cpu_capacity": 64, "memory_capacity": 128} for j in range(2)]
            })
        return scenarios
