# Complete Guide: Obtaining Real-World Resource Allocation Data

You're right to ask - this is crucial. Let me give you **exact sources, download instructions, and processing code**.

---

## OPTION 1: Google Cluster Trace (BEST - Free & Public)

### What It Is:
Google released traces from one of their production compute clusters. Contains **millions** of task and machine events.

### Download Instructions:

```python
# Method 1: Direct download (recommended for Kaggle)
import os
import urllib.request
import gzip
import pandas as pd

def download_google_trace():
    """Download Google Cluster Trace 2011 (smaller, easier)"""
    
    base_url = "https://github.com/google/cluster-data/blob/master/ClusterData2011_2"
    
    # Files we need
    files_to_download = [
        "task_events-000000000000.csv.gz",
        "machine_events-000000000000.csv.gz"
    ]
    
    os.makedirs('data/google_trace', exist_ok=True)
    
    for filename in files_to_download:
        url = f"{base_url}/{filename}?raw=true"
        output_path = f"data/google_trace/{filename}"
        
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"✅ Downloaded {filename}")
        else:
            print(f"⏭️  {filename} already exists")
    
    return "data/google_trace"

# Download
trace_dir = download_google_trace()
```

**⚠️ File Size Warning**: Full trace is ~40GB. For Kaggle, use sampled version:

```python
def load_google_trace_sample(num_tasks=1000):
    """Load a sample of Google trace for pattern extraction"""
    
    # Column names (from Google's schema)
    task_columns = [
        'timestamp', 'missing_info', 'job_id', 'task_index',
        'machine_id', 'event_type', 'user', 'scheduling_class',
        'priority', 'cpu_request', 'memory_request', 'disk_request',
        'different_machine'
    ]
    
    # Read compressed file in chunks
    chunk_size = 10000
    sampled_tasks = []
    
    try:
        for chunk in pd.read_csv(
            'data/google_trace/task_events-000000000000.csv.gz',
            compression='gzip',
            names=task_columns,
            chunksize=chunk_size,
            nrows=100000  # Only read first 100k rows
        ):
            # Filter: only SUBMIT events (event_type == 0)
            submitted = chunk[chunk['event_type'] == 0]
            sampled_tasks.append(submitted)
            
            if sum(len(df) for df in sampled_tasks) >= num_tasks:
                break
        
        all_tasks = pd.concat(sampled_tasks, ignore_index=True)
        all_tasks = all_tasks.head(num_tasks)
        
        print(f"✅ Loaded {len(all_tasks)} tasks from Google trace")
        return all_tasks
        
    except FileNotFoundError:
        print("❌ Google trace not found. Using fallback...")
        return create_synthetic_google_like_trace(num_tasks)

# Load sample
google_sample = load_google_trace_sample(1000)
print(google_sample.head())
```

### Process Into Your Format:

```python
def process_google_trace(df):
    """Convert Google trace to our scenario format"""
    
    scenarios = []
    
    # Group by time windows (e.g., 5-minute windows)
    df['time_window'] = (df['timestamp'] // 300000000).astype(int)  # 5 min windows
    
    for window_id, window_df in df.groupby('time_window'):
        
        if len(window_df) < 5:  # Skip tiny windows
            continue
        
        # Extract tasks
        tasks = []
        for idx, row in window_df.iterrows():
            # Google uses normalized units (0-1), convert to realistic values
            cpu_cores = row['cpu_request'] * 16 if pd.notna(row['cpu_request']) else 1.0
            memory_gb = row['memory_request'] * 64 if pd.notna(row['memory_request']) else 2.0
            
            tasks.append({
                'id': f"task_{row['job_id']}_{row['task_index']}",
                'cpu': max(0.1, cpu_cores),  # Ensure positive
                'memory': max(0.5, memory_gb),
                'duration': np.random.lognormal(4, 1),  # Duration not in trace
                'arrival_time': row['timestamp'],
                'priority': row['priority'] if pd.notna(row['priority']) else 5
            })
        
        # Create nodes (machines) - typical Google cluster
        num_nodes = max(3, len(tasks) // 10)
        nodes = []
        for n in range(num_nodes):
            nodes.append({
                'id': f'node_{n}',
                'cpu': np.random.choice([8, 16, 32, 64]),
                'memory': np.random.choice([16, 32, 64, 128])
            })
        
        scenarios.append({
            'id': f'google_trace_window_{window_id}',
            'source': 'google_cluster_trace',
            'tasks': tasks,
            'nodes': nodes
        })
        
        if len(scenarios) >= 100:  # Limit scenarios
            break
    
    return scenarios

# Process
google_scenarios = process_google_trace(google_sample)
print(f"✅ Extracted {len(google_scenarios)} scenarios from Google trace")
```

---

## OPTION 2: Azure Public Dataset (Good Quality)

### What It Is:
Microsoft's production VM allocation data from Azure datacenters.

### Download Instructions:

```python
def download_azure_trace():
    """Download Azure Public Dataset 2019"""
    
    # Azure dataset is on GitHub
    base_url = "https://github.com/Azure/AzurePublicDataset/raw/master/data"
    
    files = [
        "vmtable_v2.csv.gz",  # VM information
    ]
    
    os.makedirs('data/azure_trace', exist_ok=True)
    
    for filename in files:
        url = f"{base_url}/{filename}"
        output_path = f"data/azure_trace/{filename}"
        
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                return None
        else:
            print(f"⏭️  {filename} already exists")
    
    return "data/azure_trace"

def load_azure_trace_sample(num_vms=1000):
    """Load Azure VM trace sample"""
    
    try:
        df = pd.read_csv(
            'data/azure_trace/vmtable_v2.csv.gz',
            compression='gzip',
            nrows=num_vms
        )
        
        print(f"✅ Loaded {len(df)} VMs from Azure trace")
        print(f"Columns: {df.columns.tolist()}")
        return df
        
    except FileNotFoundError:
        print("❌ Azure trace not found. Using fallback...")
        return create_synthetic_azure_like_trace(num_vms)

# Download and load
download_azure_trace()
azure_sample = load_azure_trace_sample(1000)
```

### Process Azure Data:

```python
def process_azure_trace(df):
    """Convert Azure trace to our format"""
    
    scenarios = []
    
    # Azure has: vmId, subscriptionId, deploymentId, vmCreated, vmDeleted, 
    #            maxCpu, avgCpu, maxMemory, avgMemory, etc.
    
    # Group by deployment (similar VMs deployed together)
    for deployment_id, deployment_df in df.groupby('deploymentId'):
        
        if len(deployment_df) < 2:
            continue
        
        # Extract VMs as tasks
        tasks = []
        for idx, row in deployment_df.iterrows():
            # Azure reports in cores and GB
            cpu = row.get('maxCpu', row.get('avgCpu', 2))
            memory = row.get('maxMemory', row.get('avgMemory', 4))
            
            # VM lifetime
            created = pd.to_datetime(row['vmCreated'])
            deleted = pd.to_datetime(row['vmDeleted']) if pd.notna(row['vmDeleted']) else None
            
            duration = (deleted - created).total_seconds() if deleted else 3600
            
            tasks.append({
                'id': f"vm_{row['vmId']}",
                'cpu': cpu,
                'memory': memory,
                'duration': duration,
                'arrival_time': created.timestamp(),
            })
        
        # Create nodes (Azure hosts)
        num_nodes = max(2, len(tasks) // 8)  # Typical VM:host ratio
        nodes = []
        for n in range(num_nodes):
            # Azure typically has large hosts
            nodes.append({
                'id': f'azure_host_{n}',
                'cpu': np.random.choice([32, 64, 96, 128]),
                'memory': np.random.choice([128, 256, 512, 1024])
            })
        
        scenarios.append({
            'id': f'azure_deployment_{deployment_id}',
            'source': 'azure_public_dataset',
            'tasks': tasks,
            'nodes': nodes
        })
        
        if len(scenarios) >= 100:
            break
    
    return scenarios

# Process
azure_scenarios = process_azure_trace(azure_sample)
print(f"✅ Extracted {len(azure_scenarios)} scenarios from Azure trace")
```

---

## OPTION 3: Alibaba Cluster Trace (Large Scale)

### What It Is:
Production traces from Alibaba Cloud. Very large scale.

### Download:

```python
def download_alibaba_trace():
    """Download Alibaba Cluster Trace 2018"""
    
    # Available at: https://github.com/alibaba/clusterdata
    # Download link: 
    url = "http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz"
    
    os.makedirs('data/alibaba_trace', exist_ok=True)
    output_path = 'data/alibaba_trace/batch_task.tar.gz'
    
    if not os.path.exists(output_path):
        print("Downloading Alibaba trace (this may take a while)...")
        urllib.request.urlretrieve(url, output_path)
        
        # Extract
        import tarfile
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall('data/alibaba_trace/')
        
        print("✅ Downloaded and extracted Alibaba trace")
    else:
        print("⏭️  Alibaba trace already exists")
    
    return "data/alibaba_trace"

def load_alibaba_trace_sample(num_tasks=1000):
    """Load Alibaba batch task trace"""
    
    try:
        # Alibaba format: task_name, instance_num, job_name, task_type, 
        #                 status, start_time, end_time, plan_cpu, plan_mem
        
        df = pd.read_csv(
            'data/alibaba_trace/batch_task.csv',
            nrows=num_tasks
        )
        
        print(f"✅ Loaded {len(df)} tasks from Alibaba trace")
        return df
        
    except FileNotFoundError:
        print("❌ Alibaba trace not found. Using fallback...")
        return create_synthetic_alibaba_like_trace(num_tasks)
```

---

## OPTION 4: Kaggle Datasets (Easiest for You!)

Since you're using Kaggle, you can directly import datasets:

```python
# In Kaggle notebook, add these datasets:
# 1. Search: "cloud resource allocation"
# 2. Search: "datacenter traces"
# 3. Search: "vm workload"

# Example: Import from Kaggle
import kaggle

# Some available datasets on Kaggle:
kaggle_datasets = [
    'mustafakeser4/cloud-workload-traces',  # Example
    'anikannal/azure-cloud-trace',          # Example
]

for dataset in kaggle_datasets:
    try:
        kaggle.api.dataset_download_files(dataset, path='data/', unzip=True)
        print(f"✅ Downloaded {dataset}")
    except:
        print(f"⚠️  {dataset} not available")
```

---

## OPTION 5: Create Realistic Synthetic Data (BEST FALLBACK)

If you can't download real traces, create statistically accurate synthetic data:

```python
def create_realistic_synthetic_traces(num_scenarios=100):
    """
    Create synthetic traces that match real-world statistics
    Based on published research papers analyzing real traces
    """
    
    scenarios = []
    
    for i in range(num_scenarios):
        scenario_type = np.random.choice(['web', 'batch', 'ml', 'mixed'])
        
        if scenario_type == 'web':
            scenario = generate_web_workload()
        elif scenario_type == 'batch':
            scenario = generate_batch_workload()
        elif scenario_type == 'ml':
            scenario = generate_ml_workload()
        else:
            scenario = generate_mixed_workload()
        
        scenarios.append(scenario)
    
    return scenarios

def generate_web_workload():
    """Realistic web application workload"""
    
    # Based on: "Characterizing Cloud Workloads" (Google, 2011)
    # Key findings:
    # - Most tasks are short (< 1 min)
    # - Task size follows log-normal distribution
    # - Arrival rate varies diurnally
    
    num_tasks = np.random.randint(50, 200)
    num_nodes = max(3, num_tasks // 20)
    
    # Generate tasks with realistic distributions
    tasks = []
    current_time = 0
    
    for t in range(num_tasks):
        # Task size: log-normal (most small, few large)
        size_factor = np.random.lognormal(mean=0, sigma=1.0)
        
        # CPU: typically 0.5-4 cores for web apps
        cpu = np.clip(0.5 * size_factor, 0.1, 8.0)
        
        # Memory: typically 1-8 GB, correlated with CPU
        memory = cpu * np.random.uniform(1.5, 3.0)
        
        # Duration: most < 1 second, few up to minutes
        duration = np.random.lognormal(mean=-1, sigma=2)  # Mean ~0.4s
        duration = np.clip(duration, 0.01, 300)  # 10ms to 5min
        
        # Arrival pattern: Poisson with diurnal variation
        hour_of_day = (current_time / 3600) % 24
        if 8 <= hour_of_day <= 18:  # Business hours
            rate = 5.0  # tasks/second
        else:
            rate = 0.5
        
        inter_arrival = np.random.exponential(1/rate)
        current_time += inter_arrival
        
        tasks.append({
            'id': f't{t}',
            'cpu': round(cpu, 2),
            'memory': round(memory, 2),
            'duration': round(duration, 2),
            'arrival_time': round(current_time, 2),
            'priority': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        })
    
    # Nodes: heterogeneous capacities
    nodes = []
    node_types = [
        {'cpu': 4, 'memory': 8},
        {'cpu': 8, 'memory': 16},
        {'cpu': 16, 'memory': 32},
    ]
    
    for n in range(num_nodes):
        node_type = np.random.choice(node_types, p=[0.5, 0.3, 0.2])
        nodes.append({
            'id': f'n{n}',
            'cpu': node_type['cpu'],
            'memory': node_type['memory']
        })
    
    return {
        'id': f'web_synthetic_{np.random.randint(10000)}',
        'type': 'web_application',
        'tasks': tasks,
        'nodes': nodes,
        'source': 'synthetic_realistic'
    }

def generate_batch_workload():
    """Batch processing workload (MapReduce, Spark, etc.)"""
    
    # Based on: "Characterizing Private Clouds" (Yahoo, 2012)
    # - Larger tasks (hours)
    # - More uniform resource usage
    # - Predictable arrival patterns
    
    num_tasks = np.random.randint(20, 100)
    num_nodes = max(5, num_tasks // 10)
    
    tasks = []
    
    # Batch jobs arrive in waves
    num_waves = np.random.randint(2, 5)
    wave_starts = sorted(np.random.uniform(0, 7200, num_waves))
    
    tasks_per_wave = num_tasks // num_waves
    
    for wave_idx, wave_start in enumerate(wave_starts):
        for t in range(tasks_per_wave):
            # Batch tasks: larger, more uniform
            cpu = np.random.uniform(4, 32)
            memory = np.random.uniform(16, 128)
            
            # Duration: minutes to hours
            duration = np.random.uniform(300, 7200)  # 5min to 2hr
            
            # Arrival: clustered within wave
            arrival = wave_start + np.random.uniform(0, 60)
            
            tasks.append({
                'id': f't{wave_idx}_{t}',
                'cpu': round(cpu, 2),
                'memory': round(memory, 2),
                'duration': round(duration, 2),
                'arrival_time': round(arrival, 2),
                'priority': 5  # All same priority
            })
    
    # Nodes: large, uniform
    nodes = []
    for n in range(num_nodes):
        nodes.append({
            'id': f'n{n}',
            'cpu': 32,
            'memory': 128
        })
    
    return {
        'id': f'batch_synthetic_{np.random.randint(10000)}',
        'type': 'batch_processing',
        'tasks': tasks,
        'nodes': nodes,
        'source': 'synthetic_realistic'
    }

def generate_ml_workload():
    """ML training workload"""
    
    # Based on: "Analysis of Large-Scale Multi-Tenant GPU Clusters" (Microsoft, 2019)
    # - Very large resource requirements
    # - Long durations
    # - Few concurrent jobs
    
    num_tasks = np.random.randint(5, 20)  # Fewer, larger tasks
    num_nodes = max(3, num_tasks // 2)
    
    tasks = []
    current_time = 0
    
    for t in range(num_tasks):
        # ML tasks: very large
        cpu = np.random.uniform(16, 64)
        memory = np.random.uniform(64, 512)
        
        # Duration: hours
        duration = np.random.uniform(1800, 36000)  # 30min to 10hr
        
        # Arrival: somewhat clustered
        inter_arrival = np.random.exponential(300)  # Every 5min on average
        current_time += inter_arrival
        
        tasks.append({
            'id': f't{t}',
            'cpu': round(cpu, 2),
            'memory': round(memory, 2),
            'duration': round(duration, 2),
            'arrival_time': round(current_time, 2),
            'priority': np.random.choice([8, 9, 10])  # High priority
        })
    
    # Nodes: GPU-capable, very large
    nodes = []
    for n in range(num_nodes):
        nodes.append({
            'id': f'n{n}',
            'cpu': 64,
            'memory': 512,
            'gpu': 4  # 4 GPUs per node
        })
    
    return {
        'id': f'ml_synthetic_{np.random.randint(10000)}',
        'type': 'ml_training',
        'tasks': tasks,
        'nodes': nodes,
        'source': 'synthetic_realistic'
    }

def generate_mixed_workload():
    """Mix of web + batch + ML"""
    
    web = generate_web_workload()
    batch = generate_batch_workload()
    
    # Combine tasks
    all_tasks = web['tasks'] + batch['tasks']
    
    # Combine nodes (with some overlap)
    all_nodes = web['nodes'] + batch['nodes'][:len(batch['nodes'])//2]
    
    return {
        'id': f'mixed_synthetic_{np.random.randint(10000)}',
        'type': 'mixed_workload',
        'tasks': all_tasks,
        'nodes': all_nodes,
        'source': 'synthetic_realistic'
    }
```

---

## COMPLETE PRACTICAL SOLUTION FOR KAGGLE

Here's what you should actually do on Kaggle:

```python
# ========================================
# COMPLETE DATA ACQUISITION PIPELINE
# ========================================

class RealWorldDataManager:
    def __init__(self):
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_training_scenarios(self, num_scenarios=100):
        """
        Get training scenarios from best available source
        Priority: Real traces > Synthetic realistic > Basic synthetic
        """
        
        scenarios = []
        
        # Try Method 1: Download Google trace
        print("🔍 Attempting to load Google Cluster Trace...")
        try:
            google_scenarios = self._load_google_trace(num_scenarios // 2)
            scenarios.extend(google_scenarios)
            print(f"✅ Loaded {len(google_scenarios)} scenarios from Google")
        except Exception as e:
            print(f"⚠️  Google trace unavailable: {e}")
        
        # Try Method 2: Download Azure trace
        print("\n🔍 Attempting to load Azure trace...")
        try:
            azure_scenarios = self._load_azure_trace(num_scenarios // 2)
            scenarios.extend(azure_scenarios)
            print(f"✅ Loaded {len(azure_scenarios)} scenarios from Azure")
        except Exception as e:
            print(f"⚠️  Azure trace unavailable: {e}")
        
        # Method 3: Generate realistic synthetic
        if len(scenarios) < num_scenarios:
            print(f"\n🔧 Generating realistic synthetic data...")
            needed = num_scenarios - len(scenarios)
            synthetic = create_realistic_synthetic_traces(needed)
            scenarios.extend(synthetic)
            print(f"✅ Generated {len(synthetic)} realistic synthetic scenarios")
        
        print(f"\n✅ Total scenarios available: {len(scenarios)}")
        return scenarios[:num_scenarios]
    
    def _load_google_trace(self, num_scenarios):
        """Try to load Google trace"""
        
        # Check if already downloaded
        trace_file = f'{self.data_dir}/google_trace/task_events-000000000000.csv.gz'
        
        if not os.path.exists(trace_file):
            # Try to download
            url = "https://github.com/google/cluster-data/raw/master/ClusterData2011_2/task_events-000000000000.csv.gz"
            
            os.makedirs(f'{self.data_dir}/google_trace', exist_ok=True)
            
            print(f"  Downloading from {url}...")
            urllib.request.urlretrieve(url, trace_file)
        
        # Load and process
        df = load_google_trace_sample(num_scenarios * 10)  # Oversample
        scenarios = process_google_trace(df)
        
        return scenarios[:num_scenarios]
    
    def _load_azure_trace(self, num_scenarios):
        """Try to load Azure trace"""
        
        trace_file = f'{self.data_dir}/azure_trace/vmtable_v2.csv.gz'
        
        if not os.path.exists(trace_file):
            url = "https://github.com/Azure/AzurePublicDataset/raw/master/data/vmtable_v2.csv.gz"
            
            os.makedirs(f'{self.data_dir}/azure_trace', exist_ok=True)
            
            print(f"  Downloading from {url}...")
            urllib.request.urlretrieve(url, trace_file)
        
        df = load_azure_trace_sample(num_scenarios * 10)
        scenarios = process_azure_trace(df)
        
        return scenarios[:num_scenarios]
    
    def extract_real_patterns(self):
        """Extract statistical patterns from real data"""
        
        print("\n📊 Extracting real-world patterns...")
        
        patterns = {
            'task_cpu': {},
            'task_memory': {},
            'task_duration': {},
            'cpu_memory_ratio': {},
            'arrival_patterns': {}
        }
        
        # Try to extract from real traces
        try:
            google_df = load_google_trace_sample(5000)
            
            # Fit distributions
            cpu_values = google_df['cpu_request'].dropna() * 16
            memory_values = google_df['memory_request'].dropna() * 64
            
            from scipy import stats
            
            # CPU distribution
            patterns['task_cpu'] = {
                'distribution': 'lognormal',
                'params': stats.lognorm.fit(cpu_values),
                'mean': cpu_values.mean(),
                'std': cpu_values.std()
            }
            
            # Memory distribution
            patterns['task_memory'] = {
                'distribution': 'lognormal',
                'params': stats.lognorm.fit(memory_values),
                'mean': memory_values.mean(),
                'std': memory_values.std()
            }
            
            # CPU:Memory ratio
            ratios = cpu_values / memory_values
            patterns['cpu_memory_ratio'] = {
                'mean': ratios.mean(),
                'std': ratios.std(),
                'min': ratios.quantile(0.05),
                'max': ratios.quantile(0.95)
            }
            
            print("✅ Extracted patterns from Google trace")
            
        except Exception as e:
            print(f"⚠️  Using default patterns: {e}")
            
            # Default patterns from literature
            patterns = {
                'task_cpu': {
                    'distribution': 'lognormal',
                    'params': (0.5, 0, 2),  # shape, loc, scale
                    'mean': 2.5,
                    'std': 3.2
                },
                'task_memory': {
                    'distribution': 'lognormal',
                    'params': (0.6, 0, 4),
                    'mean': 5.8,
                    'std': 8.1
                },
                'cpu_memory_ratio': {
                    'mean': 0.5,  # 1 CPU : 2 GB memory
                    'std': 0.2,
                    'min': 0.25,
                    'max': 2.0
                }
            }
        
        # Save patterns
        import json
        with open(f'{self.data_dir}/real_patterns.json', 'w') as f:
            # Convert numpy types to native Python for JSON
            patterns_serializable = self._make_json_serializable(patterns)
            json.dump(patterns_serializable, f, indent=2)
        
        print("✅ Patterns saved to data/real_patterns.json")
        
        return patterns
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

# ========================================
# USAGE IN YOUR KAGGLE NOTEBOOK
# ========================================

# Initialize
data_manager = RealWorldDataManager()

# Get training scenarios (automatically tries real, falls back to synthetic)
training_scenarios = data_manager.get_training_scenarios(num_scenarios=100)

# Extract real-world patterns
real_patterns = data_manager.extract_real_patterns()

# Now you can use these!
print("\n" + "="*60)
print("DATA READY!")
print("="*60)
print(f"Training scenarios: {len(training_scenarios)}")
print(f"Real patterns extracted: {list(real_patterns.keys())}")
print("\nExample scenario:")
print(json.dumps(training_scenarios[0], indent=2)[:500] + "...")
```

---

## WHAT TO ACTUALLY DO (Step-by-Step)

### In your Kaggle notebook:

```python
# Cell 1: Setup
!pip install scipy pandas numpy

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import urllib.request

# Cell 2: Copy all the functions above
# (RealWorldDataManager, create_realistic_synthetic_traces, etc.)

# Cell 3: Get data
data_manager = RealWorldDataManager()

# This will:
# 1. Try to download Google trace
# 2. Try to download Azure trace
# 3. Fall back to realistic synthetic if needed
training_scenarios = data_manager.get_training_scenarios(100)

# Cell 4: Extract patterns
real_patterns = data_manager.extract_real_patterns()

# Cell 5: Verify quality
print("Sample task:")
print(training_scenarios[0]['tasks'][0])

print("\nReal patterns:")
print(f"CPU mean: {real_patterns['task_cpu']['mean']:.2f}")
print(f"Memory mean: {real_patterns['task_memory']['mean']:.2f}")

# Cell 6: Save for later use
with open('training_scenarios.json', 'w') as f:
    json.dump(training_scenarios, f)

print("✅ Data ready for training!")
```

---

