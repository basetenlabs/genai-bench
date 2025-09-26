"""
Baseten Training Job Configuration for GenAI Bench
"""

from truss_train import definitions
from truss.base import truss_config

# Define the benchmark training job
training_job = definitions.TrainingJob(
    image=definitions.Image(
        base_image="python:3.11-slim"
    ),
    compute=definitions.Compute(
        cpu_count=8,
        memory="32Gi"
    ),
    runtime=definitions.Runtime(
        start_commands=[
            "apt-get update && apt-get install -y git",
            "pip install uv",
            "python run_benchmark.py"
        ],
        environment_variables={
            "BASETEN_API_KEY": definitions.SecretReference(name="BASETEN_API_KEY"),
        },
        cache_config=definitions.CacheConfig(
            enabled=True,
        ),
        checkpointing_config=definitions.CheckpointingConfig(
            enabled=True,
        )
    )
)

# Create training project
training_project = definitions.TrainingProject(
    name="genai-benchmark",
    job=training_job
)
