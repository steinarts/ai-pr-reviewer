import subprocess


def ensure_model(model_name: str) -> None:
    subprocess.run(["ollama", "pull", model_name], check=True)
