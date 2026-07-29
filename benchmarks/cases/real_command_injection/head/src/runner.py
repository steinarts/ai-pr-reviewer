import subprocess


def ensure_model(model_name: str) -> None:
    command = f"ollama pull {model_name}"
    # Defect: untrusted model_name is interpolated into a shell command.
    subprocess.run(command, shell=True, check=True)
