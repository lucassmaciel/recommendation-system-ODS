This project use "uv" so you need to install it on your computer

Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

MacOS | Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

After that, you will just run one command: "uv sync"

This command will create the .venv, install the required python version for the project and install the dependencies.


For run the project, you will ne