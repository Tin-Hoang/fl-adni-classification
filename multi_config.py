# Flower Multi-Machine Configuration
# Update these values for your specific setup

import os

# Machine Configuration
SERVER_HOST = "otter30.eps.surrey.ac.uk"
CLIENT_HOSTS = [
    "otter32.eps.surrey.ac.uk",
    "otter39.eps.surrey.ac.uk"
]

# Authentication (consider using environment variables in production)
USERNAME = os.getenv("FL_USERNAME", "th01167")
PASSWORD = os.getenv("FL_PASSWORD")

# Project Configuration
PROJECT_DIR = "/user/HS402/th01167/Surrey/fl-adni-classification"
SERVER_PORT = 9092

# Virtual Environment Configuration
VENV_PATH = "/user/HS402/th01167/.venv/master/bin/python"
VENV_ACTIVATE = "/user/HS402/th01167/.venv/master/bin/activate"

# Flower Configuration
FLOWER_CONFIG = {
    "server_address": f"0.0.0.0:{SERVER_PORT}",
    "client_timeout": 300,  # 5 minutes
    "max_retry_attempts": 3
}

# SSH Configuration
SSH_CONFIG = {
    "timeout": 30,
    "banner_timeout": 30,
    "auth_timeout": 30
}

# Logging Configuration
LOG_CONFIG = {
    "server_log": "server.log",
    "client_log_prefix": "client_",
    "max_log_lines": 50
}
