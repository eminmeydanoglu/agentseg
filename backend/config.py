"""Backend configuration (env overrides)."""

import os

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5172"))

# Frontend dev server (bun run dev); the built UI is served by the backend itself
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
