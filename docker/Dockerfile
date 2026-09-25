# Sandbox image for running generated/reference implementations with --network none.
# Build: docker build -t paper-forge-sandbox:latest docker/
# ponytail: numpy-only image; add `torch --index-url https://download.pytorch.org/whl/cpu`
# when generated implementations start needing it.
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=1.24" "pytest>=8"
RUN useradd -m runner
USER runner
