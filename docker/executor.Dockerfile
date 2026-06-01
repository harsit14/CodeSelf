FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONNOUSERSITE=1

RUN useradd --create-home --uid 1000 runner

WORKDIR /workspace
USER runner

CMD ["python", "run_phase.py"]
