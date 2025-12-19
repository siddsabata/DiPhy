FROM continuumio/miniconda3:latest

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Set working directory
WORKDIR /app

# Copy scripts
COPY orchestrator.py generate_tumor.py resample_tumor.py pipeline_settings.py /app/

# Default command shows Python version
CMD ["conda", "run", "-n", "sistem", "python", "--version"]

