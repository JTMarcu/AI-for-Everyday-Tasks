FROM python:3.10-slim

# Create user to avoid permission issues
RUN useradd -m -u 1000 user

# Use the new user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy files with correct ownership
COPY --chown=user . $HOME/app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Hugging Face
EXPOSE 7860

# Run your Flask app
CMD ["python", "app.py"]
