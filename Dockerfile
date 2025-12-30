FROM python:3.12
WORKDIR /usr/src/app

# copy all, because there is a .dockerignore
COPY ./ ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "scripts/entrypoint-basic-container.py"]
# python scripts/entrypoint-basic-container.py
