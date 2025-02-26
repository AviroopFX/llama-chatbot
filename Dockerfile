FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# Create necessary directories
RUN mkdir -p logs data models uploads

# Make the start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]
