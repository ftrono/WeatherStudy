FROM python:3.12

COPY . .
RUN pip install -r requirements.txt --no-cache-dir

#Run webhook
CMD python3 main.py
