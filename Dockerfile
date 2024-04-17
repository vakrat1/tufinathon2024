FROM python:3.12
WORKDIR /
COPY . .
RUN pip install -r requirements.txt
Expose 8080
CMD ["python","controller.py"]