FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p figures outputs

CMD ["python", "src/basket_segmentation_cp1.py"]
