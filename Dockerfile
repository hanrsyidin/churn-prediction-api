# 1. Gunakan Python 3.9
FROM python:3.9

# 2. Bikin folder kerja dan atur izin akses (Penting buat Hugging Face)
WORKDIR /code
RUN chmod 777 /code

# 3. Copy file requirements
COPY ./requirements.txt /code/requirements.txt

# 4. Install library (tanpa cache biar hemat memori)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy semua file sisa (main.py, model .pkl)
COPY . .

# 6. Buka izin akses lagi untuk file yang baru dicopy
RUN chmod -R 777 /code

# 7. Jalankan Server
# PENTING: Hugging Face WAJIB pakai port 7860. Jangan ganti!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]