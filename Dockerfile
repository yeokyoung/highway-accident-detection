FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 필요한 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app/ app/
COPY static/ static/
COPY models/ models/
COPY videos/ videos/

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///./accidents.db
ENV DEBUG_MODE=False
ENV HEADLESS_MODE=True

# 포트 설정
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

