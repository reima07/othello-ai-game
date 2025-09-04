#!/bin/bash

echo "🚀 Othello AI Engine 백엔드 시작 중..."

# 백엔드 디렉토리로 이동
cd backend

# 가상환경이 없으면 생성
if [ ! -d ".venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv .venv
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source .venv/bin/activate

# 의존성 설치
echo "📚 의존성 설치 중..."
pip install -r requirements.txt

# 서버 시작
echo "🌐 서버 시작 중... (http://localhost:8000)"
echo "📖 API 문서: http://localhost:8000/docs"
echo "⏹️  중지하려면 Ctrl+C를 누르세요"
echo ""

uvicorn othello.server:app --reload --port 8000
