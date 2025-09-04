#!/bin/bash

echo "🎮 Othello AI Engine 프론트엔드 시작 중..."

# 프론트엔드 디렉토리로 이동
cd frontend

# 정적 서버 시작
echo "🌐 프론트엔드 서버 시작 중... (http://localhost:5173)"
echo "🔗 백엔드 API: http://localhost:8000"
echo "⏹️  중지하려면 Ctrl+C를 누르세요"
echo ""

python3 -m http.server 5173 -d .
