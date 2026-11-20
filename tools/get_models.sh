#!/bin/sh
# Загрузка моделей YOLOX (Apache-2.0, Megvii) в ./models.
# В изолированной сети: положите файлы в Nexus (raw-репозиторий) и укажите MODELS_URL.
#   MODELS_URL=http://nexus.local/repository/raw/yolox sh tools/get_models.sh
set -eu
URL="${MODELS_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0}"
DIR="$(dirname "$0")/../models"
mkdir -p "$DIR"
for m in ${MODELS:-yolox_s yolox_tiny}; do
  if [ -s "$DIR/$m.onnx" ]; then echo "$m.onnx уже есть"; continue; fi
  echo "загрузка $m.onnx"
  curl -fSL --retry 3 -o "$DIR/$m.onnx.part" "$URL/$m.onnx"
  mv "$DIR/$m.onnx.part" "$DIR/$m.onnx"
done
ls -la "$DIR"
cat <<TXT

Дальше (рекомендуется для Xeon с AVX-512 VNNI) — INT8 на кадрах вашего архива:
  docker compose run --rm worker-detect quantize /models/yolox_s.onnx /models/yolox_s_int8.xml 300
и в .env: DETECT_MODEL=/models/yolox_s_int8.xml
TXT
