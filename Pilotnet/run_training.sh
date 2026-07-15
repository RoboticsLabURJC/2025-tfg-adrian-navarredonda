#!/bin/bash

# Carpeta base donde están los datos
DATASET_BASE="../datasets"
TEST_BASE="../datasets/test"
VALID_BASE="../datasets/validation"

# Script de entrenamiento
TRAIN_SCRIPT="train_final.py"

# Otros parámetros
EXPERIMENT_NAME="exp_debug_$(date +%s)"

# Construir lista de --data_dir
DATA_DIRS=()
echo "Buscando datasets de TRAIN..."
for dir in "$DATASET_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Train: $dir"
    DATA_DIRS+=(--data_dir "$dir")
  fi
done

# Construir lista de --test_dir 
TEST_DIRS=()
echo "Buscando datasets de TEST..."
for dir in "$TEST_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Test: $dir"
    TEST_DIRS+=(--test_dir "$dir")
  fi
done

# Construir lista de --val_dir
VALID_DIRS=()
echo "Buscando datasets de VALIDACIÓN..."
for dir in "$VALID_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Val: $dir"
    VALID_DIRS+=(--val_dir "$dir")
  fi
done

# Comprobaciones mínimas
if [ ${#DATA_DIRS[@]} -eq 0 ]; then
  echo "No se encontraron carpetas de TRAIN en $DATASET_BASE/Deepracer_BaseMap_*"
  exit 1
fi

echo ""
echo "Iniciando entrenamiento con:"
echo "   Epochs     : 100"
echo "   Batch size : 128"
echo "   LR         : 3e-4"
echo "   Experimento: $EXPERIMENT_NAME"
echo ""

# Ejecutar entrenamiento 
python "$TRAIN_SCRIPT" \
  "${DATA_DIRS[@]}" \
  "${TEST_DIRS[@]}" \
  "${VALID_DIRS[@]}" \
  --num_epochs 100 \
  --batch_size 128 \
  --lr 3e-4 \
  --base_dir "$EXPERIMENT_NAME" \
  --comment "Sin augs · Con shuffle · sin mirror" \
  --print_terminal
