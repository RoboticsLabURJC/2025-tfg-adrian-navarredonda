import optuna
import json
from ultralytics import YOLO

DATA = "data.v4i.yolov11/data.yaml"
MODEL = "yolo11n.pt"
EPOCHS = 25

def objective(trial):

    # ----- Hiperparámetros a optimizar -----
    params = {
        # TRAINING
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
        "lrf": trial.suggest_float("lrf", 0.1, 1.0),
        "momentum": trial.suggest_float("momentum", 0.7, 0.98),
        "weight_decay": trial.suggest_float("weight_decay", 0.00001, 0.001, log=True),

        # AUGMENTATION
        "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
        "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
        "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),

        "degrees": trial.suggest_float("degrees", 0, 20),
        "translate": trial.suggest_float("translate", 0.0, 0.3),
        "scale": trial.suggest_float("scale", 0.2, 0.9),
        "shear": trial.suggest_float("shear", 0.0, 5.0),

        "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
    }

    # ----- Entrenamiento -----
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=640,
        **params
    )

    # ----- Métrica objetivo: mAP50-95 -----
    _, _, _, map5095 = results.mean_results()
    return map5095


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    # Guardar hiperparámetros en un archivo
    with open("best_hyperparameters.json", "x") as f:
        json.dump(study.best_params, f, indent=4)

    # Guardar métricas si quieres
    with open("best_map.txt", "x") as f:
        f.write(f"Best mAP50-95: {study.best_value}")

    print("Best hyperparameters:", study.best_params)
    print("Best mAP:", study.best_value)
