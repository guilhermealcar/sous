import os
from ultralytics import YOLO

def main():
    # Caminho absoluto para o data.yaml
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yaml_path = os.path.join(project_root, "data", "yolo", "dataset.yaml")
    runs_dir = os.path.join(project_root, "runs", "detect")

    print("Inicializando o treinamento otimizado na GPU...")

    # Instancia a arquitetura Small para equilibrio entre velocidade e acuracia
    model = YOLO("yolov8s.pt")

    # Hiperparametros de treino e Data Augmentation
    model.train(
        data=yaml_path,
        epochs=300,            # Aumentado significativamente
        patience=50,           # EARLY STOPPING mais tolerante
        imgsz=640,
        device=0,              
        batch=16,              
        
        # --- DATA AUGMENTATION MAIS AGRESSIVO ---
        degrees=15.0,          # Rotacao maior
        translate=0.2,         # Translacao para simular diferentes recortes
        scale=0.5,             # Escala (zoom in/out) vital para lixos de diferentes tamanhos
        fliplr=0.5,            # Espelhamento horizontal
        hsv_s=0.5,             # Alteracao de saturacao
        hsv_v=0.4,             # Alteracao de iluminacao
        mosaic=1.0,            # Mosaico ligado em 100% (vital para pequenos objetos)
        mixup=0.1,             # Leve mixup para forcar a rede a ver sobreposicoes
        
        # --- OTIMIZADORES ---
        optimizer='auto',      # Deixa o YOLO escolher o melhor (geralmente AdamW)
        lr0=0.01,              # Taxa de aprendizado inicial padrao
        weight_decay=0.0005,   # Penalidade para evitar overfitting
        
        project=runs_dir,
        name="sous_v2_augmented"
    )

    print("Pipeline de treinamento finalizado.")

if __name__ == "__main__":
    main()