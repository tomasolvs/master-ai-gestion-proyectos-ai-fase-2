from app.model import train_model
if __name__ == "__main__":
    metrics = train_model()
    print("Modelo entrenado y guardado.")
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Matriz de confusión:", metrics["confusion_matrix"])
