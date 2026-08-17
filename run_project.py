from src.eda import run_eda
from src.train import main as train_main
from src.explain import export_feature_importance

if __name__ == "__main__":
    print("=== EDA ===")
    print(run_eda())

    print("\n=== MODEL TRAINING ===")
    train_main()

    print("\n=== EXPLAINABILITY ===")
    print(export_feature_importance())

    print("\nComplete ML pipeline finished.")
