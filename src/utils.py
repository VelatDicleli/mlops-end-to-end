from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_models(models: dict, X_train, y_train, X_test, y_test, verbose=True):
    """
    Hem eğitim hem test verisi üzerinden modelleri değerlendirir.

    Parametreler:
        models (dict): {'model_adi': model_instance, ...}
        X_train, y_train: Eğitim verisi
        X_test, y_test: Test verisi
        verbose (bool): True ise çıktıları yazdırır

    Dönüş:
        dict: {
            'model_adi': {
                'train': {'mae': ..., 'rmse': ..., ...},
                'test': {'mae': ..., 'rmse': ..., ...}
            }, ...
        }
    """
    results = {}

    for name, model in models.items():
        # Train predictions
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        results[name] = {
            'train': {
                'mae': train_mae,
                'mse': train_mse,
                'rmse': train_rmse,
                'r2': train_r2
            },
            'test': {
                'mae': test_mae,
                'mse': test_mse,
                'rmse': test_rmse,
                'r2': test_r2
            }
        }

        if verbose:
            print(f"\nModel: {name}")
            print("  [TRAIN]")
            print(f"    MAE : {train_mae:.4f}")
            print(f"    RMSE: {train_rmse:.4f}")
            print(f"    R²  : {train_r2:.4f}")
            print("  [TEST]")
            print(f"    MAE : {test_mae:.4f}")
            print(f"    RMSE: {test_rmse:.4f}")
            print(f"    R²  : {test_r2:.4f}")

    return results
