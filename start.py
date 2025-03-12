import numpy as np
import pandas as pd
from preprocess import preprocess
from staking_model import StackingModel
from models import kernel_ridge_regression, kernel_logistic_regression

def main():

    # Training step 
    X_train, labels, labels_onehot = preprocess("data/Xtr0.csv", "data/Xtr1.csv", "data/Xtr2.csv", labels_1="data/Ytr0.csv", labels_2="data/Ytr1.csv", labels_3="data/Ytr2.csv")

    # Training of the STAKING MODEL 
    base_models = [kernel_ridge_regression(lambda_=0.3, kernel='spectrum', num_classes=2, k=7),
                kernel_logistic_regression(alpha0_coeff = 0, lambda_ = 10, k=3, kernel='spectrum', n_iter=100)]
    meta_model = kernel_ridge_regression(lambda_=0.1, kernel='RBF', num_classes=2, sigma= 0.1, meta_mode=True)

    model = StackingModel(base_models, meta_model)

    model.fit(X_train, labels, labels_onehot)


    # Training of the KERNEL RIDGE REGRESSION MODEL WITH SPECTRUM KERNEL
    #model = kernel_ridge_regression(lambda_=0.3, kernel='spectrum', num_classes=2, k=7)
    #model.fit(X_train, labels_onehot)

    # Testing step
    X_test = preprocess("data/Xte0.csv", "data/Xte1.csv", "data/Xte2.csv", train=False)

    # Make prediction
    Y_pred = model.predict(X_test)
    N = len(Y_pred)

    df = pd.DataFrame({
    "Id": np.arange(0, N, 1),
    "Bound": Y_pred 
    })

    # Save in a CSV file
    df.to_csv("data/Yte.csv", index=False)

    print("Predictions saved in Yte.csv")


if __name__ == "__main__":
    main()
