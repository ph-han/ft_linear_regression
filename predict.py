import numpy as np
from train import estimate_price, PRICE_NORMALIZE, KM_NORMALIZE

def predict(km, theta0=0, theta1=0):
    return estimate_price(theta0, theta1, km / KM_NORMALIZE) * PRICE_NORMALIZE

if __name__ == "__main__":
    maleize = np.float32(input("Input your car maleize : "))
    theta0, theta1 = np.load('linear_model_theta.npy')
    print(f"expect price : {predict(maleize, theta0, theta1)}")