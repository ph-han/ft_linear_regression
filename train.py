import csv
import numpy as np
import matplotlib.pyplot as plt

KM_NORMALIZE = 10000.0
PRICE_NORMALIZE = 1000.0

def read_dataset(filename):
    dataset = list()

    raw_dataset = open(filename, 'r')
    raw_dataset = csv.reader(raw_dataset, delimiter=',')
    next(raw_dataset)
    for data in raw_dataset:
        dataset.append(data)

    return np.array(dataset, np.float32)

def visualization(dataset, theta0, theta1):
    plt.scatter(dataset[:, 0] / KM_NORMALIZE, (dataset[:, 1]) / PRICE_NORMALIZE)
    x_line = np.linspace(2, 25, 100)
    y_line = estimate_price(theta0, theta1, x_line)
    plt.plot(x_line, y_line, color='red', label='regression line')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.show()

    return dataset

def estimate_price(t0, t1, km):
    return t0 + t1 * km

def gradient_theta0(predict, real):
    return np.sum(predict - real)

def gradient_theta1(predict, real, kms):
    return np.sum((predict - real) * kms)

def loss(predict, real):
    return np.sum((predict - real) ** 2 / 2)

def train(dataset, lr, step):
    kms = dataset[:, 0] / KM_NORMALIZE
    prices = dataset[:, 1] / PRICE_NORMALIZE
    size = len(kms)

    theta0, theta1 = 0.0, 0.0

    for i in range(step):
        predict = estimate_price(theta0, theta1, kms)
        tmp0 = lr * gradient_theta0(predict, prices) / size
        tmp1 = lr * gradient_theta1(predict, prices, kms) / size

        theta0 -= tmp0
        theta1 -= tmp1

        if i % 100 == 0 or i == step - 1:
            print(f"[{i}] loss: {loss(predict, prices):.4f}")

    return theta0, theta1

if __name__ == "__main__":
    dataset = read_dataset("data.csv")
    theta0, theta1 = train(dataset, 0.01, 10000)
    visualization(dataset, theta0, theta1)
    np.save('linear_model_theta.npy', [theta0, theta1])