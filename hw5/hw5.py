import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__=="__main__":
    df = pd.read_csv(sys.argv[1])

    x = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])

    plt.plot(x, y)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")

    X = []
    for i in x:
        a = [1]
        a.append(i)
        X.append(a)
    X = np.array(X)
    print("Q3a:")
    print(X)

    Y = []
    for i in y:
        Y.append(i)
    Y = np.array(Y)
    print("Q3b:")
    print(Y)

    XT = np.transpose(X)
    Z = np.dot(XT, X)
    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    PI = np.dot(I, XT)
    print("Q3e:")
    print(PI)

    hat_beta = np.dot(PI, Y)
    print("Q3f:")
    print(hat_beta)

    y_test = hat_beta[0] + hat_beta[1] * 2021
    print("Q4:", str(y_test))

    if hat_beta[1] < 0:
        print("Q5a: <")
    elif hat_beta[1] > 0:
        print("Q5a: >")
    elif hat_beta[1] == 0:
        print("Q5a: =")

    print("Q5b: The sign of beta1 is negative which means that the loss function has been minimized which maximizes the value of y. Therefore the model predicts the number of ice days in 2021 as 85 which is accurate as represented in the data.")

    x_pred = (-hat_beta[0])/hat_beta[1]
    print("Q6a:", x_pred)

    print("Q6b: The prediction made is the year 2455 which is 433 years from now when the lake stops freezing."
          " According to the data given, the number of days the lake freezes has reduced approximately by 70 days in 166 years. "
          "Looking at the trend the lake will stop freezing much earlier than 2455. Therefore x* doesnt make a very compelling prediction.")