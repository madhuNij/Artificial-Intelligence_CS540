from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    mean_var = np.mean(x, axis = 0)
    centered_dataset = x - mean_var
    return centered_dataset

def get_covariance(dataset):
    # Your implementation goes here!
    x = np.array(dataset)
    dataset_transpose = np.transpose(x)
    covariance_dataset = np.dot(dataset_transpose, x)
    S = covariance_dataset * (1/(len(x) - 1))
    return S

def get_eig(S, m):
    # Your implementation goes here!
    Lambda, U = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    idx = Lambda.argsort()[::-1]
    eigenValues = Lambda[idx]
    eigenVectors = U[:, idx]
    l = np.diag(eigenValues)
    return l, eigenVectors

def get_eig_prop(S, prop):
    # Your implementation goes here!
    Lambda, U = eigh(S)
    idx = Lambda.argsort()[::-1]
    eigenValues = Lambda[idx]
    eigenVectors = U[:, idx]
    l = np.diag(eigenValues)
    return l, eigenVectors

def project_image(image, U):
    # Your implementation goes here!
    m = len(U[0])
    x_pca = 0
    eigenvector_transpose = np.transpose(U)
    alpha = np.dot(eigenvector_transpose, image)
    for j in range(0, m):
        x_pca = x_pca + alpha[j] * eigenvector_transpose[j]
    return x_pca


def display_image(orig, proj):
    # Your implementation goes here!
    orig_img = np.transpose(np.reshape(orig, (32, 32)))
    proj_img = np.transpose(np.reshape(proj, (32, 32)))
    #fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    pos1 = ax1.imshow(orig_img, aspect = 'equal')
    pos2 = ax2.imshow(proj_img, aspect = 'equal')
    fig.colorbar(pos1, ax=ax1)
    fig.colorbar(pos2, ax=ax2)
    plt.show()

