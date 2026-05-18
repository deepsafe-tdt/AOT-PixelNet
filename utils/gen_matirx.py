import os

import numpy as np
from scipy.linalg import hadamard, toeplitz
from math import sqrt, cos, sin
np.random.seed(0)
def gen_matrix( transform_type, N,):
    if transform_type == 'HT':
        """哈达玛变换核

        """
        assert (N & (N - 1) == 0) and N != 0, "N must be power of 2"
        H = hadamard(N)
        print(H)
        return  H / sqrt(N)
    elif transform_type == 'WT':
        """ Walsh变换核"""
        assert (N & (N - 1) == 0) and N != 0, "N must be power of 2"
        H = hadamard(N) / sqrt(N)
        diffMat = np.abs(np.diff(H)) / 2
        invTimes = np.sum(diffMat, axis=1)
        W = H[invTimes.argsort(), :]
        print(W)
        return W
    elif transform_type == 'DHT':
        Matrix = np.zeros((N, N), dtype=float)
        for row in range(N):
            for col in range(N):
                Matrix[row, col] = sqrt(1 / N) * (cos(2 * np.pi * col * row / N) + sin(2 * np.pi * col * row / N))
        return Matrix



    elif transform_type == 'ST':
        """ Slant变换核
        """
        assert N % 2 == 0, "N must be even"
        arr = generate_sn(N)
        return arr

    elif transform_type == 'normal':
        "生成0-1正态分布的矩阵"
        arr = np.random.normal(0, 1, (N, N))
        print("返回μ=0, σ=1的正态分布矩阵normal：")
        print(arr)
        return arr

    elif transform_type == 'uniform':
        " 返回一个在区间[low, high]中均匀分布的数组"
        arr = np.random.uniform(-1, 1, (N, N))
        print("返回一个在区间[-1, 1]中均匀分布的数组")
        return arr

    elif transform_type == 'orthogonal':
        H = np.random.randn(N, N)
        Q, R = np.linalg.qr(H)
        print("Orthogonal Matrix:")
        print(Q)
        return Q
    elif transform_type == 'binary':
        arr = np.random.randint(2, size=(N, N))
        print("Binary (0-1) Matrix:")
        print(arr)
        return arr

    elif transform_type == 'TOE':
        c = np.random.rand(N)  # First column
        r = np.random.rand(N)  # First row
        arr = toeplitz(c, r)
        print("Toeplitz Matrix:")
        print(arr)
        return arr

    elif transform_type == 'random':
        arr = np.random.rand(N, N)
        print("Random Matrix:")
        print(arr)
        return arr

    else:
        Matrix = np.zeros((N, N), dtype=float)
        return Matrix





def generate_sn(N):
    """
    生成 N x N 的变换矩阵 S_n，其中 N = 2^n。

    参数：
        n: int, 表示矩阵的阶数 N = 2^n。

    返回：
        S_n: ndarray, 生成的变换矩阵。
    """
    a_n = np.sqrt((3 * N ** 2) / (4 * (N ** 2 - 1)))

    b_n = np.sqrt((N**2 - 4) / (4 * (N**2 - 1)))

    if N == 2:
        # 基础矩阵 S_1
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    elif N == 4:
        top_left = np.array([[1, 0], [a_n, b_n]])
        top_right = np.array([[1, 0], [-a_n, b_n]])
        bottom_left = np.array([[0, 1], [-b_n, a_n]])
        bottom_right = np.array([[0, 1], [b_n, a_n]])
        S_n = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        S_half = np.block([[generate_sn(N // 2),np.zeros(( 2, 2))], [np.zeros((2, 2)), generate_sn(N // 2)]])
        return (1 / np.sqrt(2)) * (S_n @ S_half)
    else:
        I_N_half = np.eye(N // 2 - 2)
        top_left = np.block([[np.array([[1, 0], [a_n, b_n]]),np.zeros(( 2, N // 2 -2))],
                             [np.zeros((N // 2 -2, 2)), I_N_half]])
        top_right = np.block([[np.array([[1, 0], [-a_n, b_n]]), np.zeros(( 2, N // 2 -2))],
                             [np.zeros((N // 2 -2, 2)), I_N_half]])
        bottom_left = np.block([[np.array([[0, 1], [-b_n, a_n]]), np.zeros(( 2, N // 2 -2))],
                             [np.zeros((N // 2 -2, 2)), I_N_half]])
        bottom_right = np.block([[np.array([[0, 1], [b_n, a_n]]), np.zeros(( 2, N // 2 -2))],
                             [np.zeros((N // 2 -2, 2)), -I_N_half]])
        S_n = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        S_half = np.block([[generate_sn(N // 2), np.zeros((N //2, N //2))], [np.zeros((N //2,N //2)), generate_sn(N // 2)]])
        return (1 / np.sqrt(2)) * (S_n @ S_half)


# def save_matrix_to_txt(matrix, file_path):
#     """
#         Save a matrix to a .txt file in a custom array-like format with commas.
#
#         Parameters:
#             matrix (numpy.ndarray): The matrix to save.
#             file_path (str): The full path (including filename) where the matrix will be saved.
#         """
#     try:
#         # Convert the matrix to a custom string representation
#         # Add commas between elements in each row
#         matrix_str = "[\n" + ",\n".join("[" + ", ".join(map(str, row)) + "]" for row in matrix.tolist()) + "\n]"
#
#         # Save the string to a text file
#         with open(file_path, 'w') as f:
#             f.write(matrix_str)
#         print(f"Matrix successfully saved to {file_path}")
#     except Exception as e:
#         print(f"Error saving matrix to file: {e}")
def save_matrix_to_txt(matrix, file_path):
    """
    Save a matrix to a .txt file in a custom array-like format with 8 decimal places.

    Parameters:
        matrix (numpy.ndarray): The matrix to save.
        file_path (str): The full path (including filename) where the matrix will be saved.
    """
    try:
        # Convert the matrix to a custom string representation with 8 decimal places
        matrix_str = "[\n" + ",\n".join(
            "[" + ", ".join(f"{x:.8f}" for x in row) + "]" for row in matrix.tolist()
        ) + "\n]"

        # Save the string to a text file
        with open(file_path, 'w') as f:
            f.write(matrix_str)
        print(f"Matrix successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving matrix to file: {e}")



if __name__ == "__main__":
    N = 8
    type = "WT"
    transform_matrix = gen_matrix( type,N)
    print(transform_matrix)
