import numpy as np


def find_optimal_alignment_path(y, s_prime):
    T = len(y)
   # N = len(s_prime) // 2

    # Initialize matrices
    m = np.zeros((T, len(s_prime)))
    G = {}

    # Set initial conditions
    for i in range(1, len(s_prime) + 1):
        if i in {1, 2}:
            m[0][i - 1] = y[0][i-1]
        else:
            m[0][i - 1] = 0

    # Iterative computation
    for i in range(1, len(s_prime) + 1):
        if i == 1:
            G[i] = {i}
        elif s_prime[i - 1] == 'blank' or i == 2 or s_prime[i - 1] == s_prime[i - 3]:
            G[i] = {i - 1, i}
        else:
            G[i] = {i - 2, i - 1, i}

        for t in range(1, T):
            m[t][i - 1] = \
                y[t][i - 1] * \
                max(m[t - 1][j - 1] for j in G[i])
    #print("我是G",G)
    # Backtracking
    i = np.argmax(m[T - 1, [len(s_prime) - 2, len(s_prime) - 1]]) #因为G是1开头的
   # print("我是m:",m)
    alignment_path = [i]
    i = i + 1

    for t in range(T - 1, 0, -1):
        G = [i] if i == 1 else [i - 2, i - 1, i] if s_prime[i] != 'blank' and i != 2 and s_prime[i] != s_prime[
            i - 2] else [i - 1, i]
        max_val = float('-inf')
        for j in G:
            if m[t - 1][j] > max_val:
                max_val = m[t - 1][j]
                i = j
        alignment_path.insert(0, i)

    return alignment_path

if __name__ == "__main__":
    # Example usage
   # y = [[0.75, 0.1, 0.05, 0.05, 0.05], [0.1, 0.05, 0.05, 0.05, 0.75],[0.75, 0.1, 0.05, 0.05, 0.05]]
    s_prime = ['blank', '我', 'blank', '你', 'blank', '他', 'blank', '它', 'blank', '她', 'blank']
    # 生成 (20, 5) 形状的随机数组
    y = np.random.rand(100, len(s_prime))
    # 归一化每一行，使其之和为 1
    y = y / y.sum(axis=1, keepdims=True)
   # print(y)
    alignment_path = find_optimal_alignment_path(y, s_prime)
    print("Optimal Alignment Path:", alignment_path)
    # m = np.array([[3, 4, 2, 7, 5],
    #               [1, 6, 8, 2, 9],
    #               [4, 3, 5, 1, 2],
    #               [7, 2, 9, 6, 4]])
    # print(m[4 - 1, [len(s_prime) - 2, len(s_prime) - 1]]) #[6 4]
