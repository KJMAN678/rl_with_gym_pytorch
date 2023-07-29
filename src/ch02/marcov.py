import numpy as np


def main():
    P = np.array([[0.3, 0.7], [0.2, 0.8]])
    print("Transition Matrix:\n", P)

    S = np.array([0.5, 0.5])

    for i in range(10):
        S = np.dot(S, P)
        print(f"\nIter {i}. Probability vector S = {S}")

    print(f"\nFinal Vector S={S}")


if __name__ == "__main__":
    main()
