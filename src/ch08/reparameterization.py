import matplotlib.pyplot as plt
import numpy as np


def main():
    Ns = [10, 100, 1000, 10000, 100000]
    runs = 10000

    theta = 2.0

    grad_reinforce = lambda x: np.sum((x - theta) * np.square(x)) / x.size
    grad_reparam = lambda eps: np.sum(2 * (theta + eps)) / eps.size

    mean_reinforce = np.zeros_like(Ns, dtype=np.float32)
    var_reinforce = np.zeros_like(Ns, dtype=np.float32)

    mean_reparam = np.zeros_like(Ns, dtype=np.float32)
    var_reparam = np.zeros_like(Ns, dtype=np.float32)

    runs_reinforce = np.zeros(runs)
    runs_reparam = np.zeros(runs)

    for i, N in enumerate(Ns):
        for r in range(runs):
            x = np.random.default_rng().normal(theta, 1, N)
            runs_reinforce[r] = grad_reinforce(x)
            eps = np.random.default_rng().normal(0, 1, N)
            runs_reparam[r] = grad_reparam(eps)

        mean_reinforce[i] = np.mean(runs_reinforce)
        var_reinforce[i] = np.var(runs_reinforce)

        mean_reparam[i] = np.mean(runs_reparam)
        var_reparam[i] = np.var(runs_reparam)

    print(f"mean_reinforce: {mean_reinforce}")
    print(f"nmean_reparam: {mean_reparam}")
    print(f"nratio of means: {mean_reinforce / mean_reparam}")
    print(f"var_reinforce: {var_reinforce}")
    print(f"nvar_reparam: {var_reparam}")
    print(f"nratio of variances: {var_reinforce / var_reparam}")

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(mean_reinforce, label="Reinforce")
    ax.plot(mean_reparam, label="Reparam")

    plt.title("Gradient Est(mean)")
    ax.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(var_reinforce, label="Reinforce")
    ax.plot(var_reparam, label="Reparam")

    plt.title("Gradient Est(variance)")
    ax.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == "__main__":
    main()
