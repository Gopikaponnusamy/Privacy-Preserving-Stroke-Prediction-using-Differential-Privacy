import matplotlib.pyplot as plt

def plot_results(baseline_acc, dp_results):

    eps = [x[0] for x in dp_results]
    acc = [x[1] for x in dp_results]

    fig, ax = plt.subplots(figsize=(4,2))   # 👈 SMALL SIZE

    ax.plot(eps, acc, marker='o')
    ax.axhline(y=baseline_acc, linestyle='--')

    ax.set_xlabel("Epsilon", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.tick_params(labelsize=8)

    return fig


def plot_before_after(before, after):

    fig, ax = plt.subplots(figsize=(3,2))   # 👈 EVEN SMALLER

    ax.bar(["Before", "After"], [before, after])

    ax.tick_params(labelsize=8)

    return fig