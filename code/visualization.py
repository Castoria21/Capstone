import matplotlib.pyplot as plt
plt.style.use("default")


def plot_solo_cum(ret, log_scale=True):
    plt.plot((1+ret).cumprod())
    plt.xticks(ret.index[::50], rotation=45)
    if log_scale:
        plt.yscale('log')
    plt.show()


def plot_group_cum(portfolio_df, log_scale=True):
    for i in portfolio_df:
        plt.plot((1+portfolio_df[i]).cumprod(), label=f"{i} Portfolio")
    plt.legend()
    plt.xticks(portfolio_df.index[::50], rotation=45)
    if log_scale:
        plt.yscale("log")
    plt.show()
