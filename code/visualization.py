import matplotlib.pyplot as plt
plt.style.use("default")


def plot_solo_cum(ret, title=None, log_scale=True):
    plt.figure(figsize=(12,8))
    plt.plot((1+ret).cumprod())
    plt.xticks(ret.index[::50], rotation=45)
    if log_scale:
        plt.yscale('log')
    if title != None:
        plt.title(f"{title}")
    plt.show()


def plot_group_cum(portfolio_df, feature_name, log_scale=True):
    labels = [f"Low {feature_name}"] + [f"{i+1}" for i in portfolio_df.columns[1:-1]] + [f"High {feature_name}"]
    plt.figure(figsize=(12,8))
    for i in portfolio_df:
        plt.plot((1+portfolio_df[i]).cumprod(), label=f"{labels[i]}")
    plt.legend()
    plt.xticks(portfolio_df.index[::50], rotation=45)
    plt.title(f"{feature_name} Sorted Portfolio")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    if log_scale:
        plt.yscale("log")
    plt.show()
