import os
from pathlib import Path
import seaborn as sns

colors = sns.color_palette("deep")
colors_ = sns.color_palette("pastel")

C0 = colors[0]
C1 = colors[1]
C2 = colors[2]

C0_ = colors[0]
C1_ = colors[1]
C2_ = colors[2]


def init_mpl(usetex=False):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("Greys_r")
    MPLRCPATH = Path(__file__).parent / "matplotlibrc"
    os.environ["MATPLOTLIBRC"] = str(MPLRCPATH)
    import matplotlib as mpl
    config = mpl.rc_params_from_file(MPLRCPATH, fail_on_error=True)
    mpl.rcParams = config

    from matplotlib.rcsetup import cycler

    
    my_cycler = cycler(color=colors)
    mpl.rc('axes', prop_cycle=my_cycler)
    # mpl.rcParams["font.sans-serif"] = ["Open Sans"]


    # no right and top spines
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    return plt

init_mpl()

if __name__ == "__main__":
    plt = init_mpl()
    # plot a pallette of the cycler patches
    sns.color_palette("deep")