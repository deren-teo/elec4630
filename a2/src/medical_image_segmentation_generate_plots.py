import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a2"
A2_ROOT = Path(__file__).parent.parent.resolve()

# FPS of original video, for correct time axis in plot
FPS = 29.97

### ENTRYPOINT #################################################################

def main():

    # Load the data output by the morphology and Viterbi methods
    load_fp = Path(A2_ROOT, 'output', 'cardiac_mri')
    data_morph = np.loadtxt(Path(load_fp, 'ventricle_area_morph.txt'))
    data_viter = np.loadtxt(Path(load_fp, 'ventricle_area_viterbi.txt'))
    assert data_morph.shape == data_viter.shape

    # Configure matplotlib plot style
    plt.rc('font', family='serif', size=10)
    plt.rc('text', usetex=1)
    palette = sns.color_palette('mako', n_colors=5)

    # Plot the area inside the inner wall of the left ventricle over time
    fig, ax = plt.subplots(figsize=(8, 2))

    t = np.linspace(start=0, stop=len(data_morph) / FPS, num=len(data_morph))
    sns.lineplot(x=t, y=data_morph, ax=ax, color=palette[1], label='Morphology')
    sns.lineplot(x=t, y=data_viter, ax=ax, color=palette[3], label='Viterbi')

    ax.set_title('Left ventricle area')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Area (pixels)')

    # Save the output figure
    save_fp = Path(A2_ROOT, 'output', 'cardiac_mri', 'plot_ventricle_area.png')
    fig.savefig(save_fp, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()