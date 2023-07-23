import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm


def burgers_movie(u, out, start_x, end_x, dx, t, dt, v, file_path):
    x = np.arange(start_x, end_x, dx)
    fig, ax = plt.subplots()
    im_list = []
    time_template = 'time = %.2fs'
    
    for i in tqdm(range(int(t / dt))):
        im_1, = ax.plot(x, u[i], label='Ground Truth', color='blue', animated=True)
        im_2, = ax.plot(x, out[i], label='Prediction', color='red', animated=True)
        time_text = ax.text(0.8, 0.05, time_template % (i * dt), transform=ax.transAxes)
        if i == 0:
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title('Burgers Equation mu=%f' % v)
        im_list.append([im_1, im_2, time_text])

    ani = animation.ArtistAnimation(fig, im_list, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(file_path, writer=writer)


def burgers_heatmap(u, out, start_x, end_x, dx, t, dt, v, file_path, cmap='viridis'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    heatmap_1 = ax1.imshow(u, cmap, origin='lower', aspect='auto')
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    old_xticks = ax1.get_xticks()[1:-1]
    new_xticks = np.linspace(start_x, end_x, len(old_xticks))
    new_xticks = np.round(new_xticks, 2)
    ax1.set_xticks(old_xticks, new_xticks)
    old_yticks = ax1.get_yticks()[1:-1]
    new_yticks = np.linspace(0, t, len(old_yticks))
    new_yticks = np.round(new_yticks, 3)
    ax1.set_yticks(old_yticks, new_yticks)
    fig.colorbar(heatmap_1, ax=ax1)
    
    heatmap_2 = ax2.imshow(out, cmap, origin='lower', aspect='auto')
    ax2.set_title('Prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    old_xticks = ax2.get_xticks()[1:-1]
    new_xticks = np.linspace(start_x, end_x, len(old_xticks))
    new_xticks = np.round(new_xticks, 2)
    ax2.set_xticks(old_xticks, new_xticks)
    old_yticks = ax2.get_yticks()[1:-1]
    new_yticks = np.linspace(0, t, len(old_yticks))
    new_yticks = np.round(new_yticks, 3)
    ax2.set_yticks(old_yticks, new_yticks)
    fig.colorbar(heatmap_2, ax=ax2)
    
    error = np.abs(u - out)
    heatmap_3 = ax3.imshow(error, cmap, origin='lower', aspect='auto')
    ax3.set_title('Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    old_xticks = ax3.get_xticks()[1:-1]
    new_xticks = np.linspace(start_x, end_x, len(old_xticks))
    new_xticks = np.round(new_xticks, 2)
    ax3.set_xticks(old_xticks, new_xticks)
    old_yticks = ax3.get_yticks()[1:-1]
    new_yticks = np.linspace(0, t, len(old_yticks))
    ax3.set_yticks(old_yticks, new_yticks)
    fig.colorbar(heatmap_3, ax=ax3)
    
    fig.suptitle('Burgers Equation nu=%f' % v)
    
    plt.savefig(file_path)
    plt.show()
