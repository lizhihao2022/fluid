import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
import math


def km_flow_movie(u, out, start_x, end_x, start_y, end_y, dx, dy, dt, file_path, cmap='viridis'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    im_list = []
    time_template = 'time = %.2fs'
    
    for i in tqdm(range(u.shape[-1])):
        error_i = np.abs(u[:, :, i] - out[:, :, i])
        im_1 = ax1.imshow(u[:, :, i], cmap='viridis', origin='lower', aspect='auto', animated=True)
        im_2 = ax2.imshow(out[:, :, i], cmap='viridis', origin='lower', aspect='auto', animated=True)
        im_3 = ax3.imshow(error_i, cmap='viridis', origin='lower', aspect='auto', animated=True)
        time_text = ax1.text(0.8, 0.05, time_template % (i * dt), transform=ax1.transAxes)
        if i == 0:
            old_xticks = ax1.get_xticks()[1:-1]
            new_xticks = np.linspace(start_x, end_x, len(old_xticks))
            new_xticks = np.round(new_xticks, 1)
            old_yticks = ax1.get_yticks()[1:-1]
            new_yticks = np.linspace(start_y, end_y, len(old_yticks))
            new_yticks = np.round(new_yticks, 1)
            
            ax1.set_title('Ground Truth')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_xticks(old_xticks, new_xticks)
            ax1.set_yticks(old_yticks, new_yticks)
            
            ax2.set_title('Prediction')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_xticks(old_xticks, new_xticks)
            ax2.set_yticks(old_yticks, new_yticks)
            
            ax3.set_title('Error')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_xticks(old_xticks, new_xticks)
            ax3.set_yticks(old_yticks, new_yticks)
            
        im_list.append([im_1, im_2, im_3, time_text])

    ani = animation.ArtistAnimation(fig, im_list, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(file_path, writer=writer)


def km_flow_heatmap(u, out, start_x, end_x, start_y, end_y, dx, dy, dt, file_path, cmap='viridis'):
    nt = u.shape[-1]
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    for i in range(3):
        index_i = math.floor(i / 2 * (nt - 1))
        u_i = u[:, :, index_i]
        out_i = out[:, :, index_i]
        error_i = np.abs(u_i - out_i)
    
        heatmap_1 = axes[i, 0].imshow(u_i, cmap, origin='lower', aspect='auto')
        axes[i, 0].set_title('Ground Truth t=%.2fs' % (index_i * dt))
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        old_xticks = axes[i, 0].get_xticks()[1:-1]
        new_xticks = np.linspace(start_x, end_x, len(old_xticks))
        new_xticks = np.round(new_xticks, 1)
        axes[i, 0].set_xticks(old_xticks, new_xticks)
        old_yticks = axes[i, 0].get_yticks()[1:-1]
        new_yticks = np.linspace(start_y, end_y, len(old_yticks))
        new_yticks = np.round(new_yticks, 1)
        axes[i, 0].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_1, ax=axes[i, 0])
        
        heatmap_2 = axes[i, 1].imshow(out_i, cmap, origin='lower', aspect='auto')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        old_xticks = axes[i, 1].get_xticks()[1:-1]
        new_xticks = np.linspace(start_x, end_x, len(old_xticks))
        new_xticks = np.round(new_xticks, 1)
        axes[i, 1].set_xticks(old_xticks, new_xticks)
        old_yticks = axes[i, 1].get_yticks()[1:-1]
        new_yticks = np.linspace(start_y, end_y, len(old_yticks))
        new_yticks = np.round(new_yticks, 1)
        axes[i, 1].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_2, ax=axes[i, 1])
        
        heatmap_3 = axes[i, 2].imshow(error_i, cmap, origin='lower', aspect='auto')
        axes[i, 2].set_title('Error')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        old_xticks = axes[i, 2].get_xticks()[1:-1]
        new_xticks = np.linspace(start_x, end_x, len(old_xticks))
        new_xticks = np.round(new_xticks, 1)
        axes[i, 2].set_xticks(old_xticks, new_xticks)
        old_yticks = axes[i, 2].get_yticks()[1:-1]
        new_yticks = np.linspace(start_y, end_y, len(old_yticks))
        new_yticks = np.round(new_yticks, 1)
        axes[i, 2].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_3, ax=axes[i, 2])
    
    plt.savefig(file_path)
    plt.show()
