import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
import math


def km_flow_movie(u, out, fdm, start_x, end_x, start_y, end_y, dx, dy, dt, file_path, cmap='viridis'):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    im_list = []
    time_template = 'time = %.2fs'
    
    fdm_0 = fdm[:, :, :1]
    fdm_t = fdm[:, :, -1:]
    fdm = np.concatenate([fdm_0, fdm, fdm_t], axis=-1)
     
    for i in tqdm(range(u.shape[-1])):
        error_i = np.abs(u[:, :, i] - out[:, :, i])
        im_1 = ax1.imshow(u[:, :, i], cmap='viridis', origin='lower', aspect='auto', animated=True)
        im_2 = ax2.imshow(out[:, :, i], cmap='viridis', origin='lower', aspect='auto', animated=True)
        im_3 = ax3.imshow(error_i, cmap='viridis', origin='lower', aspect='auto', animated=True)
        im_4 = ax4.imshow(fdm[:, :, i], cmap='viridis', origin='lower', aspect='auto', animated=True)
        time_text = ax1.text(0.75, 0.05, time_template % (i * dt), transform=ax1.transAxes)
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
            
            ax3.set_title('Data Error')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_xticks(old_xticks, new_xticks)
            ax3.set_yticks(old_yticks, new_yticks)
            
            ax4.set_title('Equation Error (FDM)')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            ax4.set_xticks(old_xticks, new_xticks)
            ax4.set_yticks(old_yticks, new_yticks)
            
        im_list.append([im_1, im_2, im_3, im_4, time_text])

    ani = animation.ArtistAnimation(fig, im_list, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(file_path, writer=writer)


def km_flow_heatmap(u, out, fdm, start_x, end_x, start_y, end_y, dx, dy, dt, file_path, cmap='viridis'):
    nt = u.shape[-1]
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    fdm_0 = fdm[:, :, :1]
    fdm_t = fdm[:, :, -1:]
    fdm = np.concatenate([fdm_0, fdm, fdm_t], axis=-1)
    
    for i in range(3):
        index_i = math.floor(i / 2 * (nt - 1))
        u_i = u[:, :, index_i]
        out_i = out[:, :, index_i]
        error_i = np.abs(u_i - out_i)
        fdm_i = np.abs(fdm[:, :, index_i])
    
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
        axes[i, 1].set_xticks(old_xticks, new_xticks)
        axes[i, 1].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_2, ax=axes[i, 1])
        
        heatmap_3 = axes[i, 2].imshow(error_i, cmap, origin='lower', aspect='auto')
        axes[i, 2].set_title('Data Error')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        axes[i, 2].set_xticks(old_xticks, new_xticks)
        axes[i, 2].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_3, ax=axes[i, 2])
        
        heatmap_4 = axes[i, 3].imshow(fdm_i, cmap, origin='lower', aspect='auto')
        axes[i, 3].set_title('Equation Error (FDM)')
        axes[i, 3].set_xlabel('x')
        axes[i, 3].set_ylabel('y')
        axes[i, 3].set_xticks(old_xticks, new_xticks)
        axes[i, 3].set_yticks(old_yticks, new_yticks)
        fig.colorbar(heatmap_4, ax=axes[i, 3])
        
    plt.savefig(file_path)
    plt.show()
