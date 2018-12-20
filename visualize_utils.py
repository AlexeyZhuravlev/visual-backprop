from matplotlib import pyplot as plt
import numpy as np
from visual_backprop import VisualBackpropWrapper
import torch
import torch.nn.functional as F
import os


def visualize_images_with_masks(images, masks, text, figsize=(16, 8), columns=5):
    fig=plt.figure(figsize=figsize)
    fig.suptitle(text, fontsize=14, fontweight='bold')
    columns = 3
    rows = (images.shape[0] + columns - 1) // columns
    
    red_mask = np.zeros((images.shape[0], images.shape[2], images.shape[3], 4))
    masks_np = masks.detach().cpu().numpy()
    red_mask[:,:,:,0] = np.squeeze(masks_np, 1)
    red_mask[:,:,:,3] = red_mask[:,:,:,0]
    
    for i in range(images.shape[0]):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.imshow(red_mask[i] / np.max(red_mask[i]))


def train_with_logging(model, device, train_loader, optimizer, total_num_iterations, \
                       log_freq, display_samples, save_dir='./results/'):
    model.train()
    num_iterations = 0
    while(num_iterations < total_num_iterations):
        for _, (data, target) in enumerate(train_loader):
            if num_iterations >= total_num_iterations:
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output), target)
            loss.backward()
            optimizer.step()
            if num_iterations % log_freq == 0:
                with torch.no_grad():
                    visual_backprop = VisualBackpropWrapper(model, device)
                    display_masks = visual_backprop.get_masks_for_batch(display_samples)
                    visualize_images_with_masks(display_samples, display_masks, "Iteration %d" % num_iterations)
                    plt.savefig(os.path.join(save_dir, "%05d.jpg" % (num_iterations)))
                    plt.close()
            num_iterations += 1
