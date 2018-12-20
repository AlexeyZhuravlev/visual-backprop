
import torch
import torch.nn.functional as F

class VisualBackpropWrapper():
    """
        Wraps the model and gives ability to extract importance masks
        for feature layers in convolutional networks.
        Implements idea from article: https://arxiv.org/pdf/1611.05418.pdf
        Wrapped model must have nn.Sequential fileld named "features" with all features layers
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.initialize_modules()

    """
        Initializes self.modules and its params from net structure
    """
    def initialize_modules(self):
        self.modules = []
        for _, module in self.model.features._modules.items():
            transformation_params = None
            if 'Conv2d' in str(module):
                transformation_params = (module.kernel_size, module.stride, module.padding)
            if 'Pool' in str(module):
                transformation_params = ((module.kernel_size, module.kernel_size), \
                                        (module.stride, module.stride), \
                                        (module.padding, module.padding))
            self.modules.append((module, transformation_params))

    """
        Return relevancy masks for batch of images computed by VisualBackprop
    """
    def get_masks_for_batch(self, x_batch):
        all_outputs = []
        x_batch = x_batch.to(self.device)
        for module, transformation_params in self.modules:
            x_batch = module(x_batch)
            averaged_maps = torch.mean(x_batch, 1, True)
            flatten_maps = averaged_maps.view(averaged_maps.size(0), -1)
            maps_max = torch.max(flatten_maps, 1, True)[0].unsqueeze(2).unsqueeze(3)
            maps_min = torch.min(flatten_maps, 1, True)[0].unsqueeze(2).unsqueeze(3)
            averaged_maps = (averaged_maps - maps_min) / (maps_max - maps_min + 1e-6)
            if transformation_params:
                all_outputs.append((averaged_maps, transformation_params))

        last_result = None
        for maps, (kernel_size, stride, padding) in reversed(all_outputs):
            if last_result is None:
                last_result = maps
            else:
                last_result = last_result.mul(maps)
            last_result = F.conv_transpose2d(last_result, torch.ones(1, 1, kernel_size[0], kernel_size[1], device=self.device),\
                                            stride=stride, output_padding=padding)
        return F.relu(last_result)
