
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class GenerativeResnet3Headless(nn.Module):
    """
    See https://raw.githubusercontent.com/skumra/robotic-grasping/master/inference/models/grconvnet3.py
    """

    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)


        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # freeze above params
        for param in self.parameters():
            param.requires_grad = False
        
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.head = nn.Conv2d(64, 4, 1, bias=False)

    def forward(self, x_in):
        # Freeze these layers
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x_in)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
        
        x = self.res5(x)

        # 1x1 conv to reduce feature state, init with random weights
        x = self.head(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # ignore the old head which made it larger
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = self.conv6(x)

        # if self.dropout:
        #     pos_output = self.pos_output(self.dropout_pos(x))
        #     cos_output = self.cos_output(self.dropout_cos(x))
        #     sin_output = self.sin_output(self.dropout_sin(x))
        #     width_output = self.width_output(self.dropout_wid(x))
        # else:
        #     pos_output = self.pos_output(x)
        #     cos_output = self.cos_output(x)
        #     sin_output = self.sin_output(x)
        #     width_output = self.width_output(x)

        return x


class ProcessObservation(nn.Module):
    def __init__(self, res=(224, 224)):
        super().__init__()
        self.res = res

        # Load visual model
        grconvnet3_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/nets/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97.pt'
        )
        self.feature_extractor = GenerativeResnet3Headless().eval()
        self.feature_extractor.load_state_dict(state_dict=torch.load(grconvnet3_path))

        old_img_size = (res[0], res[1], 8)
        new_img_size = (res[0]//16-1, res[1]//16-1, 8)
        self.reduce_action_space = int(np.prod(old_img_size) - np.prod(new_img_size))
    
    def __call__(self, obs):
        """
        Takes in a torch array of observations, processes the images into features.

        This assumes the observations ends in 2 rgbd images with shape (224, 244, 4)
        """
        # import pdb; pdb.set_trace()
        h, w = self.res
        px = h * w
        base_rgbd = obs[:, -px * 4:].reshape((-1, h, w, 4))
        arm_rgbd = obs[:, -px * 8: - px * 4].reshape((-1, h, w, 4))
        others = obs[:,: - px * 8]
        bs = obs.shape[0]

        # make a batch
        x = torch.cat([base_rgbd, arm_rgbd], 0)
        x = x.permute((0, 3, 1, 2)) # to ((-1, 4, x, y))
        h = self.feature_extractor(x)

        # undo fake batch
        base_h, arm_h = h[:bs].reshape((bs, -1)), h[bs:].reshape((bs, -1))
        # add features together
        y = torch.cat([others, base_h, arm_h], 1)
        return y
