import torch.nn as nn
import torch


class Cross_fusion_CNN(nn.Module):
    # Re-implemented Cross_fusion_CNN for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, input_channels2, n_classes):
        super(Cross_fusion_CNN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
                                     padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For image a (7×7×d)
        self.conv1_a = nn.Conv2d(input_channels,
                                 filters[0],
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_a = nn.Conv2d(filters[1],
                                 filters[2],
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        # For image b (7×7×d)
        self.conv1_b = nn.Conv2d(input_channels2,
                                 filters[0],
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_b = nn.Conv2d(filters[1],
                                 filters[2],
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # for image a
        # print(x1.shape)
        # print(x1.shape)
        x1 = x1[:,0,:,:,:]
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))

        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))

        x11 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x11 = self.max_pool(x11)
        x22 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x22 = self.max_pool(x22)
        x12 = self.activation(self.bn4_b(self.conv4_b(x1)))
        x12 = self.max_pool(x12)
        x21 = self.activation(self.bn4_a(self.conv4_a(x2)))
        x21 = self.max_pool(x21)

        joint_encoder_layer1 = torch.cat([x11 + x21, x22 + x12], 1)
        joint_encoder_layer2 = torch.cat([x11, x12], 1)
        joint_encoder_layer3 = torch.cat([x22, x21], 1)

        fusion1 = self.activation(self.bn5(self.conv5(joint_encoder_layer1)))
        fusion1 = self.activation(self.bn6(self.conv6(fusion1)))
        fusion1 = self.avg_pool(fusion1)
        fusion1 = self.conv7(fusion1)

        fusion2 = self.activation(self.bn5(self.conv5(joint_encoder_layer2)))
        fusion2 = self.activation(self.bn6(self.conv6(fusion2)))
        fusion2 = self.avg_pool(fusion2)
        fusion2 = self.conv7(fusion2)

        fusion3 = self.activation(self.bn5(self.conv5(joint_encoder_layer3)))
        fusion3 = self.activation(self.bn6(self.conv6(fusion3)))
        fusion3 = self.avg_pool(fusion3)
        fusion3 = self.conv7(fusion3)
        # print(fusion1.shape)
        # fusion1 = torch.squeeze(fusion1)  # For fully convolutional NN
        # fusion2 = torch.squeeze(fusion2)  # For fully convolutional NN
        # fusion3 = torch.squeeze(fusion3)  # For fully convolutional NN
        fusion1 = fusion1.view(fusion1.shape[0], fusion1.shape[1])
        fusion2 = fusion2.view(fusion2.shape[0], fusion2.shape[1])
        fusion3 = fusion3.view(fusion3.shape[0], fusion3.shape[1])
        # print(fusion1.shape)
        return (fusion1, fusion2, fusion3)


class Cross_fusion_CNN_Loss(nn.Module):
    # Loss function for paper "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    def __init__(self):
        super(Cross_fusion_CNN_Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, target):
        output1, output2, output3 = output
        loss1 = self.ce(output1, target)
        loss2 = torch.pow(output1 - output2, 2).mean()
        loss3 = torch.pow(output1 - output3, 2).mean()

        return loss1 + loss2 + loss3