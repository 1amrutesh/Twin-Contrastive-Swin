import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer_v2 import SwinTransformerV2


class CustomSwinTransformerV2(nn.Module):
    def __init__(self, variant, num_classes, img_size, window_size, pretrained=True):
        super().__init__()
        self.model = self.load_swin_transformer_v2_model(
            variant, num_classes, img_size, window_size, pretrained
        )
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    @staticmethod
    def load_swin_transformer_v2_model(
        variant, num_classes, img_size, window_size, pretrained
    ):
        model = SwinTransformerV2(
            img_size=img_size,
            num_classes=num_classes,
            window_size=window_size,
        )

        model = timm.create_model(
            variant,
            pretrained=pretrained,
            model=model,
        )

        return model

    def forward(self, x):
        return self.model(x)


class ModelFactory:
    @staticmethod
    def create_model(args):
        model_dict = {
            "swin-l": {"backbone": CustomSwinTransformerV2, "hidden_dim": 192},
        }
        if args.model not in model_dict:
            raise NotImplementedError
        model_info = model_dict[args.model]
        backbone = model_info["backbone"](
            args.variant, args.num_classes, args.img_size, args.window_size
        )
        hidden_dim = model_info["hidden_dim"]

        return backbone, hidden_dim


class Network(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(num_classes),
            nn.SiLU(),
            nn.Linear(num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.SiLU(),
            nn.Linear(num_classes, self.cluster_num),
        )
        trunc_normal_(self.cluster_projector[2].weight, std=0.02)
        trunc_normal_(self.cluster_projector[5].weight, std=0.02)

    def forward(self, x_i, x_j, return_ci=True):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_j = self.cluster_projector(h_j)

        if return_ci:
            c_i = self.cluster_projector(h_i)
            return z_i, z_j, c_i, c_j
        else:
            return z_i, z_j, c_j

    def forward_c(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return c

    def forward_zc(self, x):
        h = self.backbone(x)
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return z, c


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="swin-l", help="Model type: swin-l"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="swinv2_large_window12_192_22k",
        help="Swin Transformer V2 variant",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of output clusters"
    )
    parser.add_argument("--img_size", type=int, default=192, help="Image size")
    parser.add_argument(
        "--window_size",
        type=int,
        default=12,
        help="Window size for Swin Transformer V2",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1000,
        help="Feature dimension for instance projector",
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=2,
        help="Number of classes for cluster projector",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backbone, hidden_dim = ModelFactory.create_model(args)
    model = Network(backbone, args.num_classes, args.feature_dim, args.class_num)


if __name__ == "__main__":
    main()
