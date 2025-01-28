import torch, numpy as np
from torch import nn, autocast
from torch.distributions import Categorical

import tetris
from ev_var import kEvMatrix, kDevMatrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
kBoardShape, kMetaShape, kMovesShape, kMoveMetaShape, _ = tetris.Tetris.StateShapes()
kR = 1 if tetris.Tetris.IsNoro() else 4
kH, kW = kBoardShape[1:]

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)


class InitialEmbed(nn.Module):
    def __init__(self, feats, meta_feats, channels):
        super().__init__()
        self.embed_1 = nn.Conv2d(feats, channels, 5, padding=2)
        self.embed_2 = nn.Conv2d(feats, channels, (kH, 1))
        self.embed_3 = nn.Conv2d(feats, channels, (1, kW))
        self.meta = nn.Linear(meta_feats, channels)
        self.finish = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

    @autocast(device_type=device)
    def forward(self, obs, meta):
        x_meta = self.meta(meta)
        x = self.embed_1(obs) + self.embed_2(obs) + self.embed_3(obs) + x_meta.view(*x_meta.shape, 1, 1)
        return self.finish(x)


class EvDev(nn.Module):
    def __init__(self, in_feat, ev_rank, dev_rank):
        super().__init__()
        self.linear_ev = nn.Linear(in_feat, ev_rank)
        self.linear_dev = nn.Linear(in_feat, dev_rank)
        self.ev_mat = torch.nn.Parameter(nn.Linear(ev_rank, 215).weight)
        self.dev_mat = torch.nn.Parameter(nn.Linear(dev_rank, 215).weight)
        self.ev_mat.requires_grad = True
        self.dev_mat.requires_grad = True

    @autocast(device_type=device, enabled=False)
    def evdev_coeff(self, obs):
        obs = obs.float()
        return self.linear_ev(obs), self.linear_dev(obs)

    @autocast(device_type=device, enabled=False)
    def forward(self, obs, idx):
        obs = obs.float()
        idx = torch.clamp(idx, max=214)
        ev = (self.linear_ev(obs) * self.ev_mat[idx]).sum(axis=1)
        dev = (self.linear_dev(obs) * self.dev_mat[idx]).sum(axis=1)
        return torch.stack([ev, dev])


class PiValueHead(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.linear = nn.Linear(in_feat, 1)

    @autocast(device_type=device, enabled=False)
    def forward(self, pi, value, invalid, multiplier):
        pi = pi.float()
        pi[invalid] = -float('inf')
        v = self.linear(value.float())
        v = v.transpose(0, 1) * multiplier
        return pi, v

class Model(nn.Module):
    def __init__(self, start_blocks, end_blocks, channels):
        super().__init__()
        self.board_embed = InitialEmbed(kBoardShape[0], kMetaShape[0], channels)
        self.moves_embed = InitialEmbed(kMovesShape[0], kMoveMetaShape[0], channels)
        self.main_start = nn.Sequential(*[ConvBlock(channels) for i in range(start_blocks)])
        self.main_end = nn.Sequential(*[ConvBlock(channels) for i in range(end_blocks)])
        self.pi_logits_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(8 * kH * kW, kR * kH * kW)
        )
        self.evdev_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(2 * kH * kW, 512),
            nn.ReLU(True),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(1 * kH * kW, 256),
            nn.ReLU(True),
        )
        self.evdev_final = EvDev(512, 48, 32)
        self.pi_value_final = PiValueHead(256)

    @autocast(device_type=device)
    def evdev_coeff(self, board, board_meta):
        batch = board.shape[0]
        x = self.board_embed(board, board_meta)
        x = self.main_start(x)
        x = self.evdev_head(x)
        return self.evdev_final.evdev_coeff(x)

    @autocast(device_type=device)
    def forward(self, obs, categorical=False, pi_only=False, evdev_only=False):
        assert not (pi_only and evdev_only)
        board, board_meta, moves, moves_meta, meta_int = obs
        batch = board.shape[0]
        pi = None
        v = torch.zeros((1, batch), dtype=torch.float32, device=board.device)
        evdev = torch.zeros((2, batch), dtype=torch.float32, device=board.device)

        invalid = moves[:,2:2+kR].view(batch, -1) == 0
        x = self.board_embed(board, board_meta)
        x = self.main_start(x)
        if kR == 1:
            x = x + self.moves_embed(moves, moves_meta)
            x = self.main_end(x)
            if not pi_only:
                evdev = self.evdev_final(self.evdev_head(x), meta_int[:,0].type(torch.LongTensor))
        else:
            if not pi_only:
                evdev = self.evdev_final(self.evdev_head(x), meta_int[:,0].type(torch.LongTensor))
            if not evdev_only:
                x = x + self.moves_embed(moves, moves_meta)
                x = self.main_end(x)
        if not evdev_only:
            pi, v = self.pi_value_final(
                self.pi_logits_head(x),
                self.value_head(x),
                invalid,
                torch.exp(moves_meta[:,-1]) if kR == 1 else 1
            )
            if categorical: pi = Categorical(logits=pi)
        return pi, torch.concat([v, evdev])

def obs_to_torch(obs, device=None):
    if device is None:
        # local
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(obs[0].shape) == 3:
        return [torch.tensor(i, device=device).unsqueeze(0) for i in obs]
    else:
        return [torch.tensor(i, device=device) for i in obs]
