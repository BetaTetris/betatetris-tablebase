#!/usr/bin/env python3

import re
import argparse

import numpy as np
import torch
import onnx
from torch.distributions import Categorical, kl_divergence

from model import Model, obs_to_torch
from tetris import Tetris
from game_param import TAP_SEQUENCE_MAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('output')
parser.add_argument('--no-export', action='store_true')
args = parser.parse_args()

tetris = Tetris()
tetris.Reset(adj_delay=18, tap_sequence=TAP_SEQUENCE_MAP['30hz'].tolist())
state = tetris.GetAdjStates(0, 6, 5)
torch_state = obs_to_torch(state)

with torch.no_grad():
    state_dict = torch.load(args.model, weights_only=True, map_location=device)
    channels = state_dict['main_start.0.main.0.weight'].shape[0]
    start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
    end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])

    model = Model(start_blocks, end_blocks, channels).to(device)
    model.load_state_dict(state_dict)
    model = torch.compile(model)
    model.eval()
    orig_value = model(torch_state)

    if not args.no_export:
        onnx_program = torch.onnx.dynamo_export(model, torch_state, export_options=torch.onnx.ExportOptions(dynamic_shapes=True))
        onnx_program.save(args.output)

from onnxruntime import InferenceSession

print('Verifying consistency')
sess = InferenceSession(args.output)
onnx_value = sess.run(None, {
    f'l_args_0_{x}_': state[x] for x in range(len(state))
})
pi_orig = Categorical(logits=orig_value[0])
pi_onnx = Categorical(logits=torch.Tensor(onnx_value[0]).to(device))
pi_diff = np.nan_to_num(np.abs(orig_value[0].cpu().numpy() - onnx_value[0]))
value_diff = np.abs(orig_value[1].cpu().numpy() - onnx_value[1])
print(f'pi diff max: {pi_diff.max()}, rms: {np.sqrt(np.mean(np.square(pi_diff)))}, kl_div: {kl_divergence(pi_orig, pi_onnx).mean().item()}')
print(f'value diff max: {value_diff.max(axis=1)}, rms: {np.sqrt(np.mean(np.square(value_diff), axis=1))}')
