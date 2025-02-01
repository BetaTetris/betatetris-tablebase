import hashlib, traceback, os.path, random
from typing import Optional
from multiprocessing import shared_memory
import numpy as np
import torch
from torch.multiprocessing import Process, Pipe
from game_param import GameParamManager

import tetris

class Game:
    def __init__(self, seed: int, manager=None):
        self.args = (0, False)
        self.env = tetris.Tetris(seed)
        self.prev_soft_done = False
        self.manager = manager
        self.params = None

    def step(self, action, manager=None):
        r, x, y = action // 200, action // 10 % 20, action % 10
        reward = np.array(self.env.InputPlacement(r, x, y))
        self.reward += reward[0]

        info = None
        is_over = np.array([False, False])
        is_over[1] = self.prev_soft_done
        is_over[0] = is_over[1] or self.env.IsOver()
        num_lines = self.env.GetRunLines()
        is_short = self.params.get('is_short') == True
        self.prev_soft_done = is_short and num_lines >= 6 and self.env.GetBoard().IsClean()
        if is_over[0]:
            num_pieces = self.env.GetRunPieces()
            info = {'is_short': is_short,
                    'is_over': self.env.IsOver(),
                    'reward': self.reward,
                    'score': self.env.GetRunScore(),
                    'lines': num_lines,
                    'pieces': num_pieces}
            if self.manager:
                self.manager.UpdateState(self.params, num_pieces, num_lines)
            self.reset()
        return self.env.GetState(), reward, is_over, info

    def reset(self):
        self.reward = 0.
        self.prev_soft_done = False

        if self.manager:
            self.params = self.manager.GetNewParam()
            n_param = {i: j for i, j in self.params.items() if i not in ['tap_id', 'adj_delay_id', 'is_short']}
            self.env.Reset(**n_param)
        else:
            self.env.ResetRandom()
        return self.env.GetState()

def worker_process(remote, name: str, shms: list, idx: slice, seed: int, board_file: Optional[str]):
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shms_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    shm_obs = tuple(shms_np[:-2])
    shm_reward, shm_over = tuple(shms_np[-2:])

    # create game environments
    num = idx.stop - idx.start
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 8, 'little') + int.to_bytes(x, 8, 'little')).digest(), 'little')
    random.seed(Seed(12345))
    manager = GameParamManager(board_file)
    games = [Game(Seed(i), manager) for i in range(num)]
    # wait for instructions from the connection and execute them
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, actions, epoch = data
                result = []
                for i in range(num):
                    result.append(games[i].step(actions[i], manager))
                obs, reward, over, info = zip(*result)
                obs = tuple(zip(*obs))
                for i in range(len(obs)):
                    shm_obs[i][idx] = np.stack(obs[i])
                shm_reward[idx,step] = np.stack(reward)
                shm_over[idx,step] = np.stack(over)
                info = list(filter(lambda x: x is not None, info))
                remote.send(info)
            elif cmd == "reset":
                obs = [games[i].reset() for i in range(num)]
                obs = tuple(zip(*obs))
                for i in range(len(obs)):
                    shm_obs[i][idx] = np.stack(obs[i])
                remote.send(0)
            elif cmd == "close":
                # remote will be closed on finally
                return
            elif cmd == "set_param":
                params = data
                manager.UpdateParams(params)
                # for i in games:
                #     i.pre_trans = pre_trans
                #     pass
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        remote.close()
        for i in shms: i[0].close()

class Worker:
    """Creates a new worker and runs it in a separate process."""
    def __init__(self, *args):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process, args=(parent, *args))
        self.process.start()
