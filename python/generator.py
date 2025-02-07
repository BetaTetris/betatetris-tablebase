import math, traceback, secrets
import numpy as np, torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe
from torch.distributions import Categorical
from model import Model, obs_to_torch, kH, kW, kR

import tetris
from game import Worker

class DataGenerator:
    def __init__(self, name, model, c, game_params):
        self.model = model
        self.n_workers = c.n_workers
        self.env_per_worker = c.env_per_worker
        self.envs = c.n_workers * c.env_per_worker
        self.worker_steps = c.worker_steps
        self.gamma, self.lamda = game_params[-2:]
        self.device = next(self.model.parameters()).device
        self.total_games = 0
        self.total_long_games = 0
        self.game_params = None
        self.recv_weight_interval = c.worker_steps // c.weight_sync_per_epoch
        self.use_kl = c.use_kl

        # initialize tensors for observations
        shapes = [*[(self.envs, *i) for i in tetris.Tetris.StateShapes()],
                  (self.envs, c.worker_steps, 2),
                  (self.envs, c.worker_steps, 2)]
        types = [*[np.dtype(i) for i in tetris.Tetris.StateTypes()],
                 np.dtype('float32'),
                 np.dtype('bool')]
        self.shms = [
            shared_memory.SharedMemory(create = True, size = math.prod(shape) * typ.itemsize)
            for shape, typ in zip(shapes, types)
        ]
        np_shms = [
            np.ndarray(shape, dtype = typ, buffer = shm.buf)
            for shm, shape, typ in zip(self.shms, shapes, types)
        ]
        self.obs_np = np_shms[:-2]
        self.rewards, self.is_over = np_shms[-2:]
        # create workers
        shm = [(shm.name, shape, typ) for shm, shape, typ in zip(self.shms, shapes, types)]
        seed = secrets.randbelow(2**40)
        self.workers = [
            Worker(name, shm, self.w_range(i), seed + i, c.board_file)
            for i in range(self.n_workers)
        ]

        self.set_params(game_params)
        for i in self.workers: i.child.send(('reset', None))
        for i in self.workers: i.child.recv()

        self.obs = obs_to_torch(self.obs_np, self.device)

    def w_range(self, x): return slice(x * self.env_per_worker, (x + 1) * self.env_per_worker)

    def update_model(self, state_dict, epoch=-1):
        target_device = next(self.model.parameters()).device
        for i in state_dict:
            if state_dict[i].device != target_device:
                state_dict[i] = state_dict[i].to(target_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def set_params(self, game_params):
        if self.game_params != game_params[2:]:
            for i in self.workers:
                i.child.send(('set_param', game_params[2:]))
            self.game_params = game_params[2:]
        self.gamma, self.lamda = game_params[:2]

    def sample(self, train = True, gpu = False, epoch = 0, remote = None):
        """### Sample data with current policy"""
        actions = torch.zeros((self.worker_steps, self.envs), dtype=torch.int32, device=self.device)
        obs = [
            torch.zeros((self.worker_steps, self.envs, *shape),
                        dtype=torch.int32 if typ[:3] == 'int' else torch.float32,
                        device=self.device)
            for shape, typ in zip(tetris.Tetris.StateShapes(), tetris.Tetris.StateTypes())
        ]
        if self.use_kl:
            pi_logits = torch.zeros((self.worker_steps, self.envs, kR*kH*kW), dtype = torch.float32, device = self.device)
        log_pis = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
        values = torch.zeros((self.worker_steps, 2, self.envs), dtype = torch.float32, device = self.device)
        devs = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)

        if train:
            ret_info = {
                'reward': [],
                'scorek': [],
                'lns': [],
                'pcs': [],
                'maxk': [],
                'mil_games': [],
                'long_games': [],
                'short_finish': [],
                'perline': [],
            }
        else:
            ret_info = {}

        # sample `worker_steps` from each worker
        tot_lines = 0
        tot_score = 0
        for t in range(self.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                for i in range(len(obs)):
                    obs[i][t] = self.obs[i]
                # sample actions from $\pi_{\theta_{OLD}}$
                pi, v = self.model(self.obs, categorical=True)
                values[t] = v[:2] # remove stdev
                devs[t] = v[2]
                a = pi.sample()
                if self.use_kl: pi_logits[t] = pi.logits
                actions[t] = a
                log_pis[t] = pi.log_prob(a)
                actions_cpu = a.cpu().numpy()

            # run sampled actions on each worker
            # workers will place results in self.obs_np,rewards,is_over
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', (t, actions_cpu[self.w_range(w)], epoch)))
            for i in self.workers:
                info_arr = i.child.recv()
                # collect episode info, which is available if an episode finished
                if train:
                    self.total_games += len(info_arr)
                    for info in info_arr:
                        if info['is_short']:
                            ret_info['short_finish'].append(0.0 if info['is_over'] else 1.0)
                        else:
                            self.total_long_games += 1
                            tot_lines += info['lines']
                            tot_score += info['score']
                            ret_info['reward'].append(info['reward'])
                            ret_info['scorek'].append(info['score'] * 1e-3)
                            ret_info['lns'].append(info['lines'])
                            ret_info['pcs'].append(info['pieces'])
            self.obs = obs_to_torch(self.obs_np, self.device)

            if remote and (t + 1) % self.recv_weight_interval == 0:
                cmd, data = remote.recv()
                assert cmd == 'update_model'
                self.update_model(data[0], epoch=epoch)

        # reshape rewards & log rewards
        score_max = self.rewards[:,:,1].max()
        if train:
            ret_info['maxk'].append(score_max / 1e-2)
            ret_info['mil_games'].append(self.total_games * 1e-6)
            ret_info['long_games'].append(self.total_long_games * 1e-6)
            if tot_lines > 0:
                ret_info['perline'].append(tot_score * 1e-3 / tot_lines)

            # calculate advantages
            advantages, raw_devs, skip_mask = self._calc_advantages(self.is_over, self.rewards, values, devs)
            values_t = values.transpose(0, 1)
            advantages_t = advantages.transpose(0, 1)
            samples = {
                'obs': obs,
                'actions': actions,
                'log_pis': log_pis,
                'skip_mask': skip_mask,
                'values': values_t[0],
                'advantages': advantages_t[0],
                'raw_devs': raw_devs,
                'raw_values': values_t[1] + advantages_t[1],
            }
            if self.use_kl: samples['pi_logits'] = pi_logits
            #print('raw_devs', samples['raw_devs'].flatten())
            #print('raw_values', samples['raw_values'].flatten())
            #print('skip_mask', samples['skip_mask'].flatten())
        else:
            samples = {
                'obs': obs,
                'log_pis': log_pis,
                'values': values_t[0],
                'raw_values': values_t[1] + advantages_t[1],
            }

        # samples are currently in [time, workers] table, flatten it
        for i in samples:
            if i == 'obs':
                for j in range(len(samples[i])):
                    samples[i][j] = samples[i][j].reshape(-1, *samples[i][j].shape[2:])
            else:
                samples[i] = samples[i].reshape(-1, *samples[i].shape[2:])

        if not gpu:
            for i in samples:
                if i == 'obs':
                    for j in range(len(samples[i])):
                        samples[i][j] = samples[i][j].cpu()
                else:
                    samples[i] = samples[i].cpu()
        for i in list(ret_info):
            if ret_info[i]:
                ret_info[i] = np.mean(ret_info[i])
            else:
                del ret_info[i]
        return samples, ret_info

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: torch.Tensor, devs: torch.Tensor) -> torch.Tensor:
        """### Calculate advantages"""
        with torch.no_grad():
            # values: (t, 2, env)
            # devs: (t, env)
            # (env, t, 2) -> (t, 2, env)
            rewards = torch.permute(torch.from_numpy(rewards).to(self.device), (1, 2, 0))
            # (env, t, 2) -> (2, t, env)
            done_torch = torch.permute(torch.from_numpy(done).to(self.device), (2, 1, 0))
            #print('done', done_torch.view(2, -1))
            #print('rewards', rewards.view(-1, 2).transpose(0, 1))
            #print('values', values.view(-1, 2).transpose(0, 1))
            #print('devs', devs.flatten())
            done_neg = ~done_torch[0]
            soft_done = done_torch[1]

            # advantages table
            advantages = torch.zeros((self.worker_steps, 2, self.envs), dtype = torch.float32, device = self.device)
            raw_devs = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
            last_advantage = torch.zeros((2, self.envs), dtype = torch.float32, device = self.device)

            # $V(s_{t+1})$
            last_value = self.model(self.obs)[1]
            last_dev = last_value[2]
            last_value = last_value[:2] # remove stdev
            gammas = torch.Tensor([self.gamma, 1.0]).unsqueeze(1).to(self.device)
            lamdas = torch.Tensor([self.lamda, 1.0]).unsqueeze(1).to(self.device)

            for t in reversed(range(self.worker_steps)):
                done_mask = done_neg[t]
                soft_done_mask = soft_done[t]
                last_dev *= done_mask
                # last_value = last_value * done_mask
                # last_advantage = last_advantage * done_mask
                # $\delta_t = reward[t] - value[t] + last_value * gammas$
                # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}} (gam * lam * last_advantage)$
                last_advantage = rewards[t] - values[t] + gammas * (last_value + lamdas * last_advantage) * done_mask
                # note that we are collecting in reverse order.
                advantages[t] = last_advantage
                raw_devs[t] = last_dev
                last_value = values[t]
                # soft dones
                last_advantage[:,soft_done_mask] = 0
                last_dev[soft_done_mask] = devs[t,soft_done_mask]
            return advantages, raw_devs, done_torch[1]

    def destroy(self):
        try:
            for i in self.workers: i.child.send(('close', None))
        except: pass
        for i in self.shms:
            i.close()
            i.unlink()

def generator_process(remote, name, c, game_params, device):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        model = Model(*c.model_args()).to(device, memory_format=torch.channels_last)
        model = torch.compile(model)
        generator = DataGenerator(name, model, c, game_params)
        samples = None
        while True:
            cmd, data = remote.recv()
            if cmd == "update_model":
                generator.update_model(data[0], epoch=data[1])
            elif cmd == "set_param":
                generator.set_params(data)
            elif cmd == "start_generate":
                samples = generator.sample(remote=remote if data[0] else None, epoch=data[1])
            elif cmd == "get_data":
                remote.send(samples)
                samples = None
            elif cmd == "close":
                return
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        remote.close()

class GeneratorProcess:
    def __init__(self, model, *args):
        # args: name, c, game_params, device
        self.device = args[-1]
        self.child, parent = Pipe()
        ctx = torch.multiprocessing.get_context('spawn')
        self.process = ctx.Process(target=generator_process, args=(parent, *args))
        self.process.start()
        self.SendModel(model)

    def SendModel(self, model, epoch=-1):
        state_dict = model.state_dict()
        for i in state_dict:
            state_dict[i] = state_dict[i].cpu()
        self.child.send(('update_model', (state_dict, epoch)))

    def StartGenerate(self, epoch, update=False):
        self.child.send(('start_generate', (update, epoch)))

    def SetParams(self, game_params):
        self.child.send(('set_param', game_params))

    def GetData(self, device):
        self.child.send(('get_data', None))
        data, info = self.child.recv()
        for i in data:
            if i == 'obs':
                for j in range(len(data[i])):
                    data[i][j] = data[i][j].to(device)
            else:
                data[i] = data[i].to(device)
        return data, info

    def Close(self):
        self.child.send(('close', None))
