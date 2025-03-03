#!/usr/bin/env python3

import argparse, sys, queue, csv, re, random, math, socket, os
import numpy as np, torch
from torch.distributions import Categorical
from multiprocessing import Process, Pipe, Queue, shared_memory

import tetris

from model import Model, obs_to_torch
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = None

class RNGGym:
    def __init__(self, seed=0):
        self.reset(seed)

    def reset(self, seed):
        self.prev = -1
        self.seed = (seed >> 8) & 0xffff
        self.cnt = seed & 0xff
        self.iters = self.cnt >> 4
        if self.iters == 0: self.iters = 16
        self.iters += 3

    def advance(self):
        bit = ((self.seed >> 1) ^ (self.seed >> 9)) & 1
        self.seed = (self.seed >> 1) | (bit << 15)

    def spawn(self):
        for i in range(self.iters): self.advance()
        self.cnt = (self.cnt + 1) & 255
        ind = ((self.seed >> 8) + self.cnt) & 7
        if ind == 7 or ind == self.prev:
            self.advance()
            ind = (((self.seed >> 8) & 7) + [2,7,8,10,11,14,18,0][self.prev]) % 7
        self.prev = ind
        return ind

class RNGNormal:
    def __init__(self, seed=0):
        self.transition_matrix = [
            [1, 5, 6, 5, 5, 5, 5],
            [6, 1, 5, 5, 5, 5, 5],
            [5, 6, 1, 5, 5, 5, 5],
            [5, 5, 5, 2, 5, 5, 5],
            [5, 5, 5, 5, 2, 5, 5],
            [6, 5, 5, 5, 5, 1, 5],
            [5, 5, 5, 5, 6, 5, 1],
        ]
        self.reset(seed)

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.prev = self.rng.randrange(7)

    def spawn(self):
        self.prev = self.rng.choices(range(7), weights=self.transition_matrix[self.prev])[0]
        return self.prev

class RNGRealistic:
    def __init__(self, seed=0):
        self.transition_matrices = [[
            [0, 2, 4, 2, 2, 3, 3],
            [4, 0, 2, 3, 3, 2, 2],
            [2, 3, 1, 2, 3, 2, 3],
            [2, 3, 2, 1, 3, 2, 3],
            [3, 2, 2, 3, 1, 2, 3],
            [3, 2, 3, 3, 2, 0, 3],
            [4, 2, 2, 2, 4, 2, 0],
        ], [
            [0, 2, 4, 2, 2, 4, 2],
            [4, 0, 2, 3, 3, 2, 2],
            [2, 4, 0, 2, 3, 3, 2],
            [2, 3, 2, 1, 3, 2, 3],
            [3, 2, 3, 2, 1, 3, 2],
            [3, 2, 3, 3, 2, 0, 3],
            [3, 2, 2, 3, 3, 2, 1],
        ], [
            [0, 3, 3, 2, 3, 3, 2],
            [3, 0, 3, 3, 2, 2, 3],
            [3, 3, 0, 3, 2, 3, 2],
            [3, 2, 3, 1, 2, 3, 2],
            [2, 2, 3, 3, 0, 3, 3],
            [2, 3, 3, 2, 2, 1, 3],
            [2, 2, 2, 4, 2, 2, 2],
        ], [
            [0, 4, 2, 2, 4, 2, 2],
            [3, 0, 3, 3, 2, 2, 3],
            [3, 3, 0, 3, 3, 2, 2],
            [3, 2, 3, 1, 2, 3, 2],
            [2, 3, 2, 3, 1, 2, 3],
            [2, 3, 3, 2, 2, 1, 3],
            [2, 2, 3, 3, 2, 3, 1],
        ], [
            [1, 3, 2, 3, 3, 2, 2],
            [2, 1, 3, 2, 2, 3, 3],
            [2, 3, 1, 2, 3, 2, 3],
            [2, 3, 2, 1, 3, 2, 3],
            [2, 3, 3, 2, 1, 3, 2],
            [3, 3, 2, 2, 3, 1, 2],
            [2, 2, 4, 2, 2, 4, 0],
        ], [
            [2, 2, 2, 4, 2, 2, 2],
            [2, 1, 3, 2, 2, 3, 3],
            [3, 2, 1, 3, 2, 2, 3],
            [2, 3, 2, 1, 3, 2, 3],
            [3, 2, 3, 2, 1, 3, 2],
            [3, 3, 2, 2, 3, 1, 2],
            [2, 3, 3, 2, 3, 3, 0],
        ], [
            [1, 2, 3, 3, 2, 2, 3],
            [3, 1, 2, 2, 3, 3, 2],
            [3, 3, 0, 3, 2, 3, 2],
            [3, 2, 3, 1, 2, 3, 2],
            [3, 3, 2, 2, 2, 2, 2],
            [4, 2, 2, 3, 3, 0, 2],
            [2, 4, 2, 2, 4, 2, 0],
        ], [
            [0, 2, 4, 2, 2, 2, 4],
            [3, 1, 2, 2, 3, 3, 2],
            [2, 3, 1, 2, 2, 3, 3],
            [3, 2, 3, 1, 2, 3, 2],
            [2, 3, 2, 3, 1, 2, 3],
            [4, 2, 2, 3, 3, 0, 2],
            [3, 3, 2, 2, 4, 2, 0],
        ]]
        self.reset(seed)

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.cnt = self.rng.randrange(8)
        self.prev = -1

    def spawn(self):
        self.cnt = (self.cnt + 1) % 8
        if self.prev == -1:
            seed = self.rng.randrange(16)
            ind = (seed + self.cnt) & 7
            if ind == 7:
                ind = (seed >> 1) % 7
            self.prev = ind
        else:
            self.prev = self.rng.choices(range(7), weights=self.transition_matrices[self.cnt][self.prev])[0]
        return self.prev


TAP_SEQUENCES = {
    '30hz': np.arange(10) * 2,
    '24hz': np.array([0, 3, 5, 8, 10, 13, 15, 18, 20, 23]),
    '20hz': np.arange(10) * 3,
    '15hz': np.arange(10) * 4,
    '12hz': np.arange(10) * 5,
    '10hz': np.arange(10) * 6,
    'slow5': np.array([0, 2, 4, 6, 18, 20, 22, 24, 36, 38]),
}
ADJ_DELAYS = [0, 18, 21, 24, 30, 61]
AGGRESSION_LEVELS = {
    'high': 0,
    'mid': 1,
    'low': 2,
}


class Game:
    def __init__(self):
        self.env = tetris.Tetris()
        self.rng = RNGGym(0) if args.gym_rng else (RNGRealistic(0) if args.realistic_rng else RNGNormal(0))
        self.reset(0)

    def step(self, action, direct=False):
        LEVELS = [130, 230, 330, 430]
        old_lines = self.env.GetLines()
        old_pieces = self.env.GetPieces()
        old_run_lines = self.env.GetRunLines()
        r, x, y = action // 200, action // 10 % 20, action % 10
        if direct:
            self.env.DirectPlacement(r, x, y)
        else:
            self.env.InputPlacement(r, x, y)
        lines = self.env.GetLines()
        score = self.env.GetRunScore()
        for i in range(4):
            if old_lines < LEVELS[i] <= lines: self.stats[i] = score
        if args.detailed_history and self.env.GetPieces() != old_pieces:
            self.history += '{}{}{}{}'.format(r, '0123456789abcdefghij'[x], y, lines - old_lines)
        elif args.line_clear_history and lines > old_lines:
            self.history += str(lines - old_lines)
        if self.env.IsOver():
            for i in range(4):
                if LEVELS[i] > lines: self.stats[i] = score
            return False
        if args.keep_lines is not None and lines >= args.keep_lines + 2:
            self.env.SetLines(lines - 2)
        if self.env.GetPieces() != old_pieces:
            self.env.SetNextPiece(self.rng.spawn())
        return True

    def get_stats(self):
        lst = [self.seed] + self.stats + [self.env.GetLines()]
        if args.keep_lines is not None:
            lst.append(self.env.GetRunLines())
        if args.line_clear_history or args.detailed_history:
            lst.append(self.history)
        return lst

    def reset(self, seed, board=None, now=None):
        self.seed = seed
        self.history = ''
        self.rng.reset(seed)
        if now is None: now = self.rng.spawn()
        nxt = self.rng.spawn()
        reset_args = {}
        if tetris.Tetris.IsNoro():
            reset_args['start_level'] = args.start_level
            reset_args['do_tuck'] = not args.no_tuck
            reset_args['nnb'] = args.nnb
            reset_args['mirror'] = args.mirror
        else:
            reset_args['skip_unique_initial'] = True
            reset_args['lines'] = args.start_lines
            reset_args['adj_delay'] = args.adj_delay
            reset_args['tap_sequence'] = TAP_SEQUENCES[args.tap_speed].tolist()
            reset_args['step_reward_level'] = AGGRESSION_LEVELS[args.aggression]
            if board:
                reset_args['board'] = board
        self.env.Reset(now, nxt, **reset_args)
        self.stats = [0, 0, 0, 0]

def worker_process(remote, q_size, offset, seed_queue, shms):
    if args.server:
        host, port = args.server.split(':')
        port = int(port)
        board_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            board_conn.connect((host, port))
        except:
            board_conn = None
    else:
        board_conn = None

    games = [Game() for i in range(q_size)]
    status = np.ones((q_size, 3), dtype='bool') # is_running, need_nn, need_tablebase

    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    obs_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    obs_idx = slice(offset, q_size + offset)
    reach_end = False

    def Reset(idx):
        nonlocal games, status, reach_end
        if reach_end:
            status[idx] = False
            return
        seed, board, piece = seed_queue.get()
        if seed is None:
            reach_end = True
            status[idx] = False
            return
        board = tetris.Board(board) if board else None
        if board and board.Count() % 4 != 0:
            board = None
        games[idx].reset(seed, board, piece)
        if games[idx].env.IsOver():
            games[idx].reset(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == 'init':
            for i in range(q_size): Reset(i)
            obs = [i.env.GetState() for i in games]
            obs = tuple(zip(*obs))
            for i in range(len(obs)):
                obs_np[i][obs_idx] = np.stack(obs[i])
            remote.send(0)
        elif cmd == 'step':
            info = []
            for i in np.nonzero(status[:,1])[0]:
                if not games[i].step(data[i]):
                    info.append(games[i].get_stats())
                    Reset(i)
                if ((args.clean_only and not games[i].env.GetBoard().IsCleanForPerfect()) or
                        (args.max_lines and games[i].env.GetRunLines() >= args.max_lines)):
                    info.append(games[i].get_stats())
                    Reset(i)

            if board_conn:
                status[:,2] = status[:,0]
                for _ in range(2):
                    states = []
                    for i in np.nonzero(status[:,2])[0]:
                        env = games[i].env
                        states.append(
                            env.GetBoard().GetBytes() +
                            bytes([env.GetNowPiece(), env.GetLines() % 256, env.GetLines() // 256])
                        )
                    resp = []
                    for i in range(0, len(states), 255):
                        sz = min(255, len(states) - i)
                        query = bytes([sz]) + b''.join(states[i:i+sz])
                        board_conn.sendall(query)
                        result = board_conn.recv(22 * sz)
                        assert(len(result) == 22 * sz)
                        for j in range(sz):
                            resp.append(result[22*j:22*(j+1)])
                    for x, i in enumerate(np.nonzero(status[:,2])[0]):
                        offset = 3 * games[i].env.GetNextPiece()
                        r, x, y = resp[x][offset:offset+3]
                        if (r, x, y) == (0, 0, 0):
                            status[i,2] = False
                        else:
                            if not games[i].step(r*200 + x*10 + y, direct=True):
                                info.append(games[i].get_stats())
                                Reset(i)
                    total = status[:,0].sum()
                    total_table = status[:,2].sum()
                    if total_table < total * 0.7 or total - total_table >= 16: break
                status[:,1] = status[:,0] & ~status[:,2]
            else:
                status[:,1] = status[:,0]

            need_nn = status[:,1]
            obs = [games[i].env.GetState() for i in range(q_size) if need_nn[i]]
            obs = tuple(zip(*obs))
            for i in range(len(obs)):
                obs_np[i][obs_idx][need_nn] = np.stack(obs[i])
            remote.send((info, status[:,0], need_nn))
        elif cmd == "close":
            remote.close()
            return

class Worker:
    def __init__(self, *wargs):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process, args=(parent, *wargs))
        self.process.start()

@torch.no_grad()
def Main(models):
    assert args.batch_size % args.workers == 0
    q_size = args.batch_size // args.workers
    N = args.num

    seed_queue = Queue()
    shapes = [(args.batch_size, *i) for i in tetris.Tetris.StateShapes()]
    types = [np.dtype(i) for i in tetris.Tetris.StateTypes()]
    shms = [
        shared_memory.SharedMemory(create=True, size=math.prod(shape) * typ.itemsize)
        for shape, typ in zip(shapes, types)
    ]
    obs_np = [
        np.ndarray(shape, dtype=typ, buffer=shm.buf)
        for shm, shape, typ in zip(shms, shapes, types)
    ]
    shm_child = [(shm.name, shape, typ) for shm, shape, typ in zip(shms, shapes, types)]
    workers = [Worker(q_size, i * q_size, seed_queue, shm_child) for i in range(args.workers)]

    if args.board_file:
        board_set = set()
        try:
            with open(args.board_file, 'rb', buffering=1048576) as f:
                while True:
                    item = f.read(25)
                    if len(item) == 0: break
                    assert len(item) == 25
                    board_set.add(item)
        except FileNotFoundError:
            pass
        last_save = time.time()
        save_num = 0

    random.seed(args.seed)
    if args.seed_file:
        with open(args.seed_file, 'r') as f: seeds = list(map(int, f))
    else:
        seeds = random.sample(range(512, 2 ** 24) if args.gym_rng else range(2 ** 60), N)
    if args.sample_file:
        boards = []
        with open(args.sample_file, 'rb', buffering=1048576) as f:
            while True:
                item = f.read(26)
                if len(item) == 0: break
                assert len(item) == 26
                boards.append((item[:25], item[25] & 7))
        random.shuffle(boards)
        boards = boards * (N // len(boards)) + random.sample(boards, N % len(boards))
    elif args.board_file and args.start_from_board and len(board_set) > 0:
        pn = int(N * 0.8)
        boards = random.choices(list(board_set), k=pn) + [None] * (N - pn)
        random.shuffle(boards)
        boards = [(i, None) for i in boards]
    else:
        boards = [(None, None)] * N
    for i in zip(seeds, boards): seed_queue.put((i[0], *i[1]))
    for i in range(args.workers * 2): seed_queue.put((None, None, None))

    def Save(fname, to_sort=False):
        nonlocal last_save
        with open(fname, 'wb', buffering=1048576) as f:
            if to_sort:
                for i in sorted(board_set): f.write(i)
            else:
                for i in board_set: f.write(i)
        last_save = time.time()

    for i in workers: i.child.send(('init', None))
    for i in workers: i.child.recv()
    started = args.batch_size
    results = []

    try:
        info_arr = []
        old_finished = 0
        running = np.ones(args.batch_size, dtype='bool')
        need_nn = np.ones(args.batch_size, dtype='bool')
        while True:
            pi_np = np.zeros(args.batch_size, dtype='int')
            if np.any(need_nn):
                M = need_nn.sum()
                need_nn_ind = np.nonzero(need_nn)[0]
                M = len(need_nn_ind)
                if M > 64 and M % 64 != 0:
                    # do not change batch size too frequently
                    C = min(args.batch_size, (M + 63) // 64 * 64)
                    need_nn_ind = np.concatenate((need_nn_ind, np.zeros(C-M, dtype='int')))
                obs_torch = obs_to_torch([i[need_nn_ind] for i in obs_np])
                pi = torch.stack([model(obs_torch, pi_only=True)[0] for model in models]).mean(0)
                if args.sample_action:
                    pi = Categorical(logits=pi*0.75).sample()
                else:
                    pi = torch.argmax(pi, 1)
                pi_np[need_nn] = pi.view(-1).cpu().numpy()[:M]

            for i in range(args.workers):
                workers[i].child.send(('step', pi_np[i*q_size:(i+1)*q_size]))

            if args.board_file:
                boards = obs_torch[0][:,0].view(-1, 200).cpu().numpy().astype('bool')
                boards = np.packbits(boards, axis=1, bitorder='little')
                board_set.update(map(lambda x: x.tobytes(), boards))
                if time.time() - last_save >= 5400:
                    Save(args.board_file + f'.{save_num}')
                    save_num = 1 - save_num

            to_end = True
            for i in range(args.workers):
                info, worker_running, worker_need_nn = workers[i].child.recv()
                info_arr += info
                running[i*q_size:(i+1)*q_size] = worker_running
                need_nn[i*q_size:(i+1)*q_size] = worker_need_nn
            if running.sum() == 0: break

            if old_finished // 50 != len(info_arr) // 50:
                text = f'{len(info_arr)} / {N} games finished'
                if args.board_file: text += f'; {len(board_set)} boards collected'
                print(text, file=sys.stderr)
            old_finished = len(info_arr)
    finally:
        if args.board_file: Save(args.board_file)

        if args.output is None:
            writer = csv.writer(sys.stdout)
            writer.writerows(sorted(info_arr))
        else:
            with open(args.output, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(sorted(info_arr))

        for i in workers:
            i.child.send(('close', None))
            i.child.close()
        for i in shms:
            i.close()
            i.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('-n', '--num', type=int, default=2000)
    if tetris.Tetris.IsNoro():
        parser.add_argument('-l', '--start-level', type=int, default=0)
        parser.add_argument('--nnb', action='store_true')
        parser.add_argument('--no-tuck', action='store_true')
        parser.add_argument('--mirror', action='store_true')
    else:
        parser.add_argument('-l', '--start-lines', type=int, default=0)
        parser.add_argument('--adj-delay', type=int, default=18, choices=ADJ_DELAYS)
        parser.add_argument('--tap-speed', type=str, default='30hz', choices=list(TAP_SEQUENCES))
        parser.add_argument('--aggression', type=str, default='high', choices=list(AGGRESSION_LEVELS))
    parser.add_argument('-m', '--max-lines', type=int)
    parser.add_argument('--keep-lines', type=int)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('-w', '--workers', type=int, default=2)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-s', '--server', type=str)
    parser.add_argument('--line-clear-history', action='store_true')
    parser.add_argument('--detailed-history', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-file', type=str)
    parser.add_argument('--gym-rng', action='store_true')
    parser.add_argument('--realistic-rng', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--sample-action', action='store_true')
    parser.add_argument('--board-file', type = str)
    parser.add_argument('--sample-file', type = str)
    parser.add_argument('--start-from-board', action='store_true')
    parser.add_argument('--compile-model', action='store_true')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    models = []
    for model_file in args.models:
        with torch.no_grad():
            state_dict = torch.load(model_file, weights_only=True, map_location=device)
            channels = state_dict['main_start.0.main.0.weight'].shape[0]
            start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
            end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
            model = Model(start_blocks, end_blocks, channels).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            if args.compile_model:
                model = torch.compile(model)
            models.append(model)

    Main(models)
