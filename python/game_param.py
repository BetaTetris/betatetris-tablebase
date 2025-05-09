import os
import os.path
from filelock import FileLock
import numpy as np
from scipy.special import softmax

import tetris

TAP_SEQUENCE_MAP = {
    '30hz': np.arange(10) * 2,
    '24hz': np.array([0, 3, 5, 8, 10, 13, 15, 18, 20, 23]),
    '20hz': np.arange(10) * 3,
    '15hz': np.arange(10) * 4,
    '12hz': np.arange(10) * 5,
    '10hz': np.arange(10) * 6,
    'slow5': np.array([0, 2, 4, 6, 18, 20, 22, 24, 36, 38]),
}
AGGRESSION_LEVEL_MAP = {
    'high': 0,
} if tetris.Tetris.IsTetrisOnly() else {
    'high': 0,
    'mid': 1,
    'low': 2,
}
TAP_SEQUENCES = [
    TAP_SEQUENCE_MAP['30hz'],
    TAP_SEQUENCE_MAP['24hz'],
    TAP_SEQUENCE_MAP['20hz'],
    TAP_SEQUENCE_MAP['15hz'],
    TAP_SEQUENCE_MAP['12hz'],
    TAP_SEQUENCE_MAP['10hz'],
    TAP_SEQUENCE_MAP['slow5'],
]
ADJ_DELAYS = [0, 18, 21, 24, 30, 61]
BUCKET_INTERVAL = 5
BUCKETS = tetris.Tetris.LineCap() // BUCKET_INTERVAL - 1
LEVEL_LINES = [0, 130, 230, 330, 430]

class GameParamManager:
    def __init__(self, board_file):
        self.eof = False
        self.board_file = board_file
        self.total_cnt = 0
        self.board_cnt = 0
        self.board_short_cnt = 0
        self.param_count = np.zeros((len(TAP_SEQUENCES), len(ADJ_DELAYS), len(AGGRESSION_LEVEL_MAP), BUCKETS), dtype='int64')
        #self.param_count = np.load('cnt.npz')['arr_0']
        self.burn_over_multiplier = 0.0
        self.board_ratio = 0.0
        self.short_ratio = 0.0
        self.rng = np.random.default_rng()
        self._NextBatch()

    def SaveParams(self):
        np.savez_compressed(f'cnt{os.getpid()}.npz', self.param_count)

    def UpdateParams(self, params):
        if any([i < 0 for i in params]):
            self.param_count = np.zeros((len(TAP_SEQUENCES), len(ADJ_DELAYS), 3, BUCKETS), dtype='int64')
        params = [abs(i) for i in params]
        self.total_cnt, self.board_cnt, self.board_short_cnt = 0, 0, 0
        self.burn_over_multiplier, self.board_ratio, self.short_ratio = [abs(i) for i in params]

    def UpdateState(self, params, pieces: int, lines: int):
        self.total_cnt += pieces
        if params.get('is_board'):
            self.board_cnt += pieces
            if params['is_short']:
                self.board_short_cnt += pieces

        start_bucket = min(params['lines'] // BUCKET_INTERVAL, BUCKETS)
        end_bucket = min((params['lines'] + lines) // BUCKET_INTERVAL + 1, BUCKETS)
        self.param_count[params['tap_id'], params['adj_delay_id'], params['aggression_level'], start_bucket:end_bucket] += 1

    def GetNewParam(self):
        while True:
            data = self._GetNewBoard()
            if data is None: break
            bucket_start = LEVEL_LINES[data[2]] // BUCKET_INTERVAL
            bucket_end = LEVEL_LINES[data[2] + 1] // BUCKET_INTERVAL
            params = self._SampleDistribution(data[4], bucket_start, bucket_end)
            return {'now_piece': data[1], 'is_board': True, 'board': data[0], 'is_short': data[3], **params}
        ret = self._SampleDistribution()
        if tetris.Tetris.IsTetrisOnly() and self.rng.random() < 0.3:
            col = self.rng.integers(8)
            board_str = '.' * col + 'xx' + '.' * (8 - col)
            ret['board'] = tetris.Board(board_str)
        return ret

    def _SampleDistribution(self, cells: int = 0, bucket_start: int = 0, bucket_end: int = BUCKETS):
        n_count = np.array(self.param_count, dtype='float32')
        bucket_start = min(bucket_start, BUCKETS)
        bucket_end = min(bucket_end, BUCKETS)
        if bucket_start == bucket_end: raise RuntimeError("unexpected")
        KS_START = LEVEL_LINES[2] // BUCKET_INTERVAL
        DKS_START = LEVEL_LINES[3] // BUCKET_INTERVAL
        n_count[0,0] /= 3
        n_count[0,1] /= 6
        n_count[2,2] /= 3
        n_count[4,2] /= 6
        #n_count[:,2:6,:,KS_START:DKS_START] *= 1.6
        #n_count[:,1:6,:,DKS_START:] *= 1.8
        #n_count[:,5,:,KS_START:DKS_START] += np.sum(n_count[:,2:5,:,KS_START:DKS_START], axis=1)
        #n_count[:,2:5,:,KS_START:DKS_START] = np.inf
        #n_count[:,5,:,DKS_START:] += np.sum(n_count[:,1:5,:,DKS_START:], axis=1)
        #n_count[:,1:5,:,DKS_START:] = np.inf
        n_count[3:6,:,:,DKS_START:] = np.inf
        if bucket_start > 0: n_count[:,:,:,:bucket_start] = np.inf
        if bucket_end < BUCKETS: n_count[:,:,:,bucket_end:] = np.inf
        n_prob = softmax(-n_count * 1.0)
        c = self.rng.choice(n_prob.size, p=n_prob.flatten())
        tap_id, adj_delay_id, aggression_level, bucket = np.unravel_index(c, n_prob.shape)
        start_lines = bucket * BUCKET_INTERVAL
        if (start_lines % 2 != 0) != (cells % 4 != 0): start_lines += 1
        lines = self.rng.choice(np.arange(start_lines, (bucket + 1) * BUCKET_INTERVAL, 2))
        return {'tap_sequence': TAP_SEQUENCES[tap_id].tolist(), 'tap_id': tap_id,
                'adj_delay': ADJ_DELAYS[adj_delay_id], 'adj_delay_id': adj_delay_id,
                'aggression_level': aggression_level, 'burn_over_multiplier': self.burn_over_multiplier,
                'lines': lines}

    def _GetNewBoard(self):
        if self.eof: return None
        # print(self.board_short_cnt, self.board_cnt, self.total_cnt)
        is_board = self.board_cnt < self.board_ratio * self.total_cnt
        if not is_board: return None

        if self.data_offset >= len(self.data):
            self._NextBatch()
            if self.data_offset >= len(self.data):
                return None
        is_short = self.board_short_cnt < self.short_ratio * self.board_cnt
        b, piece, level = self.data[self.data_offset]
        cells = 200 - int.from_bytes(b, 'little').bit_count()
        self.data_offset += 1
        return tetris.Board(b), piece, level, is_short, cells

    def _NextBatch(self):
        if not self.board_file:
            self.eof = True
            return
        self.data = self.read_board_file(self.board_file)
        self.data_offset = 0

    @staticmethod
    def read_board_file(board_file: str, chunk_size: int = 4096):
        DATA_SIZE = 26
        board_file_offset_f = board_file + '.offset'
        board_file_lock = board_file + '.lock'
        lock = FileLock(board_file_lock)
        with lock:
            if os.path.isfile(board_file_offset_f):
                with open(board_file_offset_f, 'r') as f: offset = int(f.read().strip())
            else:
                offset = 0
            with open(board_file, 'rb') as f:
                f.seek(offset * DATA_SIZE)
                data = f.read(chunk_size * DATA_SIZE)
                ret = [(data[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE-1],
                        data[i*DATA_SIZE+DATA_SIZE-1]&7,
                        data[i*DATA_SIZE+DATA_SIZE-1]>>3) for i in range(len(data) // DATA_SIZE)]
            with open(board_file_offset_f, 'w') as f:
                print(0 if len(ret) == 0 else offset + len(ret), file=f)
        return ret
