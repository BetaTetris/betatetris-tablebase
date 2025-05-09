#!/usr/bin/env python3

import sys, os.path, socketserver, argparse, re, socket
import numpy as np, torch
import curses

import tetris
from model import Model, obs_to_torch
from game_param import TAP_SEQUENCE_MAP, AGGRESSION_LEVEL_MAP, ADJ_DELAYS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stdscr = None
args = None
thresholds = None
is_noro = tetris.Tetris.IsNoro()


def myprint(*pargs, posx=0, posy=0, clear=True, refresh=True):
    if args.use_curses:
        if clear: stdscr.clear()
        pstr = ' '.join(map(str, pargs))
        stdscr.addstr(posx, posy, pstr)
        if refresh: stdscr.refresh()
    else:
        print(*pargs, flush=True)


class GameConn(socketserver.BaseRequestHandler):
    def read_until(self, sz):
        data = b''
        while len(data) < sz:
            n = self.request.recv(sz - len(data))
            if len(n) == 0:
                raise ConnectionResetError()
            data += n
        return data

    @staticmethod
    def gen_seq(seq):
        if len(seq) == 0: return bytes([0xfe, 0, 0])
        return bytes([0xfe, len(seq) & 255, len(seq) >> 8]) + seq.tobytes()

    @staticmethod
    def seq_to_str(seq):
        ret = ''
        for i in seq:
            s = ''
            for val, ch in zip([1, 2, 4, 8, 16], 'LRABD'):
                if (i & val) == val: s += ch
            if s == '': s = '-'
            ret += s + ' '
        return ret.strip()

    def get_strat(self):
        with torch.no_grad():
            pi, v = self.model(obs_to_torch(self.game.GetState()))
            action = torch.argmax(pi, 1).item()
            if is_noro:
                return (action // 200, action // 10 % 20, action % 10), v[0].item() * 5
            else:
                return (action // 200, action // 10 % 20, action % 10), v[1].item()

    def strat_text(self, strat, val=None):
        ntxt = '{:8s}'.format(self.game.GetBoard().PlacementNotation(self.game.GetNowPiece(), *strat))
        if val is not None:
            ntxt += f'; val {val:6.3f}'
        return ntxt

    def print_status(self, strats, is_table, data, refresh=True):
        txt = 'Game #' + str(self.games) + '\n'
        txt += 'Games to 19: ' + str(self.num_19) + '\n'
        txt += 'Games to 29: ' + str(self.num_29) + '\n'
        txt += 'Drought: ' + str(self.drought) + '\nAgent: '
        txt += 'Tablebase' if is_table else 'Neural Net'
        txt += '\nPlacements: (where to place\n the upcoming piece based on\n its next piece)\n'
        for i in range(7):
            txt += 'TJZOSLI'[i] + ': ' + self.strat_text(strats[i], None if is_table else data[i]) + '\n'
        if is_table:
            txt += f'confidence {data}'
            if thresholds:
                multiplier = (args.buckets - 2) / (args.ratio_high - args.ratio_low)
                bias = 1 - args.ratio_low * multiplier
                value_low = 0.0 if data == 0 else (data - bias) / multiplier
                value_high = 1.0 if data == args.buckets - 1 else (data + 1 - bias) / multiplier
                scale = thresholds[self.game.GetLines()] * 100
                value_low *= scale
                value_high *= scale
                txt += f' (pure tablebase\n probability to 29:\n{value_low:6.2f}% -{value_high:6.2f}%)'
        txt += '\n'
        myprint(txt, refresh=refresh)

    def get_adj_strat(self, pos):
        with torch.no_grad():
            pi, v = self.model(obs_to_torch(self.game.GetAdjStates(*pos)))
            actions = torch.argmax(pi, 1).flatten().cpu().tolist()
            strats = [(action // 200, action // 10 % 20, action % 10) for action in actions]
            self.print_status(strats, False, v[1].flatten().cpu().tolist())
            return strats

    def print_status_noro(self, strat, v, refresh=True):
        if strat == (0, 0, 0): return
        txt = 'Placement:\n'
        txt += self.strat_text(strat, self.game.GetNowPiece(), v)
        txt += '\n'
        myprint(txt, refresh=refresh)

    def query_tablebase(self):
        query = (
            bytes([1]) +
            self.game.GetBoard().GetBytes() +
            bytes([self.game.GetNowPiece(), self.game.GetLines() % 256, self.game.GetLines() // 256])
        )
        self.board_conn.sendall(query)
        result = self.board_conn.recv(22)
        level = result[21]
        adj_strats = [tuple(result[i:i+3]) for i in range(0, 21, 3)]
        self.print_status(adj_strats, True, level, refresh=False)
        return adj_strats, level

    def get_strat_all(self):
        if self.game.GetLines() < 230 and self.board_conn:
            # tablebase
            adj_strats, level = self.query_tablebase()
            if not adj_strats[0] == (0, 0, 0):
                stdscr.refresh()
                if self.game.IsNoAdjMove(*adj_strats[0]):
                    return False, adj_strats[0]
                return True, adj_strats
        # beta
        strat, v = self.get_strat()
        if self.game.IsNoAdjMove(*strat):
            self.print_status([strat]*7, False, [v]*7)
            return False, strat
        return True, self.get_adj_strat(strat)

    def send_seq(self, seq):
        self.request.send(self.gen_seq(seq))

    def step_game(self, strat):
        if self.done: return
        self.game.InputPlacement(*strat)
        self.done = self.game.IsOver()

    def do_premove(self):
        if self.done:
            self.prev = (-1, -1, -1)
            self.send_seq([])
            return
        is_adj, strat = self.get_strat_all()
        if is_adj:
            pos, seq = self.game.GetAdjPremove(strat)
            self.prev = (pos, seq, strat)
            self.step_game(pos)
            self.send_seq(seq)
        else:
            seq = self.game.GetSequence(*strat)
            self.prev = (strat, None, None)
            self.send_seq(seq)
            self.send_seq([])

    def finish_move(self, next_piece):
        pos, seq, strats = self.prev
        if self.done:
            if seq is not None: self.send_seq([])
            return
        self.game.SetNextPiece(next_piece)
        if seq is None:
            self.step_game(pos)
            self.prev_placement = pos
        else:
            strat = strats[next_piece]
            fin_seq = self.game.FinishAdjSequence(seq, pos, strat)
            self.send_seq(fin_seq[len(seq):])
            self.step_game(strat)
            self.prev_placement = strat

    def first_piece(self):
        if self.board_conn:
            is_adj, strat = self.get_strat_all()
            if is_adj:
                pos, _ = self.game.GetAdjPremove(strat)
                strat = strat[self.game.GetNextPiece()]
                self.step_game(pos)
            seq = self.game.GetSequence(*strat)
        else:
            strat, _ = self.get_strat()
            seq = self.game.GetSequence(*strat)
            if self.game.IsAdjMove(*strat):
                self.step_game(strat)
                strat, _ = self.get_strat()
                seq = self.game.GetSequence(*strat)
        self.step_game(strat)
        self.prev_placement = strat
        self.send_seq(seq)
        self.do_premove()

    def finish_move_noro(self, next_piece):
        self.game.SetNextPiece(next_piece)
        if args.nnb:
            self.step_game(self.prev)
            self.prev_placement = self.game.GetRealPosition(self.prev)
        strat, v = self.get_strat()
        self.print_status_noro(strat, v)
        seq = self.game.GetSequence(*strat)
        self.prev = strat
        self.send_seq(seq)
        self.send_seq([])
        if args.nnb:
            self.prev = strat
        else:
            self.prev_placement = self.game.GetRealPosition(strat)
            self.step_game(strat)

    def first_piece_noro(self):
        if self.done: return
        strat, v = self.get_strat()
        self.print_status_noro(strat, v)
        self.prev_placement = self.game.GetRealPosition(strat)
        seq = self.game.GetSequence(*strat)
        self.send_seq(seq)
        self.step_game(strat)
        if args.nnb:
            strat, v = self.get_strat()
            self.print_status_noro(strat, v)
            seq = self.game.GetSequence(*strat)
            self.prev = strat
            self.send_seq(seq)
            self.send_seq([])
        else:
            self.send_seq([])

    def start_lines(self):
        if self.start_level <= 18: return 0
        if self.start_level <= 28: return 130
        if self.start_level <= 38: return 230
        return 330

    def set_effective_lines(self):
        if is_noro: return
        lines = self.game.GetRunLines()
        if self.start_level <= 18:
            pass
        elif self.start_level <= 28:
            if lines < 148: lines = 144 + (lines % 4)
        elif self.start_level <= 38 or args.no_2ks:
            lines += 40
            if lines < 248: lines = 244 + (lines % 4)
        else:
            if lines < 348: lines = 344 + (lines % 4)
        if args.no_cap:
            cap = 256 if args.no_2ks else 356
            if lines >= cap: lines = cap + (lines % 4)
        if self.game.GetLines() != lines:
            self.game.SetLines(lines)

    def handle(self):
        myprint('Connected')
        self.game = tetris.Tetris()
        while True:
            try:
                data = self.read_until(1)
                if data[0] == 0xff or data[0] == 0xef:
                    if data[0] == 0xef:
                        self.num_19 = 0
                        self.num_29 = 0
                        self.games = 0
                    self.done = False
                    cur, nxt, self.start_level = self.read_until(3)
                    if is_noro:
                        use_mirror = [
                            [0, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 1],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 1, 0, 1],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1, 0, 0],
                        ]
                        reset_args = {
                            'start_level': self.start_level,
                            'do_tuck': not args.no_tuck,
                            'nnb': args.nnb,
                            'mirror': bool(use_mirror[cur][nxt]) if args.adaptive else args.mirror,
                        }
                    else:
                        self.drought = 0
                        self.num_19 += int(self.game.GetLines() >= 130)
                        self.num_29 += int(self.game.GetLines() >= 230)
                        self.games += 1
                        reset_args = {
                            'lines': self.start_lines(),
                            'adj_delay': args.adj_delay,
                            'tap_sequence': TAP_SEQUENCE_MAP[args.tap_speed].tolist(),
                            'aggression_level': AGGRESSION_LEVEL_MAP[args.aggression],
                        }
                    self.game.Reset(cur, nxt, **reset_args)
                    myprint('New game', self.start_level, (cur, nxt))
                    if is_noro:
                        self.first_piece_noro()
                    else:
                        self.first_piece()
                elif data[0] == 0xfd:
                    r, x, y, nxt = self.read_until(4)
                    if nxt != 6: self.drought += 1
                    else: self.drought = 0
                    if (r, x, y) != self.prev_placement and not self.done:
                        myprint(f'Error: unexpected placement {(r, x, y)}; expected {self.prev_placement}')
                        self.done = True
                    if is_noro:
                        self.finish_move_noro(nxt)
                    else:
                        self.finish_move(nxt)
                        self.set_effective_lines()
                        self.do_premove()
            except ConnectionResetError:
                self.request.close()
                break
            except IOError:
                self.request.close()
                break


def GenHandler(model, board_conn):
    class Handler(GameConn):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.board_conn = board_conn
            self.games = 0
            self.num_19 = 0
            self.num_29 = 0
            self.drought = 0
            super().__init__(*args, **kwargs)
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-b', '--bind', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=3456)
    parser.add_argument('-c', '--use-curses', action='store_true')
    if is_noro:
        parser.add_argument('--nnb', action='store_true')
        parser.add_argument('--mirror', action='store_true')
        parser.add_argument('--adaptive', action='store_true')
        parser.add_argument('--no-tuck', action='store_true')
    else:
        parser.add_argument('--no-cap', action='store_true')
        parser.add_argument('--no-2ks', action='store_true')
        parser.add_argument('--adj-delay', type=int, default=18, choices=ADJ_DELAYS)
        parser.add_argument('--tap-speed', type=str, default='30hz', choices=list(TAP_SEQUENCE_MAP))
        parser.add_argument('--aggression', type=str, default='high', choices=list(AGGRESSION_LEVEL_MAP))
        parser.add_argument('-s', '--server', type=str)
        parser.add_argument('--threshold-file', type=str)
        parser.add_argument('--ratio-low', type=float, default=0.01)
        parser.add_argument('--ratio-high', type=float, default=1.0)
        parser.add_argument('--buckets', type=float, default=256)
    args = parser.parse_args()
    print(args)

    if not is_noro and args.threshold_file:
        with open(args.threshold_file, 'r') as f:
            thresholds = list(map(float, f.read().split()))

    with torch.no_grad():
        state_dict = torch.load(args.model, weights_only=True)
        channels = state_dict['main_start.0.main.0.weight'].shape[0]
        start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
        end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
        model = Model(start_blocks, end_blocks, channels).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    if not is_noro and args.server:
        host, port = args.server.split(':')
        port = int(port)
        board_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        board_conn.connect((host, port))
    else:
        board_conn = None

    # load GPU first to reduce lag
    model(obs_to_torch(tetris.Tetris().GetState()))

    if args.use_curses:
        stdscr = curses.initscr()
        curses.noecho()
        curses.curs_set(False)

    try:
        with socketserver.TCPServer((args.bind, args.port), GenHandler(model, board_conn)) as server:
            server.model = model
            myprint(f'Ready, listening on {args.bind}:{args.port}')
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.shutdown()
                if args.use_curses:
                    curses.curs_set(True)
                    curses.echo()
                    curses.endwin()
    except:
        if args.use_curses:
            curses.echo()
            curses.endwin()
        raise

