#include "frame_sequence.h"

namespace move_search {

template <int R>
struct NumTaps {
  int num_lr_tap, num_ab_tap, num_taps;
  bool is_l, is_a;

  constexpr NumTaps(int initial_rot, int initial_col, int target_rot, int target_col) :
      num_lr_tap(abs(target_col - initial_col)),
      num_ab_tap((target_rot + R - initial_rot) % R),
      num_taps(), is_l(target_col < initial_col), is_a(true) {
    if (num_ab_tap == 3) num_ab_tap = 1, is_a = false;
    num_taps = std::max(num_lr_tap, num_ab_tap);
  }
  int TotalTaps() const { return num_lr_tap + num_ab_tap; }
};

// return a frame range
template <int R>
constexpr std::pair<int, int> GetFrameRange(
    Level level, const int taps[], const std::array<Board, R>& board, bool is_tuck,
    int initial_rot, int initial_col, int initial_frame, int target_rot, int target_col) {
  auto [num_lr_tap, num_ab_tap, num_taps, is_l, is_a] = NumTaps<R>(initial_rot, initial_col, target_rot, target_col);

  Column target_column = board[target_rot].Column(target_col);
  int target_frame = initial_frame + (is_tuck ? taps[num_taps] : num_taps ? taps[num_taps - 1] : 0);
  int prev_row = GetRow(initial_frame, level);
  int cur_rot = initial_rot, cur_col = initial_col;
  if (!board[cur_rot].IsCellSet(prev_row, cur_col)) return {-1, -1};
  for (int i = 0; i < num_taps; i++) {
    int cur_row = GetRow(initial_frame + taps[i], level);
    if (!board[cur_rot].IsColumnRangeSet(prev_row, cur_row + 1, cur_col)) return {-1, -1};
    if (i < num_lr_tap) {
      is_l ? cur_col-- : cur_col++;
      if (!board[cur_rot].IsCellSet(cur_row, cur_col)) return {-1, -1};
    }
    if (i < num_ab_tap) {
      cur_rot = (cur_rot + (is_a ? 1 : R - 1)) % R;
      if (!board[cur_rot].IsCellSet(cur_row, cur_col)) return {-1, -1};
    }
    prev_row = cur_row;
  }
  if (is_tuck) {
    int cur_row = GetRow(initial_frame + taps[num_taps], level);
    if (!board[cur_rot].IsColumnRangeSet(prev_row, cur_row + 1, cur_col)) return {-1, -1};
    prev_row = cur_row;
  }
  int final_row = 31 - clz((target_column + (1 << prev_row)) ^ target_column) - 1;
  return {target_frame, GetLastFrameOnRow(final_row, level)};
}

template <int R>
void GenerateSequence(
    const int taps[], FrameSequence& seq, int initial_rot, int initial_col, int initial_frame,
    int target_rot, int target_col, size_t min_frames) {
  auto [num_lr_tap, num_ab_tap, num_taps, is_l, is_a] = NumTaps<R>(initial_rot, initial_col, target_rot, target_col);
  seq.resize(initial_frame, FrameInput{});
  for (int i = 0; i < num_taps; i++) {
    seq.resize(initial_frame + taps[i], FrameInput{});
    FrameInput cur{};
    if (i < num_lr_tap) cur |= is_l ? FrameInput::L : FrameInput::R;
    if (i < num_ab_tap) cur |= is_a ? FrameInput::A : FrameInput::B;
    seq.push_back(cur);
  }
  seq.resize(initial_frame + taps[num_taps], FrameInput{});
  if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
}

template <int R>
constexpr std::array<int, TuckTypes(R)> TuckSearchOrder() {
#ifdef DOUBLE_TUCK
  if constexpr (R == 1) {
    return {{0, 1, 2, 3}};
  } else if constexpr (R == 2) {
    return {{0, 1, 4, 5, 6, 7, 8, 2, 3}};
  } else {
    return {{0, 1, 4, 9, 5, 6, 10, 11, 7, 8, 12, 13, 2, 3}};
  }
#else
  if constexpr (R == 1) {
    return {{0, 1}};
  } else if constexpr (R == 2) {
    return {{0, 1, 2, 3, 4, 5, 6}};
  } else {
    return {{0, 1, 2, 7, 3, 4, 8, 9, 5, 6, 10, 11}};
  }
#endif
}

// should only be used for reachable positions; otherwise the result would probably be incorrect
template <int R, bool gen_seq = true>
NOINLINE int CalculateSequence(
    Level level, const int taps[], const std::array<Board, R>& board, FrameSequence& seq,
    int initial_rot, int initial_col, int initial_frame,
    const Position& target, size_t min_frames) {
  int max_height = 0;
  for (auto& i : board) max_height = std::max(max_height, i.Height());

  Column target_column = board[target.r].Column(target.y);
  int first_reachable_row = 31 - clz<uint32_t>(~(target_column << 1 | -(2 << target.x)));
  int first_reachable_frame = GetFirstFrameOnRow(first_reachable_row, level);
  int last_reachable_frame = GetLastFrameOnRow(target.x, level);
  {
    auto [frame_start, frame_end] = GetFrameRange<R>(
        level, taps, board, false, initial_rot, initial_col, initial_frame, target.r, target.y);
    if (frame_end >= first_reachable_frame && last_reachable_frame >= frame_start) {
      if constexpr (gen_seq) {
        GenerateSequence<R>(taps, seq, initial_rot, initial_col, initial_frame, target.r, target.y, min_frames);
      }
      return NumTaps<R>(initial_rot, initial_col, target.r, target.y).TotalTaps();
    }
  }

  constexpr TuckTypeTable<R> tucks;
  Frames target_frames = (2ll << GetLastFrameOnRow(target.x, level)) - (1ll << first_reachable_frame);
  for (int i : TuckSearchOrder<R>()) {
    auto& tuck = tucks.table[i];
    int intermediate_rot = (target.r + R - tuck.delta_rot) % R;
    int intermediate_col = target.y - tuck.delta_col;
    if (intermediate_col >= 10 || intermediate_col < 0) continue;
    auto [frame_start, frame_end] = GetFrameRange<R>(
        level, taps, board, true, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col);
    if (frame_start == -1) continue;
    Frames frame_mask_1 = (2ll << frame_end) - (1ll << frame_start);
    frame_mask_1 &= target_frames >> tuck.delta_frame;
    Frames frame_mask_2 = 0; // for tuck-spin on different frame
    int ret_taps = NumTaps<R>(initial_rot, initial_col, intermediate_rot, intermediate_col).TotalTaps();
#ifdef DOUBLE_TUCK
    // 0 1  2  3 4 5 6 7 8 9 10 11 12 13 ->
    // 0 1 12 13 2 3 4 5 6 7  8  9 10 11
    //     ^^^^^ double tucks
    int tuck_type_switch = i >= 2 ? (i >= 4 ? i - 2 : i + 10) : i;
#else
    int tuck_type_switch = i;
#endif
    switch (tuck_type_switch) {
      case 0: case 1: case 2: case 7: {
        ret_taps += 1;
        break; // no intermediate
      }
      case 3: case 4: case 8: case 9: {
        ret_taps += 2;
        frame_mask_1 &= ColumnToNormalFrameMask(level, board[intermediate_rot].Column(target.y));
        break;
      }
#ifdef DOUBLE_TUCK
      case 12: case 13: {
        ret_taps += 2;
        int pre_col = tuck_type_switch == 12 ? intermediate_col - 1 : intermediate_col + 1;
        Frames pre_col_mask = ColumnToDropFrameMask(level, board[target.r].Column(pre_col));
        frame_mask_1 &= pre_col_mask & pre_col_mask >> 1;
        break;
      }
#endif
      case 5: case 6: case 10: case 11: {
        ret_taps += 2;
        frame_mask_2 = frame_mask_1 & ColumnToDropFrameMask(level, board[target.r].Column(intermediate_col)); // e.g. L-A
        frame_mask_1 &= ColumnToDropFrameMask(level, board[intermediate_rot].Column(target.y)); // e.g. A-L
        break;
      }
    }
    if (frame_mask_1) {
      if constexpr (!gen_seq) return ret_taps;
      int tuck_frame = ctz(frame_mask_1);
      GenerateSequence<R>(taps, seq, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col, tuck_frame);
      switch (tuck_type_switch) {
        case 0: seq.push_back(FrameInput::L); break;
        case 1: seq.push_back(FrameInput::R); break;
        case 2: seq.push_back(FrameInput::A); break;
        case 3: seq.push_back(FrameInput::L | FrameInput::A); break;
        case 4: seq.push_back(FrameInput::R | FrameInput::A); break;
        case 5: seq.push_back(FrameInput::L); seq.push_back(FrameInput::A); break;
        case 6: seq.push_back(FrameInput::R); seq.push_back(FrameInput::A); break;
        case 7: seq.push_back(FrameInput::B); break;
        case 8: seq.push_back(FrameInput::L | FrameInput::B); break;
        case 9: seq.push_back(FrameInput::R | FrameInput::B); break;
        case 10: seq.push_back(FrameInput::L); seq.push_back(FrameInput::B); break;
        case 11: seq.push_back(FrameInput::R); seq.push_back(FrameInput::B); break;
        case 12: seq.push_back(FrameInput::L); seq.push_back(FrameInput{0}); seq.push_back(FrameInput::L); break;
        case 13: seq.push_back(FrameInput::R); seq.push_back(FrameInput{0}); seq.push_back(FrameInput::R); break;
      }
      if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
      return ret_taps;
    }
    if (frame_mask_2) {
      if constexpr (!gen_seq) return ret_taps;
      int tuck_frame = ctz(frame_mask_2);
      GenerateSequence<R>(taps, seq, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col, tuck_frame);
      switch (tuck_type_switch) {
        case 5: seq.push_back(FrameInput::A); seq.push_back(FrameInput::L); break;
        case 6: seq.push_back(FrameInput::A); seq.push_back(FrameInput::R); break;
        case 10: seq.push_back(FrameInput::B); seq.push_back(FrameInput::L); break;
        case 11: seq.push_back(FrameInput::B); seq.push_back(FrameInput::R); break;
      }
      if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
      return ret_taps;
    }
  }
  return -1;
}

constexpr std::array<uint32_t, 20> DirectionMapNoro(const Board& b, int inputs_per_row, bool do_tuck) {
  std::array<uint32_t, 20> rows = b.Rows();
  std::array<uint32_t, 20> ret = {};
  constexpr uint32_t kMask = 0x9249249;
  for (auto& row : rows) {
    row = pdep(row, kMask);
    row = row | row << 1 | row << 2;
  }
  if (do_tuck && inputs_per_row) {
    uint32_t state = 1 << 15;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      uint32_t lstate = state, rstate = state;
      // left
      for (int i = 0; i < inputs_per_row; i++) {
        lstate |= lstate >> 3 & rows[row];
        rstate |= rstate << 3 & rows[row];
        if (row == 0 && i == 0) {
          lstate |= rows[row] & (1 << 12);
          rstate |= rows[row] & (1 << 18);
        }
      }
      ret[row] = state | (lstate & ~state) << 1 | (rstate & ~state & ~lstate) << 2;
      state |= lstate | rstate;
    }
  } else if (do_tuck && !inputs_per_row) { // 29
    uint32_t state0 = 1 << 15, state1 = 1 << 15;
    for (int row = 0; row < 20 && (state0 || state1); row++) {
      state0 &= rows[row];
      state1 &= rows[row];
      uint32_t nstate0 = state0 | state1;
      uint32_t lstate = state0 >> 3 & rows[row];
      uint32_t rstate = state0 << 3 & rows[row];
      if (row == 0) {
        lstate |= rows[row] & (1 << 12);
        rstate |= rows[row] & (1 << 18);
      }
      ret[row] = nstate0 | (lstate & ~nstate0) << 1 | (rstate & ~nstate0 & ~lstate) << 2;
      state1 = lstate | rstate;
      state0 = nstate0;
    }
  } else {
    uint32_t state = 1 << 15;
    bool left = state, right = state;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      int nl = inputs_per_row ? row * inputs_per_row : (row + 1) / 2;
      int nr = inputs_per_row ? (row + 1) * inputs_per_row : (row + 2) / 2;
      if (nl <= 5 && !(1 << (5 - nl) & rows[row])) left = false;
      if (nl <= 4 && !(1 << (5 + nl) & rows[row])) right = false;
      uint32_t lstate = 0, rstate = 0;
      for (int i = nl + 1; i <= nr && i <= 5; i++) {
        if ((left || i == 1) && i <= 5 && (1 << ((5 - i) * 3) & rows[row])) {
          lstate |= 1 << ((5 - i) * 3);
          left = true;
        } else {
          left = false;
        }
        if ((right || i == 1) && i <= 4 && (1 << ((5 + i) * 3) & rows[row])) {
          rstate |= 1 << ((5 + i) * 3);
          right = true;
        } else {
          right = false;
        }
      }
      ret[row] = state | lstate << 1 | rstate << 2;
      state |= lstate | rstate;
    }
  }
  return ret;
}

} // namespace move_search

template <int R>
FrameSequence GetFrameSequence(
    Level level, const int taps[], const std::array<Board, R>& board,
    int initial_rot, int initial_col, int initial_frame,
    const Position& target, size_t min_frames = 0) {
  FrameSequence seq;
  move_search::CalculateSequence<R>(level, taps, board, seq, initial_rot, initial_col, initial_frame, target, min_frames);
  return seq;
}

FrameSequence GetFrameSequenceStart(
    Level level, const int taps[],
    const Board& b, int piece, int adj_delay, const Position& target) {
#define ONE_CASE(x) \
    case x: return GetFrameSequence<Board::NumRotations(x)>(level, taps, b.PieceMap<x>(), 0, Position::Start.y, 0, target, adj_delay);
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}

template <bool gen_seq>
int GetFrameSequenceAdj(
    Level level, const int taps[], FrameSequence& seq, const Board& b, int piece, const Position& premove,
    const Position& target) {
#define ONE_CASE(x) \
    case x: return move_search::CalculateSequence<Board::NumRotations(x), gen_seq>( \
                level, taps, b.PieceMap<x>(), seq, premove.r, premove.y, seq.size(), target, 0);
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}
template
int GetFrameSequenceAdj<true>(
    Level level, const int taps[], FrameSequence& seq, const Board& b, int piece, const Position& premove,
    const Position& target);
template
int GetFrameSequenceAdj<false>(
    Level level, const int taps[], FrameSequence& seq, const Board& b, int piece, const Position& premove,
    const Position& target);

FrameSequence GetFrameSequenceNoro(
    const Board& b, int piece, int inputs_per_row, bool do_tuck, int frames_per_drop, const Position& target) {
  if (target.r != 0) return {};
  auto dir = move_search::DirectionMapNoro(b.PieceMapNoro(piece), inputs_per_row, do_tuck);
  if ((dir[target.x] >> (target.y * 3) & 7) == 0) return {};
  Position cur = target;
  std::vector<std::string> inputs(target.x + 1);
  while (cur != Position::Start) {
    uint32_t v = dir[cur.x] >> (cur.y * 3);
    if (v & 1) {
      cur.x--;
    } else if (v & 2) {
      inputs[cur.x].push_back('L');
      cur.y++;
    } else {
      inputs[cur.x].push_back('R');
      cur.y--;
    }
  }
  for (auto& i : inputs) std::reverse(i.begin(), i.end());
  FrameSequence ret;
  constexpr int kSlowPush = 0; // 0 for fast push, >=4 for frames per row
  bool down_held = false;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (kSlowPush == 0 && down_held && frames_per_drop > 2 && inputs[i].empty()) {
      ret.resize(ret.size() + 2, FrameInput::D);
      continue;
    }
    down_held = false;
    int input_frames = std::max((int)inputs[i].size() * 2 - 1, kSlowPush >= 4 ? kSlowPush - 3 : 0);
    int blank_frames = frames_per_drop - input_frames;
    size_t offset = ret.size();
    ret.resize(offset + input_frames + std::min(blank_frames, 3), FrameInput{});
    for (size_t j = 0; j < inputs[i].size(); j++) {
      ret[offset + j * 2] = inputs[i][j] == 'L' ? FrameInput::L : FrameInput::R;
    }
    if (blank_frames >= 3) {
      ret[offset + input_frames] = FrameInput::D;
      ret[offset + input_frames + 1] = FrameInput::D;
      ret[offset + input_frames + 2] = FrameInput::D;
      down_held = true;
    }
  }
  return ret;
}

template <int R>
std::pair<Position, bool> SimulateMove(Level level, const std::array<Board, R>& board, const FrameSequence& seq, bool until_lock) {
  // TODO: fix DAS?
  Position pos = Position::Start;
  int charge = 0;
  FrameInput prev_input{};
  for (size_t frame = 0; frame < seq.size(); frame++) {
    auto& input = seq[frame];
    if (input.IsL()) {
      bool is_available = pos.y > 0 && board[pos.r].IsCellSet(pos.x, pos.y - 1);
      if (!prev_input.IsL()) {
        charge = 0;
        if (is_available) pos.y--;
      } else if (!is_available) {
        // nothing; prevent charge increase
      } else if (++charge == 6) {
        charge = 0;
        pos.y--;
      }
    } else if (input.IsR()) {
      bool is_available = pos.y < 9 && board[pos.r].IsCellSet(pos.x, pos.y + 1);
      if (!prev_input.IsR()) {
        charge = 0;
        if (is_available) pos.y++;
      } else if (!is_available) {
        // nothing; prevent charge increase
      } else if (++charge == 6) {
        charge = 0;
        pos.y++;
      }
    }
    if (input.IsA() && !prev_input.IsA()) {
      int new_r = (pos.r + 1) % R;
      if (board[new_r].IsCellSet(pos.x, pos.y)) pos.r = new_r;
    } else if (input.IsB() && !prev_input.IsB()) {
      int new_r = (pos.r + R - 1) % R;
      if (board[new_r].IsCellSet(pos.x, pos.y)) pos.r = new_r;
    }
    for (int i = 0; i < move_search::NumDrops(frame, level); i++) {
      if (pos.x == 19 || !board[pos.r].IsCellSet(pos.x + 1, pos.y)) return {pos, true};
      pos.x++;
    }
    prev_input = input;
  }
  if (!until_lock) return {pos, false};
  while (true) {
    if (pos.x == 19 || !board[pos.r].IsCellSet(pos.x + 1, pos.y)) return {pos, true};
    pos.x++;
  }
}

std::pair<Position, bool> SimulateMove(Level level, const Board& b, int piece, const FrameSequence& seq, bool until_lock) {
#define ONE_CASE(x) \
    case x: return SimulateMove<Board::NumRotations(x)>(level, b.PieceMap<x>(), seq, until_lock);
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}

namespace {

struct Counter {
  struct value_type {
    template <class T>
    value_type(const T&) {}
  };
  void push_back(const value_type&) { ++count; }
  size_t count = 0;
};

} // namespace

std::pair<size_t, FrameSequence> GetBestAdj(
    Level level, const int taps[],
    const Board& b, int piece, const PossibleMoves& moves, int adj_delay, const Position adjs[kPieces]) {
  std::vector<Position> uniq_pos(adjs, adjs + kPieces);
  std::sort(uniq_pos.begin(), uniq_pos.end());
  uniq_pos.resize(std::unique(uniq_pos.begin(), uniq_pos.end()) - uniq_pos.begin());
  std::vector<float> probs(uniq_pos.size());
  for (size_t i = 0; i < kPieces; i++) {
    for (size_t j = 0; j < uniq_pos.size(); j++) {
      if (uniq_pos[j] == adjs[i]) probs[j] += kTransitionProb[piece][i];
    }
  }
  size_t ret = 0;
  float mn = 1e5;
  FrameSequence ret_seq;
  for (size_t i = 0; i < moves.adj.size(); i++) {
    {
      Counter c;
      auto tmp_moves = moves.adj[i].second;
      std::sort(tmp_moves.begin(), tmp_moves.end());
      std::set_intersection(tmp_moves.begin(), tmp_moves.end(), uniq_pos.begin(), uniq_pos.end(), std::back_inserter(c));
      if (c.count != uniq_pos.size()) continue;
    }
    FrameSequence seq = GetFrameSequenceStart(level, taps, b, piece, adj_delay, moves.adj[i].first);
    int pre_taps = 0;
    for (auto& j : seq) {
      if (j.IsA() || j.IsB()) pre_taps++;
      if (j.IsL() || j.IsR()) pre_taps++;
    }
    float weight = pre_taps * (1. / 32);
    for (size_t j = 0; j < uniq_pos.size(); j++) {
      int num_taps = GetFrameSequenceAdj<false>(level, taps, seq, b, piece, moves.adj[i].first, uniq_pos[j]);
      weight += probs[j] * (num_taps * num_taps);
    }
    if (weight < mn) {
      mn = weight;
      ret = i;
      ret_seq.swap(seq);
    }
  }
  return {ret, ret_seq};
}
