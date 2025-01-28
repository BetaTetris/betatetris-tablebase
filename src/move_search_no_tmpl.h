#pragma once

#include <mutex>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

#include "game.h"
#include "board.h"
#include "position.h"
#include "constexpr_helpers.h"

class PossibleMoves {
  static void UniqueVector_(std::vector<Position>& p, bool unique) {
    std::sort(p.begin(), p.end());
    p.resize(std::unique(p.begin(), p.end()) - p.begin());
  }
 public:
  std::vector<Position> non_adj;
  std::vector<std::pair<Position, std::vector<Position>>> adj;
  void Normalize(bool unique = false) {
    UniqueVector_(non_adj, unique);
    for (auto& i : adj) UniqueVector_(i.second, unique);
    std::sort(adj.begin(), adj.end());
  }
};

namespace move_search {

constexpr int GetRow(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame / 3;
    case kLevel19: return frame / 2;
    case kLevel29: return frame;
    case kLevel39: return frame * 2;
  }
  unreachable();
}

constexpr bool IsDropFrame(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame % 3 == 2;
    case kLevel19: return frame % 2 == 1;
    default: return true;
  }
}

constexpr int NumDrops(int frame, Level level) {
  if (!IsDropFrame(frame, level)) return 0;
  switch (level) {
    case kLevel39: return 2;
    default: return 1;
  }
}

constexpr int GetFirstFrameOnRow(int row, Level level) {
  switch (level) {
    case kLevel18: return row * 3;
    case kLevel19: return row * 2;
    case kLevel29: return row;
    case kLevel39: return (row + 1) / 2;
  }
  unreachable();
}

constexpr int GetLastFrameOnRow(int row, Level level) {
  switch (level) {
    case kLevel18: return row * 3 + 2;
    case kLevel19: return row * 2 + 1;
    case kLevel29: return row;
    case kLevel39: return row / 2;
  }
  unreachable();
}

constexpr int abs(int x) { return x < 0 ? -x : x; }
constexpr int sgn(int x) {
  return x == 0 ? 0 : x > 0 ? 1 : -1;
}

// Check each bit in mask is set in board
template <int R>
constexpr bool Contains4(const std::array<Board, R>& board, const std::array<Board, 4>& mask) {
  bool ret = true;
  for (int i = 0; i < R; i++) ret &= (board[i] & mask[i]) == mask[i];
  return ret;
}

struct TableEntryNoTmpl {
  uint8_t rot, col, num_taps;
  std::array<Board, 4> masks_nodrop;

  bool operator<=>(const TableEntryNoTmpl&) const = default;
};

template <int R, class Entry>
constexpr int Phase1TableGen(
    Level level, const int taps[], int initial_frame, int initial_rot, int initial_col,
    int max_lr_taps, int max_ab_taps,
    Entry entries[]) {
  int sz = 0;
  static_assert(R == 1 || R == 2 || R == 4, "unexpected rotations");
  constexpr uint8_t kA = 0x1;
  constexpr uint8_t kB = 0x2;
  constexpr uint8_t kL = 0x4;
  constexpr uint8_t kR = 0x8;
  std::array<Board, R> masks[R][10] = {};
  std::array<Board, R> masks_nodrop[R][10] = {};
  uint8_t last_tap[R][10] = {};
  bool cannot_reach[R][10] = {};
  bool cannot_finish[R][10] = {};
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < R; k++) masks[i][j][k] = Board(0, 0, 0, 0);
    }
  }
  for (int col = 0; col < 10; col++) {
    for (int delta_rot = 0; delta_rot < 4; delta_rot++) {
      // piece end up at column col and rotation (initial_rot + delta_rot)
      if (R == 1 && delta_rot != 0) continue;
      if (R == 2 && delta_rot >= 2) continue;
      int rot = (initial_rot + delta_rot) % R;
      int num_lr_tap = abs(col - initial_col);
      int num_ab_tap = delta_rot == 3 ? 1 : delta_rot; // [0,1,2,1]
      if (num_lr_tap > max_lr_taps || num_ab_tap > max_ab_taps) {
        cannot_reach[rot][col] = true;
        continue;
      }
      int num_tap = std::max(num_ab_tap, num_lr_tap);
      // the frame that this tap occurred; initial_frame if no input
      int start_frame = (num_tap == 0 ? 0 : taps[num_tap - 1]) + initial_frame;
      // the frame that next input is allowed
      int end_frame = taps[num_tap] + initial_frame;
      if (num_tap) {
        if (num_tap == num_lr_tap) last_tap[rot][col] |= col > initial_col ? kR : kL;
        if (num_tap == num_ab_tap) last_tap[rot][col] |= delta_rot == 3 ? kB : kA;
      }
      // the position before this tap
      int start_row = GetRow(start_frame, level);
      if (start_row >= 20) {
        cannot_reach[rot][col] = true;
        continue;
      }
      int start_col = num_tap == num_lr_tap ? col - sgn(col - initial_col) : col;
      int start_rot = num_tap == num_ab_tap ? ((delta_rot == 2 ? 1 : 0) + initial_rot) % R : rot;
      auto& cur = masks[rot][col];
      cur[start_rot].Set(start_row, start_col);
      cur[start_rot].Set(start_row, col); // first shift
      cur[rot].Set(start_row, col); // then rotate
      masks_nodrop[rot][col] = cur;
      if (GetRow(end_frame, level) >= 20) {
        cannot_finish[rot][col] = true;
        continue;
      }
      for (int frame = start_frame; frame < end_frame; frame++) {
        int row = GetRow(frame, level);
        cur[rot].Set(row, col);
        if (IsDropFrame(frame, level)) {
          cur[rot].Set(row + 1, col);
          if (level == kLevel39) cur[rot].Set(row + 2, col);
        }
      }
    }
  }
  // start from (initial_col, initial_row) and build the entries according to
  //   ascending tap count
  auto Push = [&](uint8_t rot, uint8_t col, uint8_t orot, uint8_t ocol, uint8_t prev, uint8_t num_taps) {
    if (!cannot_reach[rot][col]) {
      if constexpr (std::is_same<Entry, TableEntryNoTmpl>::value) {
        entries[sz] = {rot, col, num_taps, {}};
      } else {
        entries[sz] = {rot, col, prev, num_taps, cannot_finish[rot][col], {}, {}};
      }
      for (int i = 0; i < R; i++) {
        if constexpr (std::is_same<Entry, TableEntryNoTmpl>::value) {
          entries[sz].masks_nodrop[i] = masks_nodrop[rot][col][i];
          if (num_taps) {
            entries[sz].masks_nodrop[i] |= masks[orot][ocol][i];
            masks[rot][col][i] |= masks[orot][ocol][i];
          }
        } else {
          entries[sz].masks[i] = masks[rot][col][i];
          entries[sz].masks_nodrop[i] = masks_nodrop[rot][col][i];
        }
      }
      sz++;
    }
  };
  Push(initial_rot, initial_col, 0, 0, 0, 0);
  for (int cur = 0; cur < sz; cur++) {
    int rot = entries[cur].rot, col = entries[cur].col, taps = entries[cur].num_taps;
    int last = last_tap[rot][col];
    bool should_l = col > 0 && (taps == 0 || (last & kL));
    bool should_r = col < 9 && (taps == 0 || (last & kR));
    bool should_a = (R > 1 && taps == 0) || (R == 4 && taps == 1 && (last & kA));
    bool should_b = R == 4 && taps == 0;
    if (should_l) Push(rot, col - 1, rot, col, cur, taps + 1); // L
    if (should_r) Push(rot, col + 1, rot, col, cur, taps + 1); // R
    if (should_a) {
      int nrot = (rot + 1) % R;
      Push(nrot, col, rot, col, cur, taps + 1); // A
      if (should_l) Push(nrot, col - 1, rot, col, cur, taps + 1); // L
      if (should_r) Push(nrot, col + 1, rot, col, cur, taps + 1); // R
    }
    if (should_b) {
      int nrot = (rot + 3) % R;
      Push(nrot, col, rot, col, cur, taps + 1); // B
      if (should_l) Push(nrot, col - 1, rot, col, cur, taps + 1); // L
      if (should_r) Push(nrot, col + 1, rot, col, cur, taps + 1); // R
    }
  }
  return sz;
}

template <class Entry>
constexpr int Phase1TableGen(
    Level level, int R, const int taps[], int initial_frame, int initial_rot, int initial_col,
    int max_lr_taps, int max_ab_taps,
    Entry entries[]) {
  if (R == 1) {
    return Phase1TableGen<1>(level, taps, initial_frame, initial_rot, initial_col, max_lr_taps, max_ab_taps, entries);
  } else if (R == 2) {
    return Phase1TableGen<2>(level, taps, initial_frame, initial_rot, initial_col, max_lr_taps, max_ab_taps, entries);
  } else {
    return Phase1TableGen<4>(level, taps, initial_frame, initial_rot, initial_col, max_lr_taps, max_ab_taps, entries);
  }
}

struct Phase1TableNoTmpl {
  std::vector<TableEntryNoTmpl> initial;
  std::vector<std::vector<TableEntryNoTmpl>> adj;
  Phase1TableNoTmpl() {}
  Phase1TableNoTmpl(Level level, int R, int adj_frame, const int taps[]) : initial(40) {
    initial.resize(10 * R);
    initial.resize(Phase1TableGen(level, R, taps, 0, 0, Position::Start.y, 9, 2, initial.data()));
    for (auto& i : initial) {
      int frame_start = std::max(adj_frame, taps[i.num_taps]);
      adj.emplace_back(10 * R);
      adj.back().resize(Phase1TableGen(level, R, taps, frame_start, i.rot, i.col, 9, 2, adj.back().data()));
    }
  }
};

// Column is simply a column; each bit is a cell, LSB is topmost
// Frames is a processed form of a column; each bit is a frame, LSB is the first frame
// Frames comes in two flavors: normal and drop mask
//   normal mask just corresponds to the row the piece is in on each frame
//   drop mask is AND of all rows that the piece will pass when dropping
// For level 18,19,29, drop mask is just (normal_mask & normal_mask >> 1);
//   drop mask exists only to make it easy to deal with level 39
using Column = uint32_t;
using Frames = uint64_t;

template <int R>
struct FrameMasks {
  Frames frame[R][10], drop[R][10];
};

constexpr Frames ColumnToNormalFrameMask(Level level, Column col) {
  switch (level) {
    case kLevel18: {
      constexpr uint64_t kMask = 0x249249249249249;
      uint64_t expanded = pdep<uint64_t>(col, kMask);
      return expanded | expanded << 1 | expanded << 2;
    }
    case kLevel19: {
      constexpr uint64_t kMask = 0x5555555555;
      uint64_t expanded = pdep<uint64_t>(col, kMask);
      return expanded | expanded << 1;
    }
    case kLevel29: return col;
    case kLevel39: {
      constexpr uint32_t kMask = 0x55555;
      return pext(col, kMask);
    }
  }
  unreachable();
}

constexpr Frames ColumnToDropFrameMask(Level level, Column col) {
  switch (level) {
    case kLevel18: [[fallthrough]];
    case kLevel19: [[fallthrough]];
    case kLevel29: {
      uint64_t mask = ColumnToNormalFrameMask(level, col);
      return mask & mask >> 1;
    }
    case kLevel39: {
      constexpr uint32_t kMask = 0x55555;
      return pext(col & col >> 1 & col >> 2, kMask);
    }
  }
  unreachable();
}

constexpr Column FramesToColumn(Level level, Frames frames) {
  switch (level) {
    case kLevel18: {
      constexpr uint64_t kMask = 0x249249249249249;
      return pext(frames | frames >> 1 | frames >> 2, kMask);
    }
    case kLevel19: {
      constexpr uint64_t kMask = 0x5555555555;
      return pext(frames | frames >> 1, kMask);
    }
    case kLevel29: return frames;
    case kLevel39: {
      constexpr uint32_t kMask = 0x55555;
      return pdep<uint32_t>(frames, kMask);
    }
  }
  unreachable();
}

constexpr int FindLockRow(uint32_t col, int start_row) {
  // given (col & 1<<row) != 0
  // col               = 00111100011101
  // 1<<row            = 00000000001000
  // col+(1<<row)      = 00111100100101
  // col^(col+(1<<row))= 00000000111000
  //              highbit=31-clz ^
  return 31 - clz<uint32_t>(col ^ (col + (1 << start_row))) - 1;
}

// note: "tuck" here means tucks, spins or spintucks
template <int R>
using TuckMask = std::array<std::array<Frames, 10>, R>;

constexpr int TuckTypes(int R) {
  return (R == 1 ? 2 : R == 2 ? 7 : 12) + (kDoubleTuckAllowed ? 2 : 0);
  // R = 1: L R (LL RR)
  // R = 2: A LA RA AL AR
  // R = 4: B LB RB BL BR
  // should also change frame_sequence.h if changed
  // it is possible to add other tuck types suck as buco-like spins
  // but we just keep it simple here
}

template <int R>
using TuckMasks = std::array<TuckMask<R>, TuckTypes(R)>;

struct TuckType {
  int delta_rot, delta_col, delta_frame;
};

template <int R>
struct TuckTypeTable {
  std::array<TuckType, TuckTypes(R)> table;
  constexpr TuckTypeTable() : table() {
    table[0] = {0, -1, 0}; // L
    table[1] = {0, 1, 0}; // R
#ifdef DOUBLE_TUCK
    table[2] = {0, -2, 2}; // L-/-L
    table[3] = {0, 2, 2}; // R-/-R
#endif
    constexpr int x = kDoubleTuckAllowed ? 2 : 0;
    if constexpr (R == 1) return;
    table[x+2] = {1, 0, 0}; // A
    table[x+3] = {1, -1, 0}; // LA
    table[x+4] = {1, 1, 0}; // RA
    table[x+5] = {1, -1, 1}; // A-L L-A
    table[x+6] = {1, 1, 1}; // A-R R-A
    if constexpr (R == 2) return;
    table[x+7] = {3, 0, 0}; // B
    table[x+8] = {3, -1, 0}; // LB
    table[x+9] = {3, 1, 0}; // RB
    table[x+10] = {3, -1, 1}; // B-L L-B
    table[x+11] = {3, 1, 1}; // B-R R-B
  }
};

template <int R>
constexpr TuckMasks<R> GetTuckMasks(const FrameMasks<R> m) {
  TuckMasks<R> ret{};
  constexpr int x = kDoubleTuckAllowed ? 2 : 0;
#pragma GCC unroll 4
  for (int rot = 0; rot < R; rot++) {
    for (int col = 0; col < 10; col++) {
      if (col > 0) ret[0][rot][col] = m.frame[rot][col] & m.frame[rot][col-1];
      if (col < 9) ret[1][rot][col] = m.frame[rot][col] & m.frame[rot][col+1];
#ifdef DOUBLE_TUCK
      if (col > 1) ret[2][rot][col] = m.frame[rot][col] & m.drop[rot][col-1] & m.drop[rot][col-1] >> 1 & m.frame[rot][col-2] >> 2;
      if (col < 8) ret[3][rot][col] = m.frame[rot][col] & m.drop[rot][col+1] & m.drop[rot][col+1] >> 1 & m.frame[rot][col+2] >> 2;
#endif
    }
  }
  if (R == 1) return ret;
#pragma GCC unroll 4
  for (int rot = 0; rot < R; rot++) {
    int nrot = (rot + 1) % R;
    for (int col = 0; col < 10; col++) {
      ret[x+2][rot][col] = m.frame[rot][col] & m.frame[nrot][col];
      if (col > 0) ret[x+3][rot][col] = ret[0][rot][col] & m.frame[nrot][col-1];
      if (col < 9) ret[x+4][rot][col] = ret[1][rot][col] & m.frame[nrot][col+1];
      if (col > 0) ret[x+5][rot][col] = m.frame[rot][col] & (m.drop[nrot][col] | m.drop[rot][col-1]) & m.frame[nrot][col-1] >> 1;
      if (col < 9) ret[x+6][rot][col] = m.frame[rot][col] & (m.drop[nrot][col] | m.drop[rot][col+1]) & m.frame[nrot][col+1] >> 1;
    }
  }
  if (R == 2) return ret;
#pragma GCC unroll 4
  for (int rot = 0; rot < R; rot++) {
    int nrot = (rot + 3) % R;
    for (int col = 0; col < 10; col++) {
      ret[x+7][rot][col] = m.frame[rot][col] & m.frame[nrot][col];
      if (col > 0) ret[x+8][rot][col] = ret[0][rot][col] & m.frame[nrot][col-1];
      if (col < 9) ret[x+9][rot][col] = ret[1][rot][col] & m.frame[nrot][col+1];
      if (col > 0) ret[x+10][rot][col] = m.frame[rot][col] & (m.drop[nrot][col] | m.drop[rot][col-1]) & m.frame[nrot][col-1] >> 1;
      if (col < 9) ret[x+11][rot][col] = m.frame[rot][col] & (m.drop[nrot][col] | m.drop[rot][col+1]) & m.frame[nrot][col+1] >> 1;
    }
  }
  return ret;
}

template <int R>
NOINLINE constexpr void SearchTucks(
    Level level,
    const Column cols[R][10],
    const TuckMasks<R> tuck_masks,
    const Column lock_positions_without_tuck[R][10],
    const Frames can_tuck_frame_masks[R][10],
    int& sz, Position* positions) {
  constexpr TuckTypeTable<R> tucks;
  Frames tuck_result[R][10] = {};
  for (int i = 0; i < TuckTypes(R); i++) {
    const auto& tuck = tucks.table[i];
    int start_col = std::max(0, -tuck.delta_col);
    int end_col = std::min(10, 10 - tuck.delta_col);
    for (int rot = 0; rot < R; rot++) {
      int nrot = (rot + tuck.delta_rot) % R;
      for (int col = start_col; col < end_col; col++) {
        tuck_result[nrot][col + tuck.delta_col] |=
            (tuck_masks[i][rot][col] & can_tuck_frame_masks[rot][col]) << tuck.delta_frame;
      }
    }
  }
  for (int rot = 0; rot < R; rot++) {
    for (int col = 0; col < 10; col++) {
      Column after_tuck_positions = FramesToColumn(level, tuck_result[rot][col]);
      Column cur = cols[rot][col];
      Column tuck_lock_positions = (after_tuck_positions + cur) >> 1 & (cur & ~cur >> 1) & ~lock_positions_without_tuck[rot][col];
      while (tuck_lock_positions) {
        int row = ctz(tuck_lock_positions);
        positions[sz++] = {rot, row, col};
        tuck_lock_positions ^= 1 << row;
      }
    }
  }
}

template <int R, class Tap, class Entry>
constexpr void CheckOneInitial(
    Level level, int adj_frame, const Tap& taps, bool is_adj,
    int total_frames, int initial_frame, const Entry& entry, const Column cols[R][10],
    Column lock_positions_without_tuck[R][10],
    Frames can_tuck_frame_masks[R][10],
    int& sz, Position* positions,
    bool& can_adj, bool& phase_2_possible) {
  int start_frame = (entry.num_taps == 0 ? 0 : taps[entry.num_taps - 1]) + initial_frame;
  int start_row = GetRow(start_frame, level);
  int end_frame = is_adj ? total_frames : std::max(adj_frame, taps[entry.num_taps]);
  // Since we verified masks_nodrop, start_row should be in col
  //if ((cols[entry.rot][entry.col] & 1 << start_row) == 0) throw std::runtime_error("unexpected");
  int lock_row = FindLockRow(cols[entry.rot][entry.col], start_row);
  int lock_frame = GetLastFrameOnRow(lock_row, level) + 1;
  if (!is_adj && lock_frame > end_frame) {
    can_adj = true;
  } else {
    positions[sz++] = {entry.rot, lock_row, entry.col};
  }
  int first_tuck_frame = initial_frame + taps[entry.num_taps];
  int last_tuck_frame = std::min(lock_frame, end_frame);
  lock_positions_without_tuck[entry.rot][entry.col] |= 1 << lock_row;
  if (last_tuck_frame > first_tuck_frame) {
    can_tuck_frame_masks[entry.rot][entry.col] = (1ll << last_tuck_frame) - (1ll << first_tuck_frame);
    phase_2_possible = true;
  }
}

template <int R>
constexpr FrameMasks<R> GetColsAndFrameMasks(Level level, const std::array<Board, R>& board, Column cols[R][10]) {
  FrameMasks<R> frame_masks = {};
  for (int rot = 0; rot < R; rot++) {
    for (int col = 0; col < 10; col++) {
      cols[rot][col] = board[rot].Column(col);
      // ColumnToNormalFrameMask<level>(col), ColumnToDropFrameMask<level>(col)
      frame_masks.frame[rot][col] = ColumnToNormalFrameMask(level, cols[rot][col]);
      frame_masks.drop[rot][col] = ColumnToDropFrameMask(level, cols[rot][col]);
    }
  }
  return frame_masks;
}

template <int R>
int DoOneSearch(
    bool is_adj, int initial_taps, Level level, int adj_frame, const int taps[],
    const std::vector<TableEntryNoTmpl>& table,
    const std::array<Board, R>& board, const Column cols[R][10],
    const TuckMasks<R> tuck_masks,
    bool can_adj[],
    Position* positions) {
  int total_frames = GetLastFrameOnRow(19, level) + 1;
  int N = table.size();
  int initial_frame = is_adj ? std::max(adj_frame, taps[initial_taps]) : 0;
  if (initial_frame >= total_frames) return 0;

  int sz = 0;
  // phase 1
  Frames can_tuck_frame_masks[R][10] = {}; // frames that can start a tuck
  Column lock_positions_without_tuck[R][10] = {};

  bool phase_2_possible = false;
  bool can_reach[R * 10] = {};
  for (int i = 0; i < N; i++) {
    can_reach[i] = Contains4<R>(board, table[i].masks_nodrop);
  }
  for (int i = 0; i < N; i++) {
    if (!can_reach[i]) continue;
    CheckOneInitial<R>(
        level, adj_frame, taps, is_adj, total_frames, initial_frame, table[i], cols,
        lock_positions_without_tuck, can_tuck_frame_masks,
        sz, positions, can_adj[i], phase_2_possible);
  }
  if (phase_2_possible) {
    SearchTucks<R>(level, cols, tuck_masks, lock_positions_without_tuck, can_tuck_frame_masks, sz, positions);
  }
  return sz;
}

template <int R>
inline PossibleMoves MoveSearchInternal(
    Level level, int adj_frame, const int taps[], const Phase1TableNoTmpl& table,
    const std::array<Board, R>& board) {
  Column cols[R][10] = {};
  auto tuck_masks = GetTuckMasks<R>(GetColsAndFrameMasks<R>(level, board, cols));
  bool can_adj[R * 10] = {}; // whether adjustment starting from this (rot, col) is possible

  PossibleMoves ret;
  Position buf[256];
  ret.non_adj.assign(buf, buf + DoOneSearch<R>(
      false, 0, level, adj_frame, taps, table.initial, board, cols, tuck_masks, can_adj, buf));

  for (size_t i = 0; i < table.initial.size(); i++) {
    auto& entry = table.initial[i];
    if (!can_adj[i]) continue;
    int x = DoOneSearch<R>(
        true, entry.num_taps, level, adj_frame, taps, table.adj[i], board, cols, tuck_masks, can_adj, buf);
    if (x) {
      int row = GetRow(std::max(adj_frame, taps[entry.num_taps]), level);
      ret.adj.emplace_back(Position{entry.rot, row, entry.col}, std::vector<Position>(buf, buf + x));
    }
  }
  return ret;
}

} // namespace move_search

using PrecomputedTable = move_search::Phase1TableNoTmpl;

template <int R>
NOINLINE PossibleMoves MoveSearch(
    Level level, int adj_frame, const int taps[], const PrecomputedTable& table,
    const std::array<Board, R>& board) {
  return move_search::MoveSearchInternal<R>(level, adj_frame, taps, table, board);
}

template <int R>
NOINLINE PossibleMoves MoveSearch(
    Level level, int adj_frame, const int taps[], const std::array<Board, R>& board) {
  PrecomputedTable table(level, R, adj_frame, taps);
  return move_search::MoveSearchInternal<R>(level, adj_frame, taps, table, board);
}

template <int R, class Taps>
inline PossibleMoves MoveSearch(
    Level level, int adj_frame, const PrecomputedTable& table, const std::array<Board, R>& board) {
  constexpr Taps taps{};
  return MoveSearch<R>(level, adj_frame, taps.data(), table, board);
}

template <int R, class Taps>
inline PossibleMoves MoveSearch(Level level, int adj_frame, const std::array<Board, R>& board) {
  constexpr Taps taps{};
  return MoveSearch<R>(level, adj_frame, taps.data(), board);
}

class PrecomputedTableTuple {
  PrecomputedTable tables[kLevels][3];
 public:
  static constexpr int TableIndex(int piece) {
    switch (piece) {
      case 0: return 2;
      case 1: return 2;
      case 2: return 1;
      case 3: return 0;
      case 4: return 1;
      case 5: return 2;
      case 6: return 1;
    }
    unreachable();
  }

  PrecomputedTableTuple() {}
  PrecomputedTableTuple(int adj_frame, const int taps[]) {
    for (int i = 0; i < kLevels; i++) {
      Level level = static_cast<Level>(i);
      tables[i][0] = PrecomputedTable(level, 1, adj_frame, taps);
      tables[i][1] = PrecomputedTable(level, 2, adj_frame, taps);
      tables[i][2] = PrecomputedTable(level, 4, adj_frame, taps);
    }
  }

  const PrecomputedTable& operator()(Level level, int table_index) const {
    return tables[static_cast<int>(level)][table_index];
  }
};

class PrecomputedTableCache {
  std::mutex mtx_;
  std::unordered_map<std::pair<std::array<int, 10>, int>, PrecomputedTableTuple> cache_;
 public:
  const PrecomputedTableTuple& operator()(decltype(cache_)::key_type&& key) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (auto it = cache_.find(key); it != cache_.end()) {
      return it->second;
    } else {
      lck.unlock();
      PrecomputedTableTuple table(key.second, key.first.data());
      lck.lock();
      it = cache_.emplace(key, std::move(table)).first;
      return it->second;
    }
  }
};

inline PossibleMoves MoveSearch(
    Level level, int adj_frame, const int taps[], const PrecomputedTableTuple& table,
    const Board& b, int piece) {
#define ONE_CASE(x) \
    case x: return MoveSearch<Board::NumRotations(x)>( \
                level, adj_frame, taps, table(level, PrecomputedTableTuple::TableIndex(x)), b.PieceMap<x>());
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}

template <class Taps>
inline PossibleMoves MoveSearch(Level level, int adj_frame, const Board& b, int piece) {
#define ONE_CASE(x) \
    case x: return MoveSearch<Board::NumRotations(x), Taps>(level, adj_frame, b.PieceMap<x>());
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}
