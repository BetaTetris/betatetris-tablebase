#include <array>
#include <set>
#include "naive_functions.h"

namespace {

constexpr int kT = 7, kN = 20, kM = 10;

using Poly = std::array<std::pair<int, int>, 4>;
const std::vector<Poly> kBlocks[kT] = {
    {{{{1, 0}, {0, 0}, {0, 1}, {0, -1}}}, // T
     {{{1, 0}, {0, 0}, {-1, 0}, {0, -1}}},
     {{{0, -1}, {0, 0}, {0, 1}, {-1, 0}}},
     {{{1, 0}, {0, 0}, {0, 1}, {-1, 0}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, 1}}}, // J
     {{{-1, 0}, {0, 0}, {1, -1}, {1, 0}}},
     {{{-1, -1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, 0}, {1, 1}}}, // Z
     {{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, -1}, {1, 0}}}}, // O
    {{{{0, 0}, {0, 1}, {1, -1}, {1, 0}}}, // S
     {{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, -1}}}, // L
     {{{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}},
     {{{-1, 1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {0, 0}, {1, 0}, {1, 1}}}},
    {{{{0, -2}, {0, -1}, {0, 0}, {0, 1}}}, // I
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};

constexpr int GetRow(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame / 3;
    case kLevel19: return frame / 2;
    case kLevel29: return frame;
    case kLevel39: return frame * 2;
  }
  __builtin_unreachable();
}
constexpr bool IsDropFrame(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame % 3 == 2;
    case kLevel19: return frame % 2 == 1;
    default: return true;
  }
}

inline std::pair<bool, Position> FreeDrop(const std::vector<ByteBoard>& b, int rot, int row, int col, Level level, int max_frame) {
  int max_row = GetRow(max_frame, level);
  while (row < max_row && row < 19 && b[rot][row+1][col]) row++;
  return {row >= max_row, {rot, row, col}};
}
inline Position FreeDrop(const std::vector<ByteBoard>& b, int rot, int row, int col) {
  while (row < 19 && b[rot][row+1][col]) row++;
  return {rot, row, col};
}
constexpr int kFinish = 1000;

void SimulateMove(
    const std::vector<ByteBoard>& b, Level level, const int taps[],
    int start_rot, int start_col, int start_frame, int end_frame, int num_lr, int num_ab, bool is_l, bool is_a,
    bool check_tuck, const std::set<Position>& non_tuck,
    std::vector<Position>& locked_placements, std::vector<std::pair<Position, int>>* continue_placements) {
  if (is_l ? (num_lr > start_col) : (num_lr > 9 - start_col)) return;
  if (is_a ? (num_ab > (int)b.size() / 2) : (num_ab > ((int)b.size() - 1) / 2)) return;
  if (is_l && num_lr == 0) return;
  if (is_a && num_ab == 0) return;
  int rot = start_rot, col = start_col, frame = start_frame;
  const int tot_taps = std::max(num_lr, num_ab);
  for (int tap = 0; tap < tot_taps; frame++) {
    int row = GetRow(frame, level);
    if (row >= 20 || !b[rot][row][col]) return;
    if (frame == taps[tap] + start_frame) {
      tap++;
      if (tap <= num_lr) {
        if (is_l) {
          if (!b[rot][row][--col]) return;
        } else {
          if (!b[rot][row][++col]) return;
        }
      }
      if (tap <= num_ab) {
        if (is_a) {
          if (!b[rot=(rot+1)%b.size()][row][col]) return;
        } else {
          if (!b[rot=(rot+b.size()-1)%b.size()][row][col]) return;
        }
      }
      if (tap == tot_taps) break;
    }
    if (IsDropFrame(frame, level)) {
      if (++row >= 20 || !b[rot][row][col]) return;
      if (level == kLevel39) {
        if (++row >= 20 || !b[rot][row][col]) return;
      }
    }
  }
  { // forward to tuck available
    auto pos = FreeDrop(b, rot, GetRow(frame, level), col, level, start_frame + taps[tot_taps]);
    if (!pos.first) {
      locked_placements.push_back(pos.second);
      return;
    }
    frame = start_frame + taps[tot_taps];
  }
  { // check continue
    auto pos = FreeDrop(b, rot, GetRow(frame, level), col, level, end_frame);
    if (pos.first) {
      continue_placements->push_back({pos.second, std::max(end_frame, frame)});
    } else {
      locked_placements.push_back(pos.second);
    }
  }
  if (!check_tuck) return;
  std::set<Position> tuck_placements;
  auto Insert = [&](const Position& p) {
    if (!non_tuck.count(p)) tuck_placements.insert(p);
  };
  for (; frame < end_frame; frame++) {
    int row = GetRow(frame, level);
    if (row >= 20 || !b[rot][row][col]) break;

    int nrow = GetRow(frame+1, level);
    int mrow = nrow - row == 2 ? nrow-1 : nrow;
#ifdef DOUBLE_TUCK
    int n2row = GetRow(frame+2, level);
    int m2row = n2row - nrow == 2 ? n2row-1 : n2row;
#endif
    int arot = (rot+1)%b.size();
    int brot = (rot+b.size()-1)%b.size();
    if (col < 9 && b[rot][row][col+1]) {
      Insert(FreeDrop(b, rot, row, col+1));
      if (b.size() >= 2 && b[arot][row][col+1]) Insert(FreeDrop(b, arot, row, col+1));
      if (b.size() >= 4 && b[brot][row][col+1]) Insert(FreeDrop(b, brot, row, col+1));
      if (nrow < 20 && b[rot][mrow][col+1] && b[rot][nrow][col+1]) {
        if (b.size() >= 2 && b[arot][nrow][col+1]) Insert(FreeDrop(b, arot, nrow, col+1));
        if (b.size() >= 4 && b[brot][nrow][col+1]) Insert(FreeDrop(b, brot, nrow, col+1));
      }
#ifdef DOUBLE_TUCK
      if (n2row < 20 && col < 8 && b[rot][mrow][col+1] && b[rot][nrow][col+1] && b[rot][m2row][col+1] &&
          b[rot][n2row][col+1] && b[rot][n2row][col+2]) {
        Insert(FreeDrop(b, rot, n2row, col+2));
      }
#endif
    }
    if (col > 0 && b[rot][row][col-1]) {
      Insert(FreeDrop(b, rot, row, col-1));
      if (b.size() >= 2 && b[arot][row][col-1]) Insert(FreeDrop(b, arot, row, col-1));
      if (b.size() >= 4 && b[brot][row][col-1]) Insert(FreeDrop(b, brot, row, col-1));
      if (nrow < 20 && b[rot][mrow][col-1] && b[rot][nrow][col-1]) {
        if (b.size() >= 2 && b[arot][nrow][col-1]) Insert(FreeDrop(b, arot, nrow, col-1));
        if (b.size() >= 4 && b[brot][nrow][col-1]) Insert(FreeDrop(b, brot, nrow, col-1));
      }
#ifdef DOUBLE_TUCK
      if (n2row < 20 && col > 1 && b[rot][mrow][col-1] && b[rot][nrow][col-1] && b[rot][m2row][col-1] &&
          b[rot][n2row][col-1] && b[rot][n2row][col-2]) {
        Insert(FreeDrop(b, rot, n2row, col-2));
      }
#endif
    }
    if (b.size() >= 2 && b[arot][row][col]) {
      Insert(FreeDrop(b, arot, row, col));
      if (nrow < 20 && b[arot][mrow][col] && b[arot][nrow][col]) {
        if (col < 9 && b[arot][nrow][col+1]) Insert(FreeDrop(b, arot, nrow, col+1));
        if (col > 0 && b[arot][nrow][col-1]) Insert(FreeDrop(b, arot, nrow, col-1));
      }
    }
    if (b.size() >= 4 && b[brot][row][col]) {
      Insert(FreeDrop(b, brot, row, col));
      if (nrow < 20 && b[brot][mrow][col] && b[brot][nrow][col]) {
        if (col < 9 && b[brot][nrow][col+1]) Insert(FreeDrop(b, brot, nrow, col+1));
        if (col > 0 && b[brot][nrow][col-1]) Insert(FreeDrop(b, brot, nrow, col-1));
      }
    }

    if (IsDropFrame(frame, level)) {
      if (++row >= 20 || !b[rot][row][col]) break;
      if (level == kLevel39) {
        if (++row >= 20 || !b[rot][row][col]) break;
      }
    }
  }
  for (auto& i : tuck_placements) locked_placements.push_back(i);
}

void MoveSearch(
    const std::vector<ByteBoard>& b, Level level, const int taps[],
    int max_lr, int max_ab, int start_rot, int start_col, int start_frame, int end_frame,
    bool check_tuck, const std::set<Position>& non_tuck,
    std::vector<Position>& locked_placements, std::vector<std::pair<Position, int>>* continue_placements) {
  if (!b[start_rot][GetRow(start_frame, level)][start_col]) return;
  for (int lr = 0; lr <= max_lr; lr++) {
    for (int ab = 0; ab <= max_ab; ab++) {
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, false, false, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, true, false, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, false, true, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, true, true, check_tuck, non_tuck, locked_placements, continue_placements);
    }
  }
}

void DFSNoro(int g, int x, int y, int taps_per_row, const ByteBoard& b, std::vector<ByteBoard>& vis) {
  int states = taps_per_row ? taps_per_row + 1 : 3;
  if (x < 0 || x >= 20 || y < 0 || y >= 10 || g >= states) return;
  if (!b[x][y] || vis[g][x][y]) return;
  vis[g][x][y] = 1;
  if (taps_per_row) {
    DFSNoro(g+1, x, y-1, taps_per_row, b, vis);
    DFSNoro(g+1, x, y+1, taps_per_row, b, vis);
    DFSNoro(0, x+1, y, taps_per_row, b, vis);
  } else {
    if (g == 1) {
      DFSNoro(2, x+1, y, taps_per_row, b, vis);
    } else {
      if (g == 0) {
        DFSNoro(1, x, y-1, taps_per_row, b, vis);
        DFSNoro(1, x, y+1, taps_per_row, b, vis);
      }
      DFSNoro(0, x+1, y, taps_per_row, b, vis);
    }
  }
}

void DFSNoro(int s, int g, int x, int y, int taps_per_row, const ByteBoard& b, std::vector<ByteBoard>& vis) {
  int row_taps_per_row = taps_per_row ? taps_per_row : (x % 2 == 0);
  if (x < 0 || x >= 20 || y < 0 || y >= 10 || g > row_taps_per_row) return;
  int id = s == 0 ? 0 : (s == 1 ? g + 1 : g + 2 + std::max(1, taps_per_row));
  if (!b[x][y] || vis[id][x][y]) return;
  vis[id][x][y] = 1;
  if (s == 1) {
    DFSNoro(s, g+1, x, y-1, taps_per_row, b, vis);
  } else if (s == 2) {
    DFSNoro(s, g+1, x, y+1, taps_per_row, b, vis);
  }
  if (g == row_taps_per_row) {
    DFSNoro(s, 0, x+1, y, taps_per_row, b, vis);
  }
  DFSNoro(0, 0, x+1, y, taps_per_row, b, vis);
}

ByteBoard MergeByteBoardsAndLock(const std::vector<ByteBoard>& v) {
  ByteBoard tmp{}, ret{};
  for (auto& x : v) {
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < 10; j++) tmp[i][j] |= x[i][j];
    }
  }
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 10; j++) ret[i][j] = tmp[i][j] && (i == 19 || !tmp[i+1][j]);
  }
  return ret;
}

} // namespace

std::vector<ByteBoard> GetPieceMap(const ByteBoard& field, int poly) {
  const size_t R = kBlocks[poly].size();
  std::vector<ByteBoard> ret(R, ByteBoard{});
  for (size_t r = 0; r < R; r++) {
    auto& pl = kBlocks[poly][r];
    for (int x = 0; x < kN; x++) {
      for (int y = 0; y < kM; y++) {
        bool flag = true;
        for (int i = 0; i < 4; i++) {
          int nx = pl[i].first + x, ny = pl[i].second + y;
          if (ny < 0 || nx >= kN || ny >= kM || (nx >= 0 && !field[nx][ny])) {
            flag = false;
            break;
          }
        }
        ret[r][x][y] = flag;
      }
    }
  }
  return ret;
}

ByteBoard NaiveNoroPossibleMoves(const ByteBoard& b, int inputs_per_row, bool do_tuck) {
  if (do_tuck) {
    std::vector<ByteBoard> vis(inputs_per_row ? inputs_per_row + 1 : 3, ByteBoard{});
    DFSNoro(0, 0, 5, inputs_per_row, b, vis);
    DFSNoro(1, 0, 4, inputs_per_row, b, vis);
    DFSNoro(1, 0, 6, inputs_per_row, b, vis);
    return MergeByteBoardsAndLock(vis);
  } else {
    std::vector<ByteBoard> vis(std::max(1, inputs_per_row) * 2 + 3, ByteBoard{});
    DFSNoro(1, 0, 0, 5, inputs_per_row, b, vis);
    DFSNoro(2, 0, 0, 5, inputs_per_row, b, vis);
    DFSNoro(1, 1, 0, 4, inputs_per_row, b, vis);
    DFSNoro(2, 1, 0, 6, inputs_per_row, b, vis);
    return MergeByteBoardsAndLock(vis);
  }
}

ByteBoard PlacePiece(const ByteBoard& b, int poly, int r, int x, int y) {
  ByteBoard field(b);
  auto& pl = kBlocks[poly][r];
  for (auto& i : pl) {
    int nx = x + i.first, ny = y + i.second;
    if (nx >= kN || ny >= kM || nx < 0 || ny < 0) continue;
    field[nx][ny] = false;
  }
  return field;
}

int ClearLines(ByteBoard& field) {
  int i = kN - 1, j = kN - 1;
  for (; i >= 0; i--, j--) {
    bool flag = false;
    for (int y = 0; y < kM; y++) flag |= field[i][y];
    if (!flag) {
      j++;
    } else if (i != j) {
      field[j] = field[i];
    }
  }
  int ans = j + 1;
  for (; j >= 0; j--) std::fill(field[j].begin(), field[j].end(), true);
  return ans;
}

PossibleMoves NaiveGetPossibleMoves(const std::vector<ByteBoard>& b, Level level, int adj_frame, const int taps[]) {
  PossibleMoves ret;

  std::vector<Position> non_tuck;
  std::set<Position> non_tuck_set;
  std::vector<std::pair<Position, int>> adj_starts;

  int max_lr = adj_frame == 0 ? 0 : 9;
  int max_ab = adj_frame == 0 ? 0 : 2;
  MoveSearch(b, level, taps, max_lr, max_ab, 0, 5, 0, kFinish, false, non_tuck_set, non_tuck, nullptr);
  non_tuck_set = std::set<Position>(non_tuck.begin(), non_tuck.end());
  MoveSearch(b, level, taps, max_lr, max_ab, 0, 5, 0, adj_frame, true, non_tuck_set, ret.non_adj, &adj_starts);

  for (auto& i : adj_starts) {
    non_tuck.clear();
    non_tuck_set.clear();
    ret.adj.push_back({i.first, {}});
    MoveSearch(b, level, taps, 9, 2, i.first.r, i.first.y, i.second, kFinish, false, non_tuck_set, non_tuck, nullptr);
    non_tuck_set = std::set<Position>(non_tuck.begin(), non_tuck.end());
    MoveSearch(b, level, taps, 9, 2, i.first.r, i.first.y, i.second, kFinish, true, non_tuck_set, ret.adj.back().second, nullptr);
  }
  return ret;
}

int NumOverhang(const ByteBoard& b) {
  int cnt = 0;
  for (int x = 1; x < kN; x++) {
    for (int y = 0; y < kM; y++) {
      if (!b[x - 1][y] && b[x][y]) cnt++;
    }
  }
  return cnt;
}

std::array<int, 10> ColumnHeights(const ByteBoard& b) {
  std::array<int, 10> ret{};
  for (int x = kN - 1; x >= 0; x--) {
    for (int y = 0; y < kM; y++) {
      if (!b[x][y]) ret[y] = kN - x;
    }
  }
  return ret;
}
