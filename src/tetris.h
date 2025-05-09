#pragma once

#include <cstring>
#include <bitset>
#include <unordered_map>
#include "hash.h"
#include "game.h"
#include "move_search.h"
#include "move_search_noro.h"
#include "frame_sequence.h"

class Tetris {
 public:
  using MoveMap = std::array<ByteBoard, 4>;
  static constexpr uint8_t kNoAdj = 1;
  static constexpr uint8_t kHasAdjReduced = 2;
  static constexpr uint8_t kHasAdjNonReduced = 3;

 private:
  Board board_;
  int lines_;
  int pieces_;
  bool is_adj_;
  int initial_move_;
  int now_piece_;
  int next_piece_;
  bool game_over_;
  PossibleMoves moves_;
  MoveMap move_map_;
  int consecutive_fail_;
  uint64_t initial_mask_;

  std::array<int, 10> tap_sequence_;
  int adj_delay_;
  const PrecomputedTableTuple* search_table_;
  static PrecomputedTableCache search_table_cache_;

  // stats
  int run_score_;
  int run_lines_;
  int run_pieces_;

  void CalculateInitialMask_() {
    if (moves_.adj.size() > 64) throw std::runtime_error("unexpected many initial placements");
    initial_mask_ = (1ll << moves_.adj.size()) - 1;
    if (moves_.adj.size() <= 1) return;

    using AdjItem = std::pair<Position, std::vector<Position>>;
    std::sort(moves_.adj.begin(), moves_.adj.end(), [](const AdjItem& x, const AdjItem& y) {
      if (x.second.size() != y.second.size()) return x.second.size() > y.second.size();
      return abs(x.first.y - 5) < abs(y.first.y - 5);
    });
    std::unordered_map<Position, uint8_t> pos_mp;
    for (auto& [_, i] : moves_.adj) {
      for (auto& j : i) pos_mp.emplace(j, pos_mp.size());
    }
    std::vector<std::bitset<256>> adj_bitset(moves_.adj.size());
    for (size_t i = 0; i < moves_.adj.size(); i++) {
      for (auto& j : moves_.adj[i].second) adj_bitset[i][pos_mp[j]] = true;
    }
    for (size_t i = 0; i < moves_.adj.size(); i++) {
      if (!(initial_mask_ >> i & 1)) continue;
      for (size_t j = 0; j < moves_.adj.size(); j++) {
        if (i == j || !(initial_mask_ >> j & 1) || moves_.adj[i].second.size() < moves_.adj[j].second.size()) continue;
        if ((adj_bitset[i] & adj_bitset[j]) == adj_bitset[j]) initial_mask_ &= ~(1ll << j);
      }
    }
  }

  void CalculateMoves_(bool regenerate) {
    if (regenerate) {
      moves_ = MoveSearch(LevelSpeed(), adj_delay_, tap_sequence_.data(), *search_table_, board_, now_piece_);
      if (moves_.non_adj.empty() && moves_.adj.empty()) {
        game_over_ = true;
        return;
      }
      CalculateInitialMask_();
    }
    memset(move_map_.data(), 0, sizeof(move_map_));
    if (!is_adj_) {
      for (auto& i : moves_.non_adj) move_map_[i.r][i.x][i.y] = kNoAdj;
      for (size_t idx = 0; idx < moves_.adj.size(); idx++) {
        auto& i = moves_.adj[idx].first;
        move_map_[i.r][i.x][i.y] = (initial_mask_ >> idx & 1) ? kHasAdjNonReduced : kHasAdjReduced;
      }
    } else {
      for (auto& i : moves_.adj[initial_move_].second) move_map_[i.r][i.x][i.y] = kNoAdj;
    }
  }

  std::pair<int, int> StepGame_(const Position& pos, int next_piece) {
    auto before_clear = board_.Place(now_piece_, pos.r, pos.x, pos.y);
    // do not allow placing pieces to be cut off from the board
    if (board_.Count() + 4 != before_clear.Count()) {
      consecutive_fail_++;
      return {-1, 0};
    }

    auto [lines, new_board] = before_clear.ClearLines();
    int delta_score = GameScore(lines_, lines);
    lines_ += lines;
    board_ = new_board;
    pieces_++;
    is_adj_ = false;
    initial_move_ = 0;
    now_piece_ = next_piece_;
    next_piece_ = next_piece;
    if (lines_ >= kLineCap || (kTetrisOnly && lines && lines != 4)) {
      game_over_ = true;
    } else {
      CalculateMoves_(true);
    }
    consecutive_fail_ = 0;
    run_score_ += delta_score;
    run_lines_ += lines;
    run_pieces_++;
    return {delta_score, lines};
  }

 public:
  Tetris() :
#if defined(TAP_SPEED) && defined(ADJ_DELAY)
      tap_sequence_(TAP_SPEED().toarray()), adj_delay_{ADJ_DELAY} {}
#else
      tap_sequence_(Tap30Hz().toarray()), adj_delay_{18} {}
#endif

  void Reset(const Board& b, int lines, int now_piece, int next_piece, const int tap_sequence[10], int adj_delay) {
    memcpy(tap_sequence_.data(), tap_sequence, sizeof(tap_sequence_));
    adj_delay_ = adj_delay;
    search_table_ = &search_table_cache_({tap_sequence_, adj_delay_});
    int pieces = (lines * 10 + b.Count()) / 4;
    if (pieces * 4 != lines * 10 + (int)b.Count()) throw std::runtime_error("Incorrect lines");
    board_ = b;
    lines_ = lines;
    pieces_ = pieces;
    is_adj_ = false;
    initial_move_ = 0;
    now_piece_ = now_piece;
    next_piece_ = next_piece;
    game_over_ = false;
    CalculateMoves_(true);
    consecutive_fail_ = 0;
    run_score_ = 0;
    run_lines_ = 0;
    run_pieces_ = 0;
  }

  void Reset(const Board& b, int lines, int now_piece, int next_piece) {
#if defined(TAP_SPEED) && defined(ADJ_DELAY)
    constexpr TAP_SPEED tap_table;
    Reset(b, lines, now_piece, next_piece, tap_table.data(), ADJ_DELAY);
#else
    constexpr Tap30Hz tap_table;
    Reset(b, lines, now_piece, next_piece, tap_table.data(), 18);
#endif
  }

  bool IsNoAdjMove(const Position& pos) const {
    return move_map_[pos.r][pos.x][pos.y] == kNoAdj;
  }
  bool IsAdjMove(const Position& pos) const {
    return move_map_[pos.r][pos.x][pos.y] >= kHasAdjReduced;
  }

  std::pair<int, int> DirectPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    uint8_t location = move_map_[pos.r][pos.x][pos.y];
    if (!(location == kNoAdj || std::any_of(moves_.adj.begin(), moves_.adj.end(), [&pos](auto i) {
            return std::find(i.second.begin(), i.second.end(), pos) != i.second.end();
          }))) {
      game_over_ = true;
      return {-1, 0};
    }
    auto ret = StepGame_(pos, next_piece);
    if (ret.first == -1) game_over_ = true;
    return ret;
  }

  // (score, lines)
  // score == -1 if invalid
  std::pair<int, int> InputPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    uint8_t location = move_map_[pos.r][pos.x][pos.y];
    if (!location) {
      consecutive_fail_++;
      return {-1, 0};
    }
    if (location == kNoAdj) {
      return StepGame_(pos, next_piece);
    } else {
      for (size_t i = 0; i < moves_.adj.size(); i++) {
        if (moves_.adj[i].first == pos) {
          initial_move_ = i;
          break;
        }
      }
      is_adj_ = true;
      CalculateMoves_(false);
      consecutive_fail_ = 0;
      return {0, 0};
    }
  }

  FrameSequence GetSequence(const Position& pos) const {
    return GetFrameSequenceStart(LevelSpeed(), tap_sequence_.data(), board_, now_piece_, adj_delay_, pos);
  }

  std::pair<Position, FrameSequence> GetAdjPremove(const Position pos[7]) const {
    auto [idx, seq] = GetBestAdj(LevelSpeed(), tap_sequence_.data(), board_, now_piece_, moves_, adj_delay_, pos);
    return {moves_.adj[idx].first, seq};
  }

  void FinishAdjSequence(FrameSequence& seq, const Position& intermediate_pos, const Position& final_pos) const {
    GetFrameSequenceAdj(LevelSpeed(), tap_sequence_.data(), seq, board_, now_piece_, intermediate_pos, final_pos);
  }

  void SetNextPiece(int piece) {
    if (piece < 0 || piece >= (int)kPieces) throw std::range_error("Invalid piece");
    next_piece_ = piece;
  }

  void SetLines(int lines) {
    if (GetLevelSpeed(GetLevelByLines(lines)) != LevelSpeed()) {
      throw std::range_error("Cannot set lines to different speed");
    }
    int pieces_diff = (lines - lines_) * 10 / 4;
    lines_ = lines;
    pieces_ += pieces_diff;
  }

  void ForceOver() { game_over_ = true; }

  const MoveMap& GetPossibleMoveMap() const { return move_map_; }
  const PossibleMoves& GetPossibleMoveList() const { return moves_; }
  uint64_t GetInitialMask() const { return initial_mask_; }
  const Board& GetBoard() const { return board_; }
  const int* GetTapSequence() const { return tap_sequence_.data(); }
  int GetAdjDelay() const { return adj_delay_; }

  int GetLevel() const { return GetLevelByLines(lines_); }
  Level LevelSpeed() const { return GetLevelSpeed(GetLevel()); }
  bool IsAdj() const { return is_adj_; }
  int GetPieces() const { return pieces_; }
  int GetLines() const { return lines_; }
  int NowPiece() const { return now_piece_; }
  int NextPiece() const { return next_piece_; }
  bool IsOver() const { return game_over_ || consecutive_fail_ >= 1; }
  Position InitialMove() const {
    if (!is_adj_) throw std::logic_error("No initial move");
    return moves_.adj[initial_move_].first;
  }

  int RunPieces() const { return run_pieces_; }
  int RunLines() const { return run_lines_; }
  int RunScore() const { return run_score_; }
};

class TetrisNoro {
 private:
  Board board_;
  int lines_;
  int start_level_;
  int pieces_;
  int now_piece_;
  int next_piece_;
  bool game_over_;
  Board moves_;
  int consecutive_fail_;

  bool do_tuck_;
  std::vector<int> inputs_per_row_;

  // stats
  int run_score_;
  int run_lines_;
  int run_pieces_;

  void CalculateMoves_() {
    moves_ = MoveSearchNoro(board_, now_piece_, InputsPerRow(), do_tuck_);
    if (moves_ == Board::Zeros) {
      game_over_ = true;
      return;
    }
  }

  std::pair<int, int> StepGame_(const Position& pos, int next_piece) {
    if (pos.r != 0) throw std::invalid_argument("pos.r must be 0");
    auto before_clear = board_.Place(now_piece_, 0, pos.x, pos.y);

    auto [lines, new_board] = before_clear.ClearLines(true);
    lines_ += lines;
    int delta_score = ScoreFromLevel(GetLevel(), lines);
    board_ = new_board;
    pieces_++;
    now_piece_ = next_piece_;
    next_piece_ = next_piece;
    CalculateMoves_();
    consecutive_fail_ = 0;
    run_score_ += delta_score;
    run_lines_ += lines;
    run_pieces_++;
    return {delta_score, lines};
  }

 public:
  TetrisNoro() : inputs_per_row_{9, 9, 9, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 0} {}

  void Reset(const Board& b, int lines, int start_level, bool do_tuck, int now_piece, int next_piece) {
    int pieces = (lines * 10 + b.Count()) / 4;
    // if (pieces * 4 != lines * 10 + (int)b.Count()) throw std::runtime_error("Incorrect lines");
    board_ = b;
    lines_ = lines;
    start_level_ = start_level;
    pieces_ = pieces;
    do_tuck_ = do_tuck;
    now_piece_ = now_piece;
    next_piece_ = next_piece;
    game_over_ = false;
    CalculateMoves_();
    consecutive_fail_ = 0;
    run_score_ = 0;
    run_lines_ = 0;
    run_pieces_ = 0;
  }

  // (score, lines)
  // score == -1 if invalid
  std::pair<int, int> InputPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    if (!moves_.Cell(pos.x, pos.y)) {
      consecutive_fail_++;
      return {-1, 0};
    }
    return StepGame_(pos, next_piece);
  }

  void SetNextPiece(int piece) {
    if (piece < 0 || piece >= (int)kPieces) throw std::range_error("Invalid piece");
    next_piece_ = piece;
  }
  void SetLines(int lines) {
    if (lines % 2 != lines_ % 2) throw std::range_error("Invalid lines");
    int pieces_diff = (lines - lines_) * 10 / 4;
    lines_ = lines;
    pieces_ += pieces_diff;
  }

  int LinesToNextSpeed() const {
    int speed = LevelSpeed();
    int next_speed = speed;
    while (next_speed < (int)inputs_per_row_.size() &&
           inputs_per_row_[speed] == inputs_per_row_[next_speed]) {
      next_speed++;
    }
    if (next_speed >= (int)inputs_per_row_.size()) return -1;
    int nlines = (lines_ + 9) * 10 / 10;
    for (; noro::GetLevelSpeed(noro::GetLevelByLines(nlines, start_level_)) != next_speed; nlines += 10);
    return nlines - lines_;
  }

  FrameSequence GetSequence(const Position& pos) const {
    return GetFrameSequenceNoro(board_, now_piece_, InputsPerRow(), do_tuck_, noro::GetFramesPerRow(GetLevel()), pos);
  }

  const Board& GetPossibleMoveMap() const { return moves_; }
  const Board& GetBoard() const { return board_; }
  int InputsPerRow() const {
    size_t speed = LevelSpeed();
    return speed >= inputs_per_row_.size() ? inputs_per_row_.back() : inputs_per_row_[speed];
  }
  int InputsPerRow(int level) const {
    size_t speed = noro::GetLevelSpeed(level);
    return speed >= inputs_per_row_.size() ? inputs_per_row_.back() : inputs_per_row_[speed];
  }
  bool DoTuck() const { return do_tuck_; }

  int GetLevel() const { return noro::GetLevelByLines(lines_, start_level_); }
  int LevelSpeed() const { return noro::GetLevelSpeed(GetLevel()); }
  int GetPieces() const { return pieces_; }
  int GetLines() const { return lines_; }
  int GetStartLevel() const { return start_level_; }
  int NowPiece() const { return now_piece_; }
  int NextPiece() const { return next_piece_; }
  bool IsOver() const { return game_over_ || consecutive_fail_ >= 1; }

  int RunPieces() const { return run_pieces_; }
  int RunLines() const { return run_lines_; }
  int RunScore() const { return run_score_; }
};
