#pragma once

#include <random>
#include "../../src/tetris.h"
#include "python.h"

class PythonTetris {
 public:
  PyObject_HEAD

 private:
  // rewards
  static constexpr int kMirrorCols_[] = {9, 9, 9, 10, 9, 9, 10};
  static constexpr int kMirrorPiece_[] = {0, 5, 4, 3, 2, 1, 6};
  static constexpr double kInvalidReward_ = -0.3;
#ifdef NO_ROTATION
  static constexpr double kRawMultiplier_ = 0.2;
  double step_reward_ = 2e-3;
#else // NO_ROTATION
#ifdef TETRIS_ONLY
  static constexpr double kRewardMultiplier_ = 2e-5; // 20 per maxout
  static constexpr double kBottomMultiplier_ = 1.1;
  static constexpr double kGameOverMultiplier_ = 1. / 16;
  static constexpr double kGameOverReward = -1.0;
  double step_reward_ = 5e-3;
#else // TETRIS_ONLY
  static constexpr double kRewardMultiplier_ = 1e-5; // 10 per maxout
  static constexpr double kBottomMultiplier_ = 1.1;
  double step_reward_ = 5e-4;
#endif // TETRIS_ONLY
  int step_reward_level_ = 0;
  double burn_over_multiplier_ = 0;
#endif // NO_ROTATION

  // states
  std::mt19937_64 rng_;
  int next_piece_;
  int piece_count_;
#ifdef NO_ROTATION
  bool is_mirror_;
  bool nnb_;
#else
  bool skip_unique_initial_;
#endif // NO_ROTATION

  int GenNextPiece_(int piece) {
#ifdef TETRIS_ONLY
    // generate more I pieces when training tetris only
    constexpr int kThresh[4] = {28, 24, 16, 8};
    constexpr double kAdd[4] = {0.035, 0.046, 0.06, 0.09};
    int level_int = static_cast<int>(tetris.LevelSpeed());
    int threshold = kThresh[level_int];
    double add = kAdd[level_int];
    if (tetris.RunLines() >= threshold) {
      float prob = add * 0.3 + add * 0.7 * std::min((tetris.RunLines() - threshold) / (threshold * 0.5), 1.0);
      if (std::uniform_real_distribution<float>(0, 1)(rng_) < prob) return 6;
    }
#endif // TETRIS_ONLY
    piece_count_ = (piece_count_ + 1) & 7;
#if false // use realistic distribution or not
    return std::discrete_distribution<int>(
        kTransitionRealisticProbInt[piece_count_][piece],
        kTransitionRealisticProbInt[piece_count_][piece] + kPieces)(rng_);
#else
    return std::discrete_distribution<int>(
        kTransitionProbInt[piece],
        kTransitionProbInt[piece] + kPieces)(rng_);
#endif
  }

  std::pair<double, double> StepAndCalculateReward_(const Position& pos, int score, int lines) {
    if (score == -1) return {kInvalidReward_, 0.0f};
#ifdef NO_ROTATION
    int pre_lines = tetris.GetLines() - lines;
    double n_reward = step_reward_;
    for (int i = pre_lines; i < pre_lines + lines; i++) {
      n_reward += std::exp(GetNoroLineRewardExp(i, tetris.GetStartLevel(), tetris.DoTuck(), nnb_));
    }
    next_piece_ = GenNextPiece_(next_piece_);
    double reward = lines * kRawMultiplier_;
#else // NO_ROTATION
    double reward = score * kRewardMultiplier_;
    double n_reward = reward;
    double n_step_reward = step_reward_;
    double bottom_multiplier = kBottomMultiplier_;
    int tap_4 = tetris.GetTapSequence()[3];
    if (step_reward_level_ == 0) {
      int now_lines = tetris.GetLines();
      int tap_mode = 0, adj_mode = 0;
      switch (tap_4) {
        case 6: tap_mode = tetris.GetTapSequence()[4] <= 10 ? 5 : 6; break;
        case 8: tap_mode = 4; break;
        case 9: tap_mode = 3; break;
        case 12: tap_mode = 2; break;
        case 15: tap_mode = 1; break;
        case 18: tap_mode = 0; break;
        default: throw std::runtime_error("unexpected tap");
      }
      switch (tetris.GetAdjDelay()) {
        case 0: adj_mode = 0; break;
        case 18: adj_mode = 1; break;
        case 21: adj_mode = 2; break;
        case 24: adj_mode = 3; break;
        case 30: adj_mode = 4; break;
        case 61: adj_mode = 5; break;
        default: throw std::runtime_error("unexpected adj");
      }
      constexpr double kOverProb[][7][6] = { // half survival chance in 100 lines
        {{0.032,0.029,0.029,0.027,0.027,0.023},
         {0.039,0.036,0.035,0.036,0.034,0.026},
         {0.044,0.039,0.038,0.039,0.038,0.028},
         {0.046,0.041,0.038,0.037,0.038,0.032},
         {0.044,0.043,0.04 ,0.04 ,0.037,0.031},
         {0.047,0.042,0.044,0.041,0.039,0.029},
         {0.042,0.038,0.038,0.037,0.037,0.027}},
        {{0.021,0.018,0.018,0.017,0.016,0.016},
         {0.029,0.026,0.024,0.023,0.021,0.02 },
         {0.035,0.03 ,0.028,0.029,0.024,0.023},
         {0.045,0.036,0.032,0.033,0.029,0.028},
         {0.045,0.036,0.035,0.032,0.03 ,0.032},
         {0.046,0.037,0.04 ,0.036,0.032,0.031},
         {0.038,0.032,0.029,0.029,0.025,0.023}},
        {{0.007,0.007,0.007,0.007,0.007,0.007},
         {0.007,0.007,0.007,0.007,0.007,0.007},
         {0.01 ,0.009,0.009,0.009,0.009,0.009},
         {0.02 ,0.015,0.016,0.016,0.016,0.015},
         {0.027,0.02 ,0.02 ,0.019,0.019,0.019},
         {0.036,0.024,0.023,0.022,0.024,0.024},
         {0.014,0.012,0.011,0.011,0.012,0.012}}
      };
      // aggressive: reduce burn reward for levels capable of consistent tetris
      if (lines != 4 && !(tetris.LevelSpeed() == kLevel39 || (
            tetris.LevelSpeed() == kLevel29 && tetris.GetTapSequence()[3] >= 12))) n_reward *= 0.1;
      int penalty_18 = std::max(kOverProb[0][tap_mode][adj_mode] - 0.01, 0.0) * 60000;
      int penalty_19 = std::max(kOverProb[1][tap_mode][adj_mode] - 0.01, 0.0) * 30000;
      int penalty_29 = std::max(kOverProb[2][tap_mode][adj_mode] - 0.01, 0.0) * 15000;
      // give negative reward and random topouts for burning
      int penalty = 0;
      if (lines && lines != 4) {
        double live_prob = 1;
        for (int i = now_lines - lines; i < now_lines; i++) {
          if (i <= 124) live_prob *= 1 - kOverProb[0][tap_mode][adj_mode], penalty += penalty_18;
          else if (i <= 224) live_prob *= 1 - kOverProb[1][tap_mode][adj_mode], penalty += penalty_19;
          else if (i <= 320) live_prob *= 1 - kOverProb[2][tap_mode][adj_mode], penalty += penalty_29;
        }
        double adjusted_over_prob = 1 - std::pow(live_prob, burn_over_multiplier_);
        if (skip_unique_initial_ && std::uniform_real_distribution<float>(0, 1)(rng_) < adjusted_over_prob) {
          tetris.ForceOver();
        }
      }
      // prevent intentional topout by providing game over penalty
      if (tetris.IsOver()) {
        penalty += penalty_18 * (124 - std::min(124, now_lines));
        penalty += penalty_19 * (224 - std::min(224, std::max(124, now_lines)));
        penalty += penalty_29 * (320 - std::min(320, std::max(224, now_lines)));
        penalty = penalty * 1.05;
      }
      n_reward -= penalty * kRewardMultiplier_;
      n_step_reward = 0;
    } else {
      double multiplier_18 = 1, multiplier_19 = 1, multiplier_29 = 1, multiplier_39 = 1;
      bool no_scale_29 = false, no_scale_39 = false;
      int now_pieces = tetris.GetPieces();
      if (tap_4 <= 6) { // 30
        multiplier_18 = step_reward_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.2 : 0.0;
        if (tetris.GetTapSequence()[4] <= 10) {
          multiplier_29 = step_reward_level_ == 2 ? 1.0 : 0.2;
        } else {
          multiplier_29 = step_reward_level_ == 2 ? 1.0 : 0.4;
        }
        no_scale_39 = step_reward_level_ == 2;
        if (now_pieces <= 330 * 10 / 4) multiplier_39 = step_reward_level_ == 2 ? 1.5 : 2.5;
      } else if (tap_4 <= 8) { // 24
        multiplier_18 = step_reward_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.2 : 0.0;
        multiplier_29 = step_reward_level_ == 2 ? 1.0 : 0.3;
        no_scale_39 = step_reward_level_ == 2;
      } else if (tap_4 <= 10) { // 20
        multiplier_18 = step_reward_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.2 : 0.0;
        multiplier_29 = step_reward_level_ == 2 ? 1.0 : 0.5;
      } else if (tap_4 <= 12) { // 15
        multiplier_18 = step_reward_level_ == 2 ? 0.25 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.3 : 0.0;
        no_scale_29 = step_reward_level_ == 2;
        if (now_pieces <= 230 * 10 / 4) multiplier_29 = step_reward_level_ == 2 ? 1.5 : 2.5;
      } else if (tap_4 <= 16) { // 12
        multiplier_18 = step_reward_level_ == 2 ? 0.35 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.5 : 0.1;
        no_scale_29 = step_reward_level_ == 2;
      } else {
        multiplier_18 = step_reward_level_ == 2 ? 0.4 : 0.0;
        multiplier_19 = step_reward_level_ == 2 ? 0.7 : 0.2;
      }
      if (now_pieces <= 120 * 10 / 4) n_step_reward *= multiplier_18;
      else if (now_pieces <= 220 * 10 / 4) n_step_reward *= multiplier_19;
      else if (now_pieces <= 314 * 10 / 4) n_step_reward *= multiplier_29;
      else n_step_reward *= multiplier_39;
      // scale reward to avoid large step reward get higher
      if ((no_scale_39 && tetris.LevelSpeed() == kLevel39) ||
          (no_scale_29 && (tetris.LevelSpeed() == kLevel29 || tetris.LevelSpeed() == kLevel39))) {
        n_reward = ScoreFromLevel(tetris.GetLevel(), 1) * lines * kRewardMultiplier_;
        bottom_multiplier = 1.0;
      }
      n_reward *= (2800 * kRewardMultiplier_) / (2800 * kRewardMultiplier_ + n_step_reward);
    }
    if (lines == 4 && pos.x >= 18) n_reward *= bottom_multiplier;
    if (!tetris.IsAdj()) {
      next_piece_ = GenNextPiece_(next_piece_);
      // scale step reward
      n_reward += n_step_reward * (tetris.GetLevel() + 1) / 30;
    }
#ifdef TETRIS_ONLY
    if (lines && lines != 4) {
      n_reward *= kGameOverMultiplier_;
    }
    if (tetris.IsOver()) n_reward += kGameOverReward;
#endif // TETRIS_ONLY
#endif // NO_ROTATION
    return {n_reward, reward};
  }

#ifndef NO_ROTATION
  std::pair<double, double> CheckReducibleInitial_() {
    if (!skip_unique_initial_ || tetris.IsAdj() || tetris.IsOver()) return {0, 0};
    auto& move_list = tetris.GetPossibleMoveList();
    auto initial_mask = tetris.GetInitialMask();
    if (!move_list.non_adj.empty() || popcount(initial_mask) != 1) return {0, 0};
    Position pos = move_list.adj[ctz(initial_mask)].first;
    auto [score, lines] = tetris.InputPlacement(pos, next_piece_);
    return StepAndCalculateReward_(pos, score, lines);
  }
#endif

 public:
#ifdef NO_ROTATION
  TetrisNoro tetris;
#else
  Tetris tetris;
#endif // NO_ROTATION

  PythonTetris(size_t seed) : rng_(seed) {
    piece_count_ = 0;
#ifdef NO_ROTATION
    Reset(Board::Ones, 0, 0, true, false, false);
#else
    Reset(Board::Ones, 0);
#endif // NO_ROTATION
  }

  void ResetRandom(const Board& b) {
#ifdef NO_ROTATION
    int start_level = std::discrete_distribution<int>({
        15, 1, 1, 1, 2, 2, 2, 2, 4, 6, // 0-9
        4, 0, 0, 4, 0, 0, 4, 0, 0, // 10-18
        4, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 19-28
        8})(rng_);
    bool do_tuck = std::discrete_distribution({1, 1})(rng_);
    bool nnb = do_tuck ? std::discrete_distribution<int>({2, 1})(rng_) :
                         std::discrete_distribution<int>({1, 1})(rng_);
    bool is_mirror = std::discrete_distribution({1, 1})(rng_);
    Reset(b, 0, start_level, do_tuck, nnb, is_mirror);
#else
    int lines = b.Count() % 4 != 0;
    lines += std::uniform_int_distribution<int>(0, kLineCap / 2 - 1)(rng_) * 2;
    Reset(b, lines, -1, -1, true);
#endif // NO_ROTATION
  }

  Position GetRealPosition(Position pos) {
#ifdef NO_ROTATION
    if (is_mirror_) pos.y = kMirrorCols_[tetris.NowPiece()] - pos.y;
#endif // NO_ROTATION
    return pos;
  }

#ifdef NO_ROTATION
  void Reset(const Board& b, int lines, int start_level, bool do_tuck, bool nnb, bool is_mirror,
             int now_piece = -1, int next_piece = -1) {
    if (now_piece == -1 || next_piece == -1) {
      piece_count_ = std::uniform_int_distribution<int>(0, 8)(rng_);
      if (now_piece == -1) now_piece = std::uniform_int_distribution<int>(0, kPieces - 1)(rng_);
      next_piece = GenNextPiece_(now_piece);
    }
    nnb_ = nnb;
    is_mirror_ = is_mirror;
    tetris.Reset(b, lines, start_level, do_tuck, now_piece, next_piece);
    next_piece_ = GenNextPiece_(next_piece);
  }
#else // NO_ROTATION
  void Reset(const Board& b, int lines, const int tap_sequence[], int adj_delay,
             int now_piece = -1, int next_piece = -1, bool skip_unique_initial = false) {
    if (now_piece == -1 || next_piece == -1) {
      piece_count_ = std::uniform_int_distribution<int>(0, 8)(rng_);
      if (now_piece == -1) now_piece = std::uniform_int_distribution<int>(0, kPieces - 1)(rng_);
      next_piece = GenNextPiece_(now_piece);
    }
    tetris.Reset(b, lines, now_piece, next_piece, tap_sequence, adj_delay);
    next_piece_ = GenNextPiece_(next_piece);
    skip_unique_initial_ = skip_unique_initial;
    CheckReducibleInitial_();
  }

  void Reset(const Board& b, int lines,
             int now_piece = -1, int next_piece = -1, bool skip_unique_initial = false) {
    constexpr TAP_SPEED tap_table;
    Reset(b, lines, tap_table.data(), ADJ_DELAY, now_piece, next_piece, skip_unique_initial);
  }

  std::pair<double, double> DirectPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.DirectPlacement(npos, next_piece_);
    return StepAndCalculateReward_(npos, score, lines);
  }

  void SetStepReward(int level) {
    int score = level == 0 ? 0 : level == 1 ? 800 : 2400;
    step_reward_ = score * kRewardMultiplier_;
    step_reward_level_ = level;
  }

  void SetBurnOverMultiplier(double mul) {
    burn_over_multiplier_ = mul;
  }
#endif // NO_ROTATION

  std::pair<double, double> InputPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.InputPlacement(npos, next_piece_);
    auto reward = StepAndCalculateReward_(npos, score, lines);
#ifdef NO_ROTATION
    return reward;
#else
    if (!skip_unique_initial_) return reward;
    auto reward_2 = CheckReducibleInitial_();
    return {reward.first + reward_2.first, reward.second + reward_2.second};
#endif
  }

  struct State {
#ifdef NO_ROTATION
    std::array<std::array<std::array<float, 10>, 20>, 2> board;
    std::array<float, 32> meta;
    std::array<std::array<std::array<float, 10>, 20>, 3> moves;
    std::array<float, 31> move_meta;
    std::array<int, 2> meta_int;
#else
    std::array<std::array<std::array<float, 10>, 20>, 6> board;
    std::array<float, 32> meta;
    std::array<std::array<std::array<float, 10>, 20>, 18> moves;
    std::array<float, 28> move_meta;
    std::array<int, 2> meta_int;
#endif
  };

#ifndef NO_ROTATION
  void GetAdjStates(const Position& pos, State states[kPieces]) const {
    if (tetris.IsAdj()) throw std::logic_error("should only called on non adj phase");
    Tetris n_tetris = tetris;
    n_tetris.InputPlacement(pos, 0);
    if (!n_tetris.IsAdj()) throw std::logic_error("not an adj placement");
    for (size_t i = 0; i < kPieces; i++) {
      n_tetris.SetNextPiece(i);
      PythonTetris::GetState(n_tetris, states[i]);
    }
  }
#endif // !NO_ROTATION

#ifdef NO_ROTATION
  void GetState(State& state, int line_reduce = 0) const {
    PythonTetris::GetState(tetris, state, nnb_, is_mirror_, line_reduce);
  }
#else // NO_ROTATION
  void GetState(State& state, int line_reduce = 0) const {
    PythonTetris::GetState(tetris, state, line_reduce, step_reward_level_);
  }
#endif // NO_ROTATION

  static double GetNoroLineRewardExp(int lines, int start_level, bool do_tuck, bool nnb) {
    constexpr int kOffset[2][2][15] = {
      { // 0,1,2,3,4,5,6, 7,8, 9, 10-12,13-15, 16-18,19, 29
        {14,14,14,14,14,14,14, 14,14, 13, 13,13, 12,12, 10}, // notuck
        {12,12,12,12,12,12,12, 12,12, 12, 10,10,  9, 9, 6}, // notuck, nnb
      }, {
        {21,21,21,21,21,21,21, 19,19, 19, 19,19, 12,12, 11}, // tuck
        {17,17,17,17,17,17,17, 17,17, 16, 15,15, 12,12, 9}, // tuck, nnb
      },
    };
    constexpr float kExpMultiplier[2][2][15] = {
      { // 0,1,2,3,4,5,6,7,8,9,10-12,13-15,16-18,19,29
        {0.33,0.33,0.33,0.33,0.33,0.33,0.33, 0.33,0.33, 0.35, 0.38,0.38, 0.38,0.38, 0.4}, // notuck
        {0.50,0.50,0.50,0.50,0.50,0.50,0.50, 0.50,0.50, 0.50, 0.50,0.50, 0.50,0.50, 0.50}, // notuck, nnb
      }, {
        {0.16,0.16,0.16,0.16,0.16,0.16,0.16, 0.16,0.16, 0.18, 0.19,0.19, 0.24,0.24, 0.33}, // tuck
        {0.20,0.20,0.20,0.20,0.20,0.20,0.20, 0.20,0.20, 0.21, 0.22,0.22, 0.40,0.40, 0.45}, // tuck, nnb
      },
    };
    constexpr float kMinExp[2][2][15] = {
      { // 0,1,2,3,4,5,6,7,8,9,10-12,13-15,16-18,19,29
        {-3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0, -3.0,-3.0, -3.0, -3.0,-3.0, -3.0,-3.0, -2.8}, // notuck
        {-2.8,-2.8,-2.8,-2.8,-2.8,-2.8,-2.8, -2.8,-2.8, -2.8, -2.8,-2.8, -2.8,-2.8, -2.8}, // notuck, nnb
      }, {
        {-3.6,-3.6,-3.6,-3.6,-3.6,-3.6,-3.6, -3.6,-3.6, -3.6, -3.5,-3.5, -3.2,-3.2, -3.0}, // tuck
        {-3.5,-3.5,-3.5,-3.5,-3.5,-3.5,-3.5, -3.5,-3.5, -3.5, -3.2,-3.2, -2.8,-2.8, -2.2}, // tuck, nnb
      },
    };

    int speed = noro::GetLevelSpeed(start_level);
    float min_exp = kMinExp[do_tuck][nnb][speed];
    int offset = kOffset[do_tuck][nnb][speed];
    float multiplier = kExpMultiplier[do_tuck][nnb][speed];
    return std::min(6.0f, std::max(0, lines - offset) * multiplier + min_exp);
  }

  static void GetState(const TetrisNoro& tetris, State& state, bool nnb, bool is_mirror, int line_reduce = 0) {
    // board: shape (2, 20, 10) [board, one]
    // meta: shape (21,) [group(5), now_piece(7), next_piece(7), nnb, do_tuck, start_speed(10)]
    // meta_int: shape (2,) [entry, now_piece]
    // moves: shape (3, 20, 10) [board, one, moves]
    // move_meta: shape (31,) [speed(10), to_transition(16), level*0.1, lines*0.01, start_lines*0.01, pieces*0.004, ln(multiplier)]
    {
      auto byte_board = tetris.GetBoard().ToByteBoard();
      for (int i = 0; i < 20; i++) {
        if (is_mirror) {
          for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][9-j];
          for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][9-j];
        } else {
          for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][j];
          for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][j];
        }
        for (int j = 0; j < 10; j++) state.board[1][i][j] = 1;
        for (int j = 0; j < 10; j++) state.moves[1][i][j] = 1;
      }
      auto move_map = tetris.GetPossibleMoveMap().ToByteBoard();
      for (int i = 0; i < 20; i++) {
        if (is_mirror) {
          for (int j = 0; j < 10; j++) {
            int ncol = kMirrorCols_[tetris.NowPiece()] - j;
            state.moves[2][i][j] = ncol >= 10 ? 0 : move_map[i][ncol];
          }
        } else {
          for (int j = 0; j < 10; j++) state.moves[2][i][j] = move_map[i][j];
        }
      }
    }

    int start_level = tetris.GetStartLevel();
    int start_speed = tetris.InputsPerRow(start_level);
    memset(state.meta.data(), 0, sizeof(state.meta));
    state.meta[0 + tetris.GetBoard().Count() / 2 % 5] = 1;
    state.meta[5 + (is_mirror ? kMirrorPiece_[tetris.NowPiece()] : tetris.NowPiece())] = 1;
    if (nnb) {
      state.meta[19] = 1;
    } else {
      state.meta[12 + (is_mirror ? kMirrorPiece_[tetris.NextPiece()] : tetris.NextPiece())] = 1;
    }
    state.meta[20] = tetris.DoTuck();
    state.meta[21] = is_mirror;
    state.meta[22 + start_speed] = 1;

    int lines = tetris.GetLines();
    int state_lines = lines - line_reduce;
    int state_level = noro::GetLevelByLines(state_lines, start_level);
    state.meta_int[0] = state_lines / 2;
    state.meta_int[1] = tetris.NowPiece();

    memset(state.move_meta.data(), 0, sizeof(state.move_meta));
    int to_transition = 0;
    state.move_meta[tetris.InputsPerRow()] = 1;
    to_transition = tetris.LinesToNextSpeed();
    if (to_transition == -1) to_transition = 1000;
    if (to_transition <= 10) { // 10..19
      state.move_meta[10 + (to_transition - 1)] = 1;
    } else if (to_transition <= 22) { // 20..23
      state.move_meta[20 + (to_transition - 11) / 3] = 1;
    } else {
      state.move_meta[24] = 1;
    }
    state.move_meta[25] = to_transition * 0.01;
    state.move_meta[26] = state_level * 0.1;
    state.move_meta[27] = state_lines * 0.01;
    state.move_meta[28] = start_level * 0.1;
    state.move_meta[29] = (tetris.GetPieces() + line_reduce * 10 / 4) * 0.004;
    state.move_meta[30] = std::max(-0.5, GetNoroLineRewardExp(state_lines + 5, start_level, tetris.DoTuck(), nnb));
  }

  static void GetState(const Tetris& tetris, State& state, int line_reduce = 0, int step_reward_level = 0) {
    // board: shape (6, 20, 10) [board, one, initial_move(4)]
    // meta: shape (32,) [now_piece(7), next_piece(7), is_adj(1), hz(7), adj_delay(6), aggro(3), pad(1)]
    // meta_int: shape (2,) [entry, now_piece]
    // moves: shape (14, 20, 10) [board, one, moves(4), adj_moves(4), initial_move(4), nonreduce_moves(4)]
    // move_meta: shape (28,) [speed(4), to_transition(21), (level-18)*0.1, lines*0.01, pieces*0.004]
    {
      auto byte_board = tetris.GetBoard().ToByteBoard();
      for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][j];
        for (int j = 0; j < 10; j++) state.board[1][i][j] = 1;
        for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][j];
        for (int j = 0; j < 10; j++) state.moves[1][i][j] = 1;
      }
      auto& move_map = tetris.GetPossibleMoveMap();
      for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 20; i++) {
          for (int j = 0; j < 10; j++) {
            state.moves[2 + r][i][j] = move_map[r][i][j] ? 1 : 0;
            state.moves[6 + r][i][j] = move_map[r][i][j] >= 2;
            state.moves[14 + r][i][j] = move_map[r][i][j] && move_map[r][i][j] != 2 ? 1 : 0;
          }
        }
        memset(state.board.data() + (2 + r), 0, sizeof(state.board[0]));
        memset(state.moves.data() + (10 + r), 0, sizeof(state.moves[0]));
      }
    }
    if (tetris.IsAdj()) {
      auto pos = tetris.InitialMove();
      state.board[2 + pos.r][pos.x][pos.y] = 1;
      state.moves[10 + pos.r][pos.x][pos.y] = 1;
    }

    memset(state.meta.data(), 0, sizeof(state.meta));
    state.meta[0 + tetris.NowPiece()] = 1;
    if (tetris.IsAdj()) {
      state.meta[7 + tetris.NextPiece()] = 1;
      state.meta[14] = 1;
    }

    int lines = tetris.GetLines();
    int state_lines = lines - line_reduce;
    int state_level = GetLevelByLines(state_lines);
    int state_speed = static_cast<int>(GetLevelSpeed(state_level));

    int tap_4 = tetris.GetTapSequence()[3];
    int tap_5 = tetris.GetTapSequence()[4];
    int adj_delay = tetris.GetAdjDelay();
    if (state_speed == 2 && adj_delay >= 20) adj_delay = 61;
    if (state_speed == 3 && adj_delay >= 10) adj_delay = 61;
    if (tap_5 <= 8) { // 30hz
      state.meta[15] = 1;
    } else if (tap_5 <= 11) { // 24hz
      state.meta[16] = 1;
    } else if (tap_5 <= 13) { // 20hz
      state.meta[17] = 1;
    } else if (tap_5 <= 16) { // 15hz
      state.meta[18] = 1;
    } else if (tap_4 <= 9) { // slow 5-tap
      state.meta[19] = 1;
    } else if (tap_5 <= 21) { // 12hz
      state.meta[20] = 1;
    } else { // 10hz
      state.meta[21] = 1;
    }
    if (adj_delay <= 4) {
      state.meta[22] = 1;
    } else if (adj_delay <= 19) {
      state.meta[23] = 1;
    } else if (adj_delay <= 22) {
      state.meta[24] = 1;
    } else if (adj_delay <= 25) {
      state.meta[25] = 1;
    } else if (adj_delay <= 32) {
      state.meta[26] = 1;
    } else {
      state.meta[27] = 1;
    }
    state.meta[28 + step_reward_level] = 1;

    state.meta_int[0] = state_lines / 2;
    state.meta_int[1] = tetris.NowPiece();

    memset(state.move_meta.data(), 0, sizeof(state.move_meta));
    int to_transition = 0;
    state.move_meta[state_speed] = 1;
    to_transition = std::max(1, kLevelSpeedLines[state_speed + 1] - state_lines);
    if (to_transition <= 10) { // 4..13
      state.move_meta[4 + (to_transition - 1)] = 1;
    } else if (to_transition <= 22) { // 14..17
      state.move_meta[14 + (to_transition - 11) / 3] = 1;
    } else if (to_transition <= 40) { // 18..20
      state.move_meta[18 + (to_transition - 22) / 6] = 1;
    } else if (to_transition <= 60) { // 21,22
      state.move_meta[21 + (to_transition - 40) / 10] = 1;
    } else {
      state.move_meta[23] = 1;
    }
    state.move_meta[24] = to_transition * 0.01;
    state.move_meta[25] = (state_level - 18) * 0.1;
    state.move_meta[26] = state_lines * 0.01;
    state.move_meta[27] = (tetris.GetPieces() + line_reduce * 10 / 4) * 0.004;
  }


#ifdef NO_ROTATION
  operator TetrisNoro() const { return tetris; }
#else
  operator Tetris() const { return tetris; }
#endif // NO_ROTATION
};

extern PyTypeObject py_tetris_class;
