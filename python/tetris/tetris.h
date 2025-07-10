#pragma once

#include <random>
#include "../../src/tetris.h"
#include "python.h"
#include "state.h"

struct Reward {
  double reward, raw_reward, live_prob, over_reward;
};

class PythonTetris {
 public:
  PyObject_HEAD

 private:
  // rewards
  static constexpr double kInvalidReward_ = -0.3;
#ifdef NO_ROTATION
  static constexpr double kRawMultiplier_ = 0.2;
  static constexpr double kStepReward_ = 2e-3;
#elif defined(TETRIS_ONLY)
  static constexpr double kRewardPerTetris_ = 0.25; // 0.25 per tetris
  static constexpr double kBurnReward_ = 0.02; // per line; to make it figure out dirty
  static constexpr double kBottomMultiplier_ = 2.0;
  static constexpr double kStepReward_ = 1e-3;
#else // normal
  static constexpr double kRewardMultiplier_ = 1e-5; // 10 per maxout
  static constexpr double kBottomMultiplier_ = 1.1;
  int aggression_level_ = 0; // 0-2, 0 is high aggression and 2 is low
  double burn_over_multiplier_ = 0;
  double step_reward_ = 5e-4;
#endif

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
#ifdef USE_PIECE_COUNT_RNG
    piece_count_ = (piece_count_ + 1) & 7;
    return std::discrete_distribution<int>(
        kTransitionRealisticProbInt[piece_count_][piece],
        kTransitionRealisticProbInt[piece_count_][piece] + kPieces)(rng_);
#else
    return std::discrete_distribution<int>(
        kTransitionProbInt[piece],
        kTransitionProbInt[piece] + kPieces)(rng_);
#endif
  }

  Reward StepAndCalculateReward_(const Position& pos, int score, int lines) {
    if (score == -1) return {kInvalidReward_, 0.0, 1.0, 0.0};

    // Generate next piece
#ifdef NO_ROTATION
    next_piece_ = GenNextPiece_(next_piece_);
#else
    if (!tetris.IsAdj()) next_piece_ = GenNextPiece_(next_piece_);
#endif

    // Reward calculation
    Reward ret = {0.0, 0.0, 1.0, 0.0};
#ifdef NO_ROTATION
    ret.raw_reward = lines * kRawMultiplier_;
    int pre_lines = tetris.GetLines() - lines;
    ret.reward = kStepReward_;
    for (int i = pre_lines; i < pre_lines + lines; i++) {
      ret.reward += std::exp(GetNoroLineRewardExp(i, tetris.GetStartLevel(), tetris.DoTuck(), nnb_));
    }
    return ret;
#elif defined(TETRIS_ONLY)
    ret.raw_reward = lines == 4 ? kRewardPerTetris_ : 0;
    ret.reward = ret.raw_reward + kStepReward_;
    // if (lines && lines < 4) ret.reward += lines * kBurnReward_;
    if (lines == 4 && pos.x >= 18) ret.reward *= kBottomMultiplier_;
    return ret;
#else // normal
    ret.raw_reward = score * kRewardMultiplier_;
    ret.reward = ret.raw_reward;
    double bottom_multiplier = kBottomMultiplier_;
    int tap_4 = tetris.GetTapSequence()[3];
    int now_lines = tetris.GetLines();
    if (aggression_level_ == 0) {
      // aggro; assign over probability for burns
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
            tetris.LevelSpeed() == kLevel29 && tetris.GetTapSequence()[3] >= 12))) ret.reward *= 0.1;
      // prevent intentional topout by providing game over penalty
      int penalty_18 = std::max(kOverProb[0][tap_mode][adj_mode] - 0.01, 0.0) * 60000;
      int penalty_19 = std::max(kOverProb[1][tap_mode][adj_mode] - 0.01, 0.0) * 30000;
      int penalty_29 = std::max(kOverProb[2][tap_mode][adj_mode] - 0.01, 0.0) * 15000;
      int penalty = 0;
      penalty += penalty_18 * (124 - std::min(124, now_lines));
      penalty += penalty_19 * (224 - std::min(224, std::max(124, now_lines)));
      penalty += penalty_29 * (320 - std::min(320, std::max(224, now_lines)));
      penalty = penalty * 1.05;
      if (tetris.IsOver()) {
        ret.reward -= penalty * kRewardMultiplier_;
      } else {
        ret.over_reward = -penalty * kRewardMultiplier_;
      }
      // give random topouts for burning
      if (lines && lines != 4) {
        for (int i = now_lines - lines; i < now_lines; i++) {
          if (i <= 124) ret.live_prob *= 1 - kOverProb[0][tap_mode][adj_mode], penalty += penalty_18;
          else if (i <= 224) ret.live_prob *= 1 - kOverProb[1][tap_mode][adj_mode], penalty += penalty_19;
          else if (i <= 320) ret.live_prob *= 1 - kOverProb[2][tap_mode][adj_mode], penalty += penalty_29;
        }
        ret.live_prob = std::pow(ret.live_prob, burn_over_multiplier_);
      }
    } else {
      double n_step_reward = step_reward_;
      double multiplier_18 = 1, multiplier_19 = 1, multiplier_29 = 1, multiplier_39 = 1;
      bool no_scale_29 = false, no_scale_39 = false;
      int now_pieces = tetris.GetPieces();
      if (tap_4 <= 6) { // 30
        multiplier_18 = aggression_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.2 : 0.0;
        if (tetris.GetTapSequence()[4] <= 10) {
          multiplier_29 = aggression_level_ == 2 ? 1.0 : 0.2;
        } else {
          multiplier_29 = aggression_level_ == 2 ? 1.0 : 0.4;
        }
        no_scale_39 = aggression_level_ == 2;
        if (now_pieces <= 330 * 10 / 4) multiplier_39 = aggression_level_ == 2 ? 1.5 : 2.5;
      } else if (tap_4 <= 8) { // 24
        multiplier_18 = aggression_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.2 : 0.0;
        multiplier_29 = aggression_level_ == 2 ? 1.0 : 0.3;
        no_scale_39 = aggression_level_ == 2;
      } else if (tap_4 <= 10) { // 20
        multiplier_18 = aggression_level_ == 2 ? 0.2 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.2 : 0.0;
        multiplier_29 = aggression_level_ == 2 ? 1.0 : 0.5;
      } else if (tap_4 <= 12) { // 15
        multiplier_18 = aggression_level_ == 2 ? 0.25 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.3 : 0.0;
        no_scale_29 = aggression_level_ == 2;
        if (now_pieces <= 230 * 10 / 4) multiplier_29 = aggression_level_ == 2 ? 1.5 : 2.5;
      } else if (tap_4 <= 16) { // 12
        multiplier_18 = aggression_level_ == 2 ? 0.35 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.5 : 0.1;
        no_scale_29 = aggression_level_ == 2;
      } else {
        multiplier_18 = aggression_level_ == 2 ? 0.4 : 0.0;
        multiplier_19 = aggression_level_ == 2 ? 0.7 : 0.2;
      }
      if (now_pieces <= 120 * 10 / 4) n_step_reward *= multiplier_18;
      else if (now_pieces <= 220 * 10 / 4) n_step_reward *= multiplier_19;
      else if (now_pieces <= 314 * 10 / 4) n_step_reward *= multiplier_29;
      else n_step_reward *= multiplier_39;
      // line out
      if ((no_scale_39 && tetris.LevelSpeed() == kLevel39) ||
          (no_scale_29 && (tetris.LevelSpeed() == kLevel29 || tetris.LevelSpeed() == kLevel39))) {
        // do not reward lines after line cap to avoid aggressive during end
        ret.reward = 3000 * kRewardMultiplier_ * (lines - std::max(0, now_lines - kLineCap));
        bottom_multiplier = 1.0;
        // minimal step reward to avoid filling up the board to lengthen game at end
        n_step_reward = 10 * kRewardMultiplier_;
      }
      // scale reward to avoid large step reward get higher overall reward
      ret.reward *= (2800 * kRewardMultiplier_) / (2800 * kRewardMultiplier_ + n_step_reward);
      if (!tetris.IsAdj()) { // add & scale step reward
        ret.reward += n_step_reward * (tetris.GetLevel() + 1) / 30;
      }
    }
    if (lines == 4 && pos.x >= 18) ret.reward *= bottom_multiplier;
    return ret;
#endif
  }

#ifndef NO_ROTATION
  Reward CheckReducibleInitial_() {
    if (!skip_unique_initial_ || tetris.IsAdj() || tetris.IsOver()) return {0, 0, 1, 0};
    auto& move_list = tetris.GetPossibleMoveList();
    auto initial_mask = tetris.GetInitialMask();
    if (!move_list.non_adj.empty() || popcount(initial_mask) != 1) return {0, 0, 1, 0};
    Position pos = move_list.adj[ctz(initial_mask)].first;
    auto [score, lines] = tetris.InputPlacement(pos, next_piece_);
    return StepAndCalculateReward_(pos, score, lines);
  }
#endif

 public:
#ifdef NO_ROTATION
  TetrisNoro tetris;

  PythonTetris(size_t seed) : rng_(seed), piece_count_(0) {
    Reset(Board::Ones, 0, 0, true, false, false);
  }

  Position GetRealPosition(Position pos) {
    if (is_mirror_) pos.y = kMirrorCols[tetris.NowPiece()] - pos.y;
    return pos;
  }

  void ResetRandom(const Board& b) {
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
  }

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
#else // !NO_ROTATION
  Tetris tetris;

  PythonTetris(size_t seed) : rng_(seed), piece_count_(0) {
    constexpr Tap30Hz taps;
    Reset(Board::Ones, 0, taps.data(), 18);
  }

  Position GetRealPosition(Position pos) { return pos; }

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

  Reward DirectPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.DirectPlacement(npos, next_piece_);
    return StepAndCalculateReward_(npos, score, lines);
  }

#ifndef TETRIS_ONLY
  void SetAggression(int level) {
    int score = level == 0 ? 0 : level == 1 ? 800 : 2400;
    step_reward_ = score * kRewardMultiplier_;
    aggression_level_ = level;
  }

  void SetBurnOverMultiplier(double mul) {
    burn_over_multiplier_ = mul;
  }
#endif // !TETRIS_ONLY
#endif // !NO_ROTATION

  Reward InputPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.InputPlacement(npos, next_piece_);
    auto reward = StepAndCalculateReward_(npos, score, lines);
#ifdef NO_ROTATION
    return reward;
#else
    if (!skip_unique_initial_) return reward;
    auto reward_2 = CheckReducibleInitial_();
    return {reward.reward + reward_2.reward, reward.raw_reward + reward_2.raw_reward,
            reward.live_prob, reward.over_reward}; // initial reduce won't cause game over
#endif
  }

  /// State generation
#ifdef NO_ROTATION
  void GetState(State& state, int line_reduce = 0) const {
    ::GetState(tetris, state, nnb_, is_mirror_, line_reduce);
  }
#else // !NO_ROTATION
  void GetState(State& state, int line_reduce = 0) const {
#ifdef TETRIS_ONLY
    ::GetState(tetris, state, line_reduce);
#else
    ::GetState(tetris, state, line_reduce, aggression_level_);
#endif
  }

  MultiState GetAdjStates(const Position& pos) const {
    if (tetris.IsAdj()) throw std::logic_error("should only called on non adj phase");
    Tetris n_tetris = tetris;
    n_tetris.InputPlacement(pos, 0);
    return ::GetAdjStates(n_tetris);
  }
#endif // !NO_ROTATION


#ifdef NO_ROTATION
  operator TetrisNoro() const { return tetris; }
#else
  operator Tetris() const { return tetris; }
#endif // NO_ROTATION
};

extern PyTypeObject py_tetris_class;
