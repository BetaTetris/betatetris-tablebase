#pragma once

#include "move_search.h"

struct FrameInput {
  static const FrameInput A;
  static const FrameInput B;
  static const FrameInput L;
  static const FrameInput R;
  static const FrameInput D;
  uint8_t value;
  FrameInput operator|=(FrameInput a) {
    value |= a.value;
    return *this;
  }
  bool IsA() const { return value & 4; }
  bool IsB() const { return value & 8; }
  bool IsL() const { return value & 1; }
  bool IsR() const { return value & 2; }
  bool IsD() const { return value & 16; }

  std::string ToString() const {
    std::string str;
    if (IsL()) str += 'L';
    if (IsR()) str += 'R';
    if (IsA()) str += 'A';
    if (IsB()) str += 'B';
    if (IsD()) str += 'D';
    return str.empty() ? "-" : str;
  }
};

inline constexpr FrameInput FrameInput::A = FrameInput{4};
inline constexpr FrameInput FrameInput::B = FrameInput{8};
inline constexpr FrameInput FrameInput::L = FrameInput{1};
inline constexpr FrameInput FrameInput::R = FrameInput{2};
inline constexpr FrameInput FrameInput::D = FrameInput{16};

inline FrameInput operator|(FrameInput a, FrameInput b) {
  return a |= b;
}

using FrameSequence = std::vector<FrameInput>;

FrameSequence GetFrameSequenceStart(
    Level level, const int taps[],
    const Board& b, int piece, int adj_delay, const Position& target);

template <bool gen_seq = true>
int GetFrameSequenceAdj(
    Level level, const int taps[], FrameSequence& seq, const Board& b, int piece, const Position& premove,
    const Position& target);

FrameSequence GetFrameSequenceNoro(
    const Board& b, int piece, int inputs_per_row, bool do_tuck, int frames_per_drop, const Position& target);

std::pair<Position, bool> SimulateMove(Level level, const Board& b, int piece, const FrameSequence& seq, bool until_lock);

struct AdjInfor {
  size_t index;
  int pre_taps;
  std::vector<std::pair<float, int>> taps; // (prob, num_taps)
  FrameSequence seq;
};

std::vector<AdjInfor> GetAdjTaps(
    Level level, const int taps[], const Board& b, int piece,
    const PossibleMoves& moves, int adj_delay, const Position adjs[kPieces]);

enum class BestAdjMode {
  kWeightedTaps,
  kPreAdjTaps,
  kWorstTaps,
  kAdjProb
};

std::pair<size_t, FrameSequence> GetBestAdj(const std::vector<AdjInfor>& infor, BestAdjMode mode);
std::pair<size_t, FrameSequence> GetBestAdj(
    Level level, const int taps[], const Board& b, int piece,
    const PossibleMoves& moves, int adj_delay, const Position adjs[kPieces], BestAdjMode mode = BestAdjMode::kWeightedTaps);
