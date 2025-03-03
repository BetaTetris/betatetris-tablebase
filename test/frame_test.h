#include <random>
#include <gtest/gtest.h>
#include "../src/move_search.h"
#include "../src/frame_sequence.h"
#include "test_boards.h"
#include "printing.h"

class FrameTest : public ::testing::Test {
 protected:
  std::mt19937_64 gen;
  void SetUp() {}
  void TearDown() override {}
};

namespace {

using mrand = std::uniform_int_distribution<int>;

void AssertTapSep(const FrameSequence& seq, int frames) {
  int prev_ab = -100, prev_lr = -100;
#ifdef DOUBLE_TUCK
  bool prev_ok_ab = true, prev_ok_lr = true;
#endif
  for (int i = 0; i < (int)seq.size(); i++) {
    if (seq[i].IsL() || seq[i].IsR()) {
#ifdef DOUBLE_TUCK
      ASSERT_TRUE(prev_ok_lr && i - prev_lr >= 2);
      prev_ok_lr = i - prev_lr >= frames;
#else
      ASSERT_TRUE(i - prev_lr >= frames);
#endif
      prev_lr = i;
    }
    if (seq[i].IsA() || seq[i].IsB()) {
#ifdef DOUBLE_TUCK
      ASSERT_TRUE(prev_ok_ab && i - prev_ab >= 2);
      prev_ok_ab = i - prev_ab >= frames;
#else
      ASSERT_TRUE(i - prev_ab >= frames);
#endif
      prev_ab = i;
    }
  }
}

} // namespace

template <class Taps, class Rand>
void RandTest(Level level, int adj_delay, const Board& b, int piece, int frame_sep, Rand& gen) {
  //auto moves = MoveSearch<level, adj_delay, Taps>(b, piece);
  Taps taps;
  auto moves = MoveSearch<Taps>(level, adj_delay, b, piece);
  for (auto& i : moves.non_adj) {
    auto seq = GetFrameSequenceStart(level, taps.data(), b, piece, adj_delay, i);
    ASSERT_TRUE((int)seq.size() >= adj_delay)
      << b << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove(level, b, piece, seq, true);
    ASSERT_EQ(sim_pos, std::make_pair(i, true))
      << b << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
  if (moves.adj.empty()) return;
  for (auto& i : moves.adj) {
    auto seq = GetFrameSequenceStart(level, taps.data(), b, piece, adj_delay, i.first);
    ASSERT_TRUE((int)seq.size() >= adj_delay)
      << b << i.first << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove(level, b, piece, seq, false);
    ASSERT_EQ(sim_pos, std::make_pair(i.first, false))
      << b << i.first << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
  auto& adj = moves.adj[mrand(0, moves.adj.size() - 1)(gen)];
  auto pre_seq = GetFrameSequenceStart(level, taps.data(), b, piece, adj_delay, adj.first);
  for (auto& i : adj.second) {
    auto seq = pre_seq;
    GetFrameSequenceAdj(level, taps.data(), seq, b, piece, adj.first, i);
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove(level, b, piece, seq, true);
    ASSERT_EQ(sim_pos, std::make_pair(i, true))
      << b << adj.first << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
}

inline void PositionTest(Level level, const TestSearchBoard& b) {
  if (!b.lvl_39_ok && level == kLevel39) return;
  if (!b.lvl_19_ok && level != kLevel18) return;
  FrameSequence seq;
  std::pair<Position, bool> sim_pos;
  if (level == kLevel18 || level == kLevel19) {
    Tap20Hz taps;
    //auto moves = MoveSearch<level, 18, Tap20Hz>(b.board, b.piece);
    auto moves = MoveSearch<Tap20Hz>(level, 18, b.board, b.piece);
    for (auto& i : moves.adj) {
      if (std::find(i.second.begin(), i.second.end(), b.pos) == i.second.end()) continue;
      seq = GetFrameSequenceStart(level, taps.data(), b.board, b.piece, 18, i.first);
      GetFrameSequenceAdj(level, taps.data(), seq, b.board, b.piece, i.first, b.pos);
      sim_pos = SimulateMove(level, b.board, b.piece, seq, true);
      ASSERT_EQ(sim_pos, std::make_pair(b.pos, true))
          << b.board << b.pos << ' ' << b.piece << ' ' << level << ' ' << seq;
    }
  }
  if (level == kLevel39) {
    seq = GetFrameSequenceStart(level, Tap30Hz().data(), b.board, b.piece, 18, b.pos);
    AssertTapSep(seq, 2);
  } else {
    seq = GetFrameSequenceStart(level, Tap20Hz().data(), b.board, b.piece, 61, b.pos);
    AssertTapSep(seq, 3);
  }
  sim_pos = SimulateMove(level, b.board, b.piece, seq, true);
  ASSERT_EQ(sim_pos, std::make_pair(b.pos, true))
      << b.board << b.pos << ' ' << b.piece << ' ' << level << ' ' << seq;
}
