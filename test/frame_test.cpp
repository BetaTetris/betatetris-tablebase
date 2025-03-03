#include "frame_test.h"

namespace {

TEST_F(FrameTest, Test30Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<Tap30Hz>(kLevel18, 18, board, mrand(0, 6)(gen), 2, gen);
    RandTest<Tap30Hz>(kLevel19, 18, board, mrand(0, 6)(gen), 2, gen);
    RandTest<Tap30Hz>(kLevel29, 18, board, mrand(0, 6)(gen), 2, gen);
    RandTest<Tap30Hz>(kLevel39, 18, board, mrand(0, 6)(gen), 2, gen);
  }
}

TEST_F(FrameTest, TestBestAdj) {
  SetUp();
  Board b = Board::Ones;
  auto moves = MoveSearch<Tap30Hz>(kLevel18, 18, b, 0);
  {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 3}, {2, 19, 3}, {2, 19, 3},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj(kLevel18, Tap30Hz().data(), b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 4));
  } {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 5}, {2, 19, 5}, {2, 19, 5},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj(kLevel18, Tap30Hz().data(), b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 5));
  } {
    std::array<Position, 7> pos{{{0, 18, 5}, {0, 18, 5}, {0, 18, 5}, {0, 18, 5},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj(kLevel18, Tap30Hz().data(), b, 0, moves, 18, pos.data()).first;
    ASSERT_TRUE(moves.adj[idx].first == Position(1, 6, 5) ||
                moves.adj[idx].first == Position(3, 6, 5));
  }
  using namespace std::literals;
  b = Board("....X.....\n"
            ".....X...."sv);
  moves = MoveSearch<Tap30Hz>(kLevel18, 18, b, 0);
  {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 3}, {2, 19, 3}, {2, 19, 3},
                                 {2, 19, 3}, {2, 19, 3}, {2, 19, 3}}};
    size_t idx = GetBestAdj(kLevel18, Tap30Hz().data(), b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 2));
  }
}

TEST_F(FrameTest, Test12HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<Tap12Hz>(kLevel18, 12, board, mrand(0, 6)(gen), 5, gen);
    RandTest<Tap12Hz>(kLevel19, 12, board, mrand(0, 6)(gen), 5, gen);
    RandTest<Tap12Hz>(kLevel29, 12, board, mrand(0, 6)(gen), 5, gen);
    RandTest<Tap12Hz>(kLevel39, 12, board, mrand(0, 6)(gen), 5, gen);
  }
}

TEST_F(FrameTest, TestTuck) {
  SetUp();
  for (auto& board : kTestTuckBoards) {
    PositionTest(kLevel18, board);
    PositionTest(kLevel19, board);
    PositionTest(kLevel29, board);
    PositionTest(kLevel39, board);
  }
}

} // namespace
