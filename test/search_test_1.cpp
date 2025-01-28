#include "search_test.h"

namespace {

TEST_F(SearchTest, Test30HzZeroAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<0, Tap30Hz>(board);
    TestZeroAdj<Tap30Hz>(board);
  }
}

TEST_F(SearchTest, Test12Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<21, Tap12Hz>(board);
  }
}

TEST_F(SearchTest, TestTuck) {
  SetUp();
  for (auto& board : kTestTuckBoards) {
    TestSearchPosition<kLevel18>(board);
    TestSearchPosition<kLevel19>(board);
    TestSearchPosition<kLevel29>(board);
    TestSearchPosition<kLevel39>(board);
  }
}

} // namespace
