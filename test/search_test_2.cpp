#include "search_test.h"

namespace {

TEST_F(SearchTest, Test30Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<18, Tap30Hz>(board);
  }
}

TEST_F(SearchTest, Test30HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<4, Tap30Hz>(board);
  }
}

} // namespace
