#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "../../src/move.h"
#include "../../src/config.h"
#include "../../src/board_set.h"

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

TEST(PipelineTest, PipelineTest) {
  spdlog::set_pattern("[%t] %+");
  spdlog::set_level(spdlog::level::debug);

  kParallel = 10;
  kDataDir = "./data_dir";
  SplitBoards(STRINGIFY(BOARD_FILE), true);

  std::vector<int> groups{0,1,2,3,4,5,6,7,8,9};
  BuildEdges(groups);

  kIOThreads = 4;
  kParallel = 4;
  RunCalculateMoves(-1, 0);

  kParallel = 10;
  MergeMoveRanges(0, 54, true);
  MergeMoveRanges(54, 111, true);
  MergeMoveRanges(111, 138, true);
  MergeFullMoveRanges(true);
}
