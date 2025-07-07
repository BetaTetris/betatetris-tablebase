#pragma once

#include <vector>
#include <filesystem>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <tsl/sparse_map.h>
#pragma GCC diagnostic pop

#include "io.h"
#include "game.h"
#include "board.h"
#include "files.h"

using BoardMapKey = std::pair<BoardTmpl<8>, uint64_t>;
using BoardMap = tsl::sparse_map<BoardTmpl<8>, uint64_t, std::hash<Board>, std::equal_to<Board>,
      std::allocator<BoardMapKey>, tsl::sh::power_of_two_growth_policy<2>,
      tsl::sh::exception_safety::basic, tsl::sh::sparsity::high>;

void SplitBoards(const std::filesystem::path&, bool is_compressed = false);

template <class Func> void ProcessBoards(int group, Func&& f) {
  constexpr size_t kBlock = 65536;
  auto fname = BoardPath(group);
  ClassReader<CompactBoard> reader(fname);
  while (true) {
    auto chunk = reader.ReadBatch(kBlock);
    for (auto& i : chunk) f(Board(i));
    if (chunk.size() < kBlock) break;
  }
}

BoardMap GetBoardMap(int group);
void WriteBoardMap();

void BuildEdges(const std::vector<int>& groups);

// ret[count/10] = offset
std::vector<size_t> GetBoardCountOffset(int group);
