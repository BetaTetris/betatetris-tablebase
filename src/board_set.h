#pragma once

#include <vector>
#include <filesystem>

#include "io.h"
#include "game.h"
#include "board.h"
#include "files.h"
#include "sparse_map.h"

using BoardMap = StaticSparseMap<BoardTmpl<8>, uint64_t>;

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
