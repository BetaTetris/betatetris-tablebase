#include "../src/io.h"
#include "../src/board.h"
#include <algorithm>

int main(int argc, char** argv) {
  if (argc < 3) return 1;

  ClassReader<CompactBoard> reader(argv[1]);
  std::vector<CompactBoard> vec;
  while (true) {
    auto boards = reader.ReadBatch(262144);
    if (boards.empty()) break;
    for (auto& b : boards) {
      Board board(b);
      if (board.Height() > 6) continue;
      vec.push_back(b);
    }
  }
  std::sort(vec.begin(), vec.end());
  CompressedClassWriter<CompactBoard> writer(argv[2], 524288, 16);
  writer.Write(vec);
}

