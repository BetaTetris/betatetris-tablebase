#include "../src/io.h"
#include "../src/supervised_data.h"
#include <random>

std::mt19937_64 gen;

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  std::vector<CompressedClassReader<SupervisedData>> readers;
  for (int i = 2; i < argc; i++) readers.emplace_back(std::string(argv[i]));
  CompressedClassWriter<SupervisedData> writer(argv[1], 65536);

  constexpr size_t kBlock = 262144;
  std::vector<SupervisedData> data;
  data.reserve(kBlock * readers.size());
  while (true) {
    data.clear();
    bool flag = true;
    for (auto& i : readers) {
      auto blk = i.ReadBatch(kBlock);
      if (blk.size() != kBlock) {
        flag = false;
        break;
      }
      for (auto& j : blk) data.emplace_back(std::move(j));
    }
    if (!flag) break;
    std::shuffle(data.begin(), data.end(), gen);
    writer.Write(data);
  }
}
