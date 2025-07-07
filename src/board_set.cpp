#include "board_set.h"

#include <fstream>
#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop
#include <spdlog/fmt/fmt.h>

#include <tsl/hopscotch_map.h>

#include "edge.h"
#include "config.h"
#include "io_hash.h"
#include "move_search.h"
#include "thread_queue.h"

namespace {

constexpr size_t kBlock = 65536;

void WriteBoardMap(int group) {
  size_t num_boards = BoardCount(BoardPath(group));
  spdlog::info("Start reading boards of group {}", group);
  if (num_boards >= (1ll << 32)) throw std::range_error("Too many boards");
  std::vector<std::pair<CompactBoard, BasicIOType<uint32_t>>> vec;
  vec.reserve(num_boards);
  uint32_t i = 0;
  ProcessBoards(group, [&i,&vec](Board&& b) {
    vec.emplace_back(b.ToBytes(), i++);
  });
  int pow = 31 - clz<uint32_t>(num_boards);
  pow = std::max(5, pow - 4); // ~32 boards / bucket
  spdlog::info("Writing board map for group {}", group);
  WriteHashMap(BoardMapPath(group), std::move(vec), 1 << pow);
  spdlog::info("Board map writing done");
}

template <class Reader> void SplitBoards(Reader& reader) {
  std::array<std::vector<CompactBoard>, kGroups> boards;
  size_t num_boards = reader.SizeUpperBound();
  if (num_boards != std::string::npos) {
    spdlog::info("Board file contains about {} boards", num_boards);
    for (auto& i : boards) i.reserve(num_boards * (1.0 / kGroups + 0.02) + 10);
  }
  spdlog::info("Start reading board file");
  while (true) {
    auto chunk = reader.ReadBatch(kBlock);
    for (auto& i : chunk) {
      int cells = i.Count();
      if (cells % kCellsMod != 0) continue;
      boards[GetGroupByCells(cells)].push_back(i);
    }
    if (chunk.size() < kBlock) break;
  }
  int threads = std::min(kParallel, kGroups);
  spdlog::info("Finish reading board file, sorting with {} threads", threads);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int i = l; i < r; i++) {
      std::sort(boards[i].begin(), boards[i].end(), [](const CompactBoard& a, const CompactBoard& b){
        return a.Count() == b.Count() ? a < b : a.Count() < b.Count();
      });
      boards[i].resize(std::unique(boards[i].begin(), boards[i].end()) - boards[i].begin());
    }
  }).wait();
  spdlog::info("Sorting finished");
  for (int group = 0; group < kGroups; group++) {
    spdlog::info("Writing group {} with {} boards", group, boards[group].size());
    ClassWriter<CompactBoard> writer(BoardPath(group));
    writer.Write(boards[group]);
  }
}

} // namespace

void SplitBoards(const std::filesystem::path& fname, bool is_compressed) {
  spdlog::info("Start preprocessing");
  if (is_compressed) {
    CompressedClassReader<CompactBoard> reader(fname);
    SplitBoards(reader);
  } else {
    ClassReader<CompactBoard> reader(fname);
    SplitBoards(reader);
  }
  spdlog::info("Done preprocessing");
}

BoardMap GetBoardMap(int group) {
  auto fname = BoardPath(group);
  size_t num_boards = BoardCount(fname);

  BoardMap ret;
  ret.reserve(num_boards);
  size_t i = 0;
  ProcessBoards(group, [&](Board&& b) { ret[b] = i++; });
  return ret;
}

void WriteBoardMap() {
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) WriteBoardMap(group);
  }).get();
}

namespace {

using TapSpeed = TAP_SPEED;
constexpr int kAdjDelay = ADJ_DELAY;

struct EdgeStats {
  std::atomic_long next, non_adj, adj_orig, adj, adj_ed, adj_ed_fin, subset;
  void Clear() {
    next.store(0);
    non_adj.store(0);
    adj_orig.store(0);
    adj.store(0);
    adj_ed.store(0);
    adj_ed_fin.store(0);
    subset.store(0);
  }
  std::string ToText() const {
    return fmt::format("next {}, non_adj {}, adj_orig {}, adj {}, adj_ed {}, adj_ed_fin {}, subset {}",
        next.load(), non_adj.load(), adj_orig.load(), adj.load(), adj_ed.load(), adj_ed_fin.load(), subset.load());
  }
} edge_stats[4];

inline std::pair<EvaluateNodeEdges, PositionNodeEdges> GetEdges(
    const Board& b, int piece, const PossibleMoves& moves, const BoardMap& mp, int level) {
  constexpr uint64_t kNone = std::numeric_limits<uint64_t>::max();
  // use position as key; note that multiple positions may lead to same board
  tsl::hopscotch_map<Position, std::pair<uint64_t, uint8_t>> mp_next;
  mp_next.reserve(48);
  for (auto& pos : moves.non_adj) mp_next[pos] = {kNone, 0};
  for (auto& adj : moves.adj) {
    for (auto& pos : adj.second) mp_next[pos] = {kNone, 0};
  }
  for (auto item = mp_next.begin(); item != mp_next.end();) {
    auto n_board = b.Place(piece, item->first.r, item->first.x, item->first.y).ClearLines();
#ifdef TETRIS_ONLY
    if (n_board.first && n_board.first != 4) {
      item = mp_next.erase(item); // only tetrises are allowed
    } else // ... if
#endif
    if (auto board_it = mp.find(n_board.second); board_it != mp.end()) {
      item.value() = {board_it->second, n_board.first};
      ++item;
    } else {
      item = mp_next.erase(item);
    }
  }
  tsl::hopscotch_map<Position, uint8_t> mp_idx;
  EvaluateNodeEdges eval_ed;
  PositionNodeEdges pos_ed;
  eval_ed.cell_count = b.Count();
  { // nexts
    uint8_t idx = 0;
    mp_idx.reserve(mp_next.size());
    eval_ed.next_ids.reserve(mp_next.size());
    pos_ed.nexts.reserve(mp_next.size());
    for (auto& item : mp_next) {
      eval_ed.next_ids.push_back(item.second);
      pos_ed.nexts.push_back(item.first);
      mp_idx[item.first] = idx++;
    }
  }
  // non-adjs
  for (const auto& pos : moves.non_adj) {
    if (auto it = mp_idx.find(pos); it != mp_idx.end()) {
      eval_ed.non_adj.push_back(it->second);
    }
  }
  // adjs
  size_t adj_eds = 0;
  auto& stats = edge_stats[level];
  {
    std::vector<Position> pos_adj;
    for (const auto& adj : moves.adj) {
      std::vector<uint8_t> ids;
      for (const auto& pos : adj.second) {
        if (auto it = mp_idx.find(pos); it != mp_idx.end()) {
          ids.push_back(it->second);
        }
      }
      if (ids.empty()) continue;
      eval_ed.adj.push_back(std::move(ids));
      pos_adj.push_back(adj.first);
      ids.clear();
    }
    stats.adj_orig += eval_ed.adj.size();
    const auto adj_mp = eval_ed.ReduceAdj();
    for (const auto& i : adj_mp) {
      pos_ed.adj.emplace_back();
      for (const auto& j : i) pos_ed.adj.back().push_back(pos_adj[j]);
    }
    for (auto& i : eval_ed.adj) adj_eds += i.size();
  }
  // stats
  stats.next += eval_ed.next_ids.size();
  stats.non_adj += eval_ed.non_adj.size();
  stats.adj += eval_ed.adj.size();
  stats.adj_ed += adj_eds;
  // subset optim
  eval_ed.CalculateSubset();
  if (adj_eds >= 1.5 * eval_ed.subset_idx_prev.size()) {
    eval_ed.adj.clear();
    eval_ed.use_subset = true;
    stats.subset += eval_ed.subset_idx_prev.size();
  } else {
    stats.adj_ed_fin += adj_eds;
  }
  return {eval_ed, pos_ed};
}

using EdgeChunk = std::pair<std::array<std::vector<std::vector<uint8_t>>, kLevels>,
                            std::array<std::vector<std::vector<uint8_t>>, kLevels>>;

EdgeChunk BuildEdgeChunk(const std::vector<Board>& boards, const BoardMap& mp) {
  EdgeChunk ret;
  auto& [eval_eds, pos_eds] = ret;
  std::array<std::vector<std::array<PossibleMoves, kPieces>>, kLevels> search_results;
  for (int i = 0; i < kLevels; i++) search_results[i].resize(boards.size());
  For<kLevels>([&](auto level_obj) {
    constexpr int level_val = level_obj.value;
    constexpr Level level = static_cast<Level>(level_obj.value);
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][0] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].TMap());
      search_results[level_val][i][1] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].JMap());
      search_results[level_val][i][5] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].LMap());
    }
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][2] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].ZMap());
      search_results[level_val][i][4] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].SMap());
      search_results[level_val][i][6] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].IMap());
    }
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][3] = MoveSearch<level, 1, ADJ_DELAY, TAP_SPEED>(boards[i].OMap());
    }
  });
  for (int level = 0; level < kLevels; level++) {
    auto& cur_eval = eval_eds[level];
    auto& cur_pos = pos_eds[level];
    auto& cur_moves = search_results[level];
    cur_eval.reserve(boards.size() * kPieces);
    cur_pos.reserve(boards.size() * kPieces);
    for (size_t i = 0; i < boards.size(); i++) {
      for (size_t j = 0; j < kPieces; j++) {
        auto [eval_ed, pos_ed] = GetEdges(boards[i], j, cur_moves[i][j], mp, level);
        cur_eval.push_back(Serialize(eval_ed));
        cur_pos.push_back(Serialize(pos_ed));
      }
    }
  }
  return ret;
}

void BuildEdges(int group) {
  spdlog::info("Start building edges for group {}", group);
  int nxt_group = NextGroup(group);
  spdlog::info("Loading board map for group {}", nxt_group);
  BoardMap mp = GetBoardMap(nxt_group);
  spdlog::info("Board map loaded with {} boards", mp.size());

  std::vector<CompressedClassWriter<EvaluateNodeEdges>> eval_writers;
  std::vector<CompressedClassWriter<PositionNodeEdges>> pos_writers;
  for (int level = 0; level < kLevels; level++) {
    eval_writers.emplace_back(EvaluateEdgePath(group, level), 512 * kPieces);
    pos_writers.emplace_back(PositionEdgePath(group, level), 512 * kPieces);
  }

  spdlog::info("Start building edges");
  for (int i = 0; i < kLevels; i++) edge_stats[i].Clear();
  auto PrintStats = [](size_t n_boards, int level, spdlog::level::level_enum log_level) {
    spdlog::log(log_level, "{} boards processed, level {}: {}",
        n_boards, level, edge_stats[level].ToText());
  };
  size_t n_boards = 0;
  auto thread_queue = MakeThreadQueue<EdgeChunk>(kParallel, [&](EdgeChunk&& chunk) {
    constexpr size_t kOutput = 131072;
    size_t sz = chunk.first[0].size() / kPieces;
    bool output = n_boards / kOutput != (n_boards + sz) / kOutput;
    n_boards += sz;
    for (int level = 0; level < kLevels; level++) {
      if (output) PrintStats(n_boards, level, spdlog::level::debug);
      eval_writers[level].WriteRaw(chunk.first[level]);
      pos_writers[level].WriteRaw(chunk.second[level]);
    }
  });
  std::vector<Board> block;
  block.reserve(1024);
  ProcessBoards(group, [&](Board&& b) {
    block.push_back(std::move(b));
    if (block.size() == 1024) {
      thread_queue.Push([block=std::move(block),&mp]() { return BuildEdgeChunk(std::cref(block), std::cref(mp)); });
      block.clear(); // recover from move
    }
  });
  thread_queue.Push([block=std::move(block),&mp]() { return BuildEdgeChunk(std::cref(block), std::cref(mp)); });
  thread_queue.WaitAll();
  for (int level = 0; level < kLevels; level++) PrintStats(n_boards, level, spdlog::level::info);
}

} // namespace

void BuildEdges(const std::vector<int>& groups) {
  std::array<BoardMap, kGroups> maps;
  for (int i : groups) BuildEdges(i);
}

std::vector<size_t> GetBoardCountOffset(int group) {
  auto fname = BoardPath(group);
  size_t num_boards = BoardCount(fname);
  ClassReader<CompactBoard> reader(fname);

  reader.Seek(num_boards - 1, 0);
  int max_count = Board(reader.ReadOne(0)).Count();
  std::vector<size_t> ret(max_count / kGroupInterval + 2);
  ret.back() = num_boards;

  for (size_t i = 1; i < ret.size() - 1; i++) {
    size_t l = 0, r = num_boards;
    while (l + 1 < r) {
      size_t m = (l + r) / 2;
      reader.Seek(m - 1);
      if ((size_t)Board(reader.ReadOne(0)).Count() >= i * kGroupInterval) {
        r = m;
      } else {
        l = m;
      }
    }
    ret[i] = l;
  }
  return ret;
}
