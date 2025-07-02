#pragma GCC optimize("fast-math")

#include "evaluate.h"

#include <cmath>
#include <set>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#pragma GCC diagnostic ignored "-Wtautological-compare"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop
#include "edge.h"
#include "game.h"
#include "board.h"
#include "config.h"
#include "board_set.h"
#include "thread_queue.h"

#pragma GCC diagnostic ignored "-Wclass-memaccess"

namespace {

struct Stats {
#ifdef TETRIS_ONLY
  static constexpr float kBucketSize = 1. / 65536;
  static constexpr int kMaximum = 1;
#else
  static constexpr int kBucketSize = 32;
  static constexpr int kMaximum = 2400000;
#endif
  static constexpr size_t kBuckets = kMaximum / kBucketSize + 1;
  std::array<std::atomic_long, kBuckets> distribution;
  std::atomic<float> maximum;

  void Update(const float val) {
    int bucket = (int)(val / kBucketSize);
    distribution[bucket]++;
    float prev_value = maximum;
    while (prev_value < val && !maximum.compare_exchange_weak(prev_value, val));
  }
  void Clear() {
    memset(distribution.data(), 0x0, sizeof(distribution));
    maximum = 0.0f;
  }
} stats;

void CalculateBlock(
    const EvaluateNodeEdgesFast* edges, size_t edges_size,
    const std::vector<NodeEval>& prev,
    int base_lines,
    NodeEval out[]) {
  if (!edges_size) return;
  if (edges_size % kPieces != 0) throw std::logic_error("unexpected: not multiples of 7");
  size_t boards = edges_size / kPieces;

  NodeEval local_val[256], adj_val[256];
  for (size_t b = 0; b < boards; b++) {
    float ev[7] = {}, var[7] = {};
    for (size_t piece = 0; piece < kPieces; piece++) {
      auto& item = edges[b * kPieces + piece];
      if (!item.next_ids_size) continue;
      for (size_t i = 0; i < item.next_ids_size; i++) {
        auto& [next, lines] = item.next_ids[i];
        local_val[i] = prev[next];
        local_val[i] += TablebaseScore(base_lines, lines);
      }
      __m256 probs = _mm256_load_ps(kTransitionProb[piece]);
      NodeEval result(_mm256_setzero_ps(), _mm256_setzero_ps());
      float mx_ev = 0.;
      auto Update = [&result,&mx_ev,&probs](const NodeEval& cur) {
        float cur_ev = cur.Dot(probs);
        if (cur_ev > mx_ev) mx_ev = cur_ev, result = cur;
      };
      for (size_t i = 0; i < item.non_adj_size; i++) {
        Update(local_val[item.non_adj[i]]);
      }
      if (item.use_subset) {
        for (size_t i = 0; i < item.subset_idx_prev_size; i++) {
          auto& [idx, prev] = item.subset_idx_prev[i];
          adj_val[i] = local_val[idx];
          if (prev != 255) adj_val[i].MaxWith(adj_val[prev]);
        }
        for (size_t i = 0; i < item.adj_subset_size; i++) {
          Update(adj_val[item.adj_subset[i]]);
        }
      } else {
        for (size_t i = 0; i < item.adj_lst_size; i++) {
          size_t start = item.adj_lst[i], end = item.adj_lst[i+1];
          NodeEval cur = local_val[item.adj[start]];
          for (size_t j = start + 1; j < end; j++) cur.MaxWith(local_val[item.adj[j]]);
          Update(cur);
        }
      }
      ev[piece] = mx_ev;
      var[piece] = result.DotVar(probs, mx_ev);
      stats.Update(mx_ev);
    }
    out[b].LoadEv(ev);
    out[b].LoadVar(var);
  }
}

void CalculateSameLines(
    int group, size_t start, size_t end, const std::vector<NodeEval>& prev, int lines,
    NodeEval out[]) {
  constexpr size_t kBatchSize = 1024;
  constexpr size_t kBlockSize = 524288;

  int level = GetLevelByLines(lines);
  auto fname = EvaluateEdgePath(group, GetLevelSpeed(level));
  using Result = std::pair<size_t, size_t>;

  BS::thread_pool io_pool(kIOThreads);
  auto thread_queue = MakeThreadQueue<Result>(kParallel,
      [&](Result range) {
        /* collect stats */
      });

  std::mutex mtx;
  std::condition_variable cv;
  size_t unfinished = 0;
  std::deque<std::function<Result()>> works;

  std::vector<std::thread> thrs;
  for (size_t block_start = start; block_start < end; block_start += kBlockSize) {
    size_t block_end = std::min(end, block_start + kBlockSize);
    unfinished++;
    io_pool.push_task([&fname,&thread_queue,&works,&unfinished,&cv,&mtx,block_start,block_end,&prev,lines,out]() {
      CompressedClassReader<EvaluateNodeEdgesFast> reader(fname);
      reader.Seek(block_start * kPieces);
      for (size_t batch_l = block_start; batch_l < block_end; batch_l += kBatchSize) {
        size_t batch_r = std::min(block_end, batch_l + kBatchSize);
        size_t num_to_read = (batch_r - batch_l) * kPieces;
        std::unique_ptr<EvaluateNodeEdgesFast[]> edges(new EvaluateNodeEdgesFast[num_to_read]);
        if (num_to_read != reader.ReadBatch(edges.get(), num_to_read)) throw std::runtime_error("read failure");
        {
          std::lock_guard lck(mtx);
          works.push_back(make_copyable_function([
                edges=std::move(edges),&works,&unfinished,&cv,&mtx,num_to_read,batch_l,batch_r,&prev,lines,out
          ]() {
            CalculateBlock(edges.get(), num_to_read, prev, lines, out + batch_l);
            return std::make_pair(batch_l, batch_r);
          }));
        }
        cv.notify_one();
      }
      {
        std::lock_guard lck(mtx);
        unfinished--;
      }
      cv.notify_one();
    });
  }
  {
    std::unique_lock lck(mtx);
    while (unfinished) {
      cv.wait(lck, [&]{ return unfinished == 0 || works.size(); });
      while (works.size()) {
        thread_queue.Push(std::move(works.front()));
        works.pop_front();
      }
    }
  }
  io_pool.wait_for_tasks();
  thread_queue.WaitAll();
}

} // namespace

std::vector<NodeEval> CalculatePiece(
    int pieces, const std::vector<NodeEval>& prev, const std::vector<size_t>& offsets) {
  int group = GetGroupByPieces(pieces);
  std::vector<NodeEval> ret(offsets.back());

  spdlog::info("Start calculate piece {}", pieces);
  stats.Clear();
  size_t start = 0, last = offsets.back();
  int cur_lines = -1; // -1 -> uninitialized
  for (size_t i = 0; i < offsets.size() - 1; i++) {
    int cells = pieces * 4 - GetCellsByGroupOffset(i, group);
    if (cells < 0) {
      last = offsets[i];
      break;
    }
    if (cells % 10) throw std::logic_error("unexpected: cells incorrect");
    int lines = cells / 10;
    if (lines >= kLineCap) {
      // lines will decrease as i increase, so this only happen at the start of the loop
      memset(ret.data() + start, 0x0, (offsets[i + 1] - start) * sizeof(NodeEval));
      start = offsets[i + 1];
      continue;
    }
    if (cur_lines != -1) {
      spdlog::debug("Calculate group {} lines {}: {} - {}", group, cur_lines, start, offsets[i]);
      CalculateSameLines(group, start, offsets[i], prev, cur_lines, ret.data());
      start = offsets[i];
    }
    cur_lines = lines;
  }
  if (cur_lines != -1) {
    spdlog::debug("Calculate group {} lines {}: {} - {}", group, cur_lines, start, last);
    CalculateSameLines(group, start, last, prev, cur_lines, ret.data());
  }
  {
    MkdirForFile(ValueStatsPath(pieces));
    std::ofstream fout(ValueStatsPath(pieces));
    for (size_t i = 0; i <= (size_t)(stats.maximum.load() / Stats::kBucketSize); i++) {
      if (long x = stats.distribution[i].load()) {
        fout << i * Stats::kBucketSize << ' ' << x << '\n';
      }
    }
  }
  std::vector<float> ev(7), var(7);
  ret[0].GetEv(ev.data());
  ret[0].GetVar(var.data());
  for (auto& i : var) i = std::sqrt(i);
  spdlog::debug("Finish piece {}: max_val {}, val0 {} {}", pieces, stats.maximum.load(), ev, var);
  return ret;
}

std::vector<NodeEval> ReadValues(int pieces, size_t total_size) {
  int group = GetGroupByPieces(pieces);
  if (!total_size) total_size = BoardCount(BoardPath(group));
  CompressedClassReader<NodeEval> reader(ValuePath(pieces));
  auto values = reader.ReadBatch(total_size);
  if (values.size() != total_size) throw std::length_error("value file length incorrect");
  return values;
}

std::vector<MoveEval> ReadValuesEvOnly(int pieces, size_t total_size) {
  constexpr size_t kBatchSize = 131072;
  int group = GetGroupByPieces(pieces);
  if (!total_size) total_size = BoardCount(BoardPath(group));
  CompressedClassReader<NodeEval> reader(ValuePath(pieces));
  std::vector<MoveEval> values;
  values.reserve(total_size);
  for (size_t i = 0; i < total_size; i += kBatchSize) {
    auto vec = reader.ReadBatch(kBatchSize);
    for (auto& x : vec) values.emplace_back(x.ev_vec);
  }
  if (values.size() != total_size) throw std::length_error("value file length incorrect");
  return values;
}

void RunEvaluate(int start_pieces, const std::vector<int>& output_locations, bool sample) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  std::vector<std::pair<uint32_t, uint8_t>> samples[kGroups];
  if (sample) {
    spdlog::info("Start reading sample files");
    for (int g = 0; g < kGroups; g++) {
      auto fname = SVDSamplePath(g);
      std::vector<uint8_t> mask(offsets[g].back());
      if (mask.size() >= (1ll << 32)) throw std::length_error("too many boards");
      if (mask.size() != std::filesystem::file_size(fname)) throw std::length_error("sample file length incorrect");
      std::ifstream(fname).read(reinterpret_cast<char*>(mask.data()), mask.size());
      for (uint64_t i = 0; i < mask.size(); i++) {
        if (!mask[i]) continue;
        for (uint8_t p = 0; p < kPieces; p++) {
          if (mask[i] >> p & 1) samples[g].push_back({(uint32_t)i, p});
        }
      }

      std::vector<uint8_t> cnt(samples[g].size());
      for (size_t i = 0, idx = 0; i < offsets[g].size() - 1; i++) {
        for (; samples[g][idx].first < offsets[g][i+1] && idx < samples[g].size(); idx++) {
          cnt[idx] = GetCellsByGroupOffset(i, g);
        }
      }
      auto cnt_fname = SVDSampleCountPath(g);
      MkdirForFile(cnt_fname);
      std::ofstream fout(cnt_fname);
      if (!fout.is_open()) throw std::runtime_error("file open failed");
      fout.write(reinterpret_cast<const char*>(cnt.data()), cnt.size());
    }
  }

  std::vector<NodeEval> values;
  if (start_pieces == -1) {
    size_t max_cells = 0;
    for (int i = 0; i < kGroups; i++) {
      max_cells = std::max(max_cells, (size_t)GetCellsByGroupOffset(offsets[i].size() - 1, i));
    }
    start_pieces = (kLineCap * 10 + max_cells + 3) / 4;
    int start_group = GetGroupByPieces(start_pieces);
    values.resize(offsets[start_group].back());
    memset(values.data(), 0x0, values.size() * sizeof(NodeEval));
  } else {
    int start_group = GetGroupByPieces(start_pieces);
    values = ReadValues(start_pieces, offsets[start_group].back());
    if (values.size() != offsets[start_group].back()) throw std::length_error("initial value file incorrect");
  }

  // really lazy here
  std::set<int> location_set(output_locations.begin(), output_locations.end());
  int last_output = *location_set.begin();
  for (int pieces = start_pieces - 1; pieces >= last_output; pieces--) {
    values = CalculatePiece(pieces, values, offsets[GetGroupByPieces(pieces)]);
    if (location_set.count(pieces)) {
      spdlog::info("Writing values of piece {}", pieces);
      CompressedClassWriter<NodeEval> writer(ValuePath(pieces), 2048);
      writer.Write(values);
    }
    if (sample) {
      spdlog::info("Writing samples of piece {}", pieces);
      ClassWriter<BasicIOType<float>> writer_ev(SVDEvPath(pieces)), writer_var(SVDVarPath(pieces));
      for (auto& [v, p] : samples[GetGroupByPieces(pieces)]) {
        float ev[8], var[8];
        values[v].GetEv(ev);
        values[v].GetVar(var);
        writer_ev.Write(ev[p]);
        writer_var.Write(var[p]);
      }
    }
  }
}
