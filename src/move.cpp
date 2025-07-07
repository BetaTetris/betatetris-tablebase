#pragma GCC optimize("fast-math")

#include "move.h"

#include <cmath>
#include <set>
#include <map>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#pragma GCC diagnostic ignored "-Wtautological-compare"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop
#include <tsl/hopscotch_map.h>
#include "config.h"
#include "evaluate.h"
#include "board_set.h"
#include "thread_queue.h"

#pragma GCC diagnostic ignored "-Wclass-memaccess"

NodeMoveBoardRange::NodeMoveBoardRange(
    const NodeMovePositionRange& range, const EvaluateNodeEdgesFast& eval_ed, const PositionNodeEdges& pos_ed) {
  tsl::hopscotch_map<Position, uint32_t> mp_idx;
  tsl::hopscotch_map<uint32_t, uint8_t> mp_pos;
  for (size_t i = 0; i < pos_ed.nexts.size(); i++) {
    mp_idx[pos_ed.nexts[i]] = eval_ed.next_ids[i].first;
  }
  for (auto& i : range.ranges) {
    ranges.push_back({i.start, i.end, {}});
    for (size_t j = 0; j < kPieces; j++) {
      auto& pos = i.pos[j];
      if (pos == Position::Invalid) {
        ranges.back().idx[j] = 0xff;
        continue;
      }
      size_t num = mp_pos.size();
      size_t idx = mp_idx[pos];
      auto it = mp_pos.insert({idx, num});
      if (it.second) {
        board_idx.push_back(idx);
      } else {
        num = it.first->second;
      }
      ranges.back().idx[j] = num;
    }
  }
  if (mp_pos.size() >= 255) throw std::length_error("Too many edges");
}

size_t NodeMoveBoardRangeFast::lines = 0;

namespace {

struct Stats {
  std::atomic<float> maximum;

  void Update(const float val) {
    float prev_value = maximum;
    while (prev_value < val && !maximum.compare_exchange_weak(prev_value, val));
  }
  void Clear() {
    maximum = 0.0f;
  }
} stats;

inline void NodeMoveIndexFromVec(NodeMoveIndex& ret, __m256i v) {
  alignas(32) uint32_t idx[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(idx), v);
  for (size_t i = 0; i < kPieces; i++) ret[i] = idx[i];
}

template <bool calculate_moves>
void CalculateBlock(
    const EvaluateNodeEdgesFast* edges, size_t edges_size,
    const std::vector<MoveEval>& prev,
    int base_lines,
    MoveEval out[], NodeMoveIndex out_idx[] = nullptr) {
  if (!edges_size) return;
  if (edges_size % kPieces != 0) throw std::logic_error("unexpected: not multiples of 7");
  size_t boards = edges_size / kPieces;

  MoveEval local_val[256], adj_val[256];
  __m256i adj_idx[256];
  for (size_t b = 0; b < boards; b++) {
    float ev[7] = {};
    for (size_t piece = 0; piece < kPieces; piece++) {
      auto& item = edges[b * kPieces + piece];
      if (!item.next_ids_size) continue;
      for (size_t i = 0; i < item.next_ids_size; i++) {
        auto& [next, lines] = item.next_ids[i];
        local_val[i] = prev[next];
        local_val[i] += TablebaseScore(base_lines, lines);
      }
      __m256 probs = _mm256_load_ps(kTransitionProb[piece]);
      __m256i res_idx = _mm256_setzero_si256();
      float mx_ev = 0.;
      auto Update = [&res_idx,&mx_ev,&probs](const MoveEval& cur, __m256i new_idx) {
        float cur_ev = cur.Dot(probs);
        if (cur_ev > mx_ev) mx_ev = cur_ev, res_idx = new_idx;
      };
      for (size_t i = 0; i < item.non_adj_size; i++) {
        auto idx = item.non_adj[i];
        if constexpr (calculate_moves) {
          Update(local_val[idx], _mm256_set1_epi32(idx));
        } else {
          mx_ev = std::max(mx_ev, local_val[idx].Dot(probs));
        }
      }
      if (item.use_subset) {
        for (size_t i = 0; i < item.subset_idx_prev_size; i++) {
          auto& [idx, prev] = item.subset_idx_prev[i];
          if (prev != 255) {
            if constexpr (calculate_moves) {
              adj_val[i] = adj_val[prev];
              adj_idx[i] = adj_val[i].MaxWithMask(local_val[idx], adj_idx[prev], idx);
            } else {
              adj_val[i] = adj_val[prev];
              adj_val[i].MaxWith(local_val[idx]);
            }
          } else {
            adj_val[i] = local_val[idx];
            if constexpr (calculate_moves) adj_idx[i] = _mm256_set1_epi32(idx);
          }
        }
        for (size_t i = 0; i < item.adj_subset_size; i++) {
          auto idx = item.adj_subset[i];
          if constexpr (calculate_moves) {
            Update(adj_val[idx], adj_idx[idx]);
          } else {
            mx_ev = std::max(mx_ev, adj_val[idx].Dot(probs));
          }
        }
      } else {
        for (size_t i = 0; i < item.adj_lst_size; i++) {
          size_t start = item.adj_lst[i], end = item.adj_lst[i+1];
          MoveEval cur = local_val[item.adj[start]];
          if constexpr (calculate_moves) {
            __m256i cur_idx = _mm256_set1_epi32(item.adj[start]);
            for (size_t j = start + 1; j < end; j++) {
              cur_idx = cur.MaxWithMask(local_val[item.adj[j]], cur_idx, item.adj[j]);
            }
            Update(cur, cur_idx);
          } else {
            for (size_t j = start + 1; j < end; j++) cur.MaxWith(local_val[item.adj[j]]);
            mx_ev = std::max(mx_ev, cur.Dot(probs));
          }
        }
      }
      ev[piece] = mx_ev;
      if constexpr (calculate_moves) {
        NodeMoveIndexFromVec(out_idx[b * kPieces + piece], res_idx);
      }
      stats.Update(mx_ev);
    }
    out[b].LoadEv(ev);
  }
}

class IndexWriterThread {
  std::mutex mtx_;
  std::condition_variable cv_;
  std::map<std::pair<size_t, size_t>, std::pair<bool, std::vector<NodeMoveIndex>>> queue_;
  std::unique_ptr<CompressedClassWriter<NodeMoveIndex>> writer_;
  bool finished_, joined_;
  std::thread thr_;

  void Worker_() {
    size_t cur = 0;
    while (true) {
      std::unique_lock lock(mtx_);
      cv_.wait(lock, [&](){ return finished_ || (queue_.size() && queue_.begin()->first.first == cur); });
      if (finished_ && queue_.empty()) break;
      if (queue_.begin()->first.first != cur) throw std::runtime_error("unexpected");
      auto r = queue_.begin()->first;
      auto work = std::move(queue_.begin()->second);
      queue_.erase(queue_.begin());
      lock.unlock();
      if (work.first) { // repeated empty
        writer_->Write(NodeMoveIndex{}, (r.second - r.first) * kPieces);
      } else {
        writer_->Write(work.second);
      }
      cur = r.second;
    }
  }
 public:
  IndexWriterThread(std::unique_ptr<CompressedClassWriter<NodeMoveIndex>>&& writer) :
      writer_(std::move(writer)), finished_(false), joined_(false), thr_([this](){ Worker_(); }) {}
  ~IndexWriterThread() { Wait(); }

  void Wait() {
    if (joined_) return;
    {
      std::lock_guard lock(mtx_);
      finished_ = true;
    }
    cv_.notify_one();
    thr_.join();
  }

  void Push(size_t start, size_t end) {
    {
      std::lock_guard lock(mtx_);
      queue_.emplace(std::piecewise_construct,
          std::forward_as_tuple(start, end),
          std::forward_as_tuple(true, std::vector<NodeMoveIndex>()));
    }
    cv_.notify_one();
  }
  void Push(size_t start, size_t end, std::vector<NodeMoveIndex>&& vec) {
    {
      std::lock_guard lock(mtx_);
      queue_.emplace(std::piecewise_construct,
          std::forward_as_tuple(start, end),
          std::forward_as_tuple(false, std::move(vec)));
    }
    cv_.notify_one();
  }
};

template <bool calculate_moves>
void CalculateSameLines(
    int group, size_t start, size_t end, const std::vector<MoveEval>& prev, int lines,
    MoveEval out[], std::optional<IndexWriterThread>& writer) {
  constexpr size_t kBatchSize = 2048;
  constexpr size_t kBlockSize = 524288;

  auto fname = EvaluateEdgePath(group, GetLevelSpeedByLines(lines));
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
  for (size_t block_start = start; block_start < end;) {
    size_t block_end = std::min(end, (block_start / kBlockSize * kBlockSize) + kBlockSize);
    unfinished++;
    io_pool.push_task([&fname,&thread_queue,&works,&unfinished,&cv,&mtx,block_start,block_end,start,&prev,lines,out,&writer]() {
      CompressedClassReader<EvaluateNodeEdgesFast> reader(fname);
      reader.Seek(block_start * kPieces);
      for (size_t batch_l = block_start; batch_l < block_end;) {
        size_t batch_r = std::min(block_end, (batch_l / kBatchSize * kBatchSize) + kBatchSize);
        size_t num_to_read = (batch_r - batch_l) * kPieces;
        std::unique_ptr<EvaluateNodeEdgesFast[]> edges(new EvaluateNodeEdgesFast[num_to_read]);
        if (num_to_read != reader.ReadBatch(edges.get(), num_to_read)) throw std::runtime_error("read failure");
        {
          std::lock_guard lck(mtx);
          works.push_back(make_copyable_function([
              edges=std::move(edges),&works,&unfinished,&cv,&mtx,num_to_read,start,batch_l,batch_r,&prev,lines,out,&writer
          ]() {
            std::vector<NodeMoveIndex> out_idx(calculate_moves ? (batch_r - batch_l) * kPieces : 0);
            auto out_ptr = calculate_moves ? out_idx.data() : nullptr;
            CalculateBlock<calculate_moves>(edges.get(), num_to_read, prev, lines, out + batch_l, out_ptr);
            if constexpr (calculate_moves) writer->Push(batch_l, batch_r, std::move(out_idx));
            return std::make_pair(batch_l, batch_r);
          }));
        }
        cv.notify_one();
        batch_l = batch_r;
      }
      {
        std::lock_guard lck(mtx);
        unfinished--;
      }
      cv.notify_one();
    });
    block_start = block_end;
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

template <bool calculate_moves>
std::vector<MoveEval> CalculatePieceMoves(
    int pieces, const std::vector<MoveEval>& prev, const std::vector<size_t>& offsets) {
  int group = GetGroupByPieces(pieces);
  std::vector<MoveEval> ret(offsets.back());
  std::optional<IndexWriterThread> writer;
  if constexpr (calculate_moves) {
    writer.emplace(std::make_unique<CompressedClassWriter<NodeMoveIndex>>(MoveIndexPath(pieces), 4096 * kPieces));
  }

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
      memset(ret.data() + start, 0x0, (offsets[i + 1] - start) * sizeof(MoveEval));
      if constexpr (calculate_moves) writer->Push(start, offsets[i + 1]);
      start = offsets[i + 1];
      continue;
    }
    if (cur_lines != -1) {
      spdlog::debug("Calculate group {} lines {}: {} - {}", group, cur_lines, start, offsets[i]);
      CalculateSameLines<calculate_moves>(group, start, offsets[i], prev, cur_lines, ret.data(), writer);
      start = offsets[i];
    }
    cur_lines = lines;
  }
  if (cur_lines != -1) {
    spdlog::debug("Calculate group {} lines {}: {} - {}", group, cur_lines, start, last);
    CalculateSameLines<calculate_moves>(group, start, last, prev, cur_lines, ret.data(), writer);
  }
  if (last < offsets.back()) {
    memset(ret.data() + last, 0x0, (offsets.back() - last) * sizeof(MoveEval));
    if constexpr (calculate_moves) writer->Push(last, offsets.back());
  }
  writer.reset(); // wait for complete
  std::vector<float> ev(7);
  ret[0].GetEv(ev.data());
  spdlog::debug("Finish piece {}: max_val {}, val0 {}", pieces, stats.maximum.load(), ev);
  return ret;
}

template <class OneClass, class PartialClass, class OneFilenameFunc, class PartialFilenameFunc>
void MergeRanges(int group, int pieces_l, int pieces_r, const std::vector<size_t>& offset, bool delete_after,
                 OneFilenameFunc&& one_filename_func, PartialFilenameFunc&& partial_filename_func, size_t index_size) {
  spdlog::info("Merging group {}: {} - {}", group, pieces_l, pieces_r);
  int orig_pieces_l = pieces_l;
  while (GetGroupByPieces(pieces_l) != group) pieces_l++;
  std::vector<CompressedClassReader<OneClass>> readers;
  for (int pieces = pieces_l; pieces < pieces_r; pieces += kGroups) {
    readers.emplace_back(one_filename_func(pieces));
  }
  if (readers.empty()) return;
  CompressedClassWriter<PartialClass> writer(
      partial_filename_func(orig_pieces_l, pieces_r, group), index_size, -2);
  std::vector<OneClass> buf(readers.size());
  for (size_t i = 0; i < offset.size() - 1; i++) {
    int start_cells = pieces_l * 4 - GetCellsByGroupOffset(i, group);
    if (start_cells % 10 != 0) throw std::logic_error("unexpected");
    int start_lines = start_cells / 10;
    uint8_t start_lines_idx = (uint32_t)start_lines / kGroupLineInterval;
    size_t begin = 0;
    size_t end = std::min(buf.size(), (size_t)(kLineCap - start_lines + kGroupLineInterval - 1) / kGroupLineInterval);
    if (start_lines < 0) {
      start_lines_idx = 0;
      begin = (-start_lines + 1) / kGroupLineInterval;
    }
    for (size_t x = 0; x < (offset[i + 1] - offset[i]) * kPieces; x++) {
      for (size_t j = 0; j < readers.size(); j++) readers[j].ReadOne(&buf[j]);
      writer.Write(PartialClass(buf.begin() + begin, buf.begin() + end, start_lines_idx));
    }
  }
  spdlog::info("Group {} merged", group);
  if (delete_after) {
    readers.clear();
    for (int pieces = pieces_l; pieces < pieces_r; pieces += kGroups) {
      std::string fname = one_filename_func(pieces);
      std::filesystem::remove(fname);
      std::filesystem::remove(fname + ".index");
    }
  }
}

void MergeFullMoveRanges(int group, const std::vector<int>& sections, bool delete_after) {
  std::vector<CompressedClassReader<PositionNodeEdges>> pos_readers;
  std::vector<CompressedClassReader<NodeMoveIndexRange>> readers;
  CompressedClassWriter<NodeMovePositionRange> writer(MovePath(group), 256 * kPieces, -2);
  for (int i = 0; i < kLevels; i++) {
    pos_readers.emplace_back(PositionEdgePath(group, i));
  }
  for (size_t i = 0; i < sections.size() - 1; i++) {
    readers.emplace_back(MoveRangePath(sections[i], sections[i+1], group));
  }
  PositionNodeEdges ed[kLevels];
  size_t n_boards = GetBoardCountOffset(group).back();
  for (size_t i = 0; i < n_boards * kPieces; i++) {
    NodeMovePositionRange range;
    for (int lvl = 0; lvl < kLevels; lvl++) ed[lvl] = pos_readers[lvl].ReadOne();
    for (auto& move_reader : readers) {
      for (auto& move : move_reader.ReadOne().ranges) {
        // exploit the fact that transitions are all at even number of lines
        // for perfect play: lines always multiples of 4 so still okay
        static_assert(std::all_of(kLevelSpeedLines, kLevelSpeedLines + kLevels, [](int x){ return x % 2 == 0; }));
        Level start_level = GetLevelSpeedByLines(move.start * kGroupLineInterval);
        Level end_level = GetLevelSpeedByLines((move.end - 1) * kGroupLineInterval);
        for (int lvl = static_cast<int>(start_level); lvl <= static_cast<int>(end_level); lvl++) {
          uint8_t start_idx = std::max((kLevelSpeedLines[lvl] + kGroupLineInterval - 1) / kGroupLineInterval, (int)move.start);
          uint8_t end_idx = move.end;
          if (lvl != kLevels - 1) {
            end_idx = std::min((kLevelSpeedLines[lvl + 1] + kGroupLineInterval - 1) / kGroupLineInterval, (int)end_idx);
          }
          MovePositionRange item{start_idx, end_idx, {}};
          if (ed[lvl].nexts.size()) {
            for (size_t j = 0; j < kPieces; j++) item.pos[j] = ed[lvl].nexts[move.idx[j]];
          } else {
            for (size_t j = 0; j < kPieces; j++) item.pos[j] = Position::Invalid;
          }
          range <<= item;
        }
      }
    }
    writer.Write(range);
  }
  spdlog::info("Group {} merged", group);
  if (delete_after) {
    readers.clear();
    for (size_t i = 0; i < sections.size() - 1; i++) {
      std::string fname = MoveRangePath(sections[i], sections[i+1], group);
      std::filesystem::remove(fname);
      std::filesystem::remove(fname + ".index");
    }
  }
}

void MergeFullThresholdRanges(const std::string& name, int group, const std::vector<int>& sections, bool delete_after) {
  std::vector<CompressedClassReader<NodePartialThreshold>> readers;
  CompressedClassWriter<NodeThreshold> writer(ThresholdPath(name, group), 256 * kPieces, -2);
  for (size_t i = 0; i < sections.size() - 1; i++) {
    readers.emplace_back(ThresholdRangePath(name, sections[i], sections[i+1], group));
  }
  size_t n_boards = GetBoardCountOffset(group).back();
  for (size_t i = 0; i < n_boards * kPieces; i++) {
    NodeThreshold range = {};
    for (auto& reader : readers) {
      auto threshold = reader.ReadOne();
      memcpy(range.data() + threshold.start, threshold.levels.data(), threshold.levels.size());
    }
    writer.Write(range);
  }
  spdlog::info("Group {} merged", group);
  if (delete_after) {
    readers.clear();
    for (size_t i = 0; i < sections.size() - 1; i++) {
      std::string fname = ThresholdRangePath(name, sections[i], sections[i+1], group);
      std::filesystem::remove(fname);
      std::filesystem::remove(fname + ".index");
    }
  }
}

std::vector<MoveEval> LoadValues(int& start_pieces, const std::vector<size_t> offsets[]) {
  std::vector<MoveEval> values;
  if (start_pieces == -1) {
    size_t max_cells = 0;
    for (int i = 0; i < kGroups; i++) {
      max_cells = std::max(max_cells, (size_t)GetCellsByGroupOffset(offsets[i].size() - 1, i));
    }
    start_pieces = (kLineCap * 10 + max_cells + 3) / 4;
    int start_group = GetGroupByPieces(start_pieces);
    values.resize(offsets[start_group].back());
    memset(values.data(), 0x0, values.size() * sizeof(MoveEval));
  } else {
    int start_group = GetGroupByPieces(start_pieces);
    values = ReadValuesEvOnly(start_pieces, offsets[start_group].back());
    if (values.size() != offsets[start_group].back()) throw std::length_error("initial value file incorrect");
  }
  return values;
}

std::vector<int> GetSections(const std::vector<std::pair<int, int>>& ranges) {
  if (ranges.empty()) throw std::runtime_error("No ranges available");
  std::vector<int> sections;
  sections.push_back(ranges[0].first);
  sections.push_back(ranges[0].second);
  for (size_t i = 1; i < ranges.size(); i++) {
    if (ranges[i - 1].second != ranges[i].first) {
      throw std::runtime_error("Ranges not consecutive or mutually exclusive");
    }
    sections.push_back(ranges[i].second);
  }
  return sections;
}

void WriteThreshold(int pieces, const std::vector<size_t>& offset, const std::vector<MoveEval>& values,
                    const std::string& name, const std::vector<float> threshold,
                    float start_ratio, float end_ratio, uint8_t buckets) {
  spdlog::info("Writing threshold of piece {}", pieces);
  int group = GetGroupByPieces(pieces);
  CompressedClassWriter<BasicIOType<uint8_t>> writer(ThresholdOnePath(name, pieces), 65536 * kPieces);
  for (size_t i = 0; i < offset.size() - 1; i++) {
    int cells = pieces * 4 - GetCellsByGroupOffset(i, group);
    std::vector<BasicIOType<uint8_t>> out((offset[i+1] - offset[i]) * kPieces, BasicIOType<uint8_t>{});
    if (cells % 10) throw std::logic_error("unexpected: cells incorrect");
    int lines = cells / 10;
    if (cells >= 0 && lines < kLineCap) {
      float thresh_low = threshold[lines] * start_ratio;
      float thresh_high = threshold[lines] * end_ratio;
      //  0 <-|-> 1 2 3 ... buckets-3 buckets-2 <-|-> buckets-1
      // thresh_low                          thresh_high
      // bucket(val) = floor( (val-thresh_low)/(thresh_high-thresh_low)*(bucket-2) + 1 )
      //             = floor( (val-thresh_low)*multiplier + 1 )
      //             = floor( val*multiplier + (1-thresh_low*multiplier) )
      float multiplier = (buckets - 2) / (thresh_high - thresh_low);
      float bias = 1 - thresh_low * multiplier;
      float mx = buckets - 1;
      for (size_t idx = offset[i]; idx < offset[i+1]; idx++) {
        __m256 bucket = _mm256_fmadd_ps(values[idx].ev_vec, _mm256_set1_ps(multiplier), _mm256_set1_ps(bias));
        bucket = _mm256_min_ps(_mm256_set1_ps(mx), _mm256_max_ps(_mm256_setzero_ps(), bucket));
        alignas(32) float val[8];
        _mm256_store_ps(val, bucket);
        for (size_t j = 0; j < kPieces; j++) out[(idx - offset[i]) * kPieces + j] = val[j];
      }
    }
    writer.Write(out);
  }
}

} // namespace

std::vector<MoveEval> CalculatePiece(
    int pieces, const std::vector<MoveEval>& prev, const std::vector<size_t>& offsets) {
  return CalculatePieceMoves<false>(pieces, prev, offsets);
}

void RunCalculateMoves(int start_pieces, int end_pieces) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  std::vector<MoveEval> values = LoadValues(start_pieces, offsets);
  for (int pieces = start_pieces - 1; pieces >= end_pieces; pieces--) {
    values = CalculatePieceMoves<true>(pieces, values, offsets[GetGroupByPieces(pieces)]);
  }
}

void MergeMoveRanges(int pieces_l, int pieces_r, bool delete_after) {
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) {
      MergeRanges<NodeMoveIndex, NodeMoveIndexRange>(
          group, pieces_l, pieces_r, GetBoardCountOffset(group), delete_after,
          MoveIndexPath, MoveRangePath, 4096 * kPieces);
    }
  }).get();
}

void MergeFullMoveRanges(bool delete_after) {
  auto sections = GetSections(GetAvailableMoveRanges());
  spdlog::info("Start merge ranges {}", sections);
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) {
      MergeFullMoveRanges(group, sections, delete_after);
    }
  }).get();
}

void RunCalculateThreshold(
    int start_pieces, int end_pieces,
    const std::string& name, const std::string& threshold_path,
    float start_ratio, float end_ratio, uint8_t buckets) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  std::vector<float> threshold(kLineCap);
  {
    std::ifstream fin(threshold_path);
    for (float& i : threshold) {
      if (!(fin >> i)) throw std::runtime_error("Invalid threshold file");
    }
  }

  std::vector<MoveEval> values = LoadValues(start_pieces, offsets);
  for (int pieces = start_pieces - 1; pieces >= end_pieces; pieces--) {
    int group = GetGroupByPieces(pieces);
    values = CalculatePieceMoves<false>(pieces, values, offsets[group]);
    WriteThreshold(pieces, offsets[group], values, name, threshold, start_ratio, end_ratio, buckets);
  }
}

void MergeThresholdRanges(const std::string& name, int pieces_l, int pieces_r, bool delete_after) {
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) {
      MergeRanges<BasicIOType<uint8_t>, NodePartialThreshold>(
          group, pieces_l, pieces_r, GetBoardCountOffset(group), delete_after,
          [&name](int piece){ return ThresholdOnePath(name, piece); },
          [&name](int pieces_l, int pieces_r, int group){
            return ThresholdRangePath(name, pieces_l, pieces_r, group);
          },
          65536 * kPieces);
    }
  }).get();
}

void MergeFullThresholdRanges(const std::string& name, bool delete_after) {
  auto sections = GetSections(GetAvailableThresholdRanges(name));
  spdlog::info("Start merge ranges {}", sections);
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) {
      MergeFullThresholdRanges(name, group, sections, delete_after);
    }
  }).get();
}
