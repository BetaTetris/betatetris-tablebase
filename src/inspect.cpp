#include "inspect.h"

#include <ranges>
#include <iostream>
#include <string_view>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#pragma GCC diagnostic ignored "-Wtautological-compare"
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop
#include "thread_pool.hpp"

#include "io.h"
#include "edge.h"
#include "play.h"
#include "board.h"
#include "files.h"
#include "config.h"
#include "evaluate.h"
#include "board_set.h"

template<> struct fmt::formatter<Position> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) const { return ctx.begin(); }
  template <typename FormatContext>
  auto format(const Position& pos, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "({},{},{})", pos.r, pos.x, pos.y);
  }
};

namespace {

void PrintGrid(const std::vector<std::string>& grid, int col_width = 12, int cols = 4) {
  for (size_t i = 0; i < grid.size(); i += cols) {
    size_t N = std::min((size_t)cols, grid.size() - i);
    std::vector<std::vector<std::string_view>> vec;
    size_t rows = 0;
    for (size_t j = i; j < i + N; j++) {
      vec.emplace_back();
      for (const auto k : std::views::split(std::string_view(grid[j]), std::string_view{"\n"})) {
        vec.back().emplace_back(k.begin(), k.end());
      }
      rows = std::max(rows, vec.back().size());
    }
    for (size_t r = 0; r < rows; r++) {
      size_t len = 0;
      for (size_t j = 0; j < N; j++) {
        size_t col_end = (j + 1) * col_width;
        if (vec[j].size() >= rows - r) {
          auto& item = vec[j][r + vec[j].size() - rows];
          len += item.size();
          std::cout << item;
        }
        if (len < col_end && j != N - 1) {
          std::cout << std::string(col_end - len, ' ');
          len = col_end;
        }
      }
      std::cout << '\n';
    }
  }
}

} // namespace

void InspectBoard(int group, const std::vector<long>& board_idx) {
  ClassReader<CompactBoard> reader(BoardPath(group));
  for (auto id : board_idx) {
    reader.Seek(id, 4096);
    try {
      auto board = Board(reader.ReadOne(4096));
      std::cout << fmt::format("Group {}, board {}:\n{{{:#x}, {:#x}, {:#x}, {:#x}}}\n",
                              group, id, board.b1, board.b2, board.b3, board.b4)
                << board.ToString();
    } catch (ReadError&) {
      std::cout << fmt::format("Group {}, board {} not found\n", group, id);
    }
  }
}

void InspectBoardStats(int group) {
  auto offsets = GetBoardCountOffset(group);
  for (size_t i = 0; i < offsets.size() - 1; i++) {
    std::cout << fmt::format("Count {}: {} boards\n", GetCellsByGroupOffset(i, group), offsets[i+1] - offsets[i]);
  }
}

void InspectEdge(int group, const std::vector<long>& board_idx, Level level, int piece) {
  int level_int = static_cast<int>(level);
  ClassReader<CompactBoard> reader_cur(BoardPath(group));
  ClassReader<CompactBoard> reader_nxt(BoardPath(NextGroup(group)));
  CompressedClassReader<EvaluateNodeEdges> reader_eval_ed(EvaluateEdgePath(group, level_int));
  CompressedClassReader<PositionNodeEdges> reader_pos_ed(PositionEdgePath(group, level_int));
  for (auto id : board_idx) {
    reader_cur.Seek(id, 4096);
    reader_eval_ed.Seek(id * kPieces + piece, 0, 0);
    reader_pos_ed.Seek(id * kPieces + piece, 0, 0);
    auto board = Board(reader_cur.ReadOne(4096));
    auto eval_ed = reader_eval_ed.ReadOne((size_t)0, 0);
    auto pos_ed = reader_pos_ed.ReadOne((size_t)0, 0);

    std::cout << fmt::format("Group {}, board {}:\n", group, id) << board.ToString();
    std::vector<std::string> next_boards;
    for (size_t i = 0; i < eval_ed.next_ids.size(); i++) {
      auto [nxt, lines] = eval_ed.next_ids[i];
      reader_nxt.Seek(nxt, 0);
      auto nboard = Board(reader_nxt.ReadOne(0));
      next_boards.push_back(nboard.ToString(false, true, false));
      const auto& pos = pos_ed.nexts[i];
      next_boards.back() += fmt::format("{},{} {},{},{}\n{}", i, lines, pos.r, pos.x, pos.y, nxt);
    }
    std::cout << "Nexts:\n";
    PrintGrid(next_boards);
    std::vector<Position> non_adj_pos;
    for (auto& i : eval_ed.non_adj) non_adj_pos.push_back(pos_ed.nexts[i]);
    std::cout << fmt::format("Non-adj: {} {}\n", eval_ed.non_adj, non_adj_pos);
    std::cout << "Adjs:\n";
    if (eval_ed.use_subset) {
      for (auto& i : eval_ed.subset_idx_prev) std::cout << fmt::format("{} {}\n", i.first, i.second);
      std::cout << fmt::format("({} before expanding)\n", eval_ed.subset_idx_prev.size());
      eval_ed.CalculateAdj();
    }
    for (size_t i = 0; i < eval_ed.adj.size(); i++) {
      std::cout << fmt::format("{}: {}\n", pos_ed.adj[i], eval_ed.adj[i]);
    }
  }
}

void InspectEdgeStats(int group, Level level) {
  int level_int = static_cast<int>(level);
  CompressedClassReader<EvaluateNodeEdgesFastTmpl<4096>> reader(EvaluateEdgePath(group, level_int));
  std::vector<size_t> num_next(256 * 7), num_piece(8);
  size_t max_buf_size = 0;

  std::vector<EvaluateNodeEdgesFastTmpl<4096>> batch(256 * 7);
  while (true) {
    size_t batch_size = reader.ReadBatch(batch.data(), 256 * 7);
    for (size_t i = 0; i < batch_size; i += 7) {
      size_t next = 0, piece = 0;
      for (size_t j = 0; j < 7; j++) {
        next += batch[i+j].next_ids_size;
        piece += batch[i+j].next_ids_size != 0;
        max_buf_size = std::max(max_buf_size, batch[i+j].buf_size);
      }
      num_next[next]++;
      num_piece[piece]++;
    }
    if (batch_size < 256 * 7) break;
  }

  std::cout << "Number of next boards:\n";
  for (size_t i = 0; i < num_next.size(); i++) {
    if (!num_next[i]) continue;
    std::cout << fmt::format("{}: {} boards\n", i, num_next[i]);
  }
  std::cout << "Number of next pieces:\n";
  for (size_t i = 0; i < num_piece.size(); i++) {
    if (!num_piece[i]) continue;
    std::cout << fmt::format("{}: {} boards\n", i, num_piece[i]);
  }
  std::cout << "Max buf size: " << max_buf_size << '\n';
}

void InspectValue(int pieces, const std::vector<long>& board_idx) {
  CompressedClassReader<NodeEval> reader(ValuePath(pieces));
  for (auto id : board_idx) {
    reader.Seek(id, 4096);
    NodeEval val = reader.ReadOne();
    std::vector<float> ev(7), var(7);
    val.GetEv(ev.data());
    val.GetVar(var.data());
    std::cout << fmt::format("{} {} {}\n", id, ev, var);
  }
}

void InspectBoard(const std::string& str) {
  Play play;
  Board b(str);
  size_t res = play.GetID(b.ToBytes());
  if (res == std::string::npos) {
    std::cout << "Board not found\n";
  } else {
    std::cout << GetGroupByCells(b.Count()) << ' ' << res << std::endl;
  }
}

void InspectMove(const std::string& str, int now_piece, int lines, bool notation) {
  Play play;
  Board b(str);
  auto res = play.GetStrat(b.ToBytes(), now_piece, lines);
  for (size_t i = 0; i < kPieces; i++) {
    if (notation) {
      std::cout << b.PlacementNotation(now_piece, res[i].r, res[i].x, res[i].y) << " \n"[i == kPieces - 1];
    } else {
      std::cout << res[i].r << ' ' << res[i].x << ' ' << res[i].y << '\n';
    }
  }
  std::cout << std::flush;
}

namespace {

void PrintMoveTree(
    const Board& b, int cur_lines, int depth, int now_piece, Play& play,
    std::string seq_str = "", std::string res_str = "") {
  seq_str += kPieceNames[now_piece];
  auto strat = play.GetStrat(b.ToBytes(), now_piece, cur_lines);
  if (strat[0] == Position::Invalid) {
    std::cout << seq_str << res_str + "\tOVER\n";
    return;
  }
  if (depth == 0) {
    std::cout << seq_str << res_str << '\t';
    std::unordered_map<Position, std::vector<int>> mp;
    for (size_t i = 0; i < kPieces; i++) mp[strat[i]].push_back(i);
    if (mp.size() == 1) {
      std::cout << b.PlacementNotation(now_piece, strat[0].r, strat[0].x, strat[0].y) << '\n';
    } else {
      std::vector<std::pair<std::vector<int>, Position>> vec;
      for (auto& i : mp) vec.emplace_back(i.second, i.first);
      std::sort(vec.begin(), vec.end());
      std::string fin;
      for (auto& [pieces, pos] : vec) {
        fin += b.PlacementNotation(now_piece, pos.r, pos.x, pos.y) + ':';
        for (int nxt : pieces) fin += kPieceNames[nxt];
        fin += '|';
      }
      fin.pop_back();
      std::cout << fin << '\n';
    }
    return;
  }
  for (size_t nxt = 0; nxt < kPieces; nxt++) {
    auto& pos = strat[nxt];
    auto nxt_board = b.Place(now_piece, pos.r, pos.x, pos.y).ClearLines();
    PrintMoveTree(nxt_board.second, cur_lines + nxt_board.first, depth - 1, nxt, play,
                  seq_str, res_str + '\t' + b.PlacementNotation(now_piece, pos.r, pos.x, pos.y));
  }
}

} // namespace

void InspectMoveTree(const std::string& str, int lines, size_t depth) {
  Board b(str);
  Play play;
  for (size_t i = 0; i < kPieces; i++) PrintMoveTree(b, lines, depth < 1 ? 0 : depth - 1, i, play);
}
