#pragma once

#include "move.h"
#include "tetris.h"
#include "io_hash.h"
#include "board_set.h"
#include "io_helpers.h"

class Play {
  std::vector<HashMapReader<CompactBoard, BasicIOType<uint32_t>>> board_hash;
  std::vector<CompressedClassReader<NodeMovePositionRange>> move_readers;
 public:
  size_t GetID(const CompactBoard& board) {
    int group = GetGroupByCells(board.Count());
    auto idx = board_hash[group][board];
    if (!idx) return std::string::npos;
    return idx.value();
  }

  std::array<Position, 7> GetStrat(const CompactBoard& board, int now_piece, int lines, size_t* move_idx_ptr = nullptr) {
    int group = GetGroupByCells(board.Count());
    auto idx = board_hash[group][board];
    if (!idx) return {Position::Invalid}; // actually {}, since Invalid is (0,0,0)
    size_t move_idx = (size_t)idx.value() * kPieces + now_piece;
    if (move_idx_ptr) *move_idx_ptr = move_idx;
    move_readers[group].Seek(move_idx, 0, 0);
    // use 1 to avoid being treated as NULL
    NodeMovePositionRange pos_ranges = move_readers[group].ReadOne(1, 0);
    for (auto& range : pos_ranges.ranges) {
      uint8_t loc = lines / kGroupLineInterval;
      if (range.start <= loc && loc < range.end) {
        return range.pos;
      }
    }
    return {Position::Invalid};
  }

  std::array<Position, kPieces> GetStrat(const Tetris& game, size_t* move_idx_ptr = nullptr) {
    return GetStrat(game.GetBoard().ToBytes(), game.NowPiece(), game.GetLines(), move_idx_ptr);
  }

  Play() {
    for (int i = 0; i < kGroups; i++) {
      board_hash.emplace_back(BoardMapPath(i));
      move_readers.emplace_back(MovePath(i));
    }
  }
};
