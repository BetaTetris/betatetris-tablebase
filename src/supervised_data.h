#include "hash.h"
#include "files.h"
#include "tetris.h"

struct SupervisedData {
  static constexpr bool kIsConstSize = true;
  static constexpr size_t NumBytes() {
    return kBoardBytes + 7 + kPieces * Position::NumBytes();
  }

  Board board;
  uint32_t tag, lines;
  uint8_t cur_piece;
  std::array<Position, kPieces> pos;

  SupervisedData() : lines(0), cur_piece(0), pos{} {}
  SupervisedData(const uint8_t buf[], size_t) : board(buf) {
    buf += kBoardBytes;
    tag = BytesToInt<uint32_t>(buf);
    lines = BytesToInt<uint16_t>(buf + 4);
    cur_piece = buf[6];
    buf += 7;
    for (size_t i = 0; i < kPieces; i++) {
      pos[i] = Position(buf + Position::NumBytes() * i, Position::NumBytes());
    }
  }
  SupervisedData(const Board& board, uint32_t tag, uint32_t lines, uint8_t cur_piece, const std::array<Position, kPieces>& pos)
      : board(board), tag(tag), lines(lines), cur_piece(cur_piece), pos(pos) {}
  void GetBytes(uint8_t ret[]) const {
    board.ToBytes(ret);
    ret += kBoardBytes;
    IntToBytes<uint32_t>(tag, ret);
    IntToBytes<uint16_t>(lines, ret + 4);
    ret[6] = cur_piece;
    ret += 7;
    for (size_t i = 0; i < kPieces; i++) {
      pos[i].GetBytes(ret + Position::NumBytes() * i);
    }
  }

  Tetris GetGame(const int tap_sequence[10], int adj_delay) const {
    Tetris ret;
    ret.Reset(board, lines, cur_piece, 0, tap_sequence, adj_delay);
    return ret;
  }

  bool operator==(const SupervisedData& x) const {
    return board == x.board && tag == x.tag && lines == x.lines && cur_piece == x.cur_piece;
  }
  bool operator!=(const SupervisedData& x) const { return !(*this == x); }
};

template<>
struct std::hash<SupervisedData> {
  size_t operator()(const SupervisedData& x) const {
    size_t seed = std::hash<Board>()(x.board);
    return Hash(Hash(x.tag, x.lines), Hash(x.cur_piece, seed));
  }
};
