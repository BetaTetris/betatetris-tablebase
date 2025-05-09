#pragma once

#include "move_search_no_tmpl.h"

namespace move_search {

template <
    int tap1, int tap2, int tap3, int tap4, int tap5,
    int tap6, int tap7, int tap8, int tap9, int tap10>
class TapTable {
  int t[10];
 public:
  constexpr TapTable() : t{tap1, tap2, tap3, tap4, tap5, tap6, tap7, tap8, tap9, tap10} {
    static_assert(
        tap1 >= 0 && tap2 >= 2 && tap3 >= 2 && tap4 >= 2 && tap5 >= 2 &&
        tap6 >= 2 && tap7 >= 2 && tap8 >= 2 && tap9 >= 2 && tap10 >= 2,
        "each tap must be at least 2 frames apart");
    for (int i = 1; i < 10; i++) t[i] += t[i - 1];
  }
  constexpr int operator[](int x) const { return t[x]; }
  constexpr int* data() { return t; }
  constexpr const int* data() const { return t; }
  constexpr std::array<int, 10> toarray() const {
    std::array<int, 10> ret;
    for (int i = 0; i < 10; i++) ret[i] = t[i];
    return ret;
  }
};

template <class T> struct IsTapTable : std::false_type {};
template <int... args> struct IsTapTable<TapTable<args...>> : std::true_type {};

template <Level level, int R, int adj_frame, class Taps>
class Search {
  PrecomputedTable table_;

 public:
  Search() : table_(level, R, adj_frame, Taps().data()) {
    static_assert(IsTapTable<Taps>::value);
  }

  PossibleMoves MoveSearch(const std::array<Board, R>& board) const {
    return ::MoveSearch<R, Taps>(level, adj_frame, table_, board);
  }
};

} // namespace move_search

template <Level level, int R, int adj_frame, class Taps>
NOINLINE PossibleMoves MoveSearch(const std::array<Board, R>& board) {
  static move_search::Search<level, R, adj_frame, Taps> search;
  return search.MoveSearch(board);
}

template <Level level, int adj_frame, class Taps>
PossibleMoves MoveSearch(const Board& b, int piece) {
#define ONE_CASE(x) \
    case x: return MoveSearch<level, Board::NumRotations(x), adj_frame, Taps>(b.PieceMap<x>());
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}

template <int adj_frame, class Taps>
PossibleMoves MoveSearch(const Board& b, Level level, int piece) {
#define LEVEL_CASE_TMPL_ARGS ,adj_frame,Taps
  DO_LEVEL_CASE(MoveSearch, b, piece);
#undef LEVEL_CASE_TMPL_ARGS
}

using Tap30Hz = move_search::TapTable<0, 2, 2, 2, 2, 2, 2, 2, 2, 2>;
using Tap20Hz = move_search::TapTable<0, 3, 3, 3, 3, 3, 3, 3, 3, 3>;
using Tap15Hz = move_search::TapTable<0, 4, 4, 4, 4, 4, 4, 4, 4, 4>;
using Tap12Hz = move_search::TapTable<0, 5, 5, 5, 5, 5, 5, 5, 5, 5>;
using TapSlow5 = move_search::TapTable<0, 2, 2, 2, 12, 2, 2, 2, 12, 2>;
