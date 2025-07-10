#pragma once

#include <algorithm>
#include "python.h"
#include "../../src/board.h"

struct PythonBoard {
  PyObject_HEAD

  BoardTmpl<8> board;

  template <class... Args> PythonBoard(Args&&... args) : board(std::forward<Args>(args)...) {}
  operator Board() const { return board; }

  // a very simple heuristic to end early
  bool IsClean() const {
    int overhangs = board.NumOverhang();
#ifdef TETRIS_ONLY
    if (overhangs >= 1) return false;
    auto heights = board.ColumnHeights();
#else
    if (overhangs >= 4) return false;
    auto heights = board.ColumnHeights();
    int max_height = *std::max_element(heights.begin(), heights.end());
    if (overhangs >= 2) return max_height <= 3;
    if (overhangs >= 1) return max_height <= 5;
    if (max_height >= 12) return false;
#endif
    int differences = 0, well = 0;
    for (int i = 0; i < 9; i++) differences += abs(heights[i] - heights[i+1]);
    if (heights[0] == 0) well = std::max(well, heights[1]);
    if (heights[9] == 0) well = std::max(well, heights[8]);
    for (int i = 1; i < 9; i++) {
      if (heights[i] == 0) well = std::max(well, heights[i-1] + heights[i+1]);
    }
    return differences - well <= 10;
  }

  bool IsCleanForPerfect() const {
    // very simple flood fill for checking all empty cells are connected (no holes)
    uint32_t cols[12], map[12] = {};
    bool updated[12] = {};
    bool flag = false;
    for (int i = 0; i < 10; i++) {
      cols[i+1] = board.Column(i);
      if (cols[i+1] == 0xfffff) flag = true;
    }
    if (!flag) return false;
    updated[6] = true;
    map[6] = 1 & cols[6];
    while (true) {
      bool flag = false;
      for (int i : {5, 6, 7, 8, 9, 4, 3, 2, 1, 0}) {
        if (!updated[i+1]) continue;
        flag = true;
        updated[i+1] = false;
        uint32_t old = map[i+1];
        map[i+1] |= ((old + cols[i+1]) ^ old ^ cols[i+1]) >> 1;
        old = map[i];
        map[i] = (map[i] | map[i+1]) & cols[i];
        updated[i] = map[i] != old;
        old = map[i+2];
        map[i+2] = (map[i+2] | map[i+1]) & cols[i+2];
        updated[i+2] = map[i+2] != old;
      }
      if (!flag) break;
    }
    for (int i = 0; i < 10; i++) {
      if (map[i+1] != cols[i+1]) return false;
    }
    return true;
  }
};

extern PyTypeObject py_board_class;
