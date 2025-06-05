#pragma once

#include <array>
#include <vector>
#include "../../src/tetris.h"

typedef struct _object PyObject;

#ifdef NO_ROTATION
struct State {
  std::array<std::array<std::array<float, 10>, 20>, 2> board;
  std::array<float, 32> meta;
  std::array<std::array<std::array<float, 10>, 20>, 3> moves;
  std::array<float, 31> move_meta;
  std::array<int, 2> meta_int;
  PyObject* ToPython() const;
};
#else // !NO_ROTATION
struct State {
  std::array<std::array<std::array<float, 10>, 20>, 6> board;
  std::array<float, 32> meta;
  std::array<std::array<std::array<float, 10>, 20>, 18> moves;
  std::array<float, 28> move_meta;
  std::array<int, 2> meta_int;
  PyObject* ToPython() const;
};
#endif // !NO_ROTATION

extern std::vector<long> kStateShapes[5];

struct MultiState {
  std::vector<decltype(State::board)> board;
  std::vector<decltype(State::meta)> meta;
  std::vector<decltype(State::moves)> moves;
  std::vector<decltype(State::move_meta)> move_meta;
  std::vector<decltype(State::meta_int)> meta_int;
  void reserve(size_t sz) {
    board.reserve(sz);
    meta.reserve(sz);
    moves.reserve(sz);
    move_meta.reserve(sz);
    meta_int.reserve(sz);
  }
  void push_back(const State& st) {
    board.push_back(st.board);
    meta.push_back(st.meta);
    moves.push_back(st.moves);
    move_meta.push_back(st.move_meta);
    meta_int.push_back(st.meta_int);
  }
  size_t size() const { return board.size(); }
  PyObject* ToPython() const;
};

static constexpr int kMirrorCols[] = {9, 9, 9, 10, 9, 9, 10};
static constexpr int kMirrorPiece[] = {0, 5, 4, 3, 2, 1, 6};
double GetNoroLineRewardExp(int lines, int start_level, bool do_tuck, bool nnb);

void GetState(const TetrisNoro& tetris, State& state, bool nnb, bool is_mirror, int line_reduce = 0);
void GetState(const Tetris& tetris, State& state, int line_reduce = 0, int step_reward_level = 0);
MultiState GetAdjStates(const Tetris& tetris);
