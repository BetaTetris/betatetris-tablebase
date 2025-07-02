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

// X macro for state members
#define STATE_MEMBERS_ \
  X(board, 0, 3, NPY_FLOAT32) \
  X(meta, 1, 1, NPY_FLOAT32) \
  X(moves, 2, 3, NPY_FLOAT32) \
  X(move_meta, 3, 1, NPY_FLOAT32) \
  X(meta_int, 4, 1, NPY_INT32)

struct MultiState {
#define X(name, id, dims, typ) std::vector<decltype(State::name)> name;
  STATE_MEMBERS_
#undef X
  void reserve(size_t sz) {
#define X(name, id, dims, typ) name.reserve(sz);
    STATE_MEMBERS_
#undef X
  }
  void push_back(const State& st) {
#define X(name, id, dims, typ) name.push_back(st.name);
    STATE_MEMBERS_
#undef X
  }
  void merge(const MultiState& st) {
    size_t offset = size(), sz = size() + st.size();
#define X(name, id, dims, typ) \
    name.resize(sz); \
    memcpy(name.data() + offset, st.name.data(), st.size() * sizeof(State::name));
    STATE_MEMBERS_
#undef X
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
