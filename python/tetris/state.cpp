#include "state.h"

#include "python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>
#include "../../src/tetris.h"

std::vector<long> kStateShapes[5] = {
  {std::tuple_size<decltype(State::board)>::value,
   std::tuple_size<decltype(State::board)::value_type>::value,
   std::tuple_size<decltype(State::board)::value_type::value_type>::value},
  {std::tuple_size<decltype(State::meta)>::value},
  {std::tuple_size<decltype(State::moves)>::value,
   std::tuple_size<decltype(State::moves)::value_type>::value,
   std::tuple_size<decltype(State::moves)::value_type::value_type>::value},
  {std::tuple_size<decltype(State::move_meta)>::value},
  {std::tuple_size<decltype(State::meta_int)>::value}
};

double GetNoroLineRewardExp(int lines, int start_level, bool do_tuck, bool nnb) {
  constexpr int kOffset[2][2][15] = {
    { // 0,1,2,3,4,5,6, 7,8, 9, 10-12,13-15, 16-18,19, 29
      {14,14,14,14,14,14,14, 14,14, 13, 13,13, 12,12, 10}, // notuck
      {12,12,12,12,12,12,12, 12,12, 12, 10,10,  9, 9, 6}, // notuck, nnb
    }, {
      {21,21,21,21,21,21,21, 19,19, 19, 19,19, 12,12, 11}, // tuck
      {17,17,17,17,17,17,17, 17,17, 16, 15,15, 12,12, 9}, // tuck, nnb
    },
  };
  constexpr float kExpMultiplier[2][2][15] = {
    { // 0,1,2,3,4,5,6,7,8,9,10-12,13-15,16-18,19,29
      {0.33,0.33,0.33,0.33,0.33,0.33,0.33, 0.33,0.33, 0.35, 0.38,0.38, 0.38,0.38, 0.4}, // notuck
      {0.50,0.50,0.50,0.50,0.50,0.50,0.50, 0.50,0.50, 0.50, 0.50,0.50, 0.50,0.50, 0.50}, // notuck, nnb
    }, {
      {0.16,0.16,0.16,0.16,0.16,0.16,0.16, 0.16,0.16, 0.18, 0.19,0.19, 0.24,0.24, 0.33}, // tuck
      {0.20,0.20,0.20,0.20,0.20,0.20,0.20, 0.20,0.20, 0.21, 0.22,0.22, 0.40,0.40, 0.45}, // tuck, nnb
    },
  };
  constexpr float kMinExp[2][2][15] = {
    { // 0,1,2,3,4,5,6,7,8,9,10-12,13-15,16-18,19,29
      {-3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0, -3.0,-3.0, -3.0, -3.0,-3.0, -3.0,-3.0, -2.8}, // notuck
      {-2.8,-2.8,-2.8,-2.8,-2.8,-2.8,-2.8, -2.8,-2.8, -2.8, -2.8,-2.8, -2.8,-2.8, -2.8}, // notuck, nnb
    }, {
      {-3.6,-3.6,-3.6,-3.6,-3.6,-3.6,-3.6, -3.6,-3.6, -3.6, -3.5,-3.5, -3.2,-3.2, -3.0}, // tuck
      {-3.5,-3.5,-3.5,-3.5,-3.5,-3.5,-3.5, -3.5,-3.5, -3.5, -3.2,-3.2, -2.8,-2.8, -2.2}, // tuck, nnb
    },
  };

  int speed = noro::GetLevelSpeed(start_level);
  float min_exp = kMinExp[do_tuck][nnb][speed];
  int offset = kOffset[do_tuck][nnb][speed];
  float multiplier = kExpMultiplier[do_tuck][nnb][speed];
  return std::min(6.0f, std::max(0, lines - offset) * multiplier + min_exp);
}

void GetState(const TetrisNoro& tetris, State& state, bool nnb, bool is_mirror, int line_reduce) {
  // board: shape (2, 20, 10) [board, one]
  // meta: shape (21,) [group(5), now_piece(7), next_piece(7), nnb, do_tuck, start_speed(10)]
  // meta_int: shape (2,) [entry, now_piece]
  // moves: shape (3, 20, 10) [board, one, moves]
  // move_meta: shape (31,) [speed(10), to_transition(16), level*0.1, lines*0.01, start_lines*0.01, pieces*0.004, ln(multiplier)]
  {
    auto byte_board = tetris.GetBoard().ToByteBoard();
    for (int i = 0; i < 20; i++) {
      if (is_mirror) {
        for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][9-j];
        for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][9-j];
      } else {
        for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][j];
        for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][j];
      }
      for (int j = 0; j < 10; j++) state.board[1][i][j] = 1;
      for (int j = 0; j < 10; j++) state.moves[1][i][j] = 1;
    }
    auto move_map = tetris.GetPossibleMoveMap().ToByteBoard();
    for (int i = 0; i < 20; i++) {
      if (is_mirror) {
        for (int j = 0; j < 10; j++) {
          int ncol = kMirrorCols[tetris.NowPiece()] - j;
          state.moves[2][i][j] = ncol >= 10 ? 0 : move_map[i][ncol];
        }
      } else {
        for (int j = 0; j < 10; j++) state.moves[2][i][j] = move_map[i][j];
      }
    }
  }

  int start_level = tetris.GetStartLevel();
  int start_speed = tetris.InputsPerRow(start_level);
  memset(state.meta.data(), 0, sizeof(state.meta));
  state.meta[0 + tetris.GetBoard().Count() / 2 % 5] = 1;
  state.meta[5 + (is_mirror ? kMirrorPiece[tetris.NowPiece()] : tetris.NowPiece())] = 1;
  if (nnb) {
    state.meta[19] = 1;
  } else {
    state.meta[12 + (is_mirror ? kMirrorPiece[tetris.NextPiece()] : tetris.NextPiece())] = 1;
  }
  state.meta[20] = tetris.DoTuck();
  state.meta[21] = is_mirror;
  state.meta[22 + start_speed] = 1;

  int lines = tetris.GetLines();
  int state_lines = lines - line_reduce;
  int state_level = noro::GetLevelByLines(state_lines, start_level);
  state.meta_int[0] = state_lines / 2;
  state.meta_int[1] = tetris.NowPiece();

  memset(state.move_meta.data(), 0, sizeof(state.move_meta));
  int to_transition = 0;
  state.move_meta[tetris.InputsPerRow()] = 1;
  to_transition = tetris.LinesToNextSpeed();
  if (to_transition == -1) to_transition = 1000;
  if (to_transition <= 10) { // 10..19
    state.move_meta[10 + (to_transition - 1)] = 1;
  } else if (to_transition <= 22) { // 20..23
    state.move_meta[20 + (to_transition - 11) / 3] = 1;
  } else {
    state.move_meta[24] = 1;
  }
  state.move_meta[25] = to_transition * 0.01;
  state.move_meta[26] = state_level * 0.1;
  state.move_meta[27] = state_lines * 0.01;
  state.move_meta[28] = start_level * 0.1;
  state.move_meta[29] = (tetris.GetPieces() + line_reduce * 10 / 4) * 0.004;
  state.move_meta[30] = std::max(-0.5, GetNoroLineRewardExp(state_lines + 5, start_level, tetris.DoTuck(), nnb));
}

void GetState(const Tetris& tetris, State& state, int line_reduce, int step_reward_level) {
  // board: shape (6, 20, 10) [board, one, initial_move(4)]
  // meta: shape (32,) [now_piece(7), next_piece(7), is_adj(1), hz(7), adj_delay(6), aggro(3), pad(1)]
  // meta_int: shape (2,) [entry, now_piece]
  // moves: shape (14, 20, 10) [board, one, moves(4), adj_moves(4), initial_move(4), nonreduce_moves(4)]
  // move_meta: shape (28,) [speed(4), to_transition(21), (level-18)*0.1, lines*0.01, pieces*0.004]
  {
    auto byte_board = tetris.GetBoard().ToByteBoard();
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][j];
      for (int j = 0; j < 10; j++) state.board[1][i][j] = 1;
      for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][j];
      for (int j = 0; j < 10; j++) state.moves[1][i][j] = 1;
    }
    auto& move_map = tetris.GetPossibleMoveMap();
    for (int r = 0; r < 4; r++) {
      for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) {
          state.moves[2 + r][i][j] = move_map[r][i][j] ? 1 : 0;
          state.moves[6 + r][i][j] = move_map[r][i][j] >= 2;
          state.moves[14 + r][i][j] = move_map[r][i][j] && move_map[r][i][j] != 2 ? 1 : 0;
        }
      }
      memset(state.board.data() + (2 + r), 0, sizeof(state.board[0]));
      memset(state.moves.data() + (10 + r), 0, sizeof(state.moves[0]));
    }
  }
  if (tetris.IsAdj()) {
    auto pos = tetris.InitialMove();
    state.board[2 + pos.r][pos.x][pos.y] = 1;
    state.moves[10 + pos.r][pos.x][pos.y] = 1;
  }

  memset(state.meta.data(), 0, sizeof(state.meta));
  state.meta[0 + tetris.NowPiece()] = 1;
  if (tetris.IsAdj()) {
    state.meta[7 + tetris.NextPiece()] = 1; // also change GetAdjStates if changed
    state.meta[14] = 1;
  }

  int lines = tetris.GetLines();
  int state_lines = lines - line_reduce;
  int state_level = GetLevelByLines(state_lines);
  int state_speed = static_cast<int>(GetLevelSpeed(state_level));

  int tap_4 = tetris.GetTapSequence()[3];
  int tap_5 = tetris.GetTapSequence()[4];
  int adj_delay = tetris.GetAdjDelay();
  if (state_speed == 2 && adj_delay >= 20) adj_delay = 61;
  if (state_speed == 3 && adj_delay >= 10) adj_delay = 61;
  if (tap_5 <= 8) { // 30hz
    state.meta[15] = 1;
  } else if (tap_5 <= 11) { // 24hz
    state.meta[16] = 1;
  } else if (tap_5 <= 13) { // 20hz
    state.meta[17] = 1;
  } else if (tap_5 <= 16) { // 15hz
    state.meta[18] = 1;
  } else if (tap_4 <= 9) { // slow 5-tap
    state.meta[19] = 1;
  } else if (tap_5 <= 21) { // 12hz
    state.meta[20] = 1;
  } else { // 10hz
    state.meta[21] = 1;
  }
  if (adj_delay <= 4) {
    state.meta[22] = 1;
  } else if (adj_delay <= 19) {
    state.meta[23] = 1;
  } else if (adj_delay <= 22) {
    state.meta[24] = 1;
  } else if (adj_delay <= 25) {
    state.meta[25] = 1;
  } else if (adj_delay <= 32) {
    state.meta[26] = 1;
  } else {
    state.meta[27] = 1;
  }
  state.meta[28 + step_reward_level] = 1;

  state.meta_int[0] = state_lines / 2;
  state.meta_int[1] = tetris.NowPiece();

  memset(state.move_meta.data(), 0, sizeof(state.move_meta));
  int to_transition = 0;
  state.move_meta[state_speed] = 1;
  to_transition = std::max(1, kLevelSpeedLines[state_speed + 1] - state_lines);
  if (to_transition <= 10) { // 4..13
    state.move_meta[4 + (to_transition - 1)] = 1;
  } else if (to_transition <= 22) { // 14..17
    state.move_meta[14 + (to_transition - 11) / 3] = 1;
  } else if (to_transition <= 40) { // 18..20
    state.move_meta[18 + (to_transition - 22) / 6] = 1;
  } else if (to_transition <= 60) { // 21,22
    state.move_meta[21 + (to_transition - 40) / 10] = 1;
  } else {
    state.move_meta[23] = 1;
  }
  state.move_meta[24] = to_transition * 0.01;
  state.move_meta[25] = (state_level - 18) * 0.1;
  state.move_meta[26] = state_lines * 0.01;
  state.move_meta[27] = (tetris.GetPieces() + line_reduce * 10 / 4) * 0.004;
}

MultiState GetAdjStates(const Tetris& tetris) {
  if (!tetris.IsAdj()) throw std::logic_error("not in adjustment");
  MultiState ret;
  ret.reserve(kPieces);
  State st{};
  GetState(tetris, st);
  for (size_t i = 0; i < kPieces; i++) st.meta[7 + i] = 0;
  for (size_t i = 0; i < kPieces; i++) {
    st.meta[7 + i] = 1;
    ret.push_back(st);
    st.meta[7 + i] = 0;
  }
  return ret;
}

PyObject* State::ToPython() const {
  PyObject *objs[5];
#define X(name, id, dims, typ) { \
  npy_intp dim_arr[dims]; \
  for (int i = 0; i < dims; i++) dim_arr[i] = kStateShapes[id][i]; \
  objs[id] = PyArray_SimpleNew(dims, dim_arr, typ); \
  memcpy(PyArray_DATA((PyArrayObject*)objs[id]), name.data(), sizeof(State::name)); \
}
  STATE_MEMBERS_
#undef X
  PyObject* ret = PyTuple_Pack(5, objs[0], objs[1], objs[2], objs[3], objs[4]);
  for (int i = 0; i < 5; i++) Py_DECREF(objs[i]);
  return ret;
}

PyObject* MultiState::ToPython() const {
  long N = size();
  PyObject *objs[5];
#define X(name, id, dims, typ) { \
  npy_intp dim_arr[dims + 1] = {N}; \
  for (int i = 0; i < dims; i++) dim_arr[i + 1] = kStateShapes[id][i]; \
  objs[id] = PyArray_SimpleNew(dims + 1, dim_arr, typ); \
  memcpy(PyArray_DATA((PyArrayObject*)objs[id]), name.data(), N * sizeof(State::name)); \
}
  STATE_MEMBERS_
#undef X
  PyObject* ret = PyTuple_Pack(5, objs[0], objs[1], objs[2], objs[3], objs[4]);
  for (int i = 0; i < 5; i++) Py_DECREF(objs[i]);
  return ret;
}
