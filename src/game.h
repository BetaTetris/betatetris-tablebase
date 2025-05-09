#pragma once

#include <algorithm>

#ifdef NO_2KS
constexpr int kLevels = 3;
#else
constexpr int kLevels = 4;
#endif

#ifdef LINE_CAP
constexpr int kLineCap = LINE_CAP;
#else
constexpr int kLineCap = 430;
#if !defined(TESTING) && !defined(NO_ROTATION)
#warning "Line cap not defined. Setting to 430."
#endif
#endif

#ifdef DOUBLE_TUCK
constexpr bool kDoubleTuckAllowed = true;
#else
constexpr bool kDoubleTuckAllowed = false;
#endif

#ifdef TETRIS_ONLY
constexpr int kGroups = 10;
constexpr bool kTetrisOnly = true;
#else
constexpr int kGroups = 5;
constexpr bool kTetrisOnly = false;
#endif

enum Level {
  kLevel18,
  kLevel19,
  kLevel29,
  kLevel39
};

alignas(32) constexpr float kTransitionProb[][8] = {
  {1./32, 5./32, 6./32, 5./32, 5./32, 5./32, 5./32}, // T
  {6./32, 1./32, 5./32, 5./32, 5./32, 5./32, 5./32}, // J
  {5./32, 6./32, 1./32, 5./32, 5./32, 5./32, 5./32}, // Z
  {5./32, 5./32, 5./32, 2./32, 5./32, 5./32, 5./32}, // O
  {5./32, 5./32, 5./32, 5./32, 2./32, 5./32, 5./32}, // S
  {6./32, 5./32, 5./32, 5./32, 5./32, 1./32, 5./32}, // L
  {5./32, 5./32, 5./32, 5./32, 6./32, 5./32, 1./32}, // I
};

constexpr int kTransitionProbInt[7][7] = {
// T  J  Z  O  S  L  I (next)
  {1, 5, 6, 5, 5, 5, 5}, // T (current)
  {6, 1, 5, 5, 5, 5, 5}, // J
  {5, 6, 1, 5, 5, 5, 5}, // Z
  {5, 5, 5, 2, 5, 5, 5}, // O
  {5, 5, 5, 5, 2, 5, 5}, // S
  {6, 5, 5, 5, 5, 1, 5}, // L
  {5, 5, 5, 5, 6, 5, 1}, // I
};

constexpr int kTransitionRealisticProbInt[8][7][7] = {{
  {0, 2, 4, 2, 2, 3, 3},
  {4, 0, 2, 3, 3, 2, 2},
  {2, 3, 1, 2, 3, 2, 3},
  {2, 3, 2, 1, 3, 2, 3},
  {3, 2, 2, 3, 1, 2, 3},
  {3, 2, 3, 3, 2, 0, 3},
  {4, 2, 2, 2, 4, 2, 0}
}, {
  {0, 2, 4, 2, 2, 4, 2},
  {4, 0, 2, 3, 3, 2, 2},
  {2, 4, 0, 2, 3, 3, 2},
  {2, 3, 2, 1, 3, 2, 3},
  {3, 2, 3, 2, 1, 3, 2},
  {3, 2, 3, 3, 2, 0, 3},
  {3, 2, 2, 3, 3, 2, 1}
}, {
  {0, 3, 3, 2, 3, 3, 2},
  {3, 0, 3, 3, 2, 2, 3},
  {3, 3, 0, 3, 2, 3, 2},
  {3, 2, 3, 1, 2, 3, 2},
  {2, 2, 3, 3, 0, 3, 3},
  {2, 3, 3, 2, 2, 1, 3},
  {2, 2, 2, 4, 2, 2, 2}
}, {
  {0, 4, 2, 2, 4, 2, 2},
  {3, 0, 3, 3, 2, 2, 3},
  {3, 3, 0, 3, 3, 2, 2},
  {3, 2, 3, 1, 2, 3, 2},
  {2, 3, 2, 3, 1, 2, 3},
  {2, 3, 3, 2, 2, 1, 3},
  {2, 2, 3, 3, 2, 3, 1}
}, {
  {1, 3, 2, 3, 3, 2, 2},
  {2, 1, 3, 2, 2, 3, 3},
  {2, 3, 1, 2, 3, 2, 3},
  {2, 3, 2, 1, 3, 2, 3},
  {2, 3, 3, 2, 1, 3, 2},
  {3, 3, 2, 2, 3, 1, 2},
  {2, 2, 4, 2, 2, 4, 0}
}, {
  {2, 2, 2, 4, 2, 2, 2},
  {2, 1, 3, 2, 2, 3, 3},
  {3, 2, 1, 3, 2, 2, 3},
  {2, 3, 2, 1, 3, 2, 3},
  {3, 2, 3, 2, 1, 3, 2},
  {3, 3, 2, 2, 3, 1, 2},
  {2, 3, 3, 2, 3, 3, 0}
}, {
  {1, 2, 3, 3, 2, 2, 3},
  {3, 1, 2, 2, 3, 3, 2},
  {3, 3, 0, 3, 2, 3, 2},
  {3, 2, 3, 1, 2, 3, 2},
  {3, 3, 2, 2, 2, 2, 2},
  {4, 2, 2, 3, 3, 0, 2},
  {2, 4, 2, 2, 4, 2, 0}
}, {
  {0, 2, 4, 2, 2, 2, 4},
  {3, 1, 2, 2, 3, 3, 2},
  {2, 3, 1, 2, 2, 3, 3},
  {3, 2, 3, 1, 2, 3, 2},
  {2, 3, 2, 3, 1, 2, 3},
  {4, 2, 2, 3, 3, 0, 2},
  {3, 3, 2, 2, 4, 2, 0}
}};

constexpr int ScoreFromLevel(int post_level, int lines) {
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (post_level + 1);
}

namespace noro {

constexpr int GetLevelByLines(int lines, int start_level) {
  int level_reduce = (start_level + 1) / 16 * 10 + std::min((start_level + 1) % 16, 10) - 1;
  return std::max(lines / 10 - level_reduce, 0) + start_level;
}

constexpr int GetLevelSpeed(int level) {
  if (level < 10) return level;
  if (level <= 12) return 10;
  if (level <= 15) return 11;
  if (level <= 18) return 12;
  if (level <= 28) return 13;
  return 14;
}

constexpr int GetFramesPerRow(int level) {
  constexpr int kSpeedTable[] = {48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 4, 3, 2, 1};
  return kSpeedTable[GetLevelSpeed(level)];
}

} // namespace noro

constexpr int kLevelSpeedLines[] = {0, 130, 230,
#ifdef NO_2KS
  kLineCap
#else
  330, kLineCap
#endif // NO_2KS
};

constexpr int GetLevelByLines(int lines) {
  if (lines < 130) return 18;
  return lines / 10 + 6;
}

constexpr Level GetLevelSpeed(int level) {
  if (level == 18) return kLevel18;
  if (level < 29) return kLevel19;
#ifdef NO_2KS
  return kLevel29;
#else
  if (level < 39) return kLevel29;
  return kLevel39;
#endif // NO_2KS
}

constexpr auto GetLevelSpeedByLines(int lines) {
  return GetLevelSpeed(GetLevelByLines(lines));
}

constexpr int GameScore(int base_lines, int lines) {
  return ScoreFromLevel(GetLevelByLines(base_lines + lines), lines);
}

constexpr int Score(int base_lines, int lines) {
#ifdef TETRIS_ONLY
  return base_lines < kLineCap && base_lines + lines >= kLineCap;
#else
  return GameScore(base_lines, lines);
#endif // TETRIS_ONLY
}

#ifdef TETRIS_ONLY
constexpr int kGroupInterval = 40;
constexpr int kGroupLineInterval = 4;
constexpr int kCellsMod = 4;
#else
constexpr int kGroupInterval = 10;
constexpr int kGroupLineInterval = 2;
constexpr int kCellsMod = 2;
#endif // TETRIS_ONLY

constexpr int GetGroupByPieces(int pieces) {
  return pieces * 4 / kGroupLineInterval % kGroups;
}

constexpr int GetGroupByCells(int cells) {
  return cells / kCellsMod % kGroups;
}

constexpr int GetCellsByGroupOffset(int offset, int group) {
  return offset * kGroupInterval + group * kCellsMod;
}

constexpr int NextGroup(int group) {
#ifdef TETRIS_ONLY
  return (group + 1) % kGroups;
#else
  return (group + 2) % kGroups;
#endif
}

// some testcases

static_assert(noro::GetLevelByLines(0, 0) == 0);
static_assert(noro::GetLevelByLines(10, 0) == 1);
static_assert(noro::GetLevelByLines(123, 0) == 12);
static_assert(noro::GetLevelByLines(99, 9) == 9);
static_assert(noro::GetLevelByLines(100, 9) == 10);
static_assert(noro::GetLevelByLines(99, 10) == 10);
static_assert(noro::GetLevelByLines(100, 10) == 11);
static_assert(noro::GetLevelByLines(99, 15) == 15);
static_assert(noro::GetLevelByLines(100, 15) == 16);
static_assert(noro::GetLevelByLines(109, 16) == 16);
static_assert(noro::GetLevelByLines(110, 16) == 17);

static_assert(GetLevelByLines(kLevelSpeedLines[0]) == 18);
static_assert(GetLevelByLines(kLevelSpeedLines[1]) == 19);
static_assert(GetLevelByLines(kLevelSpeedLines[2]) == 29);
#ifndef NO_2KS
static_assert(GetLevelByLines(kLevelSpeedLines[3]) == 39);
#endif // NO_2KS

static_assert(GetGroupByPieces(0) == 0);
#ifdef TETRIS_ONLY
static_assert(GetGroupByPieces(1) == 1);
static_assert(GetGroupByPieces(9) == 9);
static_assert(GetGroupByPieces(16) == 6);
#else
static_assert(GetGroupByPieces(1) == 2);
static_assert(GetGroupByPieces(4) == 3);
static_assert(GetGroupByPieces(9) == 3);
#endif // TETRIS_ONLY
