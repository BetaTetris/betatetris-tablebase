#pragma once

#include <tuple>
#include "hash.h"

struct Position {
  static constexpr bool kIsConstSize = true;

  uint8_t r, x, y;
  constexpr Position L() const { return Position(r, x, y - 1); }
  constexpr Position R() const { return Position(r, x, y + 1); }
  constexpr Position D() const { return Position(r, x + 1, y); }
  template <uint8_t R> constexpr Position A() const { return Position((r + 1) % R, x, y); }
  template <uint8_t R> constexpr Position B() const { return Position((r + R - 1) % R, x, y); }

  static constexpr Position FromIndex(uint32_t idx) {
    return Position(idx / 200, idx / 10 % 20, idx % 10);
  }

  constexpr auto operator<=>(const Position&) const = default;

  static constexpr size_t NumBytes() { return 2; }
  void GetBytes(uint8_t ret[]) const {
    ret[0] = r << 5 | x;
    ret[1] = y;
  }
  uint32_t Index() const { return (uint32_t)r * 200 + x * 10 + y; }
  constexpr Position() = default;
  constexpr Position(uint8_t r, uint8_t x, uint8_t y) : r(r), x(x), y(y) {}
  constexpr Position(const uint8_t data[], size_t) : r(data[0] >> 5), x(data[0] & 31), y(data[1]) {}

  static const Position Start;
  static const Position Invalid;
};

inline constexpr Position Position::Start = Position(0, 0, 5);
// all pieces has cell to the left when r=0
inline constexpr Position Position::Invalid = Position(0, 0, 0);

namespace std {

template<>
struct hash<Position> {
  constexpr size_t operator()(const Position& p) const {
    return Hash(p.r, p.x*16+p.y);
  }
};

} // namespace std
