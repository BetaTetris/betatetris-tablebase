#include <bitset>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <gtest/gtest.h>
#include "../src/sparse_map.h"

namespace {

struct ConstructCounter {
  static std::bitset<1000> construct_map;
  static size_t construct_counter;
  size_t idx;
  ConstructCounter() : idx(construct_counter++) {
    if (idx > 1000) throw std::length_error("too many constructs");
    construct_map[idx] = true;
  }
  ConstructCounter(const ConstructCounter&) = delete;
  ConstructCounter(ConstructCounter&& x) : idx(x.idx) {
    x.idx = 0;
  }
  ~ConstructCounter() noexcept(false) {
    if (!idx) return;
    if (!construct_map[idx]) throw std::runtime_error("double free");
    construct_map[idx] = false;
    idx = 0;
  }
  static void reset() { construct_counter = 1; }
  static void check_free() { ASSERT_EQ(construct_map.count(), 0); }
};

size_t ConstructCounter::construct_counter = 1;
std::bitset<1000> ConstructCounter::construct_map;

class MapTest : public ::testing::Test {
 protected:
  std::mt19937_64 gen;
  std::vector<std::pair<uint64_t, uint64_t>> pairs;
  std::unordered_map<uint64_t, uint64_t> mp_ref;
  StaticSparseMap<uint64_t, uint64_t> mp;
  void SetUp(size_t sz) {
    mp_ref.clear();
    mp_ref.reserve(sz);
    while (mp_ref.size() < sz) {
      uint64_t k = gen(), v = gen();
      mp_ref[k] = v;
    }
    pairs.assign(mp_ref.begin(), mp_ref.end());
    std::shuffle(pairs.begin(), pairs.end(), gen);
  }
  void TearDown() override {}
  void build(size_t bucket_size) {
    mp.clear(bucket_size);
    for (auto& i : pairs) mp.insert_phase1(i);
    mp.finish_phase1();
    for (auto& i : pairs) mp.insert_phase2(i);
    mp.finish_phase2();
  }
};

TEST_F(MapTest, NormalOp) {
  SetUp(10000);
  build(20000);
  for (auto& i : pairs) ASSERT_EQ(mp[i.first], i.second);
  for (size_t i = 0; i < 10000; i++) {
    uint64_t k = gen();
    ASSERT_EQ(mp.find(k) == nullptr, mp_ref.count(k) == 0);
  }
}

TEST_F(MapTest, HighLoadFactor) {
  SetUp(100);
  build(101);
  for (auto& i : pairs) ASSERT_EQ(mp[i.first], i.second);
  for (size_t i = 0; i < 100; i++) {
    uint64_t k = gen();
    ASSERT_EQ(mp.find(k) == nullptr, mp_ref.count(k) == 0);
  }
}

TEST_F(MapTest, Overload) {
  SetUp(128);
  EXPECT_THROW({ build(128); }, std::length_error);
}

TEST_F(MapTest, InconsistentInsert) {
  SetUp(100);
  mp.clear(200);
  for (auto& i : pairs) mp.insert_phase1(i);
  mp.finish_phase1();
  SetUp(100); // insert different things on phase 2
  EXPECT_THROW({
    for (auto& i : pairs) mp.insert_phase2(i);
  }, std::invalid_argument);
}

TEST_F(MapTest, CheckFree) {
  ConstructCounter::reset();
  SetUp(100);
  build(200);
  mp.clear();
  ConstructCounter::check_free();

  ConstructCounter::reset();
  SetUp(100);
  {
    StaticSparseMap<uint64_t, uint64_t> mp2(200);
    for (auto& i : pairs) mp2.insert_phase1(i);
    mp2.finish_phase1();
    for (size_t i = 0; i < 50; i++) mp2.insert_phase2(pairs[i]);
  }
  ConstructCounter::check_free();
}

} // namespace
