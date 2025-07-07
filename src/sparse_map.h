#pragma once

#include <vector>
#include <stdexcept>

#include "constexpr_helpers.h"

struct FreeDeleter {
  void operator()(void* x) { free(x); }
};

// static sparse hash with minimal memory overhead and lookup cache misses
// needs two passes on input to construct, and assumes uniqueness of input
template <class ValueType, class KeySelect, class Hash, class KeyEqual>
class StaticSparseHash {
 public:
  using key_type = typename KeySelect::key_type;
  using value_type = ValueType;
 private:
  static constexpr size_t kGroupShift = 6;
  static constexpr size_t kGroupMask = (1 << kGroupShift) - 1;

  struct Group {
    uint64_t idx, bitmap;
  };
  std::vector<uint64_t> tmp_bitmap_;
  std::vector<Group> groups_;
  std::unique_ptr<ValueType, FreeDeleter> arr_;
  uint64_t bucket_size_mask_;
  size_t size_, phase_1_cnt_, phase_2_cnt_;
  int phase_;

  size_t HashKey_(const key_type& key) const {
    return Hash()(key);
  }
  bool KeyEqual_(const key_type& key1, const key_type& key2) const {
    return KeyEqual()(key1, key2);
  }
  size_t BucketFromHash_(size_t hash) const {
    return hash & bucket_size_mask_;
  }

  template <bool tmp> uint64_t& GetBitmap_(size_t group) {
    if constexpr (tmp) return tmp_bitmap_[group];
    return groups_[group].bitmap;
  }

  template <bool tmp>
  uint64_t MarkNextEmptyPosition_(size_t bucket) {
    size_t group = bucket >> kGroupShift;
    uint64_t nmask = GetBitmap_<tmp>(group) | ((1ll << (bucket & kGroupMask)) - 1);
    while (nmask == ~(uint64_t)0) [[unlikely]] {
      group = (group + 1) & (bucket_size_mask_ >> kGroupShift);
      nmask = GetBitmap_<tmp>(group);
    }
    uint64_t pos = ctz<uint64_t>(~nmask);
    GetBitmap_<tmp>(group) |= 1ll << pos;
    return group << kGroupShift | pos;
  }

  void InsertPhase1_(const key_type& k) {
    if (phase_ != 0) throw std::logic_error("phase 1 already finished");
    if (groups_.empty()) throw std::logic_error("not initialized");
    if (phase_1_cnt_ >= bucket_size_mask_) throw std::length_error("phase 1 too many inserts");
    size_t hash_val = HashKey_(k);
    size_t bucket = BucketFromHash_(hash_val);
    MarkNextEmptyPosition_<false>(bucket);
    phase_1_cnt_++;
  }

  template <class... Args>
  void InsertPhase2_(Args&&... args) {
    if (phase_ != 1) throw std::logic_error("phase 1 haven't finished or phase 2 already finished");
    if (phase_2_cnt_ >= size_) throw std::length_error("phase 2 too many inserts");
    value_type val(std::forward<Args>(args)...);
    size_t hash_val = HashKey_(KeySelect()(val));
    size_t bucket = BucketFromHash_(hash_val);
    uint64_t position = MarkNextEmptyPosition_<true>(bucket);
    const auto& group = groups_[position >> kGroupShift];
    if ((group.bitmap & (1ll << (position & kGroupMask))) == 0) throw std::invalid_argument("inconsistent inserts");
    uint64_t idx = group.idx + popcount<uint64_t>(group.bitmap & ((1ll << (position & kGroupMask)) - 1));
    new(arr_.get() + idx) value_type(std::move(val));
    phase_2_cnt_++;
  }

  template <bool clear>
  void Reset_(size_t bucket_size) {
    if constexpr (clear) {
      if (phase_ == 1) {
        for (size_t i = 0; i < groups_.size(); i++) {
          for (size_t j = 0, idx = groups_[i].idx; j < 64; j++) {
            if (tmp_bitmap_[i] >> j & 1) arr_.get()[idx].~value_type();
            idx += groups_[i].bitmap >> j & 1;
          }
        }
      } else if (phase_ == 2) {
        for (size_t i = 0; i < size_; i++) arr_.get()[i].~value_type();
      }
      arr_.reset();
      std::vector<uint64_t>().swap(tmp_bitmap_);
    }
    if (bucket_size) {
      bucket_size = bucket_size <= (1 << kGroupShift) ? 1 << kGroupShift : 1ll << (64 - clz<uint64_t>(bucket_size - 1));
      bucket_size_mask_ = bucket_size - 1;
      groups_.resize(bucket_size >> kGroupShift);
    } else {
      groups_.clear();
    }
    if constexpr (clear) groups_.shrink_to_fit();
    tmp_bitmap_.clear();
    size_ = 0;
    phase_1_cnt_ = 0;
    phase_2_cnt_ = 0;
    phase_ = 0;
  }

  uint64_t Query_(const key_type& k) const {
    if (phase_ != 2) throw std::logic_error("not initialized yet");
    size_t hash_val = HashKey_(k);
    size_t bucket = BucketFromHash_(hash_val);
    size_t group = bucket >> kGroupShift;
    uint32_t offset = bucket & kGroupMask;
    size_t idx = groups_[group].idx + popcount<uint64_t>(groups_[group].bitmap & ((1ll << offset) - 1));
    while (true) {
      if ((groups_[group].bitmap >> offset & 1) == 0) return ~(uint64_t)0;
      if (KeyEqual_(k, KeySelect()(arr_.get()[idx]))) return idx;
      idx++;
      offset = (offset + 1) & kGroupMask;
      if (!offset) {
        group = (group + 1) & (bucket_size_mask_ >> kGroupShift);
        if (!group) idx = 0;
      }
    }
  }

 public:
  StaticSparseHash(size_t bucket_size = 0) { Reset_<false>(bucket_size); }
  StaticSparseHash(StaticSparseHash<ValueType, KeySelect, Hash, KeyEqual>&& x) :
      tmp_bitmap_(std::move(x.tmp_bitmap_)), groups_(std::move(x.groups_)), arr_(std::move(x.arr_)),
      bucket_size_mask_(x.bucket_size_mask_), size_(x.size_),
      phase_1_cnt_(x.phase_1_cnt_), phase_2_cnt_(x.phase_2_cnt_), phase_(x.phase_) {
    x.phase_ = 3;
  }
  ~StaticSparseHash() { if (phase_ != 3) Reset_<true>(0); }

  void clear(size_t bucket_size = 0) { Reset_<true>(bucket_size); }

  void insert_phase1(const key_type& k) { InsertPhase1_(k); }
  void insert_phase1(const value_type& v) { InsertPhase1_(KeySelect()(v)); }

  void finish_phase1() {
    if (phase_ != 0) throw std::logic_error("phase 1 already finished");
    size_ = 0;
    for (auto& group : groups_) {
      group.idx = size_;
      size_ += popcount<uint64_t>(group.bitmap);
    }
    tmp_bitmap_.resize(groups_.size());
    arr_.reset((ValueType*)malloc(sizeof(ValueType) * size_)); // no constructing
    phase_2_cnt_ = 0;
    phase_ = 1;
  }

  void insert_phase2(const value_type& v) { InsertPhase2_(v); }
  void insert_phase2(value_type&& v) { InsertPhase2_(std::move(v)); }
  template <class... Args>
  void insert_phase2(Args&&... args) { InsertPhase2_(std::forward<Args>(args)...); }

  void finish_phase2() {
    if (phase_ != 1) throw std::logic_error("phase 1 haven't finished or phase 2 already finished");
    if (phase_2_cnt_ != size_) throw std::length_error("inserts does not match");
    std::vector<uint64_t>().swap(tmp_bitmap_);
    phase_ = 2;
  }

  const value_type* find(const key_type& k) const {
    uint64_t res = Query_(k);
    return res == ~(uint64_t)0 ? nullptr : arr_.get() + res;
  }
  value_type* find(const key_type& k) {
    uint64_t res = Query_(k);
    return res == ~(uint64_t)0 ? nullptr : arr_.get() + res;
  }

  size_t size() const { return size_; }
};

template <class K, class V>
struct PairKeySelect {
  using key_type = K;
  const K& operator()(const std::pair<K, V>& x) const { return x.first; }
};

template <class K, class V, class Hash = std::hash<K>, class KeyEqual = std::equal_to<K>>
class StaticSparseMap : public StaticSparseHash<
    std::pair<K, V>, PairKeySelect<K, V>, Hash, KeyEqual> {
 public:
  StaticSparseMap(size_t bucket_size = 0) :
    StaticSparseHash<std::pair<K, V>, PairKeySelect<K, V>, Hash, KeyEqual>(bucket_size) {}

  const V& operator[](const K& k) const { return this->find(k)->second; }
  V& operator[](const K& k) { return this->find(k)->second; }
};
