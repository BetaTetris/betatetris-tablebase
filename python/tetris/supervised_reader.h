#pragma once

#include <random>
#include <algorithm>
#include <unordered_set>
#include "../../src/io.h"
#include "../../src/supervised_data.h"
#include "python.h"
#include "state.h"

using MultiStateY = std::vector<std::array<float, 800>>;

class PythonSupervisedDataReader {
 public:
  PyObject_HEAD

 private:
  static constexpr size_t kBufSize = 262144;
  static constexpr size_t kDataBufSize = 262144;

  static constexpr int kAdjDelays[] = {0, 18, 21, 24, 30, 61};
  static constexpr int kTapTables[][10] = {
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18},
    {0, 3, 5, 8, 10, 13, 15, 18, 20, 23},
    {0, 3, 6, 9, 12, 15, 18, 21, 24, 27},
    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36},
    {0, 5, 10, 15, 20, 25, 30, 35, 40, 45},
    {0, 6, 12, 18, 24, 30, 36, 42, 48, 54},
    {0, 2, 4, 6, 18, 20, 22, 24, 36, 38},
  };

  CompressedClassReader<SupervisedData> reader_;
  std::mt19937_64 rng_;
  std::vector<SupervisedData> data_buf_;
  std::vector<size_t> block_order_;
  size_t cur_data_, cur_idx_, cur_block_;

  MultiState st_x_;
  MultiStateY st_y_;

  void PushItem_(const SupervisedData& data) {
    auto Ind = [](const Position& a) { return a.r * 200 + a.x * 10 + a.y; };
    Tetris game = data.GetGame(kTapTables[data.tag >> 3 & 7], kAdjDelays[data.tag & 7]);
    {
      State st{};
      GetState(game, st);
      st_x_.push_back(st);
      st_y_.emplace_back();
      memset(st_y_.back().data(), 0, sizeof(st_y_.back()));
    }
    if (game.IsNoAdjMove(data.pos[0])) {
      st_y_.back()[Ind(data.pos[0])] = 1;
      return;
    }
    std::vector<Position> initial;
    uint64_t initial_mask = game.GetInitialMask();
    const PossibleMoves& moves = game.GetPossibleMoveList();
    for (int i = 0; i < 64; i++) {
      if (!(initial_mask >> i & 1)) continue;
      std::unordered_set<Position> mp(moves.adj[i].second.begin(), moves.adj[i].second.end());
      bool flag = true;
      for (size_t j = 0; j < kPieces; j++) {
        if (!mp.count(data.pos[j])) { flag = false; break; }
      }
      if (flag) initial.push_back(moves.adj[i].first);
    }
    for (auto& pos : initial) st_y_.back()[Ind(pos)] = 1. / initial.size();
    for (size_t i = 0; i < kPieces; i++) {
      Tetris ngame = game;
      ngame.SetNextPiece(i);
      ngame.InputPlacement(initial[std::uniform_int_distribution<int>(0, initial.size() - 1)(rng_)], i);
      State st{};
      GetState(ngame, st);
      st_x_.push_back(st);
      st_y_.emplace_back();
      st_y_.back()[Ind(data.pos[i])] = 1;
    }
  }

  void ReadOneData_() {
    bool err = false;
    try {
      data_buf_.push_back(reader_.ReadOne());
    } catch (ReadError&) {
      err = true;
    }
    if (err || ++cur_idx_ == reader_.ItemsPerIndex()) {
      cur_idx_ = 0;
      if (++cur_block_ == block_order_.size()) cur_block_ = 0;
      reader_.Seek(reader_.ItemsPerIndex() * block_order_[cur_block_], kBufSize, 0);
    }
  }

  void ReadOne_() {
    if (cur_data_ == data_buf_.size()) {
      data_buf_.clear();
      cur_data_ = 0;
      while (data_buf_.size() < kDataBufSize) ReadOneData_();
      std::shuffle(data_buf_.begin(), data_buf_.end(), rng_);
    }
    PushItem_(data_buf_[cur_data_++]);
  }

 public:
  PythonSupervisedDataReader(const std::string& fname, size_t seed = 0) :
      reader_(fname), rng_(seed), cur_data_(0), cur_idx_(0), cur_block_(0) {
    if (reader_.ItemsPerIndex() == 0) throw std::runtime_error("must have index file");
    block_order_.resize(reader_.NumIndex());
    for (size_t i = 0; i < block_order_.size(); i++) block_order_[i] = i;
    std::shuffle(block_order_.begin(), block_order_.end(), rng_);
    reader_.Seek(reader_.ItemsPerIndex() * block_order_[0]);
  }

  std::pair<MultiState, MultiStateY> ReadBatch(size_t sz) {
    while (st_x_.size() < sz) ReadOne_();
    if (st_x_.size() != st_y_.size()) throw std::runtime_error("unexpected");
    MultiState nx;
    MultiStateY ny;
    size_t remain = st_x_.size() - sz;
    if (remain) {
      nx.resize(remain);
      ny.resize(remain);
#define X(name, id, dims, typ) memcpy(nx.name.data(), st_x_.name.data() + sz, remain * sizeof(State::name));
      STATE_MEMBERS_
#undef X
      memcpy(ny.data(), st_y_.data() + sz, remain * sizeof(ny[0]));
    }
    nx.swap(st_x_); nx.resize(sz);
    ny.swap(st_y_); ny.resize(sz);
    return {nx, ny};
  }
};

extern PyTypeObject py_supervised_data_reader_class;
