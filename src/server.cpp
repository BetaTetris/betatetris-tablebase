#include "server.h"

#include <array>
#include <boost/asio.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop
#include "game.h"
#include "move.h"
#include "play.h"
#include "config.h"
#include "tetris.h"
#include "io_hash.h"
#include "board_set.h"

using namespace boost;
using boost::asio::ip::tcp;
using boost::system::error_code;

namespace {

struct ReadEOF {};

class ConnectionBase {
 protected:
  tcp::socket socket_;

  void Send(const char* buf, size_t len) {
    error_code ec;
    size_t sent = 0;
    while (sent < len) {
      size_t cur = socket_.write_some(asio::buffer(buf + sent, len - sent), ec);
      if (ec) throw std::runtime_error(fmt::format("Write error: {}", ec.message()));
      sent += cur;
    }
  }

  std::vector<uint8_t> ReadUntil(size_t len) {
    std::vector<uint8_t> buf(len);
    error_code ec;
    size_t received = 0;
    while (received < len) {
      size_t cur = socket_.read_some(asio::buffer(buf.data() + received, len - received), ec);
      if (ec) {
        if (ec == asio::error::eof) throw ReadEOF();
        throw std::runtime_error(fmt::format("Read error: {}", ec.message()));
      }
      received += cur;
    }
    return buf;
  }

  ConnectionBase(asio::io_context& io_context) : socket_(io_context) {}
 public:
  tcp::socket& GetSocket() { return socket_; }
};

class FCEUXConnection : public std::enable_shared_from_this<FCEUXConnection>, public ConnectionBase {
  using ConnectionBase::socket_;
  using ConnectionBase::Send;
  using ConnectionBase::ReadUntil;

  Tetris game;
  bool done;
  Play play;
  Position prev_pos, prev_placement;
  FrameSequence prev_seq;
  std::array<Position, 7> prev_strats;

  void SendSeq(const FrameSequence& seq) {
    std::vector<uint8_t> buf(seq.size() + 3);
    buf[0] = 0xfe;
    buf[1] = seq.size() & 255;
    buf[2] = seq.size() >> 8;
    static_assert(sizeof(seq[0]) == 1);
    if (seq.size()) memcpy(buf.data() + 3, seq.data(), seq.size());
    Send(reinterpret_cast<const char*>(buf.data()), buf.size());
  }

  void StepGame(const Position& pos) {
    if (done) return;
    game.InputPlacement(pos, 0);
    done = game.IsOver();
    if (done) {
      spdlog::info("Game over");
    }
  }

  void DoPremove() {
    auto strat = done ? std::array<Position, 7>{} : play.GetStrat(game);
    if (strat[0] == Position::Invalid || done) {
      if (!done) spdlog::info("Not a seen board; topping out");
      done = true;
      prev_strats[0] = {-1, 0, 0};
      SendSeq({});
      return;
    }
    if (!game.IsNoAdjMove(strat[0])) {
      auto [pos, seq] = game.GetAdjPremove(strat.data());
      prev_pos = pos;
      prev_seq = seq;
      prev_strats = strat;
      StepGame(pos);
      SendSeq(seq);
    } else {
      auto seq = game.GetSequence(strat[0]);
      prev_pos = strat[0];
      prev_seq.clear();
      prev_strats = {Position::Invalid};
      SendSeq(seq);
      SendSeq({});
    }
  }

  void FinishMove(int next_piece) {
    bool prev_none = prev_strats[0] == Position::Invalid;
    if (done) {
      if (!prev_none) SendSeq({});
      return;
    }
    game.SetNextPiece(next_piece);
    if (prev_none) {
      StepGame(prev_pos);
      prev_placement = prev_pos;
    } else {
      Position strat = prev_strats[next_piece];
      size_t old_size = prev_seq.size();
      auto fin_seq = prev_seq;
      game.FinishAdjSequence(fin_seq, prev_pos, strat);
      fin_seq.erase(fin_seq.begin(), fin_seq.begin() + old_size);
      SendSeq(fin_seq);
      StepGame(strat);
      prev_placement = strat;
    }
  }

  void FirstPiece() {
    auto strats = play.GetStrat(game);
    auto pos = strats[game.NextPiece()];
    SendSeq(game.GetSequence(pos));
    if (!game.IsNoAdjMove(strats[0])) {
      StepGame(game.GetAdjPremove(strats.data()).first);
    }
    StepGame(pos);
    prev_placement = pos;
    DoPremove();
  }

  void DoWork() {
    while (true) {
      uint8_t op = ReadUntil(1)[0];
      if (op == 0xff || op == 0xef) {
        auto data = ReadUntil(3);
        spdlog::info("New game: {}, {}", (int)data[0], (int)data[1]);
        done = false;
        game.Reset(Board::Ones, 0, data[0], data[1]);
        FirstPiece();
      } else if (op == 0xfd) {
        auto data = ReadUntil(4);
        if (!done && Position{data[0], data[1], data[2]} != prev_placement) {
          spdlog::warn("Error: unexpected placement ({}, {}, {}); expected ({}, {}, {})",
              data[0], data[1], data[2], prev_placement.r, prev_placement.x, prev_placement.y);
          done = true;
        }
        FinishMove(data[3]);
        DoPremove();
      } else {
        throw std::runtime_error("Invalid opcode");
      }
    }
  }
 public:
  FCEUXConnection(asio::io_context& io_context) : ConnectionBase(io_context), done(true) {}

  void Run(const std::string& remote_addr, int remote_port) {
    try {
      DoWork();
    } catch (ReadEOF& e) {
      spdlog::info("EOF from {}:{}", remote_addr, remote_port);
    } catch (std::exception& e) {
      spdlog::warn("{}", e.what());
    }
  }
};

class BoardConnection : public std::enable_shared_from_this<BoardConnection>, public ConnectionBase {
  using ConnectionBase::socket_;
  using ConnectionBase::Send;
  using ConnectionBase::ReadUntil;

  std::string threshold_name;

  void DoWork() {
    Play play;
    std::vector<CompressedClassReader<NodeThreshold>> readers;
    for (int i = 0; i < kGroups; i++) readers.emplace_back(ThresholdPath(threshold_name, i));
    // receive: 25 bytes board + 1 byte current piece + 2 bytes lines
    constexpr size_t kReceiveSize = 28;
    // send: 21 bytes position ((r,x,y)*7) + 1 byte threshold
    constexpr size_t kSendSize = 22;
    while (true) {
      uint32_t num_boards = ReadUntil(1)[0];
      auto data = ReadUntil(kReceiveSize * num_boards);
      std::vector<uint8_t> send_buf(kSendSize * num_boards);
      for (size_t i = 0; i < num_boards; i++) {
        const uint8_t* in_ptr = data.data() + kReceiveSize * i;
        uint8_t* out_ptr = send_buf.data() + kSendSize * i;
        CompactBoard board(in_ptr, 25);
        size_t move_idx = 0;
        int group = GetGroupByCells(board.Count());
        int piece = in_ptr[25];
        int lines = BytesToInt<uint16_t>(in_ptr + 26);
        auto strats = play.GetStrat(board, piece, lines, &move_idx);
        for (size_t j = 0; j < kPieces; j++) {
          out_ptr[j*3  ] = strats[j].r;
          out_ptr[j*3+1] = strats[j].x;
          out_ptr[j*3+2] = strats[j].y;
        }
        if (strats[0] != Position::Invalid) {
          readers[group].Seek(move_idx, 0, 0);
          out_ptr[21] = readers[group].ReadOne(1, 0)[lines / kGroupLineInterval];
        }
      }
      Send(reinterpret_cast<const char*>(send_buf.data()), send_buf.size());
    }
  }
 public:
  BoardConnection(asio::io_context& io_context) : ConnectionBase(io_context) {}
  BoardConnection(asio::io_context& io_context, const std::string& threshold_name) :
      ConnectionBase(io_context), threshold_name(threshold_name) {}

  void Run(const std::string& remote_addr, int remote_port) {
    try {
      DoWork();
    } catch (ReadEOF& e) {
      spdlog::info("EOF from {}:{}", remote_addr, remote_port);
    } catch (std::exception& e) {
      spdlog::warn("{}", e.what());
    }
  }
};

template <class Connection>
class Server {
  boost::asio::io_context& io_context_;
  tcp::acceptor acceptor_;

  template <class... Args>
  void DoAccept(Args&&... args) {
    auto new_connection = std::make_shared<Connection>(io_context_, args...);
    acceptor_.async_accept(new_connection->GetSocket(), [this,new_connection,...args=std::forward<Args>(args)](error_code ec) {
      auto& socket = new_connection->GetSocket();
      auto remote_addr = socket.remote_endpoint().address().to_string();
      auto remote_port = socket.remote_endpoint().port();
      if (!ec) {
        spdlog::info("New connection: {}:{}", remote_addr, remote_port);
        std::thread([new_connection,remote_addr,remote_port]() {
          new_connection->Run(remote_addr, remote_port);
        }).detach();
      } else {
        spdlog::info("Accept error: {}", ec.message());
      }
      DoAccept(std::forward<Args>(args)...);
    });
  }

  template <class... Args>
  void DoAcceptSynchronous(Args&&... args) {
    while (true) {
      Connection new_connection(io_context_, std::forward<Args>(args)...);
      acceptor_.accept(new_connection.GetSocket());
      auto& socket = new_connection.GetSocket();
      auto remote_addr = socket.remote_endpoint().address().to_string();
      auto remote_port = socket.remote_endpoint().port();
      spdlog::info("New connection: {}:{}", remote_addr, remote_port);
      new_connection.Run(remote_addr, remote_port);
    }
  }
 public:
  template <class... Args>
  Server(asio::io_context& io_context, const std::string& addr, int port, bool one_conn, Args&&... args) :
      io_context_(io_context),
      acceptor_(io_context, tcp::endpoint(asio::ip::make_address(addr), port)) {
    if (one_conn) {
      DoAcceptSynchronous(std::forward<Args>(args)...);
    } else {
      DoAccept(std::forward<Args>(args)...);
    }
  }
};

} // namespace

void StartFCEUXServer(const std::string& bind, int port, bool one_conn) {
  asio::io_context io_context;
  Server<FCEUXConnection> s(io_context, bind, port, one_conn);
  io_context.run();
}

void StartBoardServer(const std::string& bind, int port, const std::string& threshold_name, bool one_conn) {
  asio::io_context io_context;
  Server<BoardConnection> s(io_context, bind, port, one_conn, threshold_name);
  io_context.run();
}
