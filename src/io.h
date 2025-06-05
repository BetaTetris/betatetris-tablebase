#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <zstd.h>

#include "files.h"
#include "compressor.h"
#include "constexpr_helpers.h"

namespace io_internal {

template <class T>
inline void WriteToBuf(std::vector<uint8_t>& buf, const T& val) {
  size_t old_sz = buf.size();
  if constexpr (T::kIsConstSize) {
    buf.resize(old_sz + T::NumBytes());
    val.GetBytes(buf.data() + old_sz);
  } else {
    size_t sz = val.NumBytes();
    buf.resize(old_sz + T::kSizeNumberBytes + sz);
    uint8_t sz_buf[8] = {};
    IntToBytes<uint64_t>(sz, sz_buf);
    memcpy(buf.data() + old_sz, sz_buf, T::kSizeNumberBytes);
    val.GetBytes(buf.data() + old_sz + T::kSizeNumberBytes);
  }
}

template <class T>
inline void WriteToBufRaw(std::vector<uint8_t>& buf, const std::vector<uint8_t>& val) {
  size_t old_sz = buf.size();
  if constexpr (T::kIsConstSize) {
    buf.resize(old_sz + T::NumBytes());
    memcpy(buf.data() + old_sz, val.data(), T::NumBytes());
  } else {
    size_t sz = val.size();
    buf.resize(old_sz + T::kSizeNumberBytes + sz);
    uint8_t sz_buf[8] = {};
    IntToBytes<uint64_t>(sz, sz_buf);
    memcpy(buf.data() + old_sz, sz_buf, T::kSizeNumberBytes);
    memcpy(buf.data() + old_sz + T::kSizeNumberBytes, val.data(), sz);
  }
}

template <class T>
inline std::pair<size_t, size_t> GetNextSize(const uint8_t* buf) {
  if constexpr (T::kIsConstSize) {
    return {T::NumBytes(), 0};
  } else {
    uint8_t sz_buf[8] = {};
    memcpy(sz_buf, buf, T::kSizeNumberBytes);
    return {BytesToInt<uint64_t>(sz_buf), T::kSizeNumberBytes};
  }
}

template <class T>
struct ClassIOAttr {
  static constexpr bool kIsConstSize = T::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = [](){
    if constexpr (kIsConstSize) {
      return 0;
    } else {
      return T::kSizeNumberBytes;
    }
  }();
  static constexpr size_t kBufferSize = [](){
    if constexpr (kIsConstSize) {
      return (1048575 / T::NumBytes() + 1) * T::NumBytes();
    } else {
      return 1048576;
    }
  }();
};

template <class T>
class ClassWriterImpl {
 protected:
  static constexpr size_t kIsConstSize = ClassIOAttr<T>::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = ClassIOAttr<T>::kSizeNumberBytes;
  static constexpr size_t kBufferSize = ClassIOAttr<T>::kBufferSize;
  static constexpr size_t kIndexBufferSize = 131072;

  std::vector<uint8_t> buf;
  std::vector<uint64_t> inds;
  size_t current;
  size_t items_per_index;
  size_t current_written_size;
  std::ofstream fout;
  std::ofstream fout_ind;
  bool moved;

  void Flush() {
    if (!fout.write(reinterpret_cast<const char*>(buf.data()), buf.size())) {
      throw std::runtime_error("write failed");
    }
    current_written_size = fout.tellp();
    buf.clear();
  }

  void FlushIndex() {
    if (!HasIndex()) return;
    std::vector<uint8_t> out(inds.size() * sizeof(uint64_t));
    for (size_t i = 0; i < inds.size(); i++) {
      IntToBytes<uint64_t>(inds[i], out.data() + (i * sizeof(uint64_t)));
    }
    inds.clear();
    if (!fout_ind.write(reinterpret_cast<const char*>(out.data()), out.size())) {
      throw std::runtime_error("write failed");
    }
  }
 public:
  ClassWriterImpl(const std::string& fname, size_t items_per_index) :
      current(0), items_per_index(items_per_index), current_written_size(0), moved(false) {
    static_assert(kIsConstSize || (kSizeNumberBytes >= 1 && kSizeNumberBytes <= 8));
    MkdirForFile(fname);
    fout.open(fname, std::ios_base::out | std::ios_base::trunc);
    if (!fout.is_open()) throw std::runtime_error("cannot open file");
    if (HasIndex()) {
      fout_ind.open(fname + ".index", std::ios_base::out | std::ios_base::trunc);
      if (!fout_ind.is_open()) throw std::runtime_error("cannot open index file");
      inds.push_back(items_per_index);
    }
    buf.reserve(kBufferSize);
  }
  ClassWriterImpl(const ClassWriterImpl&) = delete;
  ClassWriterImpl(ClassWriterImpl&& x) :
      buf(std::move(x.buf)), inds(std::move(x.inds)),
      current(x.current), items_per_index(x.items_per_index),
      current_written_size(x.current_written_size),
      fout(std::move(x.fout)), fout_ind(std::move(x.fout_ind)), moved(false) {
    x.moved = true;
  }

  ~ClassWriterImpl() {
    if (moved) return;
    Flush();
    FlushIndex();
  }

  bool HasIndex() const {
    return items_per_index >= 1;
  }

  size_t Size() const {
    return current;
  }

  size_t ByteSize() {
    return current_written_size + buf.size();
  }
};

template <class T>
class ClassReaderImpl {
 protected:
  static constexpr size_t kIsConstSize = ClassIOAttr<T>::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = ClassIOAttr<T>::kSizeNumberBytes;
  static constexpr size_t kBufferSize = ClassIOAttr<T>::kBufferSize;

  std::vector<uint8_t> buf;
  size_t current;
  size_t items_per_index;
  bool eof;
  std::ifstream fin;
  std::ifstream fin_index;

  void ReadUntilSize(size_t sz, size_t buf_size) {
    if (!fin) return;
    size_t old_sz = buf.size();
    sz = std::max(sz, buf_size);
    if (sz <= old_sz) return;
    buf.resize(sz);
    fin.read(reinterpret_cast<char*>(buf.data() + old_sz), sz - old_sz);
    buf.resize(old_sz + fin.gcount());
  }
 public:
  ClassReaderImpl(const std::string& fname, bool check_index) :
      current(0), items_per_index(0), eof(false) {
    static_assert(kIsConstSize || (kSizeNumberBytes >= 1 && kSizeNumberBytes <= 8));
    fin.rdbuf()->pubsetbuf(nullptr, 0);
    fin.open(fname);
    if (!fin.is_open()) throw std::runtime_error("cannot open file");
    uint8_t sz_buf[8] = {};
    if (check_index) {
      fin_index.rdbuf()->pubsetbuf(nullptr, 0);
      fin_index.open(fname + ".index");
      if (fin_index.is_open()) {
        if (fin_index.read(reinterpret_cast<char*>(sz_buf), 8) && fin_index.gcount() == 8) {
          items_per_index = BytesToInt<uint64_t>(sz_buf);
        }
        if (!items_per_index) throw std::runtime_error("invalid index file");
      }
    }
    buf.reserve(kBufferSize);
  }
  ClassReaderImpl(const ClassReaderImpl&) = delete;
  ClassReaderImpl(ClassReaderImpl&&) = default;

  bool HasIndex() const {
    return items_per_index >= 1;
  }

  size_t Position() const {
    return current;
  }
};

} // namespace io_internal

struct ReadError : std::out_of_range {
  using std::out_of_range::out_of_range;
};

template <class T>
class ClassWriter : public io_internal::ClassWriterImpl<T> {
  using io_internal::ClassWriterImpl<T>::kIsConstSize;
  using io_internal::ClassWriterImpl<T>::kSizeNumberBytes;
  using io_internal::ClassWriterImpl<T>::kBufferSize;
  using io_internal::ClassWriterImpl<T>::kIndexBufferSize;
  using io_internal::ClassWriterImpl<T>::Flush;
  using io_internal::ClassWriterImpl<T>::FlushIndex;
  using io_internal::ClassWriterImpl<T>::current;
  using io_internal::ClassWriterImpl<T>::inds;
  using io_internal::ClassWriterImpl<T>::buf;
  using io_internal::ClassWriterImpl<T>::items_per_index;
  using io_internal::ClassWriterImpl<T>::moved;
 public:
  using io_internal::ClassWriterImpl<T>::HasIndex;
  using io_internal::ClassWriterImpl<T>::ByteSize;
  using io_internal::ClassWriterImpl<T>::Size;

  ClassWriter(const std::string& fname, size_t items_per_index = 1024) :
      io_internal::ClassWriterImpl<T>(fname, kIsConstSize ? 0 : items_per_index) {}
  ClassWriter(const ClassWriter&) = delete;
  ClassWriter(ClassWriter&&) = default;
  ~ClassWriter() {
    if (moved) return;
    inds.push_back(ByteSize());
  }

  void Write(const T& item) {
    if (HasIndex() && current % items_per_index == 0) {
      inds.push_back(ByteSize());
      if (inds.size() >= kIndexBufferSize) FlushIndex();
    }
    current++;
    size_t sz = item.NumBytes();
    if (!kIsConstSize && SizeRangeOverflow<kSizeNumberBytes>(sz)) {
      throw std::out_of_range("output size too large");
    }
    io_internal::WriteToBuf(buf, item);
    if (buf.size() >= kBufferSize) Flush();
  }

  void Write(const std::vector<T>& items) {
    for (auto& i : items) Write(i);
  }

  void Write(const T& item, size_t cnt) {
    for (size_t i = 0; i < cnt; i++) Write(item);
  }
};

template <class T>
class ClassReader : public io_internal::ClassReaderImpl<T> {
  using io_internal::ClassReaderImpl<T>::kIsConstSize;
  using io_internal::ClassReaderImpl<T>::kSizeNumberBytes;
  using io_internal::ClassReaderImpl<T>::kBufferSize;
  using io_internal::ClassReaderImpl<T>::ReadUntilSize;
  using io_internal::ClassReaderImpl<T>::buf;
  using io_internal::ClassReaderImpl<T>::eof;
  using io_internal::ClassReaderImpl<T>::current;
  using io_internal::ClassReaderImpl<T>::fin;
  using io_internal::ClassReaderImpl<T>::fin_index;
  using io_internal::ClassReaderImpl<T>::items_per_index;

  size_t current_offset;

  uint64_t GetNextSize(size_t buf_size) {
    if constexpr (kIsConstSize) return T::NumBytes();

    ReadUntilSize(current_offset + kSizeNumberBytes, buf_size);
    if (buf.size() < current_offset + kSizeNumberBytes) {
      eof = true;
      throw ReadError("no more elements");
    }
    return io_internal::GetNextSize<T>(buf.data() + current_offset).first;
  }

  void ParseBufSize(size_t& buf_size) {
    if (buf_size == std::string::npos) buf_size = kBufferSize;
  }
 public:
  using io_internal::ClassReaderImpl<T>::HasIndex;
  using io_internal::ClassReaderImpl<T>::Position;

  ClassReader(const std::string& fname) :
      io_internal::ClassReaderImpl<T>(fname, !kIsConstSize), current_offset(0) {}
  ClassReader(const ClassReader&) = delete;
  ClassReader(ClassReader&&) = default;

  void SkipOne(size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    uint64_t sz = GetNextSize(buf_size);
    if (current_offset + kSizeNumberBytes + sz > buf.size()) {
      fin.seekg(current_offset + kSizeNumberBytes + sz - buf.size(), std::ios_base::cur);
      buf.clear();
      current_offset = 0;
    } else {
      current_offset += kSizeNumberBytes + sz;
    }
    current++;
  }

  T ReadOne(size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    uint64_t sz = GetNextSize(buf_size);
    ReadUntilSize(current_offset + kSizeNumberBytes + sz, buf_size);
    if (buf.size() < current_offset + kSizeNumberBytes + sz) {
      if constexpr (kIsConstSize) {
        eof = true;
        throw ReadError("no more elements");
      } else {
        throw std::runtime_error("invalid file format");
      }
    }
    T ret(static_cast<const uint8_t*>(buf.data() + (current_offset + kSizeNumberBytes)), sz);
    current_offset += kSizeNumberBytes + sz;
    current++;
    if (current_offset + kSizeNumberBytes >= buf.size()) {
      size_t new_sz = buf.size() - current_offset;
      for (size_t i = 0; i < new_sz; i++) buf[i] = buf[i + current_offset];
      buf.resize(new_sz);
      current_offset = 0;
    }
    return ret;
  }

  std::vector<T> ReadBatch(size_t num, size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    std::vector<T> ret;
    if (num != std::string::npos) ret.reserve(num);
    try {
      for (size_t i = 0; i < num; i++) ret.push_back(ReadOne(buf_size));
    } catch (ReadError&) {}
    return ret;
  }

  void Seek(size_t location, size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    if constexpr (kIsConstSize) {
      size_t buf_start = current - current_offset / T::NumBytes();
      size_t buf_end = current + (buf.size() - current_offset) / T::NumBytes();
      if (buf_start <= location && location < buf_end) {
        current_offset += ((int64_t)location - current) * T::NumBytes();
      } else {
        buf.clear();
        eof = false;
        fin.clear();
        fin.seekg(T::NumBytes() * location);
        current_offset = 0;
      }
      current = location;
      return;
    }
    if (HasIndex()) {
      size_t target_index_idx = location / items_per_index;
      size_t current_index_idx = current / items_per_index;
      if (target_index_idx != current_index_idx || location < current) {
        buf.clear();
        // +1 because the first element is items_per_index
        fin_index.clear();
        fin_index.seekg((target_index_idx + 1) * sizeof(uint64_t));
        uint8_t sz_buf[16] = {};
        if (!fin_index.read(reinterpret_cast<char*>(sz_buf), 16) || fin_index.gcount() != 16) {
          throw std::out_of_range("index out of range");
        }
        uint64_t start = BytesToInt<uint64_t>(sz_buf), end = BytesToInt<uint64_t>(sz_buf + 8);
        fin.clear();
        fin.seekg(start);
        buf_size = end - start;
        current = target_index_idx * items_per_index;
        current_offset = 0;
      }
    } else if (location < current) {
      buf.clear();
      fin.clear();
      fin.seekg(0);
      current = 0;
      current_offset = 0;
    }
    while (location > current) SkipOne(buf_size);
  }
};

template <class T>
class CompressedClassWriter : public io_internal::ClassWriterImpl<T> {
  using io_internal::ClassWriterImpl<T>::kIsConstSize;
  using io_internal::ClassWriterImpl<T>::kSizeNumberBytes;
  using io_internal::ClassWriterImpl<T>::kBufferSize;
  using io_internal::ClassWriterImpl<T>::kIndexBufferSize;
  using io_internal::ClassWriterImpl<T>::Flush;
  using io_internal::ClassWriterImpl<T>::FlushIndex;
  using io_internal::ClassWriterImpl<T>::current;
  using io_internal::ClassWriterImpl<T>::inds;
  using io_internal::ClassWriterImpl<T>::buf;
  using io_internal::ClassWriterImpl<T>::items_per_index;
  using io_internal::ClassWriterImpl<T>::moved;

  std::unique_ptr<CompressorBase> compressor;
  std::vector<uint8_t> compress_buf;

  void PushCompressedBlock(const std::vector<uint8_t>& vec) {
    size_t old_size = buf.size();
    buf.resize(old_size + vec.size());
    memcpy(buf.data() + old_size, vec.data(), vec.size());
    inds.push_back(ByteSize());
    if (buf.size() >= kBufferSize) Flush();
  }

  void GetCompressResults(bool wait = false) {
    if (wait) {
      while (compressor->RemainingBlocks()) {
        PushCompressedBlock(compressor->GetResultBlock());
      }
    } else {
      while (true) {
        auto block = compressor->GetResultBlockNoWait();
        if (!block) return;
        PushCompressedBlock(*block);
      }
    }
  }

  void DoCompress() {
    inds.push_back(compress_buf.size());
    compressor->CompressBlock(std::move(compress_buf));
    compress_buf.clear();
    GetCompressResults();
  }
 public:
  using io_internal::ClassWriterImpl<T>::HasIndex;
  using io_internal::ClassWriterImpl<T>::ByteSize;
  using io_internal::ClassWriterImpl<T>::Size;

  CompressedClassWriter(const std::string& fname, size_t items_per_index = 1024, int compress_level = -4) :
      io_internal::ClassWriterImpl<T>(fname, items_per_index == 0 ? 1 : items_per_index),
      compressor(std::make_unique<DefaultZstdCompressor>(compress_level)) {
    inds.push_back(0);
  }
  CompressedClassWriter(const std::string& fname, size_t items_per_index, std::unique_ptr<CompressorBase>&& compressor) :
      io_internal::ClassWriterImpl<T>(fname, items_per_index == 0 ? 1 : items_per_index),
      compressor(std::move(compressor)) {
    inds.push_back(0);
  }
  CompressedClassWriter(const CompressedClassWriter&) = delete;
  CompressedClassWriter(CompressedClassWriter&& x) :
      io_internal::ClassWriterImpl<T>(std::move(x)),
      compressor(std::move(x.compressor)), compress_buf(std::move(x.compress_buf)) {}
  ~CompressedClassWriter() {
    if (moved) return;
    if (compress_buf.size()) DoCompress();
    GetCompressResults(true);
  }

  void Write(const T& item) {
    current++;
    size_t sz = item.NumBytes();
    if (!kIsConstSize && SizeRangeOverflow<kSizeNumberBytes>(sz)) {
      throw std::out_of_range("output size too large");
    }
    io_internal::WriteToBuf(compress_buf, item);
    if (current % items_per_index == 0) {
      DoCompress();
      if (inds.size() >= kIndexBufferSize) FlushIndex();
    }
  }

  void Write(const std::vector<T>& items) {
    for (auto& i : items) Write(i);
  }

  void Write(const T& item, size_t cnt) {
    for (size_t i = 0; i < cnt; i++) Write(item);
  }

  void WriteRaw(const std::vector<uint8_t>& item) {
    current++;
    size_t sz = item.size();
    if (!kIsConstSize && SizeRangeOverflow<kSizeNumberBytes>(sz)) {
      throw std::out_of_range("output size too large");
    }
    io_internal::WriteToBufRaw<T>(compress_buf, item);
    if (current % items_per_index == 0) {
      DoCompress();
      if (inds.size() >= kIndexBufferSize) FlushIndex();
    }
  }

  void WriteRaw(const std::vector<std::vector<uint8_t>>& items) {
    for (auto& i : items) WriteRaw(i);
  }
};

template <class T>
class CompressedClassReader : public io_internal::ClassReaderImpl<T> {
  using io_internal::ClassReaderImpl<T>::kIsConstSize;
  using io_internal::ClassReaderImpl<T>::kSizeNumberBytes;
  using io_internal::ClassReaderImpl<T>::kBufferSize;
  static constexpr size_t kIndexBufferSize = 131072;

  using io_internal::ClassReaderImpl<T>::ReadUntilSize;
  using io_internal::ClassReaderImpl<T>::buf;
  using io_internal::ClassReaderImpl<T>::eof;
  using io_internal::ClassReaderImpl<T>::current;
  using io_internal::ClassReaderImpl<T>::fin;
  using io_internal::ClassReaderImpl<T>::fin_index;
  using io_internal::ClassReaderImpl<T>::items_per_index;

  /*
   * buf                                     [may partial]
   *   buf_start_bytes                         fin.tallg() (bytes)
   *                       block_start                     (item idx)
   *   0               [infer from ind_buf]     buf.size() (buf idx)
   *   |----------------------->********<............|
   *
   * ind_buf                                 [may partial]
   *   ind_start           block_start=ind_start+ind_offset/2 (item idx)
   *   0                   ind_offset        ind_buf.size()   (ind_buf idx)
   *   |----------------------->*****<...............|
   *                             len=3
   * block_buf
   *   block_start       current              [always full] (item idx)
   *   0                 block_offset                       (block_buf idx)
   *   |------------------->*********<...............|
   */
  std::unique_ptr<ZSTD_DCtx, size_t(*)(ZSTD_DCtx*)> zstd_ctx;
  size_t buf_start_bytes, ind_start, ind_offset, block_start, block_offset;
  std::vector<uint8_t> block_buf;
  std::vector<uint64_t> ind_buf;

  void ReadIndUntilSize(size_t sz, size_t buf_size) {
    size_t old_sz = ind_buf.size();
    sz = std::max(sz, buf_size);
    if (sz <= old_sz) return;
    std::vector<char> tmp((sz - old_sz) * 8);
    fin_index.read(tmp.data(), tmp.size());
    tmp.resize(fin_index.gcount());
    if (tmp.size() % 8 != 0) throw std::runtime_error("unexpected index file size");
    size_t new_count = tmp.size() / 8;
    ind_buf.reserve(old_sz + new_count);
    for (size_t i = 0; i < new_count; i++) {
      ind_buf.push_back(BytesToInt<uint64_t>(reinterpret_cast<const uint8_t*>(tmp.data() + i * 8)));
    }
  }

  // if false, the offsets are in invalid state
  bool MoveToNextBlock(size_t buf_size, size_t ind_buf_size) {
    if (eof) return false;
    if (block_buf.size()) ind_offset += 2;
    // ind_buf[ind_offset:ind_offset+3] = [start_byte, orig_size, end_byte]
    if (ind_offset + 3 >= ind_buf.size()) {
      for (size_t i = ind_offset; i < ind_buf.size(); i++) ind_buf[i - ind_offset] = ind_buf[i];
      ind_buf.resize(ind_buf.size() - ind_offset);
      ind_start += (ind_offset / 2) * items_per_index;
      ind_offset = 0;
      ReadIndUntilSize(3, ind_buf_size);
      if (ind_buf.size() < 3) return false;
    }
    size_t start_offset = ind_buf[ind_offset] - buf_start_bytes;
    size_t end_offset = ind_buf[ind_offset + 2] - buf_start_bytes;
    if (end_offset >= buf.size()) {
      for (size_t i = start_offset; i < buf.size(); i++) buf[i - start_offset] = buf[i];
      buf.resize(buf.size() - start_offset);
      buf_start_bytes += start_offset;
      end_offset -= start_offset;
      start_offset = 0;
    }
    ReadUntilSize(end_offset, buf_size);
    if (buf.size() < end_offset) return false;
    block_buf.resize(ind_buf[ind_offset + 1]);
    size_t ret = ZSTD_decompressDCtx(
        zstd_ctx.get(), block_buf.data(), block_buf.size(), buf.data() + start_offset, end_offset - start_offset);
    if (ZSTD_isError(ret)) throw std::runtime_error("zstd decompress failed");
    if (ret != block_buf.size()) throw std::runtime_error("decompress: unexpected data length");
    block_start = current;
    block_offset = 0;
    return true;
  }

  uint64_t GetNextSize(size_t buf_size, size_t ind_buf_size) {
    if (block_offset == block_buf.size()) {
      if (!MoveToNextBlock(buf_size, ind_buf_size)) {
        eof = true;
        throw ReadError("no more elements");
      }
    }
    return io_internal::GetNextSize<T>(block_buf.data() + block_offset).first;
  }

  void ParseBufSize(size_t& buf_size, size_t& ind_buf_size) {
    if (buf_size == std::string::npos) buf_size = kBufferSize;
    if (ind_buf_size == std::string::npos) ind_buf_size = kIndexBufferSize;
  }

  void SeekBuf(size_t start_bytes) {
    if (!eof && buf_start_bytes <= start_bytes && start_bytes < buf_start_bytes + buf.size()) {
      // nothing
    } else {
      eof = false;
      buf_start_bytes = start_bytes;
      buf.clear();
      fin.clear();
      fin.seekg(start_bytes);
    }
  }

  size_t ReadOneCommon(size_t buf_size, size_t ind_buf_size) {
    ParseBufSize(buf_size, ind_buf_size);
    uint64_t sz = GetNextSize(buf_size, ind_buf_size);
    if (block_buf.size() < block_offset + kSizeNumberBytes + sz) {
      if constexpr (kIsConstSize) {
        eof = true;
        throw ReadError("no more elements");
      } else {
        throw std::runtime_error("invalid file format");
      }
    }
    current++;
    return sz;
  }
 public:
  using io_internal::ClassReaderImpl<T>::HasIndex;
  using io_internal::ClassReaderImpl<T>::Position;

  CompressedClassReader(const std::string& fname) :
      io_internal::ClassReaderImpl<T>(fname, true),
      zstd_ctx(ZSTD_createDCtx(), ZSTD_freeDCtx),
      buf_start_bytes(0), ind_start(0), ind_offset(0), block_start(0), block_offset(0) {
    if (items_per_index == 0) throw std::runtime_error("index file not found");
    if (!zstd_ctx) throw std::runtime_error("zstd initialize failed");
  }
  CompressedClassReader(const CompressedClassReader&) = delete;
  CompressedClassReader(CompressedClassReader&& x) :
      io_internal::ClassReaderImpl<T>(std::move(x)), zstd_ctx(std::move(x.zstd_ctx)),
      buf_start_bytes(x.buf_start_bytes), ind_start(x.ind_start), ind_offset(x.ind_offset),
      block_start(x.block_start), block_offset(x.block_offset),
      block_buf(std::move(x.block_buf)), ind_buf(std::move(x.ind_buf)) {}

  void SkipOne(size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    ParseBufSize(buf_size, ind_buf_size);
    uint64_t sz = GetNextSize(buf_size, ind_buf_size);
    block_offset += kSizeNumberBytes + sz;
    current++;
  }

  T ReadOne(size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    uint64_t sz = ReadOneCommon(buf_size, ind_buf_size);
    T ret(static_cast<const uint8_t*>(block_buf.data() + (block_offset + kSizeNumberBytes)), sz);
    block_offset += kSizeNumberBytes + sz;
    return ret;
  }

  // the destruction should be handled by caller
  void ReadOne(T* ret, size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    uint64_t sz = ReadOneCommon(buf_size, ind_buf_size);
    new(ret) T(static_cast<const uint8_t*>(block_buf.data() + (block_offset + kSizeNumberBytes)), sz);
    block_offset += kSizeNumberBytes + sz;
  }

  std::vector<T> ReadBatch(size_t num, size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    ParseBufSize(buf_size, ind_buf_size);
    std::vector<T> ret;
    if (num != std::string::npos) ret.reserve(num);
    try {
      for (size_t i = 0; i < num; i++) ret.push_back(ReadOne(buf_size, ind_buf_size));
    } catch (ReadError&) {}
    return ret;
  }

  // the destruction should be handled by caller
  size_t ReadBatch(T* ret, size_t num, size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    ParseBufSize(buf_size, ind_buf_size);
    size_t i = 0;
    try {
      for (; i < num; i++) ReadOne(ret + i, buf_size, ind_buf_size);
    } catch (ReadError&) {}
    return i;
  }

  void Seek(size_t location, size_t buf_size = std::string::npos, size_t ind_buf_size = std::string::npos) {
    ParseBufSize(buf_size, ind_buf_size);
    if (!eof && block_start <= location && location < block_start + items_per_index) {
      // same block
      if (location < current) {
        block_offset = 0;
        current = block_start;
        eof = false;
      }
    } else if (!eof && ind_buf.size() && ind_start <= location &&
               location < ind_start + (ind_buf.size() - 1) / 2 * items_per_index) {
      // different block, but in range of ind_buf
      size_t block_idx_offset = (location - ind_start) / items_per_index;
      ind_offset = block_idx_offset * 2;
      block_start = ind_start + block_idx_offset * items_per_index;
      block_buf.clear();
      block_offset = 0;
      current = block_start;
      SeekBuf(ind_buf[ind_offset]);
    } else {
      // out of ind_buf
      size_t block_idx = location / items_per_index;
      ind_buf.clear();
      ind_start = block_idx * items_per_index;
      ind_offset = 0;
      block_buf.clear();
      block_start = ind_start;
      block_offset = 0;
      current = ind_start;
      fin_index.clear();
      fin_index.seekg((block_idx * 2 + 1) * 8);
      ReadIndUntilSize(3, ind_buf_size);
      if (ind_buf.size() < 3) {
        eof = true;
        return;
      }
      SeekBuf(ind_buf[0]);
    }
    while (location > current) SkipOne(buf_size, ind_buf_size);
  }
};

template <class T>
std::vector<uint8_t> Serialize(const T& val) {
  std::vector<uint8_t> ret(val.NumBytes());
  val.GetBytes(ret.data());
  return ret;
}
