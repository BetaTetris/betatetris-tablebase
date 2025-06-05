#include <cxxabi.h>
#include <thread>
#include <iostream>
#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>

#include "move.h"
#include "files.h"
#include "prune.h"
#include "config.h"
#include "server.h"
#include "inspect.h"
#include "evaluate.h"
#include "simulate.h"
#include "board_set.h"
#include "sample_svd.h"
#include "sample_train.h"

template <class T>
std::vector<T> ParseIntList(const std::string& str) {
  std::vector<T> ret;
  for (size_t i = 0, end; i < str.size(); i = end + 1) {
    end = str.find(',', i);
    if (end == std::string::npos) end = str.size();
    std::string sub = str.substr(i, end - i);
    size_t p0 = sub.find(':');
    if (p0 == std::string::npos) {
      ret.push_back(std::stol(sub));
      continue;
    }
    size_t p1 = sub.find(':', p0 + 1);
    T range_start = std::stol(sub.substr(0, p0));
    T range_end = std::stol(sub.substr(p0 + 1, p1));
    if (p1 == std::string::npos) {
      for (T j = range_start; j < range_end; j++) ret.push_back(j);
      continue;
    }
    T step = std::stol(sub.substr(p1 + 1));
    if (step == 0) continue;
    if (step > 0) {
      for (T j = range_start; j < range_end; j += step) ret.push_back(j);
    } else {
      for (T j = range_start; j > range_end; j += step) ret.push_back(j);
    }
  }
  return ret;
}

template <class Func>
void ReadBoards(Func&& func) {
  std::string board;
  while (true) {
    std::string line;
    bool notfin = (bool)std::getline(std::cin, line);
    if (!notfin || line[0] == '-' || line[0] == '=') {
      if (!notfin) line = "";
      func(board, line);
      board = "";
      if (!notfin) return;
      continue;
    }
    if (line.size() > 10) line.resize(10);
    if (line.size() < 10) line += std::string(10 - line.size(), ' ');
    board += line + '\n';
  }
}

Level ParseLevel(int level) {
  if (level == 18) return kLevel18;
  if (level >= 19 && level < 29) return kLevel19;
#ifdef NO_2KS
  if (level >= 29) return kLevel29;
#else
  if (level >= 29 && level < 39) return kLevel29;
  if (level >= 39) return kLevel39;
#endif
  throw std::out_of_range("Invalid level");
}

int ParsePiece(const std::string& str) {
  if (str.size() > 1) throw std::invalid_argument("Invalid piece");
  switch (str[0]) {
    case 'T': return 0;
    case 'J': return 1;
    case 'Z': return 2;
    case 'O': return 3;
    case 'S': return 4;
    case 'L': return 5;
    case 'I': return 6;
    case '0': case '1': case '2': case '3':
    case '4': case '5': case '6': return str[0] - '0';
  }
  throw std::invalid_argument("Invalid piece");
}

int main(int argc, char** argv) {
  spdlog::set_pattern("[%t] %+");
  spdlog::set_level(spdlog::level::debug);

  using namespace argparse;

  ArgumentParser program("main", "1.0");

  auto DataDirArg = [](ArgumentParser& parser) {
    parser.add_argument("data_dir")
      .help("Directory for storing all results");
  };
  auto ParallelArg = [](ArgumentParser& parser) {
    parser.add_argument("-p", "--parallel")
      .help("Number of worker threads to use")
      .metavar("N")
      .default_value((int)std::thread::hardware_concurrency())
      .scan<'i', int>();
  };
  auto GroupArg = [](ArgumentParser& parser) {
    parser.add_argument("-g", "--group").required()
      .help("Group (0-" + std::to_string(kGroups - 1) + ")")
      .metavar("GROUP")
      .scan<'i', int>();
  };
  auto BoardIDArg = [](ArgumentParser& parser) {
    parser.add_argument("-b", "--board-id").required()
      .help("Board IDs (comma-separated)")
      .metavar("ID");
  };
  auto LevelArg = [](ArgumentParser& parser) {
    parser.add_argument("-l", "--level").required()
#ifdef NO_2KS
      .help("Level (18,19,29)")
      .metavar("[18/19/29]")
#else
      .help("Level (18,19,29,39)")
      .metavar("[18/19/29/39]")
#endif
      .scan<'i', int>();
  };
  auto IOThreadsArg = [](ArgumentParser& parser) {
    parser.add_argument("-i", "--io-threads")
      .help("Number of readers")
      .metavar("N")
      .scan<'i', int>()
      .default_value(4);
  };
  auto ResumeArg = [](ArgumentParser& parser) {
    parser.add_argument("-r", "--resume")
      .help("Resume from a previous checkpoint")
      .metavar("PIECES")
      .scan<'i', int>()
      .default_value(-1);
  };
  auto UntilArg = [](ArgumentParser& parser) {
    parser.add_argument("-e", "--until")
      .help("Calculate until this many pieces")
      .metavar("PIECES")
      .scan<'i', int>()
      .default_value(0);
  };
  auto NumSamplesArg = [](ArgumentParser& parser) {
    parser.add_argument("-n", "--num-samples-per-group").required()
      .help("Number of samples for each group")
      .scan<'i', long>();
  };
  auto PowArg = [](ArgumentParser& parser) {
    parser.add_argument("--pow")
      .help("Exponent used for scaling to get more high-valued samples")
      .scan<'g', float>()
      .default_value(0.5f);
  };
  auto SeedArg = [](ArgumentParser& parser) {
    parser.add_argument("--seed")
      .help("Random seed")
      .scan<'i', long>()
      .default_value(0l);
  };
  auto MergeArgs = [](ArgumentParser& parser) {
    parser.add_argument("-s", "--start").required()
      .help("Start pieces")
      .metavar("PIECES")
      .scan<'i', int>()
      .default_value(-1);
    parser.add_argument("-e", "--end").required()
      .help("End pieces")
      .metavar("PIECES")
      .scan<'i', int>()
      .default_value(-1);
    parser.add_argument("-d", "--delete")
      .help("Delete files after merge")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-w", "--whole")
      .help("Whole merge")
      .default_value(false)
      .implicit_value(true);
  };
  auto ServerArgs = [](ArgumentParser& parser) {
    parser.add_argument("-b", "--bind")
      .help("Server bind address")
      .default_value("127.0.0.1");
    parser.add_argument("-p", "--port")
      .help("Server listen port")
      .scan<'i', int>()
      .default_value(3456);
    parser.add_argument("-x", "--exclusive")
      .help("Only one connection at a time")
      .default_value(false)
      .implicit_value(true);
  };

  ArgumentParser preprocess("preprocess", "", default_arguments::help);
  preprocess.add_description("Preprocess a board file");
  DataDirArg(preprocess);
  ParallelArg(preprocess);
  preprocess.add_argument("board_file")
    .help("Board file");

  ArgumentParser board_map("board-map", "", default_arguments::help);
  board_map.add_description("Generate board map");
  DataDirArg(board_map);
  ParallelArg(board_map);

  ArgumentParser build_edges("build-edges", "", default_arguments::help);
  build_edges.add_description("Build edges from boards");
  DataDirArg(build_edges);
  ParallelArg(build_edges);
  build_edges.add_argument("-g", "--groups")
    .help("The groups to build (0-" + std::to_string(kGroups - 1) + ", comma-separated, support Python-like range)")
    .metavar("GROUP")
    .default_value("0:" + std::to_string(kGroups));

  ArgumentParser evaluate("evaluate", "", default_arguments::help);
  evaluate.add_description("Calculate values of every board");
  DataDirArg(evaluate);
  ParallelArg(evaluate);
  IOThreadsArg(evaluate);
  ResumeArg(evaluate);
  evaluate.add_argument("-c", "--checkpoints").required()
    .help("Checkpoints (in pieces) to save the evaluate result (comma-separated, support Python-like range)");
  evaluate.add_argument("-s", "--store-sample")
    .help("Store sampled values (must have sample file available)")
    .default_value(false)
    .implicit_value(true);

  ArgumentParser move_cal("move", "", default_arguments::help);
  move_cal.add_description("Calculate moves of every board");
  DataDirArg(move_cal);
  ParallelArg(move_cal);
  IOThreadsArg(move_cal);
  ResumeArg(move_cal);
  UntilArg(move_cal);

  ArgumentParser move_merge("move-merge", "", default_arguments::help);
  move_merge.add_description("Merge moves");
  DataDirArg(move_merge);
  ParallelArg(move_merge);
  MergeArgs(move_merge);

  ArgumentParser threshold_cal("threshold", "", default_arguments::help);
  threshold_cal.add_description("Calculate EV level of every board");
  DataDirArg(threshold_cal);
  ParallelArg(threshold_cal);
  IOThreadsArg(threshold_cal);
  ResumeArg(threshold_cal);
  UntilArg(threshold_cal);
  threshold_cal.add_argument("-b", "--buckets").required()
    .help("Threshold file (contains base value of each lines)")
    .scan<'i', int>();
  threshold_cal.add_argument("-f", "--threshold-file").required()
    .help("Threshold file (contains base value of each lines)");
  threshold_cal.add_argument("-l", "--ratio-low").required()
    .help("Ratio of level 0")
    .scan<'g', float>();
  threshold_cal.add_argument("-h", "--ratio-high").required()
    .help("Ratio of level (buckets-1)")
    .scan<'g', float>();
  threshold_cal.add_argument("name").required()
    .help("Name of this threshold");

  ArgumentParser threshold_merge("threshold-merge", "", default_arguments::help);
  threshold_merge.add_description("Merge thresholds");
  DataDirArg(threshold_merge);
  ParallelArg(threshold_merge);
  MergeArgs(threshold_merge);
  threshold_merge.add_argument("name").required()
    .help("Name of this threshold");

  ArgumentParser mask_threshold("mask-threshold", "", default_arguments::help);
  mask_threshold.add_description("Get prune mask from values");
  DataDirArg(mask_threshold);
  ParallelArg(mask_threshold);
  IOThreadsArg(mask_threshold);
  ResumeArg(mask_threshold);
  UntilArg(mask_threshold);
  mask_threshold.add_argument("-f", "--start-mask-file")
    .help("Start mask file")
    .default_value("");
  mask_threshold.add_argument("-t", "--threshold").required()
    .help("Threshold value")
    .scan<'g', float>();
  mask_threshold.add_argument("mask_file").required()
    .help("Mask file");

  ArgumentParser sample_svd("sample-svd", "", default_arguments::help);
  sample_svd.add_description("Sample boards for SVD");
  DataDirArg(sample_svd);
  ParallelArg(sample_svd);
  IOThreadsArg(sample_svd);
  sample_svd.add_argument("-s", "--start-pieces").required()
    .help("Start pieces for sampling (must be a saved evaluate result)")
    .metavar("PIECES")
    .scan<'i', int>();
  NumSamplesArg(sample_svd);
  PowArg(sample_svd);
  SeedArg(sample_svd);

  ArgumentParser svd("svd", "", default_arguments::help);
  svd.add_description("Run SVD for value compression");
  svd.add_argument("-r", "--ranks")
    .help("The ranks to build (comma-separated, support Python-like range)")
    .default_value("1:65");
  svd.add_argument("-t", "--training-split")
    .help("Portion of training data")
    .scan<'g', float>()
    .default_value(0.5f);
  SeedArg(svd);
  svd.add_argument("ev_var").required()
    .help("Value type (ev/var)")
    .metavar("ev/var");
  DataDirArg(svd);

  ArgumentParser sample_train("sample-train", "", default_arguments::help);
  sample_train.add_description("Sample boards for training");
  DataDirArg(sample_train);
  ParallelArg(sample_train);
  IOThreadsArg(sample_train);
  sample_train.add_argument("-s", "--start-pieces").required()
    .help("Start pieces for sampling (must be a saved evaluate result; comma-separated)");
  sample_train.add_argument("-o", "--output").required()
    .help("Output file");
  sample_train.add_argument("-z", "--zero-ratio")
    .help("Portion of samples sampled from ev=0 nodes")
    .scan<'g', float>()
    .default_value(0.5f);
  sample_train.add_argument("-h", "--zero-high-ratio")
    .help("Portion of ev=0 samples sampled from nodes with the largest other-piece ev")
    .scan<'g', float>()
    .default_value(0.4f);
  NumSamplesArg(sample_train);
  PowArg(sample_train);
  SeedArg(sample_train);

  ArgumentParser fceux_server("fceux-server", "", default_arguments::help);
  fceux_server.add_description("Server for FCEUX");
  ServerArgs(fceux_server);
  DataDirArg(fceux_server);

  ArgumentParser board_server("board-server", "", default_arguments::help);
  board_server.add_description("Server for boards");
  ServerArgs(board_server);
  DataDirArg(board_server);
  board_server.add_argument("name")
    .help("Name of the threshold")
    .default_value("");

  ArgumentParser simulate("simulate", "", default_arguments::help);
  simulate.add_description("Simulate games");
  DataDirArg(simulate);
  ParallelArg(simulate);
  simulate.add_argument("-f", "--seed-file")
    .help("File containing seeds")
    .default_value("-");
  simulate.add_argument("-o", "--output-file")
    .help("Output CSV file")
    .default_value("-");
  simulate.add_argument("--gym-rng")
    .help("Use TetrisGYM RNG (seed is decimal)")
    .default_value(false)
    .implicit_value(true);
  simulate.add_argument("--dataset-file")
    .help("Output supervised data file")
    .default_value("");
  simulate.add_argument("--dataset-tag")
    .help("Tag of supervised data")
    .default_value(0)
    .scan<'i', int>();

  ArgumentParser inspect("inspect", "", default_arguments::help);
  inspect.add_description("Inspect files");

  ArgumentParser inspect_board("board", "", default_arguments::help);
  inspect_board.add_description("Get board ID by shape");
  DataDirArg(inspect_board);

  ArgumentParser inspect_board_id("board-id", "", default_arguments::help);
  inspect_board_id.add_description("Get board(s) by ID");
  GroupArg(inspect_board_id);
  BoardIDArg(inspect_board_id);
  DataDirArg(inspect_board_id);

  ArgumentParser inspect_board_stats("board-stats", "", default_arguments::help);
  inspect_board_stats.add_description("Get board stats");
  GroupArg(inspect_board_stats);
  DataDirArg(inspect_board_stats);

  ArgumentParser inspect_edge("edge", "", default_arguments::help);
  inspect_edge.add_description("Get edges of a node");
  GroupArg(inspect_edge);
  BoardIDArg(inspect_edge);
  LevelArg(inspect_edge);
  inspect_edge.add_argument("-p", "--piece").required()
    .help("Piece (0-6 or TJZOSLI)")
    .metavar("PIECE");
  DataDirArg(inspect_edge);

  ArgumentParser inspect_edge_stats("edge-stats", "", default_arguments::help);
  inspect_edge_stats.add_description("Get edge stats");
  GroupArg(inspect_edge_stats);
  LevelArg(inspect_edge_stats);
  DataDirArg(inspect_edge_stats);

  ArgumentParser inspect_value("value", "", default_arguments::help);
  inspect_value.add_description("Get values of a node");
  BoardIDArg(inspect_value);
  inspect_value.add_argument("-p", "--pieces").required()
    .help("Location (pieces)")
    .metavar("PIECES")
    .scan<'i', int>();
  DataDirArg(inspect_value);

  ArgumentParser inspect_move("move", "", default_arguments::help);
  inspect_move.add_description("Get move by shape");
  DataDirArg(inspect_move);

  inspect.add_subparser(inspect_board);
  inspect.add_subparser(inspect_board_id);
  inspect.add_subparser(inspect_board_stats);
  inspect.add_subparser(inspect_edge);
  inspect.add_subparser(inspect_edge_stats);
  inspect.add_subparser(inspect_value);
  inspect.add_subparser(inspect_move);

  program.add_subparser(preprocess);
  program.add_subparser(board_map);
  program.add_subparser(build_edges);
  program.add_subparser(evaluate);
  program.add_subparser(move_cal);
  program.add_subparser(move_merge);
  program.add_subparser(threshold_cal);
  program.add_subparser(threshold_merge);
  program.add_subparser(mask_threshold);
  program.add_subparser(sample_svd);
  program.add_subparser(svd);
  program.add_subparser(sample_train);
  program.add_subparser(fceux_server);
  program.add_subparser(board_server);
  program.add_subparser(simulate);
  program.add_subparser(inspect);

  try {
    program.parse_args(argc, argv);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    if (program.is_subcommand_used("preprocess")) {
      std::cerr << preprocess;
    } else if (program.is_subcommand_used("board-map")) {
      std::cerr << board_map;
    } else if (program.is_subcommand_used("build-edges")) {
      std::cerr << build_edges;
    } else if (program.is_subcommand_used("evaluate")) {
      std::cerr << evaluate;
    } else if (program.is_subcommand_used("move")) {
      std::cerr << move_cal;
    } else if (program.is_subcommand_used("move-merge")) {
      std::cerr << move_merge;
    } else if (program.is_subcommand_used("threshold")) {
      std::cerr << threshold_cal;
    } else if (program.is_subcommand_used("threshold-merge")) {
      std::cerr << threshold_merge;
    } else if (program.is_subcommand_used("mask-threshold")) {
      std::cerr << mask_threshold;
    } else if (program.is_subcommand_used("sample-svd")) {
      std::cerr << sample_svd;
    } else if (program.is_subcommand_used("svd")) {
      std::cerr << svd;
    } else if (program.is_subcommand_used("sample-train")) {
      std::cerr << sample_train;
    } else if (program.is_subcommand_used("fceux-server")) {
      std::cerr << fceux_server;
    } else if (program.is_subcommand_used("simulate")) {
      std::cerr << simulate;
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board")) {
        std::cerr << inspect_board;
      } else if (subparser.is_subcommand_used("board-id")) {
        std::cerr << inspect_board_id;
      } else if (subparser.is_subcommand_used("board-stats")) {
        std::cerr << inspect_board_stats;
      } else if (subparser.is_subcommand_used("edge")) {
        std::cerr << inspect_edge;
      } else if (subparser.is_subcommand_used("edge-stats")) {
        std::cerr << inspect_edge_stats;
      } else if (subparser.is_subcommand_used("value")) {
        std::cerr << inspect_value;
      } else if (subparser.is_subcommand_used("move")) {
        std::cerr << inspect_move;
      } else {
        std::cerr << inspect;
      }
    } else {
      std::cerr << program;
    }
    return 1;
  }

  auto SetParallel = [&](const ArgumentParser& args) {
    kParallel = args.get<int>("-p");
  };
  auto SetDataDir = [&](const ArgumentParser& args) {
    kDataDir = args.get<std::string>("data_dir");
  };
  auto SetIOThreads = [&](const ArgumentParser& args) {
    kIOThreads = args.get<int>("--io-threads");
  };
  auto GetGroup = [](const ArgumentParser& args) {
    return args.get<int>("--group");
  };
  auto GetResume = [](const ArgumentParser& args) {
    return args.get<int>("--resume");
  };
  auto GetBoardID = [](const ArgumentParser& args) {
    return ParseIntList<long>(args.get<std::string>("--board-id"));
  };
  auto GetLevel = [](const ArgumentParser& args) {
    return ParseLevel(args.get<int>("--level"));
  };
  auto GetNumSamples = [](const ArgumentParser& args) {
    return args.get<long>("--num-samples-per-group");
  };
  auto GetPow = [](const ArgumentParser& args) {
    return args.get<float>("--pow");
  };
  auto GetSeed = [](const ArgumentParser& args) {
    return args.get<long>("--seed");
  };

  try {
    if (program.is_subcommand_used("preprocess")) {
      auto& args = program.at<ArgumentParser>("preprocess");
      SetParallel(args);
      SetDataDir(args);
      std::filesystem::path board_file = args.get<std::string>("board_file");
      SplitBoards(board_file);
    } else if (program.is_subcommand_used("board-map")) {
      auto& args = program.at<ArgumentParser>("board-map");
      SetParallel(args);
      SetDataDir(args);
      WriteBoardMap();
    } else if (program.is_subcommand_used("build-edges")) {
      auto& args = program.at<ArgumentParser>("build-edges");
      SetParallel(args);
      SetDataDir(args);
      auto groups = ParseIntList<int>(args.get<std::string>("--groups"));
      BuildEdges(groups);
    } else if (program.is_subcommand_used("evaluate")) {
      auto& args = program.at<ArgumentParser>("evaluate");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      auto checkpoints = ParseIntList<int>(args.get<std::string>("--checkpoints"));
      bool sample = args.get<bool>("--store-sample");
      RunEvaluate(resume, checkpoints, sample);
    } else if (program.is_subcommand_used("move")) {
      auto& args = program.at<ArgumentParser>("move");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      int until = args.get<int>("--until");
      RunCalculateMoves(resume, until);
    } else if (program.is_subcommand_used("move-merge")) {
      auto& args = program.at<ArgumentParser>("move-merge");
      SetParallel(args);
      SetDataDir(args);
      int start = args.get<int>("--start");
      int end = args.get<int>("--end");
      bool whole = args.get<bool>("--whole");
      bool delete_after = args.get<bool>("--delete");
      if (whole) {
        MergeFullMoveRanges(delete_after);
      } else if (start != -1 || end != -1) {
        MergeMoveRanges(start, end, delete_after);
      } else {
        throw std::runtime_error("start / end not given");
      }
    } else if (program.is_subcommand_used("threshold")) {
      auto& args = program.at<ArgumentParser>("threshold");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      int until = args.get<int>("--until");
      std::string name = args.get<std::string>("name");
      std::string threshold_path = args.get<std::string>("--threshold-file");
      float start_ratio = args.get<float>("--ratio-low");
      float end_ratio = args.get<float>("--ratio-high");
      int buckets = args.get<int>("--buckets");
      buckets = std::max(3, std::min(255, buckets));
      RunCalculateThreshold(resume, until, name, threshold_path, start_ratio, end_ratio, buckets);
    } else if (program.is_subcommand_used("threshold-merge")) {
      auto& args = program.at<ArgumentParser>("threshold-merge");
      SetParallel(args);
      SetDataDir(args);
      std::string name = args.get<std::string>("name");
      int start = args.get<int>("--start");
      int end = args.get<int>("--end");
      bool whole = args.get<bool>("--whole");
      bool delete_after = args.get<bool>("--delete");
      if (whole) {
        MergeFullThresholdRanges(name, delete_after);
      } else if (start != -1 || end != -1) {
        MergeThresholdRanges(name, start, end, delete_after);
      } else {
        throw std::runtime_error("start / end not given");
      }
    } else if (program.is_subcommand_used("mask-threshold")) {
      auto& args = program.at<ArgumentParser>("mask-threshold");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      int until = args.get<int>("--until");
      std::string start_mask_path = args.get<std::string>("--start-mask-file");
      std::string mask_path = args.get<std::string>("mask_file");
      float threshold = args.get<float>("--threshold");
      ThresholdMask(
          start_mask_path == "" ? SameValueMask(kAllZeroValue) : ReadMask(start_mask_path),
          resume, until, threshold, mask_path);
    } else if (program.is_subcommand_used("sample-svd")) {
      auto& args = program.at<ArgumentParser>("sample-svd");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int pieces = args.get<int>("--start-pieces");
      long num_samples = GetNumSamples(args);
      float smooth_pow = GetPow(args);
      long seed = GetSeed(args);
      RunSample(pieces, num_samples, smooth_pow, seed);
    } else if (program.is_subcommand_used("svd")) {
      auto& args = program.at<ArgumentParser>("svd");
      SetDataDir(args);
      auto ranks = ParseIntList<int>(args.get<std::string>("--ranks"));
      float training_split = args.get<float>("--training-split");
      long seed = GetSeed(args);
      bool is_ev = args.get<std::string>("ev_var") == "ev";
      DoSVD(is_ev, training_split, ranks, seed);
    } else if (program.is_subcommand_used("sample-train")) {
      auto& args = program.at<ArgumentParser>("sample-train");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      auto pieces = ParseIntList<int>(args.get<std::string>("--start-pieces"));
      long num_samples = GetNumSamples(args);
      float smooth_pow = GetPow(args);
      float zero_ratio = args.get<float>("--zero-ratio");
      float zero_high_ratio = args.get<float>("--zero-high-ratio");
      long seed = GetSeed(args);
      std::filesystem::path output = args.get<std::string>("--output");
      SampleTrainingBoards(pieces, num_samples, zero_ratio, zero_high_ratio, smooth_pow, seed, output);
    } else if (program.is_subcommand_used("fceux-server")) {
      auto& args = program.at<ArgumentParser>("fceux-server");
      SetDataDir(args);
      int port = args.get<int>("--port");
      std::string addr = args.get<std::string>("--bind");
      bool one_conn = args.get<bool>("--exclusive");
      StartFCEUXServer(addr, port, one_conn);
    } else if (program.is_subcommand_used("board-server")) {
      auto& args = program.at<ArgumentParser>("board-server");
      SetDataDir(args);
      int port = args.get<int>("--port");
      std::string addr = args.get<std::string>("--bind");
      std::string threshold_name = args.get<std::string>("name");
      bool one_conn = args.get<bool>("--exclusive");
      StartBoardServer(addr, port, threshold_name, one_conn);
    } else if (program.is_subcommand_used("simulate")) {
      auto& args = program.at<ArgumentParser>("simulate");
      SetParallel(args);
      SetDataDir(args);
      std::string seed_file = args.get<std::string>("--seed-file");
      std::string output_file = args.get<std::string>("--output-file");
      std::string dataset_file = args.get<std::string>("--dataset-file");
      bool gym_rng = args.get<bool>("--gym-rng");
      int tag = args.get<int>("--dataset-tag");
      OutputSimulate(seed_file, output_file, gym_rng, dataset_file, (uint32_t)tag);
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board")) {
        auto& args = subparser.at<ArgumentParser>("board");
        SetDataDir(args);
        ReadBoards([](const std::string& str, const std::string&){ InspectBoard(str); });
      } else if (subparser.is_subcommand_used("board-id")) {
        auto& args = subparser.at<ArgumentParser>("board-id");
        auto group = GetGroup(args);
        auto board_id = GetBoardID(args);
        SetDataDir(args);
        InspectBoard(group, board_id);
      } else if (subparser.is_subcommand_used("board-stats")) {
        auto& args = subparser.at<ArgumentParser>("board-stats");
        auto group = GetGroup(args);
        SetDataDir(args);
        InspectBoardStats(group);
      } else if (subparser.is_subcommand_used("edge")) {
        auto& args = subparser.at<ArgumentParser>("edge");
        auto group = GetGroup(args);
        auto board_id = GetBoardID(args);
        Level level = GetLevel(args);
        int piece = ParsePiece(args.get<std::string>("--piece"));
        SetDataDir(args);
        InspectEdge(group, board_id, level, piece);
      } else if (subparser.is_subcommand_used("edge-stats")) {
        auto& args = subparser.at<ArgumentParser>("edge-stats");
        auto group = GetGroup(args);
        Level level = GetLevel(args);
        SetDataDir(args);
        InspectEdgeStats(group, level);
      } else if (subparser.is_subcommand_used("value")) {
        auto& args = subparser.at<ArgumentParser>("value");
        int pieces = args.get<int>("--pieces");
        auto board_id = GetBoardID(args);
        SetDataDir(args);
        InspectValue(pieces, board_id);
      } else if (subparser.is_subcommand_used("move")) {
        auto& args = subparser.at<ArgumentParser>("move");
        SetDataDir(args);
        ReadBoards([](const std::string& str, const std::string& line) {
          std::stringstream ss(line);
          std::string tmp;
          int piece, lines;
          ss >> tmp >> piece >> lines;
          InspectMove(str, piece, lines);
        });
      } else {
        std::cerr << inspect;
        return 1;
      }
    } else {
      std::cerr << program;
      return 1;
    }
  } catch (std::exception& e) {
    int status;
    char* type = abi::__cxa_demangle(abi::__cxa_current_exception_type()->name(), 0, 0, &status);
    spdlog::error("Exception {} when running command: {}", type, e.what());
    if (type) free(type);
    return 1;
  }
}
