#pragma once

#include <cstdlib>

#include "move_search.h"

void InspectBoard(int group, const std::vector<long>& board_idx);
void InspectBoardStats(int group);
void InspectEdge(int group, const std::vector<long>& board_idx, Level level, int piece);
void InspectEdgeStats(int group, Level level);
void InspectValue(int pieces, const std::vector<long>& board_idx);
void InspectBoard(const std::string& str);
void InspectMove(const std::string& str, int now_piece, int lines, bool notation = false);
void InspectMoveTree(const std::string& str, int lines, size_t depth);
