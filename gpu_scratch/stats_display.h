#pragma once
#include "control.h"
#include "numeric_typedefs.h"

#include <chrono>

void initStats();
void nextSection(const std::chrono::nanoseconds);
void updateCounters(const u64);
void updateDuration();
void setSubstatus(const char *);
void setIteration(const u64);