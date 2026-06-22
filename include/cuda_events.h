#pragma once

#include <cuda_runtime.h>

// Global CUDA Events

extern cudaEvent_t startEvent;

extern cudaEvent_t stopEvent;

// Create

void initCudaEvents();

// Destroy

void cleanupCudaEvents();