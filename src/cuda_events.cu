#include "cuda_events.h"

cudaEvent_t startEvent;
cudaEvent_t stopEvent;

void initCudaEvents()
{
    cudaEventCreate(
        &startEvent
    );

    cudaEventCreate(
        &stopEvent
    );
}

void cleanupCudaEvents()
{
    cudaEventDestroy(
        startEvent
    );

    cudaEventDestroy(
        stopEvent
    );
}