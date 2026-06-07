// On-device benchmark for the classical (emlearn C) models on the classic ESP32.
// Times DecisionTree and RandomForest inference and reports static RAM. No tensor arena
// is needed: the only working memory is the feature buffer plus a shallow call stack.
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "DecisionTree_uthar.h"
#include "RandomForest_uthar.h"

#define NFEAT 450               // UT-HAR statistical features = 90 subcarriers x 5 stats
static int16_t feats[NFEAT];

static void bench(const char *name, int32_t (*predict)(const int16_t*, int32_t)) {
    volatile int32_t acc = 0;
    acc += predict(feats, NFEAT);   // warm-up
    const int BATCHES = 20, PER = 50, N = BATCHES * PER;
    int64_t t = 0;
    for (int b = 0; b < BATCHES; b++) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < PER; i++) acc += predict(feats, NFEAT);
        t += esp_timer_get_time() - t0;
        vTaskDelay(1);              // feed the watchdog; not timed
    }
    printf("model = %s\n", name);
    printf("latency_us = %.3f\n", (double)t / N);
    printf("latency_ms = %.5f\n", (double)t / 1000.0 / N);
    printf("checksum = %d\n", (int)acc);
}

void app_main(void) {
    for (int i = 0; i < NFEAT; i++) feats[i] = (int16_t)((i * 7) % 11 - 5);
    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    bench("DecisionTree_uthar", DecisionTree_uthar_predict);
    bench("RandomForest_uthar", RandomForest_uthar_predict);
    size_t free_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    printf("feature_buffer_bytes = %d\n", (int)sizeof(feats));
    printf("heap_delta_bytes = %d  (0 = no dynamic allocation)\n", (int)(free_before - free_after));
    printf("DONE\n");
}
