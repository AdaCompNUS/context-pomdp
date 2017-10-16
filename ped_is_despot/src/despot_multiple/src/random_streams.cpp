#include "random_streams.h"

RandomStreams::RandomStreams(int num_streams, int length, unsigned seed)
    : streams_(num_streams) {
  for (int i = 0; i < num_streams; i++) {
    unsigned i_seed = seed ^ i;
    rand_r(&i_seed);
    streams_[i].resize(length);
    for (int j = 0; j < length; j++)
      streams_[i][j] = (double)rand_r(&i_seed) / RAND_MAX;
  }
}
