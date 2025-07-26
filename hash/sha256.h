#ifndef SHA256_H
#define SHA256_H

#include <cstdint>

// SHA256 hash function
void sha256(const unsigned char* data, size_t len, unsigned char* hash);

#endif // SHA256_H
