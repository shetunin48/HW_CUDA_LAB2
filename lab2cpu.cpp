
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <chrono>


struct uchar4 {
    uint8_t x, y, z, w; // для хранения пикселей 
};

inline int clamp(int val, int max_val) {
    if (val < 0) return 0;
    if (val >= max_val) return max_val - 1;
    return val;
}


inline float get_y(const uchar4* data, int w, int h, int x, int y) {
    int cx = clamp(x, w);
    int cy = clamp(y, h);
    uchar4 p = data[cy * w + cx];
    return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}

int main(int argc, char* argv[]) {
    int blocksPerGrid = 1024;
    int threadsPerBlock = 1024;
    std::string filename = "";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            filename = argv[++i];
        } else if (std::strcmp(argv[i], "-grid") == 0 && i + 1 < argc) {
            blocksPerGrid = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-block") == 0 && i + 1 < argc) {
            threadsPerBlock = std::atoi(argv[++i]);
        }
    }

    char in_path[256], out_path[256];
    scanf("%s %s", in_path, out_path);

    int w, h;
    FILE *fp = fopen(in_path, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float g00 = get_y(data, w, h, x - 1, y - 1);  // достаточно ли float..
            float g10 = get_y(data, w, h, x, y - 1);
            float g20 = get_y(data, w, h, x + 1, y - 1);

            float g01 = get_y(data, w, h, x - 1, y);
            float g21 = get_y(data, w, h, x + 1, y);

            float g02 = get_y(data, w, h, x - 1, y + 1);
            float g12 = get_y(data, w, h, x, y + 1);
            float g22 = get_y(data, w, h, x + 1, y + 1);

            float gx = -g00 + g20 - 2.0 * g01 + 2.0 * g21 - g02 + g22;
            float gy = -g00 - 2.0 * g10 - g20 + g02 + 2.0 * g12 + g22;

            float g = sqrtf(gx * gx + gy * gy);
            if (g > 255.0) {
                g = 255.0;
            }

            uchar4 p = data[y * w + x];
            out[y * w + x] = uchar4{(uint8_t)g, (uint8_t)g, (uint8_t)g, (uint8_t)p.w};
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration elapsed = end - start;
    double time_ms = elapsed.count();

    std::cerr << "TIME_MS: " << time_ms << "\n";

    fp = fopen(out_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(out, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(out);
    return 0;
}
