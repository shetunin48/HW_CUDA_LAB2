
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdlib>

#define CSC(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)


__device__ float get_y(cudaTextureObject_t tex, float x, float y) {
    uchar4 p = tex2D<uchar4>(tex, x + 0.5, y + 0.5);
    return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;

    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            float g00 = get_y(tex, x - 1, y - 1); // достаточно ли float..
            float g10 = get_y(tex, x, y - 1);
            float g20 = get_y(tex, x + 1, y - 1);

            float g01 = get_y(tex, x - 1, y);
            float g21 = get_y(tex, x + 1, y);

            float g02 = get_y(tex, x - 1, y + 1);
            float g12 = get_y(tex, x, y + 1);
            float g22 = get_y(tex, x + 1, y + 1);

            float gx = -g00 + g20 - 2.0 * g01 + 2.0 * g21 - g02 + g22;
            float gy = -g00 - 2.0 * g10 - g20 + g02 + 2.0 * g12 + g22;

            float g = sqrtf(gx * gx + gy * gy);
            if (g > 255.0) {
                g = 255.0;
            }

            uchar4 p = tex2D<uchar4>(tex, x + 0.5, y + 0.5);
            out[y * w + x] = make_uchar4((unsigned char)g, (unsigned char)g, (unsigned char)g, p.w);
        }
    }
}


int main(int argc, char* argv[]) {
    int blocksPerGrid = 1024; // База
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

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));

    kernel<<< blocksPerGrid, threadsPerBlock >>>(tex, dev_out, w, h);

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    float time_ms = 0.0f;
    CSC(cudaEventElapsedTime(&time_ms, start, stop));

    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    fprintf(stderr, "TIME_MS: %f\n", time_ms);

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(out_path, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
