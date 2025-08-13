#include "GeccAdapter.h"
#include "../CommonUtils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>

// 静态成员初始化
bool GeccAdapter::initialized = false;
bool GeccAdapter::gpuInitialized = false;
bool GeccAdapter::perfMonitoringEnabled = false;
bool GeccAdapter::useGPU = false;
int GeccAdapter::batchSize = 1024;
int GeccAdapter::windowSize = 4;
std::string GeccAdapter::lastError;
PerformanceMonitor GeccAdapter::perfMonitor;
void* GeccAdapter::gpuContext = nullptr;
int GeccAdapter::gpuDeviceId = 0;

// 初始化gECC库
bool GeccAdapter::Initialize() {
    if(initialized) return true;

    try {
        // 初始化gECC有限域
        GeccField::initialize();

        // 初始化椭圆曲线
        GeccEC::initialize();

        initialized = true;
        ClearError();

        std::cout << "gECC库初始化成功" << std::endl;
        return true;

    } catch(const std::exception& e) {
        SetError("gECC初始化失败: " + std::string(e.what()));
        return false;
    }
}

void GeccAdapter::Cleanup() {
    // 目前gECC库没有提供清理函数，此函数备用
}

// 坐标转换: Kangaroo Point -> gECC Affine
GeccAffine GeccAdapter::ToGeccAffine(const Point& p) {
    if(!ValidatePoint(p) || p.isZero()) {
        return GeccAffine::zero();
    }

    GeccAffine result;

    // The Int class uses 64-bit limbs, while gECC's Secp256k1Fp uses 32-bit limbs.
    // We need to convert from one to the other.
    // NB64BLOCK is the number of 64-bit limbs in Int (typically 4 for 256-bit).
    // GeccField::LIMBS is the number of 32-bit limbs in gECC's field element (typically 8 for 256-bit).

    static_assert(NB64BLOCK * 2 == GeccField::LIMBS, "Limb size mismatch between Int and GeccField");

    for (int i = 0; i < NB64BLOCK; i++) {
        result.x.digits[2 * i] = (uint32_t)p.x.bits64[i];
        result.x.digits[2 * i + 1] = (uint32_t)(p.x.bits64[i] >> 32);
        result.y.digits[2 * i] = (uint32_t)p.y.bits64[i];
        result.y.digits[2 * i + 1] = (uint32_t)(p.y.bits64[i] >> 32);
    }

    // 转换为Montgomery形式
    result.x = result.x.inplace_to_montgomery();
    result.y = result.y.inplace_to_montgomery();

    return result;
}

// 坐标转换: gECC Affine -> Kangaroo Point
Point GeccAdapter::FromGeccAffine(const GeccAffine& p) {
    Point result;
    if(p.is_zero()) {
        result.Clear();
        return result;
    }

    // 从Montgomery形式转换
    GeccField x_normal = p.x.from_montgomery();
    GeccField y_normal = p.y.from_montgomery();

    // Combine 32-bit limbs into 64-bit limbs
    static_assert(NB64BLOCK * 2 == GeccField::LIMBS, "Limb size mismatch between Int and GeccField");

    for (int i = 0; i < NB64BLOCK; i++) {
        result.x.bits64[i] = (uint64_t)x_normal.digits[2 * i] | ((uint64_t)x_normal.digits[2 * i + 1] << 32);
        result.y.bits64[i] = (uint64_t)y_normal.digits[2 * i] | ((uint64_t)y_normal.digits[2 * i + 1] << 32);
    }

    // Set z coordinate to 1 for affine points
    result.z.SetInt32(1);

    return result;
}

GeccEC GeccAdapter::ToGeccJacobian(const Point& p) {
    if (p.isZero()) {
        return GeccEC::zero();
    }
    // Assuming the point p is in affine form (Z=1)
    return ToGeccAffine(p).to_jacobian();
}

Point GeccAdapter::FromGeccJacobian(const GeccEC& p) {
    if (p.is_zero()) {
        Point pt;
        pt.Clear();
        return pt;
    }
    // This is a costly operation, should be used sparingly
    return FromGeccAffine(p.to_affine());
}


// 高效椭圆曲线加法
Point GeccAdapter::Add(const Point& p1, const Point& p2) {
    if (!initialized) Initialize();
    if (p1.isZero()) return p2;
    if (p2.isZero()) return p1;

    GeccAffine gecc_p1 = ToGeccAffine(p1);
    GeccAffine gecc_p2 = ToGeccAffine(p2);

    // gECC's `+` operator handles the addition logic, including point doubling if p1 == p2.
    GeccAffine gecc_result = gecc_p1 + gecc_p2;

    return FromGeccAffine(gecc_result);
}

Point GeccAdapter::AddMixed(const Point& jacobian_p, const Point& affine_p) {
    if (!initialized) Initialize();
    if (jacobian_p.isZero()) return affine_p;
    if (affine_p.isZero()) return jacobian_p;

    GeccEC jacobian = ToGeccJacobian(jacobian_p);
    GeccAffine affine = ToGeccAffine(affine_p);

    // Use gECC's mixed addition
    GeccEC result = jacobian + affine;

    return FromGeccJacobian(result);
}

// 高效椭圆曲线倍点
Point GeccAdapter::Double(const Point& p) {
    if (!initialized) Initialize();
    if (p.isZero()) return p;

    GeccAffine gecc_p = ToGeccAffine(p);

    // Use gECC's affine doubling
    GeccAffine gecc_result = gecc_p.affine_dbl();

    return FromGeccAffine(gecc_result);
}

Point GeccAdapter::Negate(const Point& p) {
    Point result = p;
    result.y.ModNeg();
    return result;
}

bool GeccAdapter::IsEqual(const Point& p1, const Point& p2) {
    return p1.equals(p2);
}

bool GeccAdapter::IsZero(const Point& p) {
    return p.isZero();
}


// 高效标量乘法 (使用窗口方法)
Point GeccAdapter::ScalarMult(const Int& scalar, const Point& base) {
    if (!initialized) Initialize();
    if (scalar.IsZero() || base.isZero()) {
        Point zero;
        zero.Clear();
        return zero;
    }

    // 使用窗口方法进行标量乘法
    return ScalarMultWindow(scalar, base, windowSize);
}

// 窗口方法标量乘法实现 (当前为简单的double-and-add)
Point GeccAdapter::ScalarMultWindow(const Int& scalar, const Point& base, int windowSize) {
    GeccEC result = GeccEC::zero();
    GeccAffine base_affine = ToGeccAffine(base);

    // Get the bit length of the scalar
    int bitLength = scalar.GetBitLength();

    // Perform double-and-add
    for (int i = bitLength - 1; i >= 0; i--) {
        result = result.dbl();
        if (scalar.GetBit(i)) {
            result = result + base_affine;
        }
    }

    return FromGeccJacobian(result);
}


std::vector<Point> GeccAdapter::PrecomputeTable(const Point& base, int tableSize) {
    // Implement precomputation table generation if needed
    return std::vector<Point>();
}

// 批量椭圆曲线加法
std::vector<Point> GeccAdapter::BatchAdd(
    const std::vector<Point>& p1,
    const std::vector<Point>& p2
) {
    //GECC_PERF_START("ECC_BatchAdd");

    if(p1.size() != p2.size()) {
        SetError("批量加法: 输入向量大小不匹配");
        return {};
    }

    std::vector<Point> results;
    results.reserve(p1.size());

    if(useGPU && gpuInitialized && p1.size() >= batchSize) {
        // 使用GPU加速批量运算
        //results = GPUBatchAdd(p1, p2);
    } else {
        // 使用CPU批量运算
        for(size_t i = 0; i < p1.size(); i++) {
            results.push_back(Add(p1[i], p2[i]));
        }
    }

    //GECC_PERF_END("ECC_BatchAdd");
    return results;
}

std::vector<Point> GeccAdapter::BatchDouble(const std::vector<Point>& points) {
    // Implement batch double if needed
    return std::vector<Point>();
}

std::vector<Point> GeccAdapter::BatchScalarMult(
    const std::vector<Int>& scalars,
    const std::vector<Point>& bases
) {
    // Implement batch scalar multiplication if needed
    return std::vector<Point>();
}


bool GeccAdapter::InitializeGPU(int deviceId) {
    // Implement GPU initialization if needed
    return false;
}

std::vector<Point> GeccAdapter::GPUBatchScalarMult(
    const std::vector<Int>& scalars,
    const Point& base,
    int batchSize
) {
    // Implement GPU batch scalar multiplication if needed
    return std::vector<Point>();
}


void GeccAdapter::SetError(const std::string& error) {
    lastError = error;
    // Optionally log the error
    // CommonUtils::reportError("GeccAdapter", error);
}

bool GeccAdapter::ValidatePoint(const Point& p) {
    // Add point validation logic if necessary
    return true;
}

bool GeccAdapter::ValidateScalar(const Int& scalar) {
    // Add scalar validation logic if necessary
    return true;
}


// 性能监控实现
void PerformanceMonitor::StartTimer(const std::string& operation) {
    std::lock_guard<std::mutex> lock(mutex);
    startTimes[operation] = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::EndTimer(const std::string& operation) {
    auto endTime = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(mutex);
    auto it = startTimes.find(operation);
    if(it != startTimes.end()) {
        auto duration = std::chrono::duration<double>(endTime - it->second).count();
        totalTimes[operation] += duration;
        operationCounts[operation]++;
        startTimes.erase(it);
    }
}

void PerformanceMonitor::ReportPerformance() {
    std::lock_guard<std::mutex> lock(mutex);

    std::cout << "\n=== gECC性能报告 ===" << std::endl;
    std::cout << std::setw(20) << "操作"
              << std::setw(15) << "总时间(s)"
              << std::setw(10) << "次数"
              << std::setw(15) << "平均时间(ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for(const auto& pair : totalTimes) {
        const std::string& op = pair.first;
        double totalTime = pair.second;
        uint64_t count = operationCounts[op];
        double avgTime = (count > 0) ? (totalTime * 1000.0 / count) : 0.0;

        std::cout << std::setw(20) << op
                  << std::setw(15) << std::fixed << std::setprecision(6) << totalTime
                  << std::setw(10) << count
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                  << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
}

void PerformanceMonitor::Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    startTimes.clear();
    totalTimes.clear();
    operationCounts.clear();
}
