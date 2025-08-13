#ifndef GECCADAPTERH
#define GECCADAPTERH

#include "Point.h"
#include "Int.h"
#include "gecc.h"
#include "gecc/arith/fp.h"
#include "gecc/arith/ec.h"
#include <chrono>
#include <map>
#include <string>

// gECC类型定义 - SECP256K1曲线
DEFINE_FP(Secp256k1Fp, SECP256K1_FP, u32, 32, ColumnMajorLayout<1>, 8);
DEFINE_EC(Secp256k1, Jacobian, Secp256k1Fp, SECP256K1_EC, 2);

using GeccField = Secp256k1Fp;
using GeccEC = Secp256k1_Jacobian;
using GeccAffine = GeccEC::Affine;

// 性能监控类
class PerformanceMonitor {
public:
    void StartTimer(const std::string& operation);
    void EndTimer(const std::string& operation);
    void ReportPerformance();
    void Reset();

private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> startTimes;
    std::map<std::string, double> totalTimes;
    std::map<std::string, uint64_t> operationCounts;
    mutable std::mutex mutex;
};

class GeccAdapter {
public:
    // 初始化和清理
    static bool Initialize();
    static void Cleanup();
    static bool IsInitialized() { return initialized; }

    // 坐标转换函数
    static GeccAffine ToGeccAffine(const Point& p);
    static GeccEC ToGeccJacobian(const Point& p);
    static Point FromGeccAffine(const GeccAffine& p);
    static Point FromGeccJacobian(const GeccEC& p);

    // 基础椭圆曲线运算
    static Point Add(const Point& p1, const Point& p2);
    static Point AddMixed(const Point& jacobian, const Point& affine);
    static Point Double(const Point& p);
    static Point Negate(const Point& p);
    static bool IsEqual(const Point& p1, const Point& p2);
    static bool IsZero(const Point& p);

    // 高级椭圆曲线运算
    static Point ScalarMult(const Int& scalar, const Point& base);
    static Point ScalarMultWindow(const Int& scalar, const Point& base, int windowSize = 4);
    static std::vector<Point> PrecomputeTable(const Point& base, int tableSize);

    // 批量运算
    static std::vector<Point> BatchAdd(
        const std::vector<Point>& p1,
        const std::vector<Point>& p2
    );
    static std::vector<Point> BatchDouble(const std::vector<Point>& points);
    static std::vector<Point> BatchScalarMult(
        const std::vector<Int>& scalars,
        const std::vector<Point>& bases
    );

    // GPU加速运算
    static bool InitializeGPU(int deviceId = 0);
    static std::vector<Point> GPUBatchScalarMult(
        const std::vector<Int>& scalars,
        const Point& base,
        int batchSize = 1024
    );

    // 性能监控和调试
    static void EnablePerfMonitoring(bool enable) { perfMonitoringEnabled = enable; }
    static void ReportPerformance() { if(perfMonitoringEnabled) perfMonitor.ReportPerformance(); }
    static void ResetPerformanceCounters() { perfMonitor.Reset(); }

    // 错误处理
    static std::string GetLastError() { return lastError; }
    static void ClearError() { lastError.clear(); }

    // 配置选项
    static void SetUseGPU(bool use) { useGPU = use; }
    static void SetBatchSize(int size) { batchSize = size; }
    static void SetWindowSize(int size) { windowSize = size; }

private:
    // 内部状态
    static bool initialized;
    static bool gpuInitialized;
    static bool perfMonitoringEnabled;
    static bool useGPU;
    static int batchSize;
    static int windowSize;
    static std::string lastError;
    static PerformanceMonitor perfMonitor;

    // 内部辅助函数
    static bool ValidatePoint(const Point& p);
    static bool ValidateScalar(const Int& scalar);
    static void SetError(const std::string& error);

    // GPU相关
    static void* gpuContext;
    static int gpuDeviceId;
};

// 性能测试宏
#define GECC_PERF_START(op) \
    if(GeccAdapter::perfMonitoringEnabled) GeccAdapter::perfMonitor.StartTimer(op)

#define GECC_PERF_END(op) \
    if(GeccAdapter::perfMonitoringEnabled) GeccAdapter::perfMonitor.EndTimer(op)

#endif // GECCADAPTERH
