#pragma once
#include <string>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <nvml.h>

// ============================================================
// BenchResult: 所有 12 个字段的聚合结构体
// ============================================================
struct BenchResult {
    std::string kernel_name;        //测试核函数名称
    std::string version;            //版本
    std::string dtype;              //数据类型
    std::string shape;              //输入形状
    float time_ms_median = 0.0f;    //中位数
    float time_ms_mean   = 0.0f;    //平均值
    float time_ms_stddev = 0.0f;    //标准差
    float bandwidth_GBps = 0.0f;    //带宽
    float tflops         = 0.0f;    //浮点运算性能
    float pct_of_peak    = 0.0f;    //峰值百分比
    float pct_of_ref     = 0.0f;    //参考百分比
    int   sm_clock_MHz   = 0;       //SM 时钟频率
    std::string timestamp;          //时间戳
    std::string git_commit;         //Git 提交 ID
};

// ============================================================
// BenchReporter: 收集信息 + CSV 输出
// ============================================================
class BenchReporter {
public:
    // --- 手动设置 ---
    void set_metadata(const std::string& name, const std::string& ver,
                      const std::string& dtype, const std::string& shape) {
        r.kernel_name = name;
        r.version     = ver;
        r.dtype       = dtype;
        r.shape       = shape;
    }

    void set_time(float median, float mean, float stddev) {
        r.time_ms_median = median;
        r.time_ms_mean   = mean;
        r.time_ms_stddev = stddev;
    }

    void set_performance(float bw, float tf, float pct_peak, float pct_ref) {
        r.bandwidth_GBps = bw;
        r.tflops         = tf;
        r.pct_of_peak    = pct_peak;
        r.pct_of_ref     = pct_ref;
    }

    // --- 自动获取 ---
    void auto_fill() {
        r.timestamp  = get_timestamp();
        r.git_commit = get_git_commit();
        r.sm_clock_MHz = get_sm_clock();
    }

    // --- 输出 ---
    // 打印 CSV 表头（只调用一次）
    static void print_header() {
        std::printf("kernel_name,version,dtype,shape,"
                    "time_ms_median,time_ms_mean,time_ms_stddev,"
                    "bandwidth_GBps,tflops,pct_of_peak,pct_of_ref,"
                    "sm_clock_MHz,timestamp,git_commit\n");
    }

    // 打印一行 CSV
    void print_csv() const {
        std::printf("%s,%s,%s,%s,"
                    "%.3f,%.3f,%.3f,"
                    "%.2f,%.2f,%.2f,%.2f,"
                    "%d,%s,%s\n",
                    r.kernel_name.c_str(), r.version.c_str(),
                    r.dtype.c_str(), r.shape.c_str(),
                    r.time_ms_median, r.time_ms_mean, r.time_ms_stddev,
                    r.bandwidth_GBps, r.tflops,
                    r.pct_of_peak, r.pct_of_ref,
                    r.sm_clock_MHz,
                    r.timestamp.c_str(), r.git_commit.c_str());
    }

    // 追加写入 CSV 文件
    void save_csv(const std::string& path) const {
        std::ofstream f(path, std::ios::app);
        if (!f.is_open()) {
            std::fprintf(stderr, "Failed to open %s\n", path.c_str());
            return;
        }
        f << r.kernel_name << "," << r.version << ","
          << r.dtype << "," << r.shape << ","
          << r.time_ms_median << "," << r.time_ms_mean << ","
          << r.time_ms_stddev << ","
          << r.bandwidth_GBps << "," << r.tflops << ","
          << r.pct_of_peak << "," << r.pct_of_ref << ","
          << r.sm_clock_MHz << ","
          << r.timestamp << "," << r.git_commit << "\n";
    }

    // 打印可读摘要（人类友好，非 CSV）
    void print_summary() const {
        std::printf("\n");
        std::printf("  Benchmark: %s (%s)  dtype=%s  shape=%s\n",
                    r.kernel_name.c_str(), r.version.c_str(),
                    r.dtype.c_str(), r.shape.c_str());
        std::printf("  ┌─────────────────┬──────────────┐\n");
        std::printf("  │ Time (median)   │ %10.3f ms│\n", r.time_ms_median);
        std::printf("  │ Time (mean)     │ %10.3f ms│\n", r.time_ms_mean);
        std::printf("  │ Stddev          │ %10.3f ms│\n", r.time_ms_stddev);
        std::printf("  ├─────────────────┼──────────────┤\n");
        if (r.bandwidth_GBps > 0)
            std::printf("  │ Bandwidth       │ %10.2f GB/s│\n", r.bandwidth_GBps);
        if (r.tflops > 0)
            std::printf("  │ TFLOPS          │ %10.2f     │\n", r.tflops);
        if (r.pct_of_peak > 0)
            std::printf("  │ Pct of peak     │ %9.1f %% │\n", r.pct_of_peak);
        if (r.pct_of_ref > 0)
            std::printf("  │ Pct of ref      │ %9.1f %% │\n", r.pct_of_ref);
        std::printf("  ├─────────────────┼──────────────┤\n");
        std::printf("  │ SM Clock        │ %10d MHz│\n", r.sm_clock_MHz);
        std::printf("  │ Git commit      │ %12s │\n", r.git_commit.c_str());
        std::printf("  └─────────────────┴──────────────┘\n");
        std::printf("\n");
    }

private:
    BenchResult r;

    static std::string get_timestamp() { // 获取当前时间戳
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        return std::string(buf);
    }

    static std::string get_git_commit() { // 获取当前 Git 提交 ID
        char buf[64] = {0};
        FILE* fp = popen("git log -1 --format=%h 2>/dev/null", "r");
        if (fp) {
            if (fgets(buf, sizeof(buf), fp)) {
                // 去掉末尾换行
                size_t len = strlen(buf);
                if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
            }
            pclose(fp);
        }
        if (buf[0] == '\0') return "N/A";
        return std::string(buf);
    }

    static int get_sm_clock() {// 获取当前 SM 频率（MHz）
        // 优先用 NVML 直接读取（比 popen 快得多）
        nvmlInit();
        nvmlDevice_t dev;
        unsigned int mhz = 0;
        if (nvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS) {
            nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
        }
        nvmlShutdown();
        return static_cast<int>(mhz);
    }
};