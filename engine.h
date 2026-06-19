#ifndef ENGINE_H
#define ENGINE_H

#include <QObject>
#include <QTimer>
#include <QImage>
#include <QString>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>

// Upgraded Flat 32x32 Surface Matrix (1,024 elements)
struct CustomMatrix32 {
    int rows = 32;
    int cols = 32;
    std::vector<float> data;
    CustomMatrix32() : data(32 * 32, 0.0f) {}
    
    inline float get(int r, int c) const { return data[r * 32 + c]; }
    void set(int r, int c, float val) { data[r * 32 + c] = val; }
};

// Upgraded Flat 64x64x32 Volumetric Depth Tensor (131,072 elements)
struct VoxelVolume64 {
    std::vector<float> voxels;
    VoxelVolume64() : voxels(64 * 64 * 32, 0.0f) {}
    
    // Cache-localized 3D index flattener
    inline int to1D(int x, int y, int z) const { return (z * 64 * 64) + (y * 64) + x; }
    
    float getVoxel(int x, int y, int z) const { return voxels[to1D(x, y, z)]; }
    void setVoxel(int x, int y, int z, float val) { voxels[to1D(x, y, z)] = val; }
};

enum class TrackingState {
    LookingCenter,
    LookingUp,
    LookingDown,
    LookingLeft,
    LookingRight,
    TrainingMode, // 10-second multi-pose phase
    Complete
};

class FaceEngine : public QObject {
    Q_OBJECT
public:
    explicit FaceEngine(QObject *parent = nullptr);
    ~FaceEngine();

    bool initialize();
    void saveModel(const QString &filePath);

signals:
    void frameProcessed(const QImage &image, QString instruction, int percentage);

private slots:
    void processFrameAt30FPS(); // 30 FPS Camera & UI Loop

private:
    cv::VideoCapture videoCapture;
    QTimer *cameraTimer;
    cv::CascadeClassifier faceCascade;

    TrackingState currentState;
    int progressPercentage;
    int totalFramesTracked; 
    
    // Multi-threaded Infrastructure
    std::thread solverThread;
    std::mutex matrixMutex;
    std::atomic<bool> runSolver{true};
    std::atomic<bool> newDataAvailable{false};
    
    CustomMatrix32 sharedFaceMatrix; // Guarded by mutex
    VoxelVolume64 globalDepthVolume; // Written by 10 FPS background thread

    int trainingFrames30FPS;
    int trainingSteps10FPS;

    void runReverseAlgebraLoop10FPS(); 
    CustomMatrix32 cvMatToCustomMatrix32(const cv::Mat &src);
    void executeReverseLinearAlgebra(const CustomMatrix32& surfaceInput, VoxelVolume64& volumeOutput);
    void analyzeHeadPoseWithMatrix32(const CustomMatrix32 &faceMat, int &yaw, int &pitch);
    QImage matToQImage(const cv::Mat &mat);
};

#endif // ENGINE_H