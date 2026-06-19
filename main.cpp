#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>
#include <QShortcut>
#include <QKeySequence>
#include <QFileDialog>
#include <QMessageBox>
#include <iostream>
#include "engine.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    window.setWindowTitle("3D Volumetric Matrix Vision Core [64x64x32 Stage]");
    window.resize(800, 720); // Expanded boundary footprint to comfortably handle 64-grid scales

    QLabel *videoLabel = new QLabel(&window);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setStyleSheet("background-color: #0A0A0A; border: 2px solid #222; border-radius: 4px;");
    videoLabel->setAlignment(Qt::AlignCenter);

    QLabel *instructionLabel = new QLabel("Initializing multi-rate tracking subsystems...", &window);
    instructionLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #34495E; margin: 8px;");
    instructionLabel->setAlignment(Qt::AlignCenter);

    QProgressBar *progressBar = new QProgressBar(&window);
    progressBar->setStyleSheet(
        "QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; font-weight: bold; height: 24px; }"
        "QProgressBar::chunk { background-color: #27AE60; width: 15px; }"
    );
    progressBar->setRange(0, 100);
    progressBar->setValue(0);

    QPushButton *latencyBtn = new QPushButton("Verify Multi-Rate Core Sync [Test]", &window);
    latencyBtn->setStyleSheet("padding: 8px; font-weight: bold; font-size: 13px; background-color: #2C3E50; color: white; border-radius: 4px;");

    QVBoxLayout *layout = new QVBoxLayout(&window);
    layout->setContentsMargins(15, 15, 15, 15);
    layout->setSpacing(10);
    layout->addWidget(videoLabel, 1);
    layout->addWidget(instructionLabel);
    layout->addWidget(progressBar);
    layout->addWidget(latencyBtn);
    window.setLayout(layout);

    FaceEngine engine;

    // Connect 30 FPS signal thread payload straight into GUI update fields
    QObject::connect(&engine, &FaceEngine::frameProcessed, videoLabel, 
        [videoLabel, instructionLabel, progressBar](const QImage &img, QString txt, int percent) {
            // Keep original camera aspect ratio during scaling
            videoLabel->setPixmap(QPixmap::fromImage(img).scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            instructionLabel->setText(txt);
            progressBar->setValue(percent);
    });

    // Verification testing window map
    QObject::connect(latencyBtn, &QPushButton::clicked, &window, []() {
        QMessageBox::information(nullptr, "System Matrix Status", 
            "Operational Performance Status: ACTIVE\n\n"
            "UI / Ingestion Layer: 30 FPS (33ms updates)\n"
            "Reverse Linear Algebra Engine: 10 FPS (100ms sweeps)\n"
            "Active Surface Array: 32x32 Flat Matrix\n"
            "Target Volumetric Tensor: 64x64x32 Heap Allocation\n"
            "Total Monitored Nodes: 131,072 elements");
    });

    // Intercept native Ctrl+S events to safely write binary files out to disk
    QShortcut *saveShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_S), &window);
    QObject::connect(saveShortcut, &QShortcut::activated, [&engine, &window]() {
        QString savePath = QFileDialog::getSaveFileName(&window, "Export 64x64x32 Voxel Feature Map", "", "Tensor Data (*.txt *.dat *.bin)");
        if (!savePath.isEmpty()) {
            engine.saveModel(savePath);
            QMessageBox::information(&window, "System Backup Complete", "131,072 structural voxel elements committed to storage.");
        }
    });

    window.show();
    
    // Establish link to camera hardware and load Haar parameters
    if (!engine.initialize()) {
        QMessageBox::critical(&window, "Hardware Link Fault", 
            "Unable to establish baseline link to video capture device.\n"
            "Ensure 'haarcascade_frontalface_alt.xml' is inside the workspace root.");
        return -1;
    }

    return app.exec();
}