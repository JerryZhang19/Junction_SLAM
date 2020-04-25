//
// Created by gaoxiang on 19-5-2.
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace simpleslam {
class Map;

/**
 * Backend
 * Has a seperate thread, triggered when map is updated
 * Map update is triggered by Frontend
 */ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// Constructor, start backend thread and do nothing.
    Backend();

    void SetCamera(Camera::Ptr left) {
        cam_ = left;
    }

    /// Set Map
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// Trigger map update and start optimization
    void UpdateMap();

    /// Close backend tread
    void Stop();

   private:
    /// Bakend tread
    void BackendLoop();

    /// Do optimization
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H