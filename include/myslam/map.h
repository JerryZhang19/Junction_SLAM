#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map_element.h"

namespace simpleslam {

/**
 * @brief Map
 * Interaction with Map：
 * Frontend calls InsertKeyframe and InsertMapPoint, Backend manage structure of map, mark as outlier and remove.
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;
    typedef std::unordered_map<unsigned long, Junction3D::Ptr> JunctionsType;

    Map() {}

    void InsertKeyFrame(Frame::Ptr frame);

    void InsertMapPoint(MapPoint::Ptr map_point);

    void InsertMapJunction(Junction3D::Ptr);


    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }

    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    JunctionsType GetAllJunctions() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return junctions_;
    }

    JunctionsType GetactiveJUnctions() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_junctions_;
    }

    void CleanMap();

   private:
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;         // all landmarks
    LandmarksType active_landmarks_;  // active landmarks
    KeyframesType keyframes_;         // all key-frames
    KeyframesType active_keyframes_;  // all key-frames
    JunctionsType junctions_;         // all landmarks
    JunctionsType active_junctions_;  // active landmarks

    Frame::Ptr current_frame_ = nullptr;

    // settings
    int num_active_keyframes_ = 7;  // number of activated keyframes
};
}  // namespace myslam

#endif  // MAP_H
