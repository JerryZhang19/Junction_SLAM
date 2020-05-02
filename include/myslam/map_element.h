#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace simpleslam {

    struct Frame;
    struct Feature;
    struct Junction2D;

/**
 * Map elements class
 */
    struct MapPoint {
    private:
        Vec3 pos_ = Vec3::Zero();  // Position in world
        std::mutex data_mutex_;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0;  // ID
        bool is_outlier_ = false;
        int observed_times_ = 0;  // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_;

        MapPoint() {}

        MapPoint(long id, Vec3 position);

        Vec3 Pos() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Vec3 &pos) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        void AddObservation(std::shared_ptr<Feature> feature) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
    // namespace myslam


    struct Junction3D {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Junction3D> Ptr;
        unsigned long id_ = 0;
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();
        std::vector<Vec3> endpoints_;

        std::mutex data_mutex_;
        int observed_times_ = 0;  // being observed by feature matching algo.
        std::list<std::weak_ptr<Junction2D>> observations_;

        Junction3D(){}

        Junction3D(long id, const Vec3 &pos, const std::vector<Vec3>& endpoints) : id_(id), pos_(pos), endpoints_(endpoints){}

        Vec3 Pos() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Vec3 &pos) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        std::vector<Vec3> Endpoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return endpoints_;
        }

        void  SetEndpoints(const std::vector<Vec3> &endpoints) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            endpoints_ = endpoints;
        };

        int Degree(){
            std::unique_lock<std::mutex> lck(data_mutex_);
            return endpoints_.size();
        }

        void AddObservation(std::shared_ptr<Junction2D> j2d) {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(j2d);
            observed_times_++;
        }

        void RemoveObservation(std::shared_ptr<Junction2D> j2d);

        std::list<std::weak_ptr<Junction2D>> GetObs() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static Junction3D::Ptr CreateNewJunction3D();

    };
}
#endif  // MYSLAM_MAPPOINT_H
