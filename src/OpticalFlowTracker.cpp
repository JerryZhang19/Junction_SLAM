//
// Created by jerry on 5/9/20.
//

#include "myslam/OpticalFlowTracker.h"
#include "myslam/feature.h"
using namespace std;
using namespace cv;

namespace simpleslam {

    inline float GetPixelValue(const cv::Mat &img, float x, float y) {
        // boundary check
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= img.cols) x = img.cols - 1;
        if (y >= img.rows) y = img.rows - 1;
        uchar *data = &img.data[int(y) * img.step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[img.step] +
                xx * yy * data[img.step + 1]
        );
    }


    void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
        // parameters
        int half_patch_size = 4;
        int iterations = 10;
        for (size_t i = range.start; i < range.end; i++) {
            auto kp = kp1[i];
            double dx = 0, dy = 0; // dx,dy need to be estimated
            if (has_initial) {
                dx = kp2[i].pt.x - kp.pt.x;
                dy = kp2[i].pt.y - kp.pt.y;
            }

            double cost = 0, lastCost = 0;
            bool succ = true; // indicate if this point succeeded

            // Gauss-Newton iterations
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
            Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
            Eigen::Vector2d J;  // jacobian
            for (int iter = 0; iter < iterations; iter++) {
                if (inverse == false) {
                    H = Eigen::Matrix2d::Zero();
                    b = Eigen::Vector2d::Zero();
                } else {
                    // only reset b
                    b = Eigen::Vector2d::Zero();
                }

                cost = 0;

                // compute cost and jacobian
                for (int x = -half_patch_size; x < half_patch_size; x++)
                    for (int y = -half_patch_size; y < half_patch_size; y++) {
                        double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                       GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                        if (inverse == false) {
                            J = -1.0 * Eigen::Vector2d(
                                    0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                           GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                    0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                           GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                            );
                        } else if (iter == 0) {
                            // in inverse mode, J keeps same for all iterations
                            // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                            J = -1.0 * Eigen::Vector2d(
                                    0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                           GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                    0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                           GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                            );
                        }
                        // compute H, b and set cost;
                        b += -error * J;
                        cost += error * error;
                        if (inverse == false || iter == 0) {
                            H += J * J.transpose();
                        }
                    }
                // compute update

                Eigen::Vector2d update;
                Eigen::EigenSolver<Eigen::Matrix2d> es(H);
                auto eigenvalues = es.eigenvalues();
                double lambda1 = eigenvalues[0].real();
                double lambda2 = eigenvalues[1].real();
                double lambda_max = lambda1 > lambda2 ? lambda1 : lambda2;
                double lambda_min = lambda1 < lambda2 ? lambda1 : lambda2;
                double ratio = lambda_max / lambda_min;

                ostream_mux.lock();
                //cout << ratio << endl;
                ostream_mux.unlock();

                if (is_edge&&(ratio>500||lambda_min==0)) {
                    int index = lambda1 > lambda2 ? 0 : 1;
                    Eigen::Vector2d v = es.eigenvectors().col(index).real();
                    update = (v.dot(b) / lambda_max) * v;
                } else {
                    update = H.ldlt().solve(b);
                }


                if (std::isnan(update[0])) {
                    cout << "update is nan" << endl;
                    succ = false;
                    break;
                }

                if (iter > 0 && cost > lastCost) {
                    break;
                }

                // update dx, dy
                dx += update[0];
                dy += update[1];
                lastCost = cost;
                succ = true;

                if (update.norm() < 1e-2) {
                    // converge
                    break;
                }
            }

            success[i] = succ;

            kp2[i].pt = kp.pt + Point2f(dx, dy);
        }
    }


    void OpticalFlowSingleLevel(
            const Mat &img1,
            const Mat &img2,
            const vector<KeyPoint> &kp1,
            vector<KeyPoint> &kp2,
            vector<bool> &success,
            bool inverse, bool has_initial, bool is_edge) {
        kp2.resize(kp1.size());
        success.resize(kp1.size());
        OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial, is_edge);
        cv::parallel_for_(Range(0, kp1.size()),
                          std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
    }


    void OpticalFlowMultiLevel(
            const Mat &img1,
            const Mat &img2,
            const vector<KeyPoint> &kp1,
            vector<KeyPoint> &kp2,
            vector<bool> &success,
            bool inverse, bool is_edge) {

        // parameters
        int pyramids;
        double pyramid_scale;
        vector<double> scales;

        if(is_edge)
        {
            pyramids = 1;
            pyramid_scale = 0.5;
            scales = {1.0}; // effective gradient scale for edge usually small
        }
        else{
            pyramids = 4;
            pyramid_scale = 0.5;
            scales = {1.0,0.5,0.25,0.125};
        }

        // create pyramids
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        vector<Mat> pyr1, pyr2; // image pyramids
        for (int i = 0; i < pyramids; i++) {
            if (i == 0) {
                pyr1.push_back(img1);
                pyr2.push_back(img2);
            } else {
                Mat img1_pyr, img2_pyr;
                cv::resize(pyr1[i - 1], img1_pyr,
                           cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
                cv::resize(pyr2[i - 1], img2_pyr,
                           cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
                pyr1.push_back(img1_pyr);
                pyr2.push_back(img2_pyr);
            }
        }
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        //cout << "build pyramid time: " << time_used.count() << endl;

        // coarse-to-fine LK tracking in pyramids
        vector<KeyPoint> kp1_pyr, kp2_pyr;
        for (auto &kp:kp1) {
            auto kp_top = kp;
            kp_top.pt *= scales[pyramids - 1];
            kp1_pyr.push_back(kp_top);
            kp2_pyr.push_back(kp_top);
        }

        for (int level = pyramids - 1; level >= 0; level--) {
            // from coarse to fine
            success.clear();
            t1 = chrono::steady_clock::now();
            OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true, is_edge);
            t2 = chrono::steady_clock::now();
            auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            //cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

            if (level > 0) {
                for (auto &kp: kp1_pyr)
                    kp.pt /= pyramid_scale;
                for (auto &kp: kp2_pyr)
                    kp.pt /= pyramid_scale;
            }
        }

        for(int i=0;i<kp2_pyr.size();i++)
        {
            auto kp = kp2_pyr[i];
            if(kp.pt.x>img2.cols)
                success[i]=false;
            if(kp.pt.y>img2.rows)
                success[i]=false;
            kp2.push_back(kp);
        }

    }

    void TrackEdge(
            const Mat &img1,
            const Mat &img2,
            const Vec2& position,
            const Vec2& endpoint,
            vector<Vec2> &kps
    )
    {
        vector<KeyPoint> kp1;
        vector<KeyPoint> kp2;
        vector<bool> succ;
        for(int i=1;i<10;i++)
        {
            float x = position[0]/10*(10-i)+endpoint[0]/10*i;
            float y = position[1]/10*(10-i)+endpoint[1]/10*i;
            kp1.push_back(KeyPoint(x,y,0));
        }
        OpticalFlowMultiLevel(img1, img2, kp1, kp2, succ, false,true);
        for (int i=0;i<kp2.size();i++)
        {
            if(succ[i])
                kps.push_back({kp2[i].pt.x,kp2[i].pt.y});
        }
    }

    bool BoundaryCheck(const Mat& img, Vec2 pos)
    {
        if(pos[0]>=0&&pos[0]<img.cols&&pos[1]>=0&&pos[1]<img.rows)
            return true;
        return false;
    }



    void TrackCenter(
            const Mat &img1,
            const Mat &img2,
            const Vec2& position,
            KeyPoint &kp,
            bool &succ
    )
    {
        vector<KeyPoint> kps;
        vector<bool> success;
        OpticalFlowMultiLevel(img1, img2, {KeyPoint(position[0],position[1],0)}, kps, success, false,false);
        succ = success[0];
        kp = kps[0];
    }

    std::shared_ptr<Junction2D> TrackJunction(
            const Mat &img1,
            const Mat &img2,
            std::shared_ptr<Junction2D> junction1
            )
    {
        Vec2 center = junction1->position_;
        KeyPoint center_kp;
        bool succ_center;
        TrackCenter(img1,img2,center,center_kp,succ_center);
        Vec2 center_tracked{center_kp.pt.x,center_kp.pt.y};
        if(!succ_center)
            return NULL;
        auto junction2 = std::make_shared<Junction2D>();
        junction2->position_ = center_tracked;
        for (Vec2 endpoint: junction1->endpoints_)
        {
            vector<Vec2> tracked_middle_points;
            TrackEdge(img1,img2,center,endpoint,tracked_middle_points);
            int num_tracked = tracked_middle_points.size();
            Eigen::MatrixXd M;
            M.resize(2,num_tracked);
            for(int i=0;i<tracked_middle_points.size();i++)
                M.col(i) = tracked_middle_points[i] - center_tracked;
            //Do PCA
            Eigen::Matrix2d H  = M*M.transpose();
            Eigen::EigenSolver<Eigen::Matrix2d> es(H);
            auto eigenvalues = es.eigenvalues();
            double lambda1 = eigenvalues[0].real();
            double lambda2 = eigenvalues[1].real();
            int index = lambda1>lambda2? 0:1;
            Vec2 v = es.eigenvectors().col(index).real();
            //Vec2 displacement = v*v.dot(*(tracked_middle_points.end()-1)-center_tracked);
            Vec2 displacement = v*(endpoint-center).norm();
            LOG(INFO)<<"Quality Ratio:"<<(index?(lambda2/lambda1):(lambda1/lambda2));
            int iters = 0;
            while(!BoundaryCheck(img2,center_tracked+displacement))
            {
                displacement*=0.95;
                iters++;
                if(iters>=100)
                    break;
            }
            if(displacement.norm()<15||iters>=100)  //edge too short or out of bound
                continue;
            junction2->endpoints_.emplace_back(center_tracked+displacement);
        }
        LOG(INFO)<<"track junction done!";
        return junction2;
    }
}
