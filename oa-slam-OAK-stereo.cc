
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdlib.h>     /* srand, rand */

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <ImageDetections.h>
#include <System.h>
#include "Osmap.h"
#include <nlohmann/json.hpp>
#include <experimental/filesystem>
#include "Utils.h"

#include <depthai/depthai.hpp>

using json = nlohmann::json;

namespace fs = std::experimental::filesystem;

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight, int start_id, int end_id,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadImages(const string &strPathLeft, const string &strPathRight, int start_id, int end_id,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    vstrImageLeft.clear();
    vstrImageRight.clear();
    vTimeStamps.clear();
    
    
    vTimeStamps.reserve(2000);
    vstrImageLeft.reserve(2000);
    vstrImageRight.reserve(2000);
    for (int i = start_id; i <= end_id; i++){

            stringstream ss_left, ss_right;
            ss_left << i;
            ss_right << i;
            vstrImageLeft.push_back(strPathLeft + "/img" + ss_left.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/img" + ss_right.str() + ".png");
            double t = ((double)i)*(1/20.0);
            vTimeStamps.push_back(t);

    }
}

int main(int argc, char **argv)
{
    srand(time(nullptr));
    std::cout << "C++ version: " << __cplusplus << std::endl;

    if(argc != 6)
    {
        cerr << endl << "Usage:\n"
                        " ./oa-slam\n"
                        "      vocabulary_file\n"
                        "      camera_file\n"
                        "      detections_file (.json file with detections or .onnx yolov5 weights)\n"
                        "      categories_to_ignore_file (file containing the categories to ignore (one category_id per line))\n"
                        "      relocalization_mode ('points', 'objects' or 'points+objects')\n";
        return 1;
    }

    std::string vocabulary_file = string(argv[1]);
    std::string parameters_file = string(argv[2]);
    std::string detections_file(argv[3]);
    std::string categories_to_ignore_file(argv[4]);
    string reloc_mode = string(argv[5]);

    // possible to pass 'webcam_X' where 'X' is the webcam id

    //string output_folder = output_name;
    //if (output_folder.back() != '/')
    //    output_folder += "/";
    //fs::create_directories(output_folder);


    // Load categories to ignore
    std::ifstream fin(categories_to_ignore_file);
    vector<int> classes_to_ignore;
    if (!fin.is_open()) {
        std::cout << "Warning !! Failed to open the file with ignore classes. No class will be ignore.\n";
    } else {
        int cat;
        while (fin >> cat) {
            std::cout << "Ignore category: " << cat << "\n";
            classes_to_ignore.push_back(cat);
        }
    }


    // Load object detections
    auto extension = get_file_extension(detections_file);
    std::shared_ptr<ORB_SLAM2::ImageDetectionsManager> detector = nullptr;
    if (extension == "onnx") { // load network
        detector = std::make_shared<ORB_SLAM2::ObjectDetector>(detections_file, classes_to_ignore);
    } else if (extension == "json") { // load from external detections file
        
    } else {
        std::cout << "Invalid detection file. It should be .json or .onnx\n"
                      "No detections will be obtained.\n";
    }


    // Relocalization mode
    ORB_SLAM2::enumRelocalizationMode relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    if (reloc_mode == string("points"))
        relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    else if (reloc_mode == std::string("objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS;
    else if (reloc_mode == std::string("points+objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS_POINTS;
    else {
        std::cerr << "Error: Invalid parameter for relocalization mode. "
                     "It should be 'points', 'objects' or 'points+objects'.\n";
        return 1;
    }

    /// ////////////////////// Set up OAK-D stream ////////////////////// ///
    /// ////////////////////// Set up OAK-D stream ////////////////////// /// 
    /// c++ code to get rectified left and right color image from OAK-D camera ///
    /// ////////////////////// Set up OAK-D stream ////////////////////// ///
	/*
	vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    
    string leftpath = "/home/duong/Desktop/Dataset/image0";
    string rightpath = "/home/duong/Desktop/Dataset/image1";
    int startid = 7;
    int endid = 915;
    
    LoadImages(leftpath, rightpath, startid, endid, vstrImageLeft, vstrImageRight, vTimeStamp);
	
	if(vstrImageLeft.empty() || vstrImageRight.empty())
    {
        cerr << "ERROR: No images in provided path." << endl;
        return 1;
    }

    if(vstrImageLeft.size()!=vstrImageRight.size())
    {
        cerr << "ERROR: Different number of left and right images." << endl;
        return 1;
    }
	*/
	
	// Create a pipeline
    dai::Pipeline p;

    // Create the mono camera nodes for left and right cameras
    auto monoLeft = p.create<dai::node::MonoCamera>();
    auto monoRight = p.create<dai::node::MonoCamera>();
    auto xoutLeft = p.create<dai::node::XLinkOut>();
    auto xoutRight = p.create<dai::node::XLinkOut>();
    auto stereo = p.create<dai::node::StereoDepth>();
    auto xoutDepth = p.create<dai::node::XLinkOut>();
    
     // Create outputs for both left and right camera streams
    xoutLeft->setStreamName("left");
    xoutRight->setStreamName("right");
    xoutDepth->setStreamName("depth");

    // Set the camera resolution for left and right cameras
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B); // Left camera
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C); // Right camera
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

	// StereoDepth
    stereo->initialConfig.setConfidenceThreshold(200);
    stereo->setRectifyEdgeFillColor(0);  // black, to better see the cutout
    stereo->initialConfig.setLeftRightCheckThreshold(5);
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(false);
    stereo->setSubpixel(true);

    // Link the cameras to the outputs
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    
    stereo->rectifiedLeft.link(xoutLeft->input);
    stereo->rectifiedRight.link(xoutRight->input);
    
    stereo->depth.link(xoutDepth->input);
    

    // Connect to the device and start the pipeline
    dai::Device d(p);
    
    // Get the output queues for left and right cameras
    auto qLeft = d.getOutputQueue("left", 2, false);
    auto qRight = d.getOutputQueue("right", 2, false);
	auto stereoQueue = d.getOutputQueue("depth", 2, false);
	
	/// ////////////////////// END Set up OAK-D stream ////////////////////// ///
	
	
	/// /////////////////////// READ RECTIFICATION ///////////////////////// ///
	cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
	
	//const int nImages = vstrImageLeft.size();
	
	/// ///////////////////////  OA SLAM   //////////////////////////////// ///
	
    // Create system
    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, ORB_SLAM2::System::STEREO, true, false, false);
    SLAM.SetRelocalizationMode(relocalization_mode);

    // Vector for tracking time statistics
    //vector<float> vTimesTrack;
    //vTimesTrack.reserve(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);

    // Main loop
    cv::Mat im;
    //std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    //poses.reserve(nImages);
    //std::vector<std::string> filenames;
    //filenames.reserve(nImages);
    //std::vector<double> timestamps;
    //timestamps.reserve(nImages);
    
    int ni = 0;   
    cv::Mat imLeftRect, imRightRect;
    
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    
    while (1)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        // Read left and right images from file
        //imLeft = cv::imread(vstrImageLeft[ni],cv::IMREAD_UNCHANGED);
        //imRight = cv::imread(vstrImageRight[ni],cv::IMREAD_UNCHANGED);
		
		// Get the latest frames from left and right cameras
        auto frameLeft = qLeft->get<dai::ImgFrame>();
        auto frameRight = qRight->get<dai::ImgFrame>();
		
        
        // Convert frames to OpenCV Mat format
        cv::Mat imLeft = frameLeft->getCvFrame();//(frameLeft->getHeight(), frameLeft->getWidth(), CV_8UC1, frameLeft->getData());
        cv::Mat imRight = frameRight->getCvFrame();//(frameRight->getHeight(), frameRight->getWidth(), CV_8UC1, frameRight->getData());

		
        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: " << endl;
            continue;
        }

        if(imRight.empty())
        {
            cerr << endl << "Failed to load image at: " << endl;
            continue;
        }

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

       
        // Get the current time in seconds since epoch
		std::chrono::steady_clock::time_point tnow = std::chrono::steady_clock::now();		
		double tframe = std::chrono::duration_cast<std::chrono::duration<double> >(tnow - t0).count();
		std::cout << "tframe = " << tframe << "\n";
		
		cv::Mat imLeftRect_color, imRightRect_color;
        
        cv::cvtColor(imLeftRect, imLeftRect_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(imRightRect, imRightRect_color, cv::COLOR_GRAY2BGR);

// Get object detections
std::vector<ORB_SLAM2::Detection::Ptr> detections;
/*
if (detector) {
    detections = detector->detect(imLeftRect_color);  // from neural network

    for (std::vector<ORB_SLAM2::Detection::Ptr>::iterator it = detections.begin(); it != detections.end(); ++it) {
        ORB_SLAM2::Detection::Ptr det = *it;

        // Resize mask nếu cần
        if (!det->mask.empty() && det->mask.size() != imLeftRect.size()) {
            cv::resize(det->mask, det->mask, imLeftRect.size(), 0, 0, cv::INTER_NEAREST);
        }

        // Debug mask nếu cần
        if (!det->mask.empty()) {
            std::cout << "Segmentation mask size: " << det->mask.cols << "x" << det->mask.rows << "\n";

            // Convert mask về 8-bit để hiển thị
            //cv::Mat visMask;
            //det->mask.convertTo(visMask, CV_8U, 255);
            //cv::imshow("Segmentation Mask", visMask);
            //cv::waitKey(1);
        } else {
            std::cout << "Warning: Detection không có mask.\n";
        }
        
    }
}*/
if (detector) {
    detections = detector->detect(imLeftRect_color);  // from neural network

    cv::Mat imLeftWithMasks = imLeftRect_color.clone();  // Ảnh để vẽ mask

    for (auto& det : detections) {
        if (!det || det->mask.empty()) {
            std::cout << "Warning: Detection không hợp lệ hoặc không có mask.\n";
            continue;
        }

        // Tính toán bbox (đảm bảo nằm trong ảnh)
        int x = std::max(int(det->bbox(0)), 0);
        int y = std::max(int(det->bbox(1)), 0);
        int width = std::min(int(det->bbox(2) - det->bbox(0)), imLeftRect.cols - x);
        int height = std::min(int(det->bbox(3) - det->bbox(1)), imLeftRect.rows - y);
        if (width <= 0 || height <= 0) continue;

        cv::Rect bbox(x, y, width, height);

        // Resize mask về đúng bbox nếu chưa đúng size
        cv::Mat resizedMask;
        if (det->mask.size() != bbox.size())
            cv::resize(det->mask, resizedMask, bbox.size(), 0, 0, cv::INTER_NEAREST);
        else
            resizedMask = det->mask;

        if (resizedMask.type() != CV_8UC1)
            resizedMask.convertTo(resizedMask, CV_8UC1);

        // Tạo mask màu ngẫu nhiên
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        cv::Mat colorMask(bbox.size(), CV_8UC3, color);

        // Áp dụng mask màu lên vùng bbox trong ảnh gốc
        cv::Mat roi = imLeftWithMasks(bbox);
        cv::Mat blended;
        colorMask.copyTo(blended, resizedMask);  // chỉ giữ lại phần mask
        cv::addWeighted(roi, 1.0, blended, 0.5, 0, roi);
    }

    cv::imshow("Image with Segmentation Masks", imLeftWithMasks);
    cv::waitKey(1);
}





              		
		std::cout << "GRAY TO COLOR CONVERTED" << std::endl;
		
        // Pass the image and detections to the SLAM system
        cv::Mat m = SLAM.TrackStereo_object(imLeftRect, imRightRect, tframe, detections, false);

        //if (m.rows && m.cols)
        //    poses.push_back(ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m));
        //else
        //    poses.push_back(Eigen::Matrix4d::Identity());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //vTimesTrack.push_back(ttrack);
        std::cout << "time = " << ttrack << "\n";

        if (SLAM.ShouldQuit())
            break;
            
    }

    // Stop all threads
    SLAM.Shutdown();


    // Save camera tracjectory
	/*
    // TXT files
    std::ofstream file(output_folder + "camera_poses_" + output_name + ".txt");
    std::ofstream file_tum(output_folder + "camera_poses_" + output_name + "_tum.txt");    // output poses in the TUM RGB-D format
    json json_data;
    for (unsigned int i = 0; i < poses.size(); ++i)
    {
        Eigen::Matrix4d m = poses[i];
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = m.block<3, 3>(0, 0).transpose();
        pose.block<3, 1>(0, 3) = -m.block<3, 3>(0, 0).transpose() * m.block<3, 1>(0, 3);

        file << i << " " << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << " "
                  << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << " "
                  << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << "\n";


        json R({{m(0, 0), m(0, 1), m(0, 2)},
                {m(1, 0), m(1, 1), m(1, 2)},
                {m(2, 0), m(2, 1), m(2, 2)}});
        json t({m(0, 3), m(1, 3), m(2, 3)});
        json image_data;
        image_data["file_name"] = filenames[i];
        image_data["R"] = R;
        image_data["t"] = t;
        json_data.push_back(image_data);

        auto q = Eigen::Quaterniond(pose.block<3, 3>(0, 0));
        auto p = pose.block<3, 1>(0, 3);
        file_tum << std::fixed << timestamps[i] << " " << p[0] << " " << p[1] << " " << p[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n"; 
    }
    file.close();
    file_tum.close();


    // JSON files
    std::ofstream json_file(output_folder + "camera_poses_" + output_name + ".json");
    json_file << json_data;
    json_file.close();


    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl << endl;

    // Save camera trajectory, points and objects
    SLAM.SaveKeyFrameTrajectoryTUM(output_folder + "keyframes_poses_" + output_name + "_tum.txt");
    SLAM.SaveKeyFrameTrajectoryJSON(output_folder + "keyframes_poses_" + output_name + ".json", filenames);
    SLAM.SaveMapPointsOBJ(output_folder + "map_points_" + output_name + ".obj");
    SLAM.SaveMapObjectsOBJ(output_folder + "map_objects_" + output_name + ".obj");
    SLAM.SaveMapObjectsTXT(output_folder + "map_objects_" + output_name + ".txt");
    std::cout << "\n";

    // Save a reloadable map
    osmap.mapSave(output_folder + "map_" + output_name);
	*/
    return 0;
}
