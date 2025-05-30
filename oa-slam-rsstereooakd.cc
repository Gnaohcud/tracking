/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <stdlib.h>     /* srand, rand */

#include<opencv2/core/core.hpp>
//#include <librealsense2/rs.hpp>
#include <depthai/depthai.hpp>
#include <ImageDetections.h>
#include <System.h>
#include "Osmap.h"
#include <nlohmann/json.hpp>
#include <experimental/filesystem>
#include "Utils.h"

using json = nlohmann::json;

namespace fs = std::experimental::filesystem;

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

dai::Pipeline pipeline;
void setupOAKD() {
        // Tạo pipeline cho thiết bị DepthAI
        dai::Pipeline pipeline;

        // --- Cấu hình camera RGB ---
        auto colorCam = pipeline.create<dai::node::ColorCamera>();
        // Bạn có thể cấu hình độ phân giải, kích thước preview nếu cần
        colorCam->setBoardSocket(dai::CameraBoardSocket::RGB);
        // Sử dụng stream video để lấy frame chất lượng gốc
        auto xoutColor = pipeline.create<dai::node::XLinkOut>();
        xoutColor->setStreamName("color");
        // Liên kết output video của colorCam với xoutColor
        colorCam->video.link(xoutColor->input);

        // --- Cấu hình camera mono (trái) ---
        auto monoLeft = pipeline.create<dai::node::MonoCamera>();
        monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
        // Cấu hình độ phân giải của mono camera (có thể chọn độ phân giải khác)
        monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        auto xoutMono = pipeline.create<dai::node::XLinkOut>();
        xoutMono->setStreamName("mono");
        // Liên kết output của monoLeft với xoutMono
        monoLeft->out.link(xoutMono->input);

        // Khởi tạo thiết bị với pipeline đã cấu hình
        dai::Device device(pipeline);

        // Lấy queue dữ liệu video từ các stream
        auto colorQueue = device.getOutputQueue("color", 30, false);
        auto monoQueue  = device.getOutputQueue("mono", 30, false);

        std::cout << "Kết nối thành công. Nhấn ESC để thoát." << std::endl;

}
int main(int argc, char **argv)
{
    srand(time(nullptr));
    std::cout << "C++ version: " << __cplusplus << std::endl;

    if(argc != 8)
    {
        cerr << endl << "Usage:\n"
                        " ./oa-slam\n"
                        "      vocabulary_file\n"
                        "      camera_file\n"
                        "      path_to_image_sequence (.txt file listing the images or a folder with rgb.txt or 'webcam')\n"
                        "      detections_file (.json file with detections or .onnx yolov5 weights)\n"
                        "      categories_to_ignore_file (file containing the categories to ignore (one category_id per line))\n"
                        "      relocalization_mode ('points', 'objects' or 'points+objects')\n"
                        "      output_name \n";
        return 1;
    }

    std::string vocabulary_file = string(argv[1]);
    std::string parameters_file = string(argv[2]);
    string path_to_images = string(argv[3]);
    std::string detections_file(argv[4]);
    std::string categories_to_ignore_file(argv[5]);
    string reloc_mode = string(argv[6]);
    string output_name = string(argv[7]);

    // possible to pass 'webcam_X' where 'X' is the webcam id
    bool use_webcam = false;
 /*   int webcam_id_1 = 0;
    int webcam_id_2 = 0;
    if (path_to_images.size() >= 6 && path_to_images.substr(0, 6) == "webcam") {
        use_webcam = true;
        if (path_to_images.size() > 7) {
            webcam_id_1 = std::stoi(path_to_images.substr(7));
            webcam_id_2 = std::stoi(path_to_images.substr(9));
        }
    }*/

    // Possible to pass a file listing images instead of a folder containing a file rgb.txt or "webcam"
    std::string image_list_file = "rgb.txt";
    int nn = path_to_images.size();
    if (!use_webcam && get_file_extension(path_to_images) == "txt") {
        int pos = path_to_images.find_last_of('/');
        image_list_file = path_to_images.substr(pos+1);
        path_to_images = path_to_images.substr(0, pos+1);
    }

    if (!use_webcam && path_to_images.back() != '/')
        path_to_images += "/";

    string output_folder = output_name;
    if (output_folder.back() != '/')
        output_folder += "/";
    fs::create_directories(output_folder);


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
    bool detect_from_file = false;
    if (extension == "onnx") { // load network
        detector = std::make_shared<ORB_SLAM2::ObjectDetector>(detections_file, classes_to_ignore);
        detect_from_file = false;
    } else if (extension == "json") { // load from external detections file
        detector = std::make_shared<ORB_SLAM2::DetectionsFromFile>(detections_file, classes_to_ignore);
        detect_from_file = true;
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

    // Load images_webcam1
    cv::VideoCapture cap1,cap2;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    int nImages = 10000;
    /*
    if (!use_webcam) {
        string strFile = path_to_images + image_list_file;
        LoadImages(strFile, vstrImageFilenames, vTimestamps);
        nImages = vstrImageFilenames.size();
    } else {
        if (cap1.open(webcam_id_1)) {
            std::cout << "Opened webcam: " << webcam_id_1 << "\n";
            cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        } else {
            std::cerr << "Failed to open webcam: " << webcam_id_1 << "\n";
            return -1;
        }
    
    // Load images_webcam2
	
        if (cap2.open(webcam_id_2)) {
            std::cout << "Opened webcam: " << webcam_id_2 << "\n";
            cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        } else {
            std::cerr << "Failed to open webcam: " << webcam_id_2 << "\n";
            return -1;
        }
        
    }*/

    
    if (!use_webcam) {
        std::string strFile = path_to_images + image_list_file;
        LoadImages(strFile, vstrImageFilenames, vTimestamps);
        nImages = vstrImageFilenames.size();
    } else {
        setupOAKD(); // Khởi động RealSense thay vì dùng OpenCV
    dai::Device device(pipeline);
    auto rgbQueue = device.getOutputQueue("rgb", 1, false);
    auto monoQueue = device.getOutputQueue("mono", 1, false);
    }
    
    // Create system
    
    cout << "START INIT OA" << endl;
     
    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, ORB_SLAM2::System::STEREO, true, false, false);
    
    cout << "END INIT OA" << endl;
    SLAM.SetRelocalizationMode(relocalization_mode);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.reserve(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);

    // Main loop
    cv::Mat im1,im2;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    poses.reserve(nImages);
    std::vector<std::string> filenames;
    filenames.reserve(nImages);
    std::vector<double> timestamps;
    timestamps.reserve(nImages);
    int ni = 0;
    dai::Device device(pipeline);
    auto colorQueue = device.getOutputQueue("color", 30, false);
    auto monoQueue = device.getOutputQueue("mono", 1, false);
    while (1)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::string filename1,filename2;
        if (use_webcam) {
            auto inColorFrame = colorQueue->get<dai::ImgFrame>();
            cv::Mat colorFrame = inColorFrame->getCvFrame();

            auto inMonoFrame = monoQueue->get<dai::ImgFrame>();
            cv::Mat monoFrame = inMonoFrame->getCvFrame();

            // Hiển thị các frame trên cửa sổ riêng biệt
            cv::imshow("Camera RGB", colorFrame);
            cv::imshow("Camera Mono (Trái)", monoFrame);
        
        // Nếu nhấn 'q', thoát vòng lặp
        if (cv::waitKey(1) == 'q') break;
            continue;
        }
        else
        {
            filename1 = path_to_images + vstrImageFilenames[ni*2];
            filename2 = path_to_images + vstrImageFilenames[ni*2+1];
            im1 = cv::imread(filename1, cv::IMREAD_UNCHANGED);  
            im2 = cv::imread(filename2, cv::IMREAD_UNCHANGED);// read image from disk
        }
        double tframe = ni < vTimestamps.size() ? vTimestamps[ni] : std::time(nullptr);
        timestamps.push_back(tframe);
        if(im1.empty()||im2.empty())
        {
			cerr << "Failed to load images: " << filename1 << " or " << filename2 << std::endl;
            return 1;
        }
        filenames.push_back(filename1);
        filenames.push_back(filename2);

        // Get object detections
        std::vector<ORB_SLAM2::Detection::Ptr> detections1;//, detections2;
        if (detector) {
            if (detect_from_file)
                detections1 = detector->detect(filename1);
                //detections2 = detector->detect(filename2); // from detections file
            }
            else{
                cout << "DETECT" << endl;
                detections1 = detector->detect(im1);
                //detections2 = detector->detect(im2);  // from neural network
        }

        // Pass the image and detections to the SLAM system
        cout << "FEED TO SLAM" << endl;
        cv::Mat m1 = SLAM.TrackStereo_object(im1, im2, tframe, detections1, false);
        //cv::Mat m2 = SLAM.TrackMonocular(im2, tframe, detections2, false);

        if (m1.rows && m1.cols)
            poses.push_back(ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m1));
        else
            poses.push_back(Eigen::Matrix4d::Identity());
        //if (m2.rows && m2.cols)
        //    poses.push_back(ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m2));
        //else
        //    poses.push_back(Eigen::Matrix4d::Identity());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack.push_back(ttrack);
        std::cout << "time = " << ttrack << "\n";

        if (SLAM.ShouldQuit())
            break;

        ++ni;
        if (ni >= nImages/2)
            break;
    }

    // Stop all threads
    SLAM.Shutdown();


    // Save camera tracjectory

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

    return 0;
}


void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    string s0;
    double t = 0;
    int n = 0;
    bool found_timestamps = false;
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty() && s[0] != '#')
        {
            stringstream ss;
            ss << s;
            string sRGB;
            if (ss.str().find(' ') != std::string::npos) {
                ss >> t;
                found_timestamps = true;
            }
            ss >> sRGB;

            vTimestamps.push_back(t);
            vstrImageFilenames.push_back(sRGB);
            if (!found_timestamps)
                t += 0.033;
            ++n;
        }
    }
    f.close();
}
