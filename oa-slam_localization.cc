
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include <experimental/filesystem>

#include <System.h>
#include "Osmap.h"
#include <nlohmann/json.hpp>
#include "Utils.h"



#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <mutex>

using json = nlohmann::json;
namespace fs = std::experimental::filesystem;

using namespace std;

class ROS2ImageReader : public rclcpp::Node {
public:
    ROS2ImageReader(const std::string &topic_name, const std::string &node_name)
        : Node(node_name) {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic_name, 10,
            std::bind(&ROS2ImageReader::imageCallback, this, std::placeholders::_1));
    }

    bool getLatestImage(cv::Mat &image_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!latest_image_.empty()) {
            image_out = latest_image_.clone();
            return true;
        }
        return false;
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            auto cv_img = cv_bridge::toCvCopy(msg, msg->encoding);
            std::lock_guard<std::mutex> lock(mutex_);
            latest_image_ = cv_img->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    cv::Mat latest_image_;
    std::mutex mutex_;
};

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    srand(time(nullptr));
    std::cout << "C++ version: " << __cplusplus << std::endl;

    if(argc != 9)
    {
        cerr << endl << "Usage:\n"
                        " ./oa-slam_localization\n"
                        "      vocabulary_file\n"
                        "      camera_file\n"
                        "      detections_file (.json file with detections or .onnx yolov5 weights)\n"
                        "      categories_to_ignore_file (file containing the categories to ignore (one category_id per line))\n"
                        "      map_file (.yaml)\n"
                        "      relocalization_mode ('points', 'objects' or 'points+objects')\n"
                        "      output_name \n"
                        "      force_relocalization_on_each_frame (0 or 1)\n";
        return 1;
    }



    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::string vocabulary_file = string(argv[1]);
    std::string parameters_file = string(argv[2]);
    std::string detections_file(argv[3]);
    std::string categories_to_ignore_file(argv[4]);
    string map_file = string(argv[5]);
    string reloc_mode = string(argv[6]);
    string output_name = string(argv[7]);
    bool force_reloc = std::stoi(argv[8]);


	rclcpp::init(argc, argv);

auto left_reader = std::make_shared<ROS2ImageReader>("/camera/camera/infra1/image_rect_raw", "left_reader");
auto right_reader = std::make_shared<ROS2ImageReader>("/camera/camera/infra2/image_rect_raw", "right_reader");

rclcpp::executors::MultiThreadedExecutor executor;
executor.add_node(left_reader);
executor.add_node(right_reader);

std::thread ros_spin_thread([&executor]() {
    executor.spin();
});
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
	

    string output_folder = output_name;
    if (output_folder.back() != '/')
        output_folder += "/";
    fs::create_directories(output_folder);

    // Get map folder absolute path
    int l = map_file.find_last_of('/') + 1;
    std::string map_folder = map_file.substr(0, l);
    if (map_folder[0] != '/') {
        fs::path map_folder_abs = fs::current_path() / map_folder;
        map_folder = map_folder_abs.string();
    }

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

    // Load images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    int nImages = 10000;

    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, ORB_SLAM2::System::STEREO, true, false, false);
    SLAM.SetRelocalizationMode(relocalization_mode);
    SLAM.map_folder = map_folder;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.reserve(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);
    std::cout << "Start loading map" << std::endl;
    osmap.mapLoad(map_file);
    std::cout << "End of loading map" << std::endl;
    SLAM.ActivateLocalizationMode();

    // SLAM.remove_nth_object_by_cat(71, 2); // remove some objects from the loaded map

    // Main loop
    cv::Mat im;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    poses.reserve(nImages);
    std::vector<double> reloc_times;
    reloc_times.reserve(nImages);
    std::vector<bool> reloc_status;
    reloc_status.reserve(nImages);
    int ni = 0;
        cv::Mat imLeftRect, imRightRect;
    
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    
while (rclcpp::ok()) 
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
std::string filename;

cv::Mat imLeft, imRight;
if (!left_reader->getLatestImage(imLeft) || !right_reader->getLatestImage(imRight)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    continue;
}
        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
        cv::Mat imLeftRect_color, imRightRect_color;


        cv::cvtColor(imLeftRect, imLeftRect_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(imRightRect, imRightRect_color, cv::COLOR_GRAY2BGR);
        double tframe = ni < vTimestamps.size() ? vTimestamps[ni] : std::time(nullptr);


        // Get object detections
        std::vector<ORB_SLAM2::Detection::Ptr> detections;
        if (detector) {
    detections = detector->detect(imLeftRect_color);  // from neural network   
        }

        // Pass the image and detections to the SLAM system
        cv::Mat m = SLAM.TrackStereo_object(imLeftRect, imRightRect, tframe, detections, false);
        reloc_times.push_back(SLAM.relocalization_duration);
        reloc_status.push_back(SLAM.relocalization_status);

        if (m.rows && m.cols)
            poses[ni] = ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m);
        else
            poses.push_back(Eigen::Matrix4d::Identity());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack.push_back(ttrack);

        if (SLAM.ShouldQuit())
            break;

        ++ni;
        if (ni >= nImages)
            break;
    }

    // Stop all threads
    SLAM.Shutdown();
  
    // Save camera trajectory
    json json_data;
    for (size_t i = 0; i < poses.size(); ++i)
    {
        Eigen::Matrix4d m = poses[i];
        json R({{m(0, 0), m(0, 1), m(0, 2)},
                {m(1, 0), m(1, 1), m(1, 2)},
                {m(2, 0), m(2, 1), m(2, 2)}});
        json t({m(0, 3), m(1, 3), m(2, 3)});
        json image_data;
        image_data["R"] = R;
        image_data["t"] = t;
        json_data.push_back(image_data);
    }

    std::ofstream json_file(output_folder + "camera_poses_" + output_name + ".json");
    json_file << json_data;
    json_file.close();
    std::cout << "Saved " << poses.size() << " poses.\n";


    // Relocalization time statistics
    std::ofstream file_times(output_folder + "relocalization_times.txt");
    for (int i = 0; i < reloc_times.size(); ++i) {
        file_times << reloc_times[i] << " " << (int)reloc_status[i] << "\n";
    }
    file_times.close();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    return 0;
}



	 	
		
