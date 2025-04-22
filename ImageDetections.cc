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


#include "ImageDetections.h"

#include <experimental/filesystem>
#include <unordered_set>


namespace fs = std::experimental::filesystem;


namespace ORB_SLAM2
{

std::ostream& operator <<(std::ostream& os, const Detection& det)
{
    os << "Detection:  cat = " << det.category_id << "  score = "
       << det.score << "  bbox = " << det.bbox.transpose();
    return os;
}


DetectionsFromFile::DetectionsFromFile(const std::string& filename, const std::vector<int>& cats_to_ignore)
    : ImageDetectionsManager()
{
    std::unordered_set<unsigned int> to_ignore(cats_to_ignore.begin(), cats_to_ignore.end());
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Warning failed to open file: " << filename << std::endl;
        return ;
    }
    fin >> data_;

    for (auto& frame : data_)
    {
        std::string name = frame["file_name"].get<std::string>();
        name = fs::path(name).filename();
        frame_names_.push_back(name);

        std::vector<Detection::Ptr> detections;
        for (auto& d : frame["detections"])
        {
            double score = d["detection_score"].get<double>();
            unsigned int cat = d["category_id"].get<unsigned int>();
            if (to_ignore.find(cat) != to_ignore.end())
                continue;
            auto bb = d["bbox"];
            Eigen::Vector4d bbox(bb[0], bb[1], bb[2], bb[3]);
            detections.push_back(std::shared_ptr<Detection>(new Detection(cat, score, bbox)));
        }
        detections_[name] = detections;
    }
}


std::vector<Detection::Ptr> DetectionsFromFile::detect(const std::string& name) const {
    std::string basename = fs::path(name).filename();

    if (detections_.find(basename) == detections_.end())
        return {};
    return detections_.at(basename);
}
std::vector<Detection::Ptr> DetectionsFromFile::detect(unsigned int idx) const {
    if (idx < 0 || idx >= frame_names_.size()) {
        std::cerr << "Warning invalid index: " << idx << std::endl;
        return {};
    }
    return this->detect(frame_names_[idx]);
}


#ifdef USE_DNN

ObjectDetector::ObjectDetector(const std::string& model, const std::vector<int>& cats_to_ignore) 
    : network_(std::make_unique<cv::dnn::Net>()), ignored_cats_(cats_to_ignore.begin(), cats_to_ignore.end()), ImageDetectionsManager()
{
    if (model.substr(model.size()-4) == "onnx")
    {
        //*network_ =  cv::dnn::readNet(model);
        std::cout << "Start to read onnx model" << std::endl;
        *network_ =  cv::dnn::readNetFromONNX(model);
        std::cout << "Read model successful" << std::endl;
    }
    else
    {
        *network_ =  cv::dnn::readNetFromDarknet(model + ".cfg", model + ".weights");
    }
    network_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    network_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Object detector created" << std::endl;
}
cv::Mat compute_mask(const cv::Mat& protos, const cv::Mat& coeffs, const cv::Rect& box, const cv::Size& image_size);

std::vector<Detection::Ptr> ObjectDetector::detect(cv::Mat img) const
{

    // Settings
    const int INPUT_WIDTH = 640.0/2; // size of image passed to the network (reducing may be faster to process)
    const int INPUT_HEIGHT = 640.0/2;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.45;

    cv::Mat result;
    cv::dnn::blobFromImage(img, result, 1./255, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    
    std::vector<cv::Mat> predictions;
    // for (auto x : network_->getUnconnectedOutLayersNames())
    //     std::cout << x  << std::endl;

    auto outnames = network_->getUnconnectedOutLayersNames().back();    // output only the last layer
    network_->setInput(result,"images");
    network_->forward(predictions, network_->getUnconnectedOutLayersNames());

//    network_->forward(predictions, outnames);
    const cv::Mat& output = predictions[0]; // [1, num_detections, 5 + classes + 32]
    const cv::Mat& mask_protos = predictions[1]; // [1, 32, 160, 160] (ví dụ)
    // output should have size: 1 x nb_detections x (5 + nb_classes)

    float x_factor = (float)(img.cols) / INPUT_WIDTH;
    float y_factor = (float)(img.rows) / INPUT_HEIGHT;
//    float *data = (float *)output.data;

    const int rows = output.size[1];
    const int cols = output.size[2];
    
    int num_classes = cols -5 -32;

    std::vector<int> class_ids;
    class_ids.reserve(1000);
    std::vector<float> confidences;
    confidences.reserve(1000);
    std::vector<cv::Rect> boxes;
    boxes.reserve(1000);
    
    int nb_classes = cols - 5;
    std::vector<cv::Mat> mask_coeff_list;
    float* data = (float*)output.data;


    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, nb_classes, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (ignored_cats_.count(class_id.x) == 0 && max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(max_class_score);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));  
                
                cv::Mat coeffs(1, 32, CV_32F);
                memcpy(coeffs.data, data + 5 + num_classes, sizeof(float) * 32);
                mask_coeff_list.push_back(coeffs);  
            }
        }
        data += cols;
    }

    // Filter detection with NMS
    std::vector<Detection::Ptr> detections;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        cv::Rect& bb = boxes[idx];
        Eigen::Vector4d bbox(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
        cv::Mat seg_mask = compute_mask(mask_protos, mask_coeff_list[idx], bb, img.size());
        
        auto det = std::make_shared<Detection>(class_ids[idx], confidences[idx], bbox);
        det->mask = seg_mask;
        detections.push_back(det);
    }
    return detections;
}
#endif
cv::Mat compute_mask(const cv::Mat& protos, const cv::Mat& coeffs, const cv::Rect& box, const cv::Size& image_size)
{
    // protos: [1, 32, H, W]
    cv::Mat proto = protos.reshape(1, {32, protos.size[2] * protos.size[3]});  // [32, H*W]
    cv::Mat mask_flat = coeffs * proto;  // [1, H*W]
    mask_flat = mask_flat.reshape(1, protos.size[2]);  // [H, W]

    // Resize mask to match the image size
    cv::Mat mask_resized;
    cv::resize(mask_flat, mask_resized, image_size);

    // Check if the bounding box is inside the image bounds
    int x = std::max(box.x, 0);
    int y = std::max(box.y, 0);
    int width = std::min(box.width, mask_resized.cols - x);
    int height = std::min(box.height, mask_resized.rows - y);

    // Create valid bounding box within the image
    cv::Rect valid_box(x, y, width, height);

    // Crop according to the valid bounding box
    cv::Mat crop = mask_resized(valid_box);

    // Return binarized mask
    return crop > 0.5;
}


}
