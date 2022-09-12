
#include <iostream>
//#include </usr/include/opencv2/highgui.hpp>
//#include </usr/include/opencv2/core.hpp>
//#include </usr/include/opencv/cv.hpp>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>    // to processs the K function
#include <cmath>     //for K function
#include <cstdlib>   //for K function
#include <algorithm> //for sort
//#include </usr/include/boost/math/distributions/empirical_cumulative_distribution_function.hpp>
// using boost::math::empirical_cumulative_distribution_function;
// using namespace cv::imread;
// using namespace cv::Mat;
// using namespace cv; // This can import all functions of cv but may take some memory

// Function that returns the CDF of an input cv::Mat
std::vector<double> GetCDF(cv::Mat _Mat1)
{
    int block_width_and_height = 7;
    std::vector<int> vec = _Mat1.isContinuous() ? _Mat1 : _Mat1.clone();

    // sorting the vector
    std::sort(vec.begin(), vec.end());

    // seeing the vector
    // for (auto i = vec.begin(); i < vec.end(); i++)
    //{
    //    std::cout << " " << *i;
    //}

    // CDF
    // the sum of it all
    int tot = 0;
    for (auto i = vec.begin(); i < vec.end(); i++)
    {
        tot = tot + *i;
    }

    std::vector<double> cdf1(block_width_and_height * block_width_and_height);
    double ksum = 0;
    int pos = 0;
    // getting the CDF
    for (int i = 0; i < 49; i++)
    {
        ksum = ksum + vec[i] / (double)tot;
        cdf1[pos] = ksum;
        pos = pos + 1;
    }
    return cdf1;
}

// get ROI Function
cv::Mat GetROI(int i_or, int j_or, cv::Mat _img2, int N)
{
    int block_width_and_height = N;
    int roi_origin_x = i_or; // maybe erase this lines
    int roi_origin_y = j_or;
    cv::Rect roi(roi_origin_x, roi_origin_y, block_width_and_height, block_width_and_height);
    // Create the cv::Mat with the ROI you need, where "image" is the cv::Mat you want to extract the ROI from
    cv::Mat image_roi = _img2(roi).clone(); //.clone()
                                            // The clone is to make a deep copy of the pixels
    // cv::imshow("Patch", image_roi);
    // int k = cv::waitKey(0);
    cv::Mat mat = image_roi.cv::Mat::reshape(1, image_roi.total() * image_roi.channels());
    return mat;
}

// function that gets the max difference
double GetMaxDiff(std::vector<double> vec1, std::vector<double> vec2, int n) // n is the total number of data in the vec
{
    double maxdiff = 0;
    for (int i = 0; i < n * n; i++)
    {
        if (std::abs(vec1[i] - vec2[i]) > maxdiff)
        {
            maxdiff = std::abs(vec1[i] - vec2[i]);
        }
    }
    return maxdiff; // std::max_element(diffe.begin(), diffe.end());
}

int main()
{
    // Read the image
    std::string image_path = "000031.jpg"; // 029831
    cv::Mat img1 = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    /// verify that the image was read
    if (img1.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    // Getting the ROI of an image at 0,0 and 10,10
    // cv::Mat mat = GetROI(0, 0, img1, 7);
    // cv::Mat mat1 = GetROI(57, 0, img1, 7);

    // NLM
    // vector to save the distances
    std::vector<double> dist(112360);
    int vecpos = 0;
    int Totalwidth = 74; // because it's expanded to get the original image in the algorithm
    int ke;
    int me;
    int ve;
    int we;
    for (ke = 0; ke < Totalwidth - 21; ke++)
    {
        for (me = 0; me < Totalwidth - 21; me++) // iterate over all posible windows
        {
            // window cv::Mat Reference_patch = GetROI(k+ 21/2-7/2, m + 21/2-7/2, img1, 21);
            cv::Mat Reference_patch = GetROI(ke + 21 / 2 - 7 / 2, me + 21 / 2 - 7 / 2, img1, 7);
            std::vector<double> cdfvec1 = GetCDF(Reference_patch);
            int Muestras = 0;
            // compare to all other patches
            for (ve = ke; ve < ke + 21 - 7; ve++)
            {

                for (we = me; we < me + 21 - 7; we++)
                {
                    cv::Mat patch_to_compare = GetROI(ve, we, img1, 7);
                    std::vector<double> cdfvec2 = GetCDF(patch_to_compare);
                    double d = GetMaxDiff(cdfvec1, cdfvec2, 7); // distance between the two patches
                    if (Muestras < 20)
                    {
                        if (d < 0.0076114) //
                        {
                            dist[vecpos] = ve + 7 / 2;
                            dist[vecpos + 1] = we + 7 / 2;
                            vecpos = vecpos + 2;
                            Muestras = Muestras + 1;
                        } // alpha=0.5 the difference is significant                         // saving all the distances now
                    }
                }
            }
            if (Muestras < 20)
            {
                while (Muestras < 20)
                {
                    dist[vecpos] = ke + 21 / 2;
                    dist[vecpos + 1] = me + 21 / 2;
                    vecpos = vecpos + 2;
                    Muestras = Muestras + 1;
                }
            }
            // iterating over all window and comparing each patch
        }
    }

    // in one window get all the patches to compare

    // viewing the vector
    for (auto i = dist.begin(); i < dist.end(); i++)
    {
        std::cout << " " << *i;
    }
    return 1;
}

// Reading input arguments whencompiling https://stackoverflow.com/questions/10424747/running-c-programs-with-variable-inputs
//  copy faster the submatrix https://stackoverflow.com/questions/30952048/fastest-way-to-copy-some-rows-from-one-matrix-to-another-in-opencv
//  https://github.com/tumi8/topas/blob/master/detectionmodules/statmodules/wkp-module/ks-test.cpp