
#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp> // opencv libraries
#include <opencv2/highgui/highgui.hpp>
#include <vector>    // to processs
#include <algorithm> //for sort
#include <chrono>    // for counting the elapsed time

// Function that returns the CDF of an input cv::Mat
std::vector<double> GetCDF(cv::Mat _Mat1)
{
    int block_width_and_height = 7;
    std::vector<int> vec = _Mat1.isContinuous() ? _Mat1 : _Mat1.clone();

    // sorting the vector
    std::sort(vec.begin(), vec.end());

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

int main(int argc, char *argv[]) // to read the image file and maybe numbers of the NLM parameters
{                                // Time variables
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //  Read the image
    std::string image_path = argv[2]; // "000031.jpg"
    cv::Mat img1 = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    /// verify that the image was read
    if (img1.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }
    // Getting the ROI of an image at 0,0 and 10,10
    // cv::Mat mat = GetROI(0, 0, img1, 7);
    // cv::Mat mat1 = GetROI(57, 0, img1, 7);

    // NLM

    // int Totalwidth=argv[3];
    // int window_size =argv[4];
    // int patch_size = argv[5];

    int Totalwidth = 85; // because it's expanded to get the original image in the algorithm, it should be 85
    int window_size = 21;
    int patch_size = 7;
    // vector to save the distances
    std::vector<double> dist((Totalwidth - window_size) * (Totalwidth - window_size) * 20); // This is all he values it should save (64*64*20*2) in the image proposed

    // The NLM loop that iterates over all the patches and fills the vector
    //#pragma omp parallel for // parallelize this for loop
    for (int ke = 0; ke < Totalwidth - window_size; ke++)
    {
        for (int me = 0; me < Totalwidth - window_size; me++) // iterate over all posible windows
        {
            // window cv::Mat Reference_patch = GetROI(k+ 21/2-7/2, m + 21/2-7/2, img1, 21);
            cv::Mat Reference_patch = GetROI(ke + window_size / 2 - patch_size / 2, me + window_size / 2 - patch_size / 2, img1, patch_size);
            std::vector<double> cdfvec1 = GetCDF(Reference_patch);
            int Muestras = 0;
            // compare to all other patches
            for (int ve = ke; ve < ke + window_size - patch_size; ve++)
            {

                for (int we = me; we < me + window_size - patch_size; we++)
                {
                    cv::Mat patch_to_compare = GetROI(ve, we, img1, patch_size);
                    std::vector<double> cdfvec2 = GetCDF(patch_to_compare);
                    double d = GetMaxDiff(cdfvec1, cdfvec2, patch_size); // distance between the two patches
                    if (Muestras < 20)
                    {
                        if (d < 0.0076114) // from the distribution this value is taken
                        {
                            dist[20 * (ke) + (Totalwidth - window_size) * 20 * (me) + Muestras] = patch_to_compare.data[3 * 7 + 3]; // save the center pixelvalue
                            Muestras = Muestras + 1;
                        } // alpha=0.5 the difference is significant                         // saving all the distances now
                    }
                }
            }
            if (Muestras < 20)
            {
                while (Muestras < 20)
                {
                    dist[20 * (ke) + (Totalwidth - window_size) * 20 * (me) + Muestras] = Reference_patch.data[3 * 7 + 3]; // save the center pixel value of the reference
                    Muestras = Muestras + 1;
                }
            }
            // iterating over all window and comparing each patch
        }
    }
    // in one window get all the patches to compare

    // getting and printing the time
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Parallel Elapsed time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;

    // returning the vector in the format (x,y) from left to right and top to bottom) 53 column and rows with 20 values saved of positions of similar patches
    for (auto i = dist.begin(); i < dist.end(); i++)
    {
        std::cout << " " << *i;
    }
    return 1;
}

// Compilation:
// g++ NLMparalel.cpp -fpermissive $(pkg-config --cflags --libs opencv) -fopenmp -O3 -o Roi

// Usage:
// ./Roi 2 000031.jpg

//  copy faster the submatrix https://stackoverflow.com/questions/30952048/fastest-way-to-copy-some-rows-from-one-matrix-to-another-in-opencv