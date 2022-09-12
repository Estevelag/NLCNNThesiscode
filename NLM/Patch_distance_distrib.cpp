
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

void print(std::vector<int> const &a)
{
    std::cout << "The vector elements are : ";

    for (int i = 0; i < a.size(); i++)
        std::cout << a.at(i) << ' ';
}
void mMultiply(double *A, double *B, double *C, int m)
{
    int i, j, k;
    double s;
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
        {
            s = 0.;
            for (k = 0; k < m; k++)
                s += A[i * m + k] * B[k * m + j];
            C[i * m + j] = s;
        }
}

void mPower(double *A, int eA, double *V, int *eV, int m, int n)
{
    double *B;
    int eB, i;
    if (n == 1)
    {
        for (i = 0; i < m * m; i++)
            V[i] = A[i];
        *eV = eA;
        return;
    }
    mPower(A, eA, V, eV, m, n / 2);
    B = (double *)malloc((m * m) * sizeof(double));
    mMultiply(V, V, B, m);
    eB = 2 * (*eV);
    if (n % 2 == 0)
    {
        for (i = 0; i < m * m; i++)
            V[i] = B[i];
        *eV = eB;
    }
    else
    {
        mMultiply(A, B, V, m);
        *eV = eA + eB;
    }
    if (V[(m / 2) * m + (m / 2)] > 1e140)
    {
        for (i = 0; i < m * m; i++)
            V[i] = V[i] * 1e-140;
        *eV += 140;
    }
    free(B);
}
// Function that calculates the K value
double K(int n, double d)
{
    int k, m, i, j, g, eH, eQ;
    double h, s, *H, *Q;
    // OMIT NEXT LINE IF YOU REQUIRE >7 DIGIT ACCURACY IN THE RIGHT TAIL
    s = d * d * n;
    if (s > 7.24 || (s > 3.76 && n > 99))
        return 1 - 2 * exp(-(2.000071 + .331 / sqrt((double)n) + 1.409 / n) * s);
    k = (int)(n * d) + 1;
    m = 2 * k - 1;
    h = k - n * d;
    H = (double *)malloc((m * m) * sizeof(double));
    Q = (double *)malloc((m * m) * sizeof(double));
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            if (i - j + 1 < 0)
                H[i * m + j] = 0;
            else
                H[i * m + j] = 1;
    for (i = 0; i < m; i++)
    {
        H[i * m] -= pow(h, i + 1);
        H[(m - 1) * m + i] -= pow(h, (m - i));
    }
    H[(m - 1) * m] += (2 * h - 1 > 0 ? pow(2 * h - 1, m) : 0);
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            if (i - j + 1 > 0)
                for (g = 1; g <= i - j + 1; g++)
                    H[i * m + j] /= g;
    eH = 0;
    mPower(H, eH, Q, &eQ, m, n);
    s = Q[(k - 1) * m + k - 1];
    for (i = 1; i <= n; i++)
    {
        s = s * i / n;
        if (s < 1e-140)
        {
            s *= 1e140;
            eQ -= 140;
        }
    }
    s *= pow(10., eQ);
    free(H);
    free(Q);
    return s;
}

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
    std::vector<double> dist(550564);
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
            for (ve = ke; ve < ke + 21 - 7; ve++)
            {
                for (we = me; we < me + 21 - 7; we++)
                {
                    cv::Mat patch_to_compare = GetROI(ve, we, img1, 7);
                    std::vector<double> cdfvec2 = GetCDF(patch_to_compare);
                    double d = GetMaxDiff(cdfvec1, cdfvec2, 7); // distance between the two patches
                    dist[vecpos] = d;                           // saving all the distances now
                    vecpos++;
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

    cv::Mat mat = GetROI(0, 0, img1, 7);
    cv::Mat mat1 = GetROI(57, 0, img1, 7);
    // using the function to get the CDF of mat and mat1
    std::vector<double> cdfvec1 = GetCDF(mat);
    std::vector<double> cdfvec2 = GetCDF(mat1);
    double d = GetMaxDiff(cdfvec1, cdfvec2, 7); // Max Distance of the two cumulative distributions of the ROI

    int k;
    for (k = 0; k <= 10; k++)
    {
    }
    // viewing the cdf1
    std::cout << " /n El CDFfunc1 es: " << std::endl;
    // Print the cdf
    for (auto i = cdfvec1.begin(); i < cdfvec1.end(); i++)
    {
        std::cout << " " << *i;
    }
    std::cout << " Termina el vector" << std::endl;

    // viewing the cdf2
    std::cout << " /n El CDFfunc2 es: " << std::endl;
    // Print the cdf
    for (auto i = cdfvec2.begin(); i < cdfvec2.end(); i++)
    {
        std::cout << " " << *i;
    }
    std::cout << " Termina el vector" << std::endl;

    std::cout << "La distancia maxima es: " << d << std::endl;
    // performing the two sided test
    // value to test https://www.statisticshowto.com/wp-content/uploads/2016/07/k-s-test-table-p-value.png

    if (d < 0.274357) // alpha 0.05 1.358*sqrt(2*49/(49*49))=0.274357, alpha 0.01 = 0.328906
    {
        std::cout << "Se rechaza Diferencia de distribución" << std::endl;

    } // alpha=0.5 the difference is significant

    double Pval = 1 - K(49, d / 49); // P-value of the Kolmogorov-Smirnov test
    std::cout << "El P-value es: " << Pval << std::endl;
    return 0;
}

// Missing the iteration of the patches, getting the vector of a patch as a function, and having two patches. and The proof hypothesis

// not function that turns the mat into a cdf
//    std::vector<int> vec = mat.isContinuous() ? mat : mat.clone();
//
// sorting the vector
//    std::sort(vec.begin(), vec.end());
//    for (auto i = vec.begin(); i < vec.end(); i++)
//    {       std::cout << " " << *i;
//    }
// CDF
// the sum of it all
//    int tot = 0;
//    for (auto i = vec.begin(); i < vec.end(); i++)
//    {
//        tot = tot + *i;
//    }
//  std::vector<double> cdf1(block_width_and_height * block_width_and_height);
//    double ksum = 0;
//    int pos = 0;
//    // getting the CDF
//    for (int i = 0; i < 49; i++)
//    {
//        ksum = ksum + vec[i] / (double)tot;
//        cdf1[pos] = ksum;
//        pos = pos + 1;
//    }
//    std::cout << "/n El CDF es: ";
//    for (auto i = cdf1.begin(); i < cdf1.end(); i++)
//    {
//        std::cout << " " << *i;
//    }

// copy faster the submatrix https://stackoverflow.com/questions/30952048/fastest-way-to-copy-some-rows-from-one-matrix-to-another-in-opencv
// https://github.com/tumi8/topas/blob/master/detectionmodules/statmodules/wkp-module/ks-test.cpp