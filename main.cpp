//
//  main.cpp
//  LFdepthMap
//
//  Created by alwi husada on 11/30/16.
//  Copyright © 2016 alwi husada. All rights reserved.
//

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <glob.h>
#include <armadillo>
#include <math.h>
#include <time.h>
#include "GCoptimization.h"
#include <iomanip>

using namespace std;
using namespace cv;

#define IM_DIR "/Users/alwihusada/Desktop/Project/Datasets/medieval2/"
#define IM_EXT "png"

struct cv_params
{
    /* Parameters for syntatic Datasets */
//    const vector<float> Sc = {4,4};
//    const float alpha      = 0.3;
//    const double tau1      = 1;
//    const double tau2      = 3;
//    const double delta     = 0.03;
//    const int dataType     = 1;
    
    /* Parameters for Lytro Datasets (ie. Flowers)*/
    vector<float> Sc     = {3,3};
    float alpha          = 0.5;
    const double tau1    = 0.5;
    const double tau2    = 0.5;
    const double delta   = 0.02;
    const int dataType   = 2;
    
    const int numLabel   = 75;
    const vector<float> kernel = {1,1};
} cv_param;

struct cagg_params
{
    const int r = 5;
    const double eps  = 0.0001;
} cagg_param;

struct wmf_params
{
    const int denum_r = 40;
    const double eps  = 0.0001;
} wmf_param;

struct ir_params
{
    const int denum_r = 100;
    const double eps  = 0.0001;
    const int iternum = 4;
} ir_param;

static Mat convarma2cv(arma::mat arma_mat)
{
    cv::Mat opencv_mat((int)arma_mat.n_rows, (int)arma_mat.n_cols, CV_64F, arma_mat.memptr() );
    return opencv_mat;
}

static arma::mat convcv2arma(Mat opencv_mat)
{
    Mat opencv_tran;
    transpose(opencv_mat, opencv_tran);
    arma::mat arma_mat( reinterpret_cast<double*>(opencv_tran.data), opencv_tran.rows, opencv_tran.cols );
    return arma_mat;
}

static void meshgrid(const Mat1d &xgv, const Mat1d &ygv, Mat1d &X, Mat1d &Y)
{
    cv::repeat(xgv.reshape(1,1), ygv.total(), 1, X);
    cv::repeat(ygv.reshape(1,1).t(), 1, xgv.total(), Y);
}

/********************************** Pre-PROCESSING generate Light field images **************************************/
vector< vector<Mat> > make4dLight(string path_images , string ext)
{
    Mat src;
    vector<String> filenames;
    glob(path_images+"*"+ext, filenames);
    src = imread(filenames[0]);
    int N = sqrt(filenames.size());
    vector< vector<Mat> > lightimg(N , vector<Mat> (N));
    for (int i=0 ; i < N ; i++){
       for (int j =0 ; j < N; j++) {
          arma::uword n =arma::sub2ind(arma::size(N,N), j, i);
          src = imread(filenames[n]);
          lightimg[i][j].push_back(src);
       }
    }
    return lightimg;
}

void viewLightField (vector< vector<Mat> > LF)
{
    Mat img;
    size_t sz =LF.size();
    Mat im = LF[0][0];
    int imrows = im.rows;
    int imcols = im.cols;
    Mat resultImg(static_cast<int>(imrows * sz),static_cast<int>(imcols * sz),CV_64F);
    vector< vector<Mat> > bigimg(static_cast<int>(imrows * sz), vector<Mat>(imcols * sz));
    for (int i = 0 ; i < sz ; i++){
        for (int j =0 ; j < sz ; j++) {
            img = LF[i][j];
            namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
            imshow( "Figure 1", img );
            waitKey(80);
        }
    }
}

/********************************** COST VOLUME ***********************************
% 2015.05.12 Hae-Gon Jeon
% Accurate Depth Map Estimation from a Lenslet Light Field Camera
% CVPR 2015
***********************************************************************************/

void swapQuadrant(Mat magI)
{
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2)); // crop if it has an odd number of rows or columns
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void createKernel(Mat &kernel ,vector<double> delta)
{
    vector<double> veccl(kernel.cols);
    vector<double> vecrw(kernel.rows);
    int cy = kernel.cols/2;
    int cx = kernel.rows/2;
    int i;
    for (i=0; i < cy; i++) {
        veccl[i]=i;
    }
    while (i< kernel.cols) {
        veccl[i]= -cy;
        cy--;
        i++;
    }
    int j;
    for (j=0; j < cx; j++) {
        vecrw[j]=j;
    }
    while (j< kernel.rows) {
        vecrw[j]= -cx;
        cx--;
        j++;
    }
    Mat1d X, Y;
    meshgrid(Mat1d(veccl), Mat1d(vecrw), X, Y);
    kernel = (delta[0]*X/kernel.rows) + (delta[1]*Y/kernel.cols);
}

Mat fft2(Mat src)
{
    Mat planes[]       = {Mat_<double>(src ), Mat::zeros(src.size(), CV_64F)};
    Mat complex_fft2;
    merge(planes, 2, complex_fft2);
    dft(complex_fft2, complex_fft2, DFT_COMPLEX_OUTPUT);
    
    return complex_fft2;
}

Mat pixel_shift(Mat fft2,Mat src, vector<double> delta, int mode)
{
 
    const double phase = 2.0;
    Mat planes[]       = {Mat_<double>(src ), Mat::zeros(src.size(), CV_64F)};
    Mat kernel(fft2.rows, fft2.cols, CV_64F, Scalar::all(0));
    createKernel(kernel , delta);

    Mat cx_kernel(fft2.rows, fft2.cols, fft2.type(), Scalar::all(0));
    complex<double> cx(0.0,1.0);
    for (int i=0; i< fft2.rows;i++){
        for (int j=0; j< fft2.cols; j++){
            cx_kernel.at<complex<double>>(i,j) = exp(cx * phase * M_PI *  kernel.at<double>(i,j));
        }
    }
    Mat complexI;
    merge(planes, 2, complexI);
    mulSpectrums(fft2, cx_kernel, complexI, 0);
    idft(complexI, complexI, DFT_COMPLEX_OUTPUT+DFT_SCALE,0);
    for (int i=0; i< complexI.rows;i++){
        for (int j=0; j< complexI.cols; j++){
            complexI.at<complex<double>>(i,j) = complexI.at<complex<double>>(i,j) * exp(-cx*phase);
        }
    }
    split(complexI, planes);
    magnitude(planes[0], planes[1], complexI);
    if (mode == 0) {
        cv::max(complexI, 0.00, complexI);
        cv::min(complexI, 1.00, complexI);
    }
    //    Mat ret_val;
    //    complexI(Range(0, complexI.rows - (m - src.rows)), Range(0, complexI.cols-(n - src.rows))).copyTo(ret_val);
    return complexI;
}

vector< Mat > costVolume(vector< vector<Mat> > LF, cv_params &param)
{
    vector<float> Sc     = param.Sc;
    float alpha          = param.alpha;
    const double tau1    = param.tau1;
    const double tau2    = param.tau1;
    const double delta   = param.delta;
    const int dataType   = param.dataType;
    const int numLabel   = param.numLabel;
    vector<float> kernel = param.kernel;
    
    Mat p_im;
    LF[Sc[0]][Sc[1]].convertTo(p_im, CV_64F, 1.0/255.0);
    
    size_t N             = LF.size();
    const vector<int> windowSize = {1,1};
    vector<Mat> cost_vol(numLabel);
    vector<Mat> cost_result(numLabel);
    Mat grad_sum         = Mat(p_im.rows,p_im.cols, CV_64FC3, 0.0);
    int grad_scale       = 1;
    int grad_delta       = 0;
    int grad_ddepth      = CV_64F;
    Mat grad_x;
    Mat grad_y;
    Mat src;
    
    Sobel(p_im, grad_x, grad_ddepth, 1, 0, 3, grad_scale, grad_delta, BORDER_REPLICATE );
    Sobel(p_im, grad_y, grad_ddepth, 0, 1, 3, grad_scale, grad_delta, BORDER_REPLICATE );
    
    vector <Mat> gradX_cen_gray(3);
    vector <Mat> gradY_cen_gray(3);
    vector <Mat> gray_cen(3);
    
    split (grad_x, gradX_cen_gray);
    split (grad_y, gradY_cen_gray);
    split (p_im, gray_cen);
    
    vector<Mat> target_grad_x(3);
    vector<Mat> target_grad_y(3);
    vector<Mat> Ftarget(3);
    vector<Mat> Ftarget_x(3);
    vector<Mat> Ftarget_y(3);
    vector<Mat> bgr(3);
    
    vector<Mat> fft_src_res(3);
    vector<Mat> fft_gradX_res(3);
    vector<Mat> fft_gradY_res(3);
    
    for (int i=0 ; i < N ; i++){
        for (int j =0 ; j < N; j++) {
            
            if (Sc[0] == i && Sc[1] == j){
                continue;
            }
            vector<int> ind = {j,i};
            vector<int> flags = {i,j};
            vector<double> vn ;
           
            src = LF[i][j];
            src.convertTo(src, CV_64F, 1.0/255.0);
            split(src, bgr);
 
            Sobel(bgr[0], target_grad_x[0], grad_ddepth, 1, 0, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            Sobel(bgr[1], target_grad_x[1], grad_ddepth, 1, 0, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            Sobel(bgr[2], target_grad_x[2], grad_ddepth, 1, 0, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            
            Sobel(bgr[0], target_grad_y[0], grad_ddepth, 0, 1, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            Sobel(bgr[1], target_grad_y[1], grad_ddepth, 0, 1, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            Sobel(bgr[2], target_grad_y[2], grad_ddepth, 0, 1, 3, grad_scale, grad_delta, BORDER_REPLICATE );
            
            for (int i=0; i < bgr.size(); i++) {
                fft_src_res[i] = fft2(bgr[i]);
            }
            for (int i=0; i < bgr.size(); i++) {
                fft_gradX_res[i] = fft2(target_grad_x[i]);
            }
            for (int i=0; i < bgr.size(); i++) {
                fft_gradY_res[i] = fft2(target_grad_y[i]);
            }
            subtract(ind, Sc ,vn );
            vector <double> deltarc;
            double beta = abs(vn[0])/ (abs(vn[0])+ abs(vn[1]));
            for (int cn = 1; cn <= numLabel; cn++) {
                if (dataType==0) {
                    deltarc = { -delta*vn[0]*((double)cn - (double)numLabel/2), delta*vn[1]*((double)cn - (double)numLabel/2)};
                } else if (dataType==1){
                    deltarc = { -delta*vn[0]*((double)cn - (double)numLabel/2), -delta*vn[1]*((double)cn - (double)numLabel/2)};
                } else {
                    deltarc = { -delta*vn[0]*(double)cn, -delta*vn[1]*(double)cn};
                }
                // ++++++++++++++ ORIGINAL image DFT +++++++++++++++++++++++++++++++++++++++++
                vector< Mat> abs_diff(3);
                for (int i=0; i < bgr.size(); i++ ) {
                    Ftarget[i] = pixel_shift(fft_src_res[i], bgr[i], deltarc, 0);
                    abs_diff[i] = Ftarget[i] - gray_cen[i];
                    pow(abs_diff[i],2, abs_diff[i]);
                }
    
                Mat color_src = abs_diff[0] + abs_diff[1] + abs_diff[2];
                sqrt(color_src, color_src);
                cv::min(color_src, (double)tau1, color_src);
                filter2D(color_src, color_src, -1, 1, Point(-1,-1), 0, BORDER_REPLICATE);
                
                // ++++++++++++++ Grdient X direc. image DFT ++++++++++++++++++++++++++++++++
                vector<Mat> abs_diff_gradX(3);
                for (int i=0; i < bgr.size(); i++ ) {
                    Ftarget_x[i] = pixel_shift(fft_gradX_res[i], target_grad_x[i], deltarc, 1);
                    abs_diff_gradX[i] = Ftarget_x[i] - gradX_cen_gray[i];
                    pow(abs_diff_gradX[i],2, abs_diff_gradX[i]);
                }
                Mat color_gradX  = abs_diff_gradX[0] + abs_diff_gradX[1] + abs_diff_gradX[2];
                sqrt(color_gradX, color_gradX);
                cv::min(color_gradX, (double)tau2, color_gradX);
                filter2D(color_gradX, color_gradX, -1, 1, Point(-1,-1), 0, BORDER_REPLICATE);

                // ++++++++++++++ Grdient Y direc. image DFT ++++++++++++++++++++++++++++++++
                vector<Mat> abs_diff_gradY(3);
                for (int i=0; i < bgr.size(); i++ ) {
                    Ftarget_y[i] = pixel_shift(fft_gradY_res[i],target_grad_y[i], deltarc, 1);
                    abs_diff_gradY[i] = Ftarget_y[i] - gradY_cen_gray[i];
                    pow(abs_diff_gradY[i],2,abs_diff_gradY[i]);
                }
                
                Mat color_gradY  = abs_diff_gradY[0] + abs_diff_gradY[1] + abs_diff_gradY[2];
                sqrt(color_gradY, color_gradY);
                cv::min(color_gradY, (double)tau2, color_gradY);
                filter2D(color_gradY, color_gradY, -1, 1, Point(-1,-1), 0, BORDER_REPLICATE);
                
                Mat grad1 = (beta*color_gradX)+((1-beta)*color_gradY);
                 /*++++++++++++++ Cost VOLUME ++++++++++++++++++++++++++++++++*/
                Mat grad_sum_temp =((1.0-alpha)*color_src) + (alpha*grad1);
                cost_vol[cn-1]=grad_sum_temp;
                
                if (flags[0] == 0 &&  flags[1] == 0)
                    cost_result[cn-1].push_back(cost_vol[cn-1]);
                else
                    cost_result[cn-1] +=  cost_vol[cn-1];

            }
            cout << "COST VOLUME .... " << i*N+j <<" / "<<N*N-1<<endl;
//            for(int i=0; i<numLabel; ++i)
//                cost_result[i] = cost_vol[i] + cost_result[i];
        }
    }
    return cost_result;
}

/************************************* COST AGGREGATION *******************************************
  C Rhemann et al.,
  Fast cost-volume filtering for visual correspondence and beyond
  CVPR 2011
**************************************************************************************************/

/*
 * Domain Transform Filter (RF)
 * Recursive Filter : (http://inf.ufrgs.br/~eslgastal/DomainTransform/ )
 *
 * Eduardo SL Gastal and Manuel M Oliveira. Domain transform for edge-aware image and video
 * processing. In ACM Transactions on Graphics (ToG), volume 30, page 69. ACM, 2011.
 *
 */
Mat  transformedDomain_recursive_f(Mat &I, Mat img, double sigma)
{
    double a = exp(-sqrt(2) / sigma);
    Mat V = Mat(img.rows, img.cols, img.type());
    for (int i =0; i< img.rows; i ++){
        for (int j =0; j< img.rows; j ++){
            V.at<double>(i,j) = pow(a, img.at<double>(i,j));
        }
    }
    for (int i= 1; i < I.cols; i++){
        I.col(i) =  I.col(i) + V.col(i).mul( (I.col(i-1) - I.col(i)));
    }
    for (int width = I.cols -2; width >= 0; width--){
        I.col(width) =  I.col(width) + V.col(width+1).mul((I.col(width+1) - I.col(width)));
    }
    return I;
}

Mat domain_transform_f(const Mat &I, double sigma_s, double sigma_r,int num_iter, Mat im_joint)
{
    Mat im = I.clone();
    Mat hor_0 = im_joint.colRange(0, im_joint.cols-1 ).clone();
    Mat hor_1 = im_joint.colRange(1, im_joint.cols ).clone();
    Mat ver_0 = im_joint.rowRange(0, im_joint.rows-1).clone();
    Mat ver_1 = im_joint.rowRange(1, im_joint.rows).clone();
    
    Mat dIcdx;
    Mat dIcdy;
    absdiff(hor_0, hor_1 , dIcdx);
    absdiff(ver_0, ver_1 , dIcdy);

    Mat cols= Mat::zeros(im_joint.rows, 1, CV_64F);
    Mat rows= Mat::zeros(1, im_joint.cols, CV_64F);
    
    transform(dIcdx, dIcdx, Matx13d(1,1,1));
    transform(dIcdy, dIcdy, Matx13d(1,1,1));
    
    Mat dIdx;
    Mat dIdy;
    hconcat(cols, dIcdx, dIdx);
    vconcat(rows, dIcdy, dIdy);
    
    Mat dHdx = (1 + sigma_s/sigma_r * dIdx);
    Mat dVdy = (1 + sigma_s/sigma_r * dIdy);
    transpose(dVdy, dVdy);
    
    double sigma_H = sigma_s;
    
    for (int i=0; i< num_iter; i++){
        double sigma_H_i =sigma_H * sqrt(3) * pow(2,(num_iter - (i + 1))) / sqrt(pow(4,num_iter )- 1);
        transformedDomain_recursive_f(im, dHdx, sigma_H_i);
        transpose(im, im);
        transformedDomain_recursive_f(im, dVdy, sigma_H_i);
        transpose(im, im);
    }
    return im;
}

/*
 * Guided Filter
 * Modified from this implementation: ( https://github.com/atilimcetin/guided-filter )
 */
class GuidedF
{
public:
    GuidedF(Mat &guide_im, double eps, int r);
    Mat filter(const Mat &p)const;
    
private:
    float eps;
    int r;
    Mat N;
    Mat invrr, invrg,  invrb, invgg, invgb, invbb;
    Mat mean_I_r,  mean_I_g, mean_I_b;
    vector<Mat> Ichannels;
};

static Mat box_filter(Mat mat_src, double r)
{
    arma::mat arma_kernel = convcv2arma(mat_src);
    arma::mat imDst(arma_kernel.n_cols,arma_kernel.n_rows);
    imDst.zeros();
    arma::mat imcum = arma::cumsum(arma_kernel);
    imDst.rows(0, r) = imcum.rows(r, 2*r);
    imDst.rows(r+1, arma_kernel.n_rows-1-r) = imcum.rows(2*r+1, arma_kernel.n_rows-1) - imcum.rows(0,(arma_kernel.n_rows-1)-2*r-1);
    imDst.rows((arma_kernel.n_rows-1)-r+1, arma_kernel.n_rows-1) = repmat(imcum.row(arma_kernel.n_rows-1), r, 1) - imcum.rows((arma_kernel.n_rows-1)-2*r,(arma_kernel.n_rows-1)-r-1);
    imcum = arma::cumsum(imDst, 1);
    imDst.cols(0,r) = imcum.cols(r,2*r);
    imDst.cols( r+1, arma_kernel.n_cols-1-r) = imcum.cols(2*r+1,arma_kernel.n_cols-1) - imcum.cols(0, (arma_kernel.n_cols-1)-2*r-1);
    imDst.cols(arma_kernel.n_cols-1-r+1,arma_kernel.n_cols-1) = repmat(imcum.col(arma_kernel.n_cols-1), 1, r) - imcum.cols( arma_kernel.n_cols-1-2*r,arma_kernel.n_cols-1-r-1);
    
    mat_src = convarma2cv(imDst).clone();
    return mat_src;
}

GuidedF::GuidedF(Mat &guide_im, double eps, int r):eps(eps),r(r)
{
    split(guide_im, Ichannels);
    
    //N = Mat::zeros(guide_im.rows, guide_im.cols, CV_32SC1);
    Mat N_ones = Mat::ones(guide_im.rows, guide_im.cols, CV_64FC1);
    N = box_filter(N_ones, r);
    
    mean_I_r = box_filter(Ichannels[0], r)/N;
    mean_I_g = box_filter(Ichannels[1], r)/N;
    mean_I_b = box_filter(Ichannels[2], r)/N;
    
    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    Mat var_I_rr = box_filter(Ichannels[0].mul(Ichannels[0]), r)/N - mean_I_r.mul(mean_I_r) + eps;
    Mat var_I_rg = box_filter(Ichannels[0].mul(Ichannels[1]), r)/N - mean_I_r.mul(mean_I_g);
    Mat var_I_rb = box_filter(Ichannels[0].mul(Ichannels[2]), r)/N - mean_I_r.mul(mean_I_b);
    Mat var_I_gg = box_filter(Ichannels[1].mul(Ichannels[1]), r)/N - mean_I_g.mul(mean_I_g) + eps;
    Mat var_I_gb = box_filter(Ichannels[1].mul(Ichannels[2]), r)/N - mean_I_g.mul(mean_I_b);
    Mat var_I_bb = box_filter(Ichannels[2].mul(Ichannels[2]), r)/N - mean_I_b.mul(mean_I_b) + eps;
    
    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);
    
    Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);
    
    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

Mat GuidedF::filter(const Mat &p)const
{
    Mat dispI;
    p.convertTo(dispI,Ichannels[0].type() );
    
    Mat mean_p = box_filter(dispI, r)/N;
    
    Mat mean_Ip_r = box_filter(Ichannels[0].mul(dispI), r)/N;
    Mat mean_Ip_g = box_filter(Ichannels[1].mul(dispI), r)/N;
    Mat mean_Ip_b = box_filter(Ichannels[2].mul(dispI), r)/N;
    
    // covariance of (I, p) in each local patch.
    Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);
    
    Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);
    
    Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);
    
    return (box_filter(a_r, r).mul(Ichannels[0])
            + box_filter(a_g, r).mul(Ichannels[1])
            + box_filter(a_b, r).mul(Ichannels[2])
            + box_filter(b, r))/N;
}

void winner_take_all(vector<Mat> cost_result, Mat LF_I)
{
    int cols     = cost_result[0].cols;
    int rows     = cost_result[0].rows;
    int labels   = (int)cost_result.size();

    Mat tempMat;
    Mat dispar = Mat(rows, cols, CV_8UC1, Scalar(0));

    for (int i=0; i<labels; i++) {
        if (i==0) cost_result[i].copyTo(tempMat);
        for (int j=0; j<rows; j++) {
            for (int k=0;k<cols;k++) {
                if (cost_result[i].at<double>(j,k) < tempMat.at<double>(j,k)){
                    tempMat.at<double>(j,k) = cost_result[i].at<double>(j,k);
                    dispar.at<char>(j,k) = static_cast<char>(i);
                }
            }
        }
    }
    
    double min_x;
    double max_x;
    cv::minMaxIdx(dispar, &min_x, &max_x);
    cv::Mat adjMap;
    //cv::convertScaleAbs(ret_val, adjMap, 255 / max);
    dispar.convertTo(adjMap,CV_8UC1, 255 / (max_x-min_x), -min_x);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, COLORMAP_PARULA);
    namedWindow("WIN_ta_BW_1",CV_WINDOW_AUTOSIZE);
    namedWindow("WIN_ta_CM_1",CV_WINDOW_AUTOSIZE);
    cv::imshow("WIN_ta_BW_1 ", adjMap);
    cv::imshow("WIN_ta_CM_1", falseColorsMap);
    waitKey(45);
}

vector< Mat > costAgg(vector< Mat > cost_result, Mat LF_I, cagg_params &param)
{
    const int r = param.r;
    double eps  = param.eps;
    GuidedF GuidedF(LF_I, eps, r);
    for (int i=0 ; i < cost_result.size() ; i++){
        /**** Guided filter ****/
        cost_result[i] = GuidedF.filter(cost_result[i]);
        
        /**** Bilateral filter ****/
        //Mat costslice;
        //Mat costslice_r;
        //cost_result[i].convertTo(costslice, CV_8U);
        //bilateralFilter(costslice, costslice_r, 5, 15, 15);
        //costslice_r.convertTo(cost_result[i], cost_result[i].type());
        
        /**** Domain Transform (Recursive Filter) ****/
        //cost_result[i] = domain_transform_f(cost_result[i], 60, 0.2, 3, LF_I);
    }
    
    return cost_result;
}


/********************************* GRAPH CUT  *******************************************
 * GRAPH CUT
 * gco-v3.0: Multi-label optimization (http://vision.csd.uwo.ca/code/)
 *
 * [4] Fast Approximate Energy Minimization with Label Costs.
 * A. Delong, A. Osokin, H. N. Isack, Y. Boykov. In CVPR, June 2010.
 *
 ***************************************************************************************/

vector<double> vectorise_f(Mat mat)
{
    arma::mat arma_mat  = convcv2arma(mat);
    arma::vec arma_vect = arma::vectorise(arma_mat);
    vector<double> vect(arma_vect.size());
    for (int i=0; i< arma_vect.size(); ++i){
        vect[i] = arma_vect(i);
    }
    return vect;
}

vector<double> line_space(double start, double end, int dim)
{
    arma::vec a = arma::linspace<arma::vec>(start, end, dim);
    vector<double> vect(a.size());
    for (int i=0; i < a.size(); i++) {
        vect[i] = a[i];
    }
    return vect;
}

vector<int> quantiz(vector<double> seg, vector<double> partition)
{
    vector<int32_t> indices(seg.size(),0);
    for (int i=0; i< partition.size(); ++i){
        for(int j=0;j<seg.size();++j){
            if (seg[j] > partition[i] ){
                indices[j] = indices[j]+ 1;
            }
        }
    }
    return indices;
}

Mat graphCuts(vector<Mat> img, Mat guide_im)
{
    //Assuming that all images in vector<Mat> have same size. eg all 75 images have dim. 512x512
    Mat img_cost   = img[0];
    int width      = img_cost.cols;
    int height     = img_cost.rows;
    int num_pixels = width*height;
    int num_labels = (int) img.size();
    int *result    = new int[num_pixels];
    int numQuantiz = 3*num_labels;
   
    vector<double> vec_vect(num_pixels*num_labels);
    
    Mat data_src = Mat(num_labels, width*height, CV_64F);;
    int ind = 0;
    for (int l = 0; l < num_labels; l++ ){
        vector<double> vect(img[l].rows*img[l].cols);
        vect= vectorise_f(img[l]);
        for (int i = 0;i<vect.size();i++) {
            data_src.at<double>(ind, i)= vect[i];
        }
        ind++;
    }
    // Data cost parameters
    double min                = 0.0;
    double max                = 0.0;
    minMaxLoc(data_src, &min, &max);
    vector<double> linespaced =line_space(min, max, numQuantiz);
    vec_vect=vectorise_f(data_src);
    vector<int> indices       =quantiz(vec_vect ,linespaced);

    int *data = new int[num_pixels*num_labels];
    for ( int i = 0; i < num_pixels*num_labels; i++ ) {
        data[i] = (int32_t)indices[i]*2;
    }
    
    // Smoothness parameters
    arma::vec smooth_vect = arma::linspace<arma::vec>(0, num_labels-1, num_labels );
    arma::mat cir_mat = toeplitz(smooth_vect);
    arma::Mat<int> smoothness = arma::conv_to<arma::Mat<int>>::from(cir_mat);
    for (int a =0 ; a< smoothness.n_rows;++a) {
        for (int b =0 ; b< smoothness.n_cols;++b) {
            if (smoothness(a,b) > 10) smoothness(a,b) = 10;
        }
    }

    int *smooth = new int[num_labels*num_labels];
    for ( int l1 = 0; l1 < smoothness.n_rows; l1++ ) {
        for (int l2 = 0; l2 < smoothness.n_cols; l2++ ) {
            smooth[l1+l2*num_labels] = (int)smoothness(l1,l2);
        }
    }
  
    Mat hor_0 = guide_im.colRange(0, guide_im.cols-1 ).clone();
    Mat hor_1 = guide_im.colRange(1, guide_im.cols ).clone();
    Mat ver_0 = guide_im.rowRange(0, guide_im.rows-1).clone();
    Mat ver_1 = guide_im.rowRange(1, guide_im.rows).clone();
    
    vector <Mat> hor_0gray, hor_1gray,hor_sum(3),ver_0gray, ver_1gray, ver_sum(3);
    
    split(hor_0, hor_0gray);
    split(hor_1, hor_1gray);
    split(ver_0, ver_0gray);
    split(ver_1, ver_1gray);
    
    for (int cn=0; cn < 3; cn++){
        hor_sum[cn] = hor_0gray[cn] - hor_1gray[cn];
        pow(hor_sum[cn], 2, hor_sum[cn]);
    }
    
    Mat hor_w = hor_sum[0] + hor_sum[1] + hor_sum[2];
    sqrt(hor_w, hor_w);

    minMaxLoc(hor_w, &min, &max);
    vector<double> hor_linespaced = line_space(0, max, numQuantiz);
    transpose(hor_w, hor_w);
    vector<double> hor_vect       = vectorise_f(hor_w);
    vector<int> q_hor_w           = quantiz(hor_vect ,hor_linespaced);
    auto biggest_h                = max_element(begin(q_hor_w), end(q_hor_w));
    subtract(*biggest_h, q_hor_w, q_hor_w);
    
    
    for (int cn=0; cn < 3; cn++){
        ver_sum[cn] = ver_0gray[cn] - ver_1gray[cn];
        pow(ver_sum[cn], 2, ver_sum[cn]);
    }
    
    Mat vert_w = ver_sum[0] + ver_sum[1] + ver_sum[2];
    sqrt(vert_w, vert_w);
    
    minMaxLoc(vert_w, &min, &max);
    vector<double> vert_linespaced = line_space(0, max, numQuantiz);
    transpose(vert_w, vert_w);
    vector<double> vert_vect       = vectorise_f(vert_w);
    vector<int> q_vert_w           = quantiz(vert_vect ,vert_linespaced);
    auto biggest                   = max_element(begin(q_vert_w), end(q_vert_w));
    subtract(*biggest, q_vert_w, q_vert_w);

    try{
        GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
        gc->setDataCost(data);
        gc->setSmoothCost(smooth);
        
        // now set up a grid neighborhood system
        // first set up horizontal neighbors
        int count = 0;
        for (int y = 0; y < height; y++ ) {
            for (int  x = 1; x < width; x++ ){
                int p1 = x-1+y*width;
                int p2 =x+y*width;
                gc->setNeighbors(p1,p2,q_hor_w[count]*0.009);
                count++;
            }
        }

        // next set up vertical neighbors
        int count_v = 0;
        for (int y = 1; y < height; y++ ) {
            for (int  x = 0; x < width; x++ ){
                int p1 = x+(y-1)*width;
                int p2 =x+y*width;
                gc->setNeighbors(p1,p2,q_vert_w[count_v]*0.009);
                count_v++;
            }
        }
        
        printf("\nBefore optimization energy is %lld",gc->compute_energy());
        gc->expansion();// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        //gc->swap(2);
        printf("\nAfter optimization energy is %lld \n",gc->compute_energy());
        for ( int  i = 0; i < num_pixels; i++ ) {
            result[i] = gc->whatLabel(i);
        }
        delete gc;
    }
    catch (GCException e){
        e.Report();
    }
    
    Mat ret_val = Mat(width,height,CV_32SC1, result).clone() ;
    transpose(ret_val, ret_val);

    return ret_val;
}

/*********************************************** WMT **********************************************
 * Z Ma et al.,
 * Constant Time Weighted Median Filtering for Stereo Matching and Beyond
 * ICCV 2013
 **************************************************************************************************/

void weighted_median_f(Mat &refined_mat , Mat guide_im, wmf_params &param)
{
    Mat dispImg = refined_mat.clone();
    
    const double eps = param.eps ;
    int r = ceil(max(guide_im.rows,guide_im.cols)/param.denum_r);//40: follow papers: Ziyang Ma et al.
    double min;
    double max;
    minMaxLoc(dispImg, &min, &max);
    max = max + 1;
    
    Mat unfilter_im = Mat::zeros(dispImg.rows,dispImg.cols, CV_64F);
    Mat accum_im = Mat::zeros(dispImg.rows,dispImg.cols, CV_64FC1);
    Mat dispImg_out = Mat::zeros(dispImg.rows,dispImg.cols, dispImg.type());
    
    GuidedF GuidedF(guide_im, eps, r);
    for (int dis=0; dis < max ;dis++) {

        unfilter_im.setTo(1, dispImg == dis);
        //Mat filtered_im = domain_transform_f(unfilter_im, 60, 0.2, 3, guide_im);
        Mat filtered_im =GuidedF.filter(unfilter_im);
        unfilter_im = 0;
        accum_im =  accum_im + filtered_im;

        for (int i=0; i < accum_im.rows; i++){
            for (int j=0; j< accum_im.cols; j++){
                if (accum_im.at<double>(i,j) > 0.5 && dispImg_out.at<int>(i,j) == 0)
                    dispImg_out.at<int>(i,j) = dis;
            }
        };
        cout << "MEDIAN FILT ... " << dis << " / " << max <<endl;
    }
    
    Mat adj_Map;
    dispImg_out.convertTo(adj_Map,CV_8UC1);
    medianBlur(adj_Map, refined_mat, 3);
}

/*************************** ITERATIVE REFINEMENT  ******************************
% 2010.06.14 Jaesik Park
% implementation of Spatial-Depth Super Resolution for Range Images
% CVPR 2007
*********************************************************************************/

vector<Mat> makeCostVolume(Mat &disp_I, double disp_max)
{
    const int search_r = 5;
    const int eta      = 1;
    vector<Mat> costVol_out(disp_max);
    Mat diff_depthDis;
    Mat temp = Mat(disp_I.rows, disp_I.cols, disp_I.type());
    
    for (int i =0; i< disp_max; i++ ){
        subtract((double)i, disp_I, diff_depthDis);
        pow(diff_depthDis, 2 , diff_depthDis);
        cv::min( (double)eta*search_r, diff_depthDis, temp);
        costVol_out[i].push_back(temp);
    }

    return costVol_out;
}

vector<Mat> WMF4IterRefine(vector<Mat> cost, Mat guide_im, double disp_max, double eps, int r)
{
    GuidedF GuidedF(guide_im, eps, r);
    
    for (int dis=0; dis < disp_max ;dis++) {
        //for (int i=0; i < cost.size(); i++){
            cost[dis] =GuidedF.filter(cost[dis]);
//             cost[dis] = domain_transform_f(cost[dis], 60, 0.2, 3, guide_im);
        //}
    }
    
    return cost;
}

Mat selectMinimumCost(vector<Mat> cost, double disp_max)
{
    Mat dmap = Mat::zeros(cost[0].rows, cost[0].cols, cost[0].type());
    Mat minmap = Mat::ones(cost[0].rows, cost[0].cols, cost[0].type())*100;
    
    for (int dis=0; dis< cost.size(); dis++) {
        for (int j=0; j< minmap.rows; j++){
            for (int k=0; k< minmap.cols; k++){
                if (cost[dis].at<double>(j,k) < minmap.at<double>(j,k)) {
                    minmap.at<double>(j,k) = cost[dis].at<double>(j,k);
                    dmap.at<double>(j,k) = (double)dis;
                }
            }
        }
    }
    return dmap;
}

Mat disp2Idx(Mat depth, double disp_max)
{
    Mat idmap = Mat::zeros(depth.rows, depth.cols, depth.type());
    for (int dis= 0; dis < disp_max; dis++){
        for (int i=0; i< depth.rows; i++){
            for (int j=0; j< depth.rows; j++){
                if (depth.at<double>(i,j) == dis)
                    idmap.at<double>(i,j) = (double) dis;
            }
        }
    }
    return idmap;
}

Mat subpixelRefinement(Mat dmap_h, vector<Mat> cost, double disp_max)
{
    //Mat idmap = disp2Idx(dmap_h, disp_max);
    Mat idmap = dmap_h.clone();
    Mat costslice_m;
    Mat costslice_c;
    Mat costslice_p;
    Mat depth_sub = Mat::zeros(dmap_h.rows, dmap_h.cols, dmap_h.type());
    
    for (int dis = 1; dis < disp_max-1; dis++){
        costslice_m = cost[dis-1];
        costslice_c = cost[dis];
        costslice_p = cost[dis+1];
        
        for (int i=0; i< idmap.rows; i++){
            for (int j =0; j< idmap.cols; j++){
                if (idmap.at<double>(i,j) == (double)dis){
                    double f_dm= costslice_m.at<double>(i,j);
                    double f_dc= costslice_c.at<double>(i,j);
                    double f_dp= costslice_p.at<double>(i,j);
                    
                    depth_sub.at<double>(i,j) = (double)(dis) - (f_dp-f_dm)/(2.0*(f_dp+f_dm)-2.0*f_dc);
                }
            }
        }
    }
    return depth_sub;
}

void iter_refine(Mat &dispMap , Mat guide_im, ir_params &param)
{
    Mat disp_I;
    dispMap.convertTo(disp_I, CV_64FC1);
    
   // disp_I = disp_I + 1;
    double disp_min;
    double disp_max;
    minMaxLoc(disp_I, &disp_min, &disp_max);
    disp_max = disp_max+1;
    
    const double eps = param.eps;
    int r = ceil(max(guide_im.rows,guide_im.cols)/param.denum_r);
    
    vector<Mat> cost = makeCostVolume(disp_I, disp_max);
    Mat dmap_h;
    Mat dmap_r = Mat::zeros(dispMap.rows, dispMap.cols, dispMap.type());
    
    int iternum = param.iternum;
    
    for (int i =0; i < iternum; i++){
        cost   = WMF4IterRefine(cost, guide_im, disp_max, eps, r);
        dmap_h = selectMinimumCost(cost, disp_max);
        dmap_r = subpixelRefinement(dmap_h, cost, disp_max);
        cost   = makeCostVolume(dmap_r, disp_max);
        
        cout << "ITER REFINE .... " << i <<" / "<<iternum<<endl;
    }
    
    dispMap = dmap_r.clone();
}

/************************************** MSD ***********************************************/

void mean_square_diff (const Mat &im1 , const Mat &im2)
{
    size_t rows = im1.rows;
    size_t cols = im1.cols;
    Mat result;
    absdiff(im1, im2, result);
    pow(result, 2, result);
    double err =sum(result)[0];
    
    double mse = err / (rows * cols);

    cout << "Means Square Errors: " << mse <<endl;
    
}

/************************************** MAIN ***********************************************/
int main(int ac, char** av)
{
    clock_t tStart;
    vector< vector<Mat> > LF = make4dLight(IM_DIR, IM_EXT);
    Mat im_guide;
    LF[cv_param.Sc[0]][cv_param.Sc[1]].convertTo(im_guide, CV_64F, 1.0/255.0);
    //viewLightField(LF);
    
    tStart = clock();
    vector< Mat > cost_vol =costVolume(LF, cv_param);
    cout << "COST VOL. running time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << endl;
    //winner_take_all(cost_vol, im_guide);

    tStart = clock();
    vector< Mat > cost_agg = costAgg(cost_vol, im_guide, cagg_param);
    cout << "COST AGG. running time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << endl;
    winner_take_all(cost_agg, im_guide);
    
    tStart = clock();
    //Mat refined_gc = Mat(im_guide.rows,im_guide.cols,CV_32SC1);
    Mat refined_gc = graphCuts(cost_agg, im_guide);
    cout << "GRAPH CUT running time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << endl;
    
    tStart = clock();
    weighted_median_f(refined_gc, im_guide, wmf_param);
    cout << "WMT. running time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << endl;

    tStart = clock();
    iter_refine(refined_gc, im_guide, ir_param);
    cout << "ITERATIVE REF. running time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << endl;
    
    double min_x;
    double max_x;
    cv::minMaxIdx(refined_gc, &min_x, &max_x);
    cv::Mat adjMap;
    refined_gc.convertTo(adjMap,CV_8UC1, 255 / (max_x-min_x), -min_x);

    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_PARULA);
    namedWindow( "FIN_BW",CV_WINDOW_AUTOSIZE);
    namedWindow( "FIN_CM",CV_WINDOW_AUTOSIZE);
    cv::imshow("FIN_BW", adjMap);
    cv::imshow("FIN_CM", falseColorsMap);
    waitKey();
    
    /* MEANS SQUARE DIFFERENCE */
//    Mat im1 =imread("/Users/alwihusada/Desktop/Matlab_res/myimage_ir.png");
//    Mat im1_bw;
//    Mat im1_d;
//    cvtColor( im1, im1_bw, CV_BGR2GRAY );
//    im1_bw.convertTo(im1_d, CV_64F, 1.0/255.0);
//    
//    Mat im2 =imread("/Users/alwihusada/Desktop/C++_res/myimage_ir.png");
//    Mat im2_bw;
//    Mat im2_d;
//    cvtColor( im2, im2_bw, CV_BGR2GRAY );
//    im2_bw.convertTo(im2_d, CV_64FC1, 1.0/255.0);
//
//    mean_square_diff(im1_d, im2_d);
    return 0;
}
