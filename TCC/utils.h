//C++ and C librarys
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
//opencv librarys
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#define FEATURE_SIZE 805
#define GLCM_HORIZONTAL 0
#define GLCM_VERTICAL 1
#define GLCM_LEFT_DIAGONAL 2
#define GLCM_RIGHT_DIAGONAL 3

double features[FEATURE_SIZE];

using namespace std;
using namespace cv;

struct Haralick{
    double angularMoment;
    double entropy;
    double contrast;
    double homogeneity;
};

 String toString(int n){
    std::stringstream nStr;
    nStr << n;
    return nStr.str();
 }

void getNextClass(String &arffClass, int &limit){

     if(arffClass == "integral"){
        arffClass = "integralDef";
        limit = 33;
     }
     else if(arffClass == "integralDef"){
       arffClass = "normal";
       limit = 37;
     }
     else if(arffClass == "normal"){
       arffClass = "normalDef";
       limit = 35;
     }
     else if(arffClass == "normalDef"){
       arffClass = "recheio";
       limit = 47;
     }
     else if(arffClass == "recheio"){
       arffClass = "recheioDef";
       limit = 46;
     }
     else{//The images are over.
        limit = 0;
     }
}

void getHistogram(Mat img,vector<int> &hist,int histSize = 256, bool accumulate = false) {

    if(img.channels() > 1){
        cout << "Err image has more than one channel" << endl;
        return;
    }
    //Generates the color histogram accumulated
    for(int i=0; i<histSize; i++){
        hist.push_back(0);
    }

    for(int i=0; i<img.rows; i++){
       for(int j=0; j<img.cols; j++){
            hist.at(img.at<uchar>(i,j))++;
       }
    }

    if(accumulate){
        for(int i=1; i<histSize; i++){
            hist.at(i) += hist.at(i-1);
        }
    }
}

//Gray Level Co-occurrence Matrix: GLCM
void getGLCM(Mat img,vector<vector<double> > &glcm,int direction){

    Point Q;

    //Initialize the GLCm with zeros.
    for(int i=0; i < glcm.size(); i++){
        for(int j=0; j < glcm.size(); j++){
            glcm[i][j] = 0;
        }
    }

    switch(direction){
        case GLCM_VERTICAL://90 º
            Q = {1,0};
        break;
        case GLCM_LEFT_DIAGONAL://135 º
            Q = {1,1};
        break;
        case GLCM_RIGHT_DIAGONAL://45 º
            Q = {1,-1};
        break;
        default:// 0º
            Q = {0,1};
        break;

    }

    if(direction != GLCM_RIGHT_DIAGONAL){
        //Calculates the GLCM
        for(int i=0; i < img.rows; i++){
            for(int j=0; j < img.cols; j++){
                glcm[img.at<uchar>(i,j)][img.at<uchar>(i+Q.x,j+Q.y)]++;
            }
        }
    }else{
        //Calculates the GLCM
        for(int i=0; i < img.rows; i++){
            for(int j=img.cols-1; j > 0; j--){
                glcm[img.at<uchar>(i,j)][img.at<uchar>(i+Q.x,j+Q.y)]++;
            }
        }
    }

    //Normalize the GLCM.
    for(int i=0; i < glcm.size(); i++){
        for(int j=0; j < glcm.size(); j++){
            glcm[i][j] /= img.total();
        }
    }
}

void createHaralickDescriptor(Mat img,Haralick &h,int glcmDirection){

    vector<vector<double> > glcm(256,vector<double>(256));

    if(img.channels() > 1){
        cout << "Err calculating GLCM. Number of channels from the source image is larger than 1" << endl;
        return;
    }

    if(img.depth() != CV_8U){
        cout << "Err getting GLCM. Depth from the source image is larger than 8 bits" << endl;
        return;
    }

    h.angularMoment = h.contrast = h.entropy = h.homogeneity = 0.0;

    getGLCM(img,glcm,glcmDirection);

    for(int i = 0; i < glcm.size(); i++){
        for(int j = 0; j < glcm.size(); j++){
            h.angularMoment += pow(glcm[i][j],2);
            h.contrast += pow((i-j),2)*glcm[i][j];
            h.entropy += (glcm[i][j] == 0) ? 0 : glcm[i][j]*log10(glcm[i][j]) * -1;
            h.homogeneity += (1/(1+pow((i-j),2)))*glcm[i][j];
        }
    }
}

Mat preProcessing(Mat img){

    Mat out,mask,channels[3];

    out = img.clone();

    //Median Filter
    medianBlur(img,img,11);
    split(img,channels);

    //Apply the Otsu threshold in the Red channel because the background is green, so it will be at the low intensitys.
    threshold(channels[2],mask,0,1,THRESH_OTSU);

    for(int i=0; i<3;i++)
        multiply(channels[i],mask,channels[i]);

    merge(channels,3,out);

    return out;
}

//Hu Moment
void shapeFeature(Mat img){

    Mat channels[3];
    double hu[7];
    split(img,channels);
    for(int i=0; i<3; i++){
     HuMoments(moments(channels[i]),hu);
     for(int j = (i*7); j < ((i+1)*7); j++ ){
        features[j] = hu[j%7];
     }
    }
}

//Haralick
void textureFeature(Mat img){

    Mat grayImg;
    Haralick h,h1,h2,h3;

    cvtColor(img,grayImg,CV_BGR2GRAY);

    createHaralickDescriptor(grayImg,h,GLCM_HORIZONTAL);
    createHaralickDescriptor(grayImg,h1,GLCM_VERTICAL);
    createHaralickDescriptor(grayImg,h2,GLCM_LEFT_DIAGONAL);
    createHaralickDescriptor(grayImg,h3,GLCM_RIGHT_DIAGONAL);

    features[21] = h.angularMoment;
    features[22] = h.contrast;
    features[23] = h.entropy;
    features[24] = h.homogeneity;

    features[25] = h1.angularMoment;
    features[26] = h1.contrast;
    features[27] = h1.entropy;
    features[28] = h1.homogeneity;

    features[29] = h2.angularMoment;
    features[30] = h2.contrast;
    features[31] = h2.entropy;
    features[32] = h2.homogeneity;

    features[33] = h2.angularMoment;
    features[34] = h2.contrast;
    features[35] = h2.entropy;
    features[36] = h2.homogeneity;
}

//GCH - Global Color Histogram
void colorFeature(Mat img){

    vector<Mat> channels;
    vector<int> histB,histG,histR;
    bool accumulate = true;
    int histSize = 256;

    split(img,channels);

    getHistogram(channels.at(0),histB,histSize,accumulate);
    getHistogram(channels.at(1),histG,histSize,accumulate);
    getHistogram(channels.at(2),histR,histSize,accumulate);

    for(int i=0; i<histSize;i++){
        features[i+37] = histB.at(i);
        features[i+37+256] = histG.at(i);
        features[i+37+256+256] = histR.at(i);
    }
}

bool generateArffHeader(const char* archive,String relation, String classes){
    ofstream arff;
    arff.open(archive);
    if(!arff.is_open())
        return false;
    arff << "@RELATION " << relation << endl;
    for(int i=0; i < FEATURE_SIZE; i++){
        if(i < 21)
            arff << "@ATTRIBUTE shape_" << i + 1 << " NUMERIC" << endl;
        else if(i > 36)
            arff << "@ATTRIBUTE color_" << i + 1 << " NUMERIC" << endl;
        else
            arff << "@ATTRIBUTE texture_" << i + 1 << " NUMERIC" << endl;
    }
    arff << "@ATTRIBUTE class " << classes << endl;
    arff << "@DATA" << endl;
    arff.close();
    return true;
}

bool generateArffData(const char* archive, String class_name){
    ofstream arff;
    arff.open(archive, ofstream::out | ofstream::app);
    if(!arff.is_open())
        return false;
    for(int i=0; i < FEATURE_SIZE; i++){
          arff << features[i] << ",";
    }
    arff << class_name << endl;
    arff.close();
    return true;
}
