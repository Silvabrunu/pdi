#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

#define STEP 5
#define JITTER 3
#define RAIO 5

int main(int argc, char** argv){
  vector<int> yrange;
  vector<int> xrange;

  Mat imagemOriginal, borderImagemOriginal;
  Mat imagemComPontilhismo;
  int width, height;
  int gray;
  int x, y;

  imagemOriginal= imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);

  srand(time(0));

  if(!imagemOriginal.data){
      cout << "nao abriu" << argv[1] << endl;
    cout << argv[0] << " imagemOriginal.jpg";
    exit(0);
  }

  width = imagemOriginal.size().width;
  height = imagemOriginal.size().height;

  xrange.resize(height/STEP);
  yrange.resize(width/STEP);

  iota(xrange.begin(), xrange.end(), 0);
  iota(yrange.begin(), yrange.end(), 0);

  for(uint i=0; i<xrange.size(); i++){
    xrange[i]= xrange[i]*STEP+STEP/2;
    yrange[i]= yrange[i]*STEP+STEP/2;
  }

  imagemOriginal.copyTo(imagemComPontilhismo);
  random_shuffle(xrange.begin(), xrange.end());

  for(auto i : xrange){
    random_shuffle(yrange.begin(), yrange.end());
    for(auto j : yrange){
      x = i+rand()%(2*JITTER)-JITTER+1;
      y = j+rand()%(2*JITTER)-JITTER+1;
      gray = imagemOriginal.at<uchar>(x,y);
      circle(imagemComPontilhismo,
             cv::Point(y,x),
             RAIO,
             CV_RGB(gray,gray,gray),
             -1,
             CV_AA);
    }
  }

  imshow("Imagem com Pontilhismo", imagemComPontilhismo);
  imwrite("imgfiltroPontilhismo.jpg", imagemComPontilhismo);

  for(int it=0; it<5; it++){
     Canny(imagemOriginal, borderImagemOriginal, 10*it, 50*it);
     int raio = 5-it;

     for(int i=0; i<height; i++ ){
        for(int j=0; j<width; j++){
           if(borderImagemOriginal.at<uchar>(i,j)>0){
              gray = imagemOriginal.at<uchar>(i,j);
              circle(imagemComPontilhismo,
                     cv::Point(j,i),
                     raio,
                     CV_RGB(gray,gray,gray),
                     -1,
                     CV_AA);
           }
        }
     }
  }

  imshow("Imagem com CannyPoints", imagemComPontilhismo);
  imwrite("imgfiltroCannyPoints.jpg", imagemComPontilhismo);

  waitKey();
  return 0;
}