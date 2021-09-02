#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

int c_slider = 0;
int c_slider_max = 100;

int gamaL_slider = 0;
int gamaL_slider_max = 100;

int gamaH_slider = 0;
int gamaH_slider_max = 100;

int d0_slider = 0;
int d0_slider_max = 100;

Mat image1, image2, blended;
Mat imageTop;

char TrackbarName[50];

Mat padded, filter, mag, complexImage;
Mat image, imagegray, tmp, imageO, imagegrayO;
Mat_<float> realInput, zeros;
vector<Mat> planos;
int dft_M, dft_N;

void deslocaDFT(Mat& image ){
    Mat tmp, A, B, C, D;

    // se a imagem tiver tamanho impar, recorta a regiao para
    // evitar cópias de tamanho desigual
    image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols/2;
    int cy = image.rows/2;

    // reorganiza os quadrantes da transformada
    // A B   ->  D C
    // C D       B A
    A = image(Rect(0, 0, cx, cy));
    B = image(Rect(cx, 0, cx, cy));
    C = image(Rect(0, cy, cx, cy));
    D = image(Rect(cx, cy, cx, cy));

    // A <-> D
    A.copyTo(tmp);  D.copyTo(A);  tmp.copyTo(D);

    // C <-> B
    C.copyTo(tmp);  B.copyTo(C);  tmp.copyTo(B);
}

void on_trackbar(int, void*){

    imageO.copyTo(image);
    image.convertTo(image, CV_32F);
    log(image, image);

    dft_M = getOptimalDFTSize(image.rows);
    dft_N = getOptimalDFTSize(image.cols);

    // realiza o padding da imagem
    copyMakeBorder(image, padded, 0,
                   dft_M - image.rows, 0,
                   dft_N - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // parte imaginaria da matriz complexa (preenchida com zeros)
    zeros = Mat_<float>::zeros(padded.size());

    // prepara a matriz complexa para ser preenchida
    complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));

    // a função de transferência (filtro frequencial) deve ter o
    // mesmo tamanho e tipo da matriz complexa
    filter = complexImage.clone();

    // cria uma matriz temporária para criar as componentes real
    // e imaginaria do filtro ideal
    tmp = Mat(dft_M, dft_N, CV_32F);

    cvtColor(imageO, imagegrayO, CV_BGR2GRAY);
    imshow("original", imagegrayO);
    cvtColor(image, imagegray, CV_BGR2GRAY);

    // realiza o padding da imagem
    copyMakeBorder(imagegray, padded, 0,
                   dft_M - image.rows, 0,
                   dft_N - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));
    // limpa o array de matrizes que vao compor a
    // imagem complexa
    planos.clear();
    // cria a compoente real
    realInput = Mat_<float>(padded);
    // insere as duas componentes no array de matrizes
    planos.push_back(realInput);
    planos.push_back(zeros);

    // combina o array de matrizes em uma unica
    // componente complexa
    merge(planos, complexImage);

    // calcula o dft
    dft(complexImage, complexImage);

    // realiza a troca de quadrantes
    deslocaDFT(complexImage);

    float d0, gamaH, gamaL;
    gamaL = gamaL_slider/10.0;
    gamaH = gamaH_slider/10.0;
    d0 = d0_slider/10.0;

    // prepara o filtro passa-baixas ideal
    for(int i=0; i<dft_M; i++){

        for(int j=0; j<dft_N; j++){
            tmp.at<float> (i,j) = (gamaH - gamaL) *
                    (1.0 - exp(-1.0 * (float)c_slider * ((pow(((float) i - dft_M/2.0),2)+pow(((float) j - dft_N/2.0),2))/(d0*d0)) ))
                    + gamaL;
        }
    }

    // cria a matriz com as componentes do filtro e junta
    // ambas em uma matriz multicanal complexa
    Mat comps[]= {tmp, tmp};
    merge(comps, 2, filter);

    // aplica o filtro frequencial
    mulSpectrums(complexImage,filter,complexImage,0);

    // troca novamente os quadrantes
    deslocaDFT(complexImage);

    // calcula a DFT inversa
    idft(complexImage, complexImage);

    // limpa o array de planos
    planos.clear();

    // separa as partes real e imaginaria da
    // imagem filtrada
    split(complexImage, planos);

    // normaliza a parte real para exibicao
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);
    planos[0].convertTo(planos[0], CV_32F);
    imshow("addweighted", planos[0]);
}

int main(int argvc, char** argv){

    imageO = imread("biel.png");

    namedWindow("addweighted", 1);
    sprintf( TrackbarName, "C x %d", c_slider_max );
    createTrackbar( TrackbarName, "addweighted",
                    &c_slider,
                    c_slider_max,
                    on_trackbar);
    on_trackbar(c_slider, 0 );

    sprintf( TrackbarName, "GAMA L x %d", gamaL_slider_max );
    createTrackbar( TrackbarName, "addweighted",
                    &gamaL_slider,
                    gamaL_slider_max,
                    on_trackbar);
    on_trackbar(gamaL_slider, 0 );

    sprintf( TrackbarName, "GAMA H x %d", gamaH_slider_max );
    createTrackbar( TrackbarName, "addweighted",
                    &gamaH_slider,
                    gamaH_slider_max,
                    on_trackbar);
    on_trackbar(gamaH_slider, 0 );

    sprintf( TrackbarName, "D0 x %d", d0_slider_max );
    createTrackbar( TrackbarName, "addweighted",
                    &d0_slider,
                    d0_slider_max,
                    on_trackbar);
    on_trackbar(d0_slider, 0 );


    waitKey(0);
    return 0;
}