#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"
#include "minst.h"
#include "opencl.hpp"
#define AvePool 0//����ػ�����,ƽ���ػ�
#define MaxPool 1//���ػ�
#define MinPool 2//��С�ػ�
#define BATCHSIZE 50//������ѵ��ʱÿ���ö��ٸ�ԭʼ���ݽ���ѵ��������ĿӦ�ܱ�ԭʼ��Ŀ����
// �����
typedef struct convolutional_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	// ��������ģ���Ȩ�طֲ���������һ����ά����
	// ���СΪinChannels*outChannels*mapSize*mapSize��С
	// ��������ά���飬��Ҫ��Ϊ�˱���ȫ���ӵ���ʽ��ʵ���Ͼ���㲢û���õ�ȫ���ӵ���ʽ
	// �����������DeapLearningToolboox���CNN���ӣ����õ�����ȫ����
	float**** mapData;     //�������ģ�������
	float**** dmapData;    //�������ģ������ݵľֲ��ݶ�

	float* basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	float* dbasicData;   //ƫ�õ��ݶȣ�ƫ�õĴ�С��ΪoutChannels
	bool isFullConnect; //�Ƿ�Ϊȫ����
	bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	// �������ߵĴ�Сͬ�����ά����ͬ
	float*** v; // ���뼤���������ֵ
	float*** y; // ���������Ԫ�����

	// ������صľֲ��ݶ�
	float*** d; // ����ľֲ��ݶ�,��ֵ  
}CovLayer;

// ������ pooling
typedef struct pooling_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int poolType;     //Pooling�ķ���
	float* basicData;   //ƫ��,ʵ����û���õ�

	float*** y; // ������������Ԫ�����,�޼����
	float*** d; // ����ľֲ��ݶ�,��ֵ
}PoolLayer;

// ����� ȫ���ӵ�������
typedef struct nn_layer{
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	float** wData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	float* basicData;   //ƫ�ã���СΪoutputNum��С

	float** dwData; // Ȩ�������ݶȣ�Ϊһ��inputNum*outputNum��С
	float* dbasicData;   //ƫ���ݶȣ���СΪoutputNum��С
	// �������ߵĴ�Сͬ�����ά����ͬ
	float* v; // ���뼤���������ֵ
	float* y; // ���������Ԫ�����
	float* d; // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
}OutLayer;

typedef struct cnn_network{//����CNN����������һ��,�������������,�����Ŀ
	int layerNum;//����Ŀ
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;
	float* e; // ѵ�����
	float* L; // ˲ʱ�������
}CNN;
typedef struct cnn_arr
{
	CNN** cnn;
}CNNA;
//ѵ������
typedef struct train_opts{
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;
void cnncpy(CNN** cnnarray,FILE *fp);
void cnnsetup(CNN** cnnarray,nSize inputSize,int outputSize, FILE *fp);//��ʼ��CNN�Ĳ���
/*	
	CNN�����ѵ������
	inputData��outputData�ֱ����ѵ������
	dataNum����������Ŀ
*/
void cnntrain(CNNA *cnns,ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum);
// ����cnn����,inputData��outputData�ֱ������Լ����ݵ�x��y
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum, FILE *fp);
// ����cnn
void savecnn(CNN* cnn, const char* filename);
// ����cnn������
void importcnn(CNN* cnn, const char* filename);

// ��ʼ�������
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// ��ʼ��������
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// ��ʼ�������
OutLayer* initOutLayer(int inputNum,int outputNum);
void cnnupdategrad(CNN** cnnarray);
// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input,float bas); // sigma�����

void cnnff(CNN* cnn,float* inputData); // �����ǰ�򴫲�
void cnnbp(CNN* cnn,float* outputData); // ����ĺ��򴫲�
void cnnapplygrads(CNN* cnn,CNNOpts opts,float* inputData);//���������Ȩֵ
void cnnclear(CNN* cnn); // ������vyd����

/*
	Pooling Function
	input ��������
	inputNum ����������Ŀ
	mapSize ��ƽ����ģ������
*/
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // ��ƽ��ֵ

/* 
	����ȫ����������Ĵ���
	nnSize������Ĵ�С
*/
void nnff(float* output,float* input,float** wdata,nSize nnSize); // ����ȫ�����������ǰ�򴫲�

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // ����CNN�����е��������
void cnntrain1(CNN* cnn, float* inputData, float* LabelData, CNNOpts opts);
void int2str(const int &int_temp, string &string_temp);
#endif
