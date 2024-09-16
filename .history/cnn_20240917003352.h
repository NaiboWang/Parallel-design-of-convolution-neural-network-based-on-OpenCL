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
#define AvePool 0//定义池化类型,平均池化
#define MaxPool 1//最大池化
#define MinPool 2//最小池化
#define BATCHSIZE 50//定义批训练时每次用多少个原始数据进行训练，此数目应能被原始数目整除
// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	// 关于特征模板的权重分布，这里是一个四维数组
	// 其大小为inChannels*outChannels*mapSize*mapSize大小
	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	// 这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
	float**** mapData;     //存放特征模块的数据
	float**** dmapData;    //存放特征模块的数据的局部梯度

	float* basicData;   //偏置，偏置的大小，为outChannels
	float* dbasicData;   //偏置的梯度，偏置的大小，为outChannels
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	float*** v; // 进入激活函数的输入值
	float*** y; // 激活函数后神经元的输出

	// 输出像素的局部梯度
	float*** d; // 网络的局部梯度,δ值  
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	float* basicData;   //偏置,实际上没有用到

	float*** y; // 采样函数后神经元的输出,无激活函数
	float*** d; // 网络的局部梯度,δ值
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct nn_layer{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	float** wData; // 权重数据，为一个inputNum*outputNum大小
	float* basicData;   //偏置，大小为outputNum大小

	float** dwData; // 权重数据梯度，为一个inputNum*outputNum大小
	float* dbasicData;   //偏置梯度，大小为outputNum大小
	// 下面三者的大小同输出的维度相同
	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
	float* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network{//整个CNN的最外面那一层,包括五个层和误差,层的数目
	int layerNum;//层数目
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;
	float* e; // 训练误差
	float* L; // 瞬时误差能量
}CNN;
typedef struct cnn_arr
{
	CNN** cnn;
}CNNA;
//训练参数
typedef struct train_opts{
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;
void cnncpy(CNN** cnnarray,FILE *fp);
void cnnsetup(CNN** cnnarray,nSize inputSize,int outputSize, FILE *fp);//初始化CNN的参数
/*	
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
*/
void cnntrain(CNNA *cnns,ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum);
// 测试cnn函数,inputData，outputData分别存入测试集数据的x和y
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum, FILE *fp);
// 保存cnn
void savecnn(CNN* cnn, const char* filename);
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename);

// 初始化卷积层
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// 初始化采样层
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// 初始化输出层
OutLayer* initOutLayer(int inputNum,int outputNum);
void cnnupdategrad(CNN** cnnarray);
// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas); // sigma激活函数

void cnnff(CNN* cnn,float* inputData); // 网络的前向传播
void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
void cnnapplygrads(CNN* cnn,CNNOpts opts,float* inputData);//更新网络的权值
void cnnclear(CNN* cnn); // 将数据vyd清零

/*
	Pooling Function
	input 输入数据
	inputNum 输入数据数目
	mapSize 求平均的模块区域
*/
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // 求平均值

/* 
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
void nnff(float* output,float* input,float** wdata,nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // 保存CNN网络中的相关数据
void cnntrain1(CNN* cnn, float* inputData, float* LabelData, CNNOpts opts);
void int2str(const int &int_temp, string &string_temp);
#endif
