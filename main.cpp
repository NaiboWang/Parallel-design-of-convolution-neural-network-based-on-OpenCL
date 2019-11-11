#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"//导入CNN文件
#include "minst.h"//导入手写数字输入文件
#include "opencl.hpp"

//批训练globalworksize BATCHSIZE优化
/*主函数*/
int main()
{
	FILE  *fp = NULL;
	fp = fopen("E:\\CNNData\\test1.txt", "w");
	GTime time = GTime("readtime:", fp), time2 = GTime("totaltime:", fp);
	initOpenCL(fp);
	LabelArr trainLabel=read_Lable("E:\\CNN\\train-labels.idx1-ubyte");//读入训练集的标签y
	ImgArr trainImg=read_Img("E:\\CNN\\train-images.idx3-ubyte");//读入训练集的原始图像x
	LabelArr testLabel=read_Lable("E:\\CNN\\t10k-labels.idx1-ubyte");//读入测试集的标签y
	ImgArr testImg=read_Img("E:\\CNN\\t10k-images.idx3-ubyte");//读入训练集的原始图像x
	time.endT();
	printf("readtime:%f ms\n", time.getDu());
	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};//记录图像大小为28x28
	int outSize=testLabel->LabelPtr[0].l;//记录标签大小为10
	//主要对细粒度缓冲进行优化,细粒度系统只有INTEL平台支持
	CNNA* cnna;
	// CNN结构的初始化
	CNN** cnnarray = (CNN **)malloc(sizeof(CNN*)*BATCHSIZE);//分配二维数组来保存BATCHSIZE个CNN网络
	cnna = (CNNA*)malloc(sizeof(CNNA));
	cnna->cnn = cnnarray;
	//先用INTEL平台试下支持细粒度系统的OPENCL操作,因为最简单
	for (int i = 0; i < BATCHSIZE; i++)
	{
		cnnarray[i] = (CNN*)malloc(sizeof(CNN));//分配CNN元素大小的空间
		//cnnarray[i] = (CNN *)clSVMAlloc(conte)
	}
	cnnsetup(cnnarray,inputSize,outSize,fp);//初始化CNN网络，初始化所有BATCHSIZE个CNN网络
	//第2到BATCHSIZE个神经网络是完全复制第一个神经网络的参数的，但这里也只是复制卷积核参数以及每一层的大小变量以及偏置b,其他的参数不需要复制
	// CNN训练
	//
	int test = 0;//测试位，为1只测试
	if (!test)
	{
		CNNOpts opts;
		opts.numepochs=10;//训练次数,默认为1
		opts.alpha=1.0;//学习率
		int trainNum=60000;//暂时未知,应该指的是训练集的训练数量
						   //导入网络
		//importcnn(cnnarray[0], "E:\\minst.cnn");
		time2.startT();
		cnntrain(cnna,trainImg,trainLabel,opts,trainNum, fp,testImg, testLabel,10000);//训练CNN网络
		time2.endT();
		printf("traintotaltime:%f", time2.getDu());
		savecnn(cnnarray[0],"minst.cnn");//保存CNN网络
		// 保存训练误差
		FILE  *fp2=NULL;
		fp2=fopen("E:\\cnnL.ma","wb");
		if(fp2==NULL)
			printf("write file failed\n");
		fwrite(cnnarray[0]->L,sizeof(float),trainNum,fp2);
		fclose(fp2);
	}
	// CNN测试
	if (test)
		importcnn(cnnarray[0], "E:\\1196.cnn");
	cnncpy(cnnarray, fp);
	int testNum=10000;//测试集的测试数量
	float incorrectRatio=0.0;//错误率,默认为0
	incorrectRatio=cnntest(cnnarray[0],testImg,testLabel,testNum, fp);//测试CNN网络,输出错误率
	cout <<"error:"<< incorrectRatio << endl;
	fprintf(fp, "训练之后的最终误差：%f\n", incorrectRatio);
	fclose(fp);
	system("pause");
	return 0;
}