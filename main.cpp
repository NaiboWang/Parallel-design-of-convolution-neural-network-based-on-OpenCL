#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"//����CNN�ļ�
#include "minst.h"//������д���������ļ�
#include "opencl.hpp"

//��ѵ��globalworksize BATCHSIZE�Ż�
/*������*/
int main()
{
	FILE  *fp = NULL;
	fp = fopen("E:\\CNNData\\test1.txt", "w");
	GTime time = GTime("readtime:", fp), time2 = GTime("totaltime:", fp);
	initOpenCL(fp);
	LabelArr trainLabel=read_Lable("E:\\CNN\\train-labels.idx1-ubyte");//����ѵ�����ı�ǩy
	ImgArr trainImg=read_Img("E:\\CNN\\train-images.idx3-ubyte");//����ѵ������ԭʼͼ��x
	LabelArr testLabel=read_Lable("E:\\CNN\\t10k-labels.idx1-ubyte");//������Լ��ı�ǩy
	ImgArr testImg=read_Img("E:\\CNN\\t10k-images.idx3-ubyte");//����ѵ������ԭʼͼ��x
	time.endT();
	printf("readtime:%f ms\n", time.getDu());
	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};//��¼ͼ���СΪ28x28
	int outSize=testLabel->LabelPtr[0].l;//��¼��ǩ��СΪ10
	//��Ҫ��ϸ���Ȼ�������Ż�,ϸ����ϵͳֻ��INTELƽ̨֧��
	CNNA* cnna;
	// CNN�ṹ�ĳ�ʼ��
	CNN** cnnarray = (CNN **)malloc(sizeof(CNN*)*BATCHSIZE);//�����ά����������BATCHSIZE��CNN����
	cnna = (CNNA*)malloc(sizeof(CNNA));
	cnna->cnn = cnnarray;
	//����INTELƽ̨����֧��ϸ����ϵͳ��OPENCL����,��Ϊ���
	for (int i = 0; i < BATCHSIZE; i++)
	{
		cnnarray[i] = (CNN*)malloc(sizeof(CNN));//����CNNԪ�ش�С�Ŀռ�
		//cnnarray[i] = (CNN *)clSVMAlloc(conte)
	}
	cnnsetup(cnnarray,inputSize,outSize,fp);//��ʼ��CNN���磬��ʼ������BATCHSIZE��CNN����
	//��2��BATCHSIZE������������ȫ���Ƶ�һ��������Ĳ����ģ�������Ҳֻ�Ǹ��ƾ���˲����Լ�ÿһ��Ĵ�С�����Լ�ƫ��b,�����Ĳ�������Ҫ����
	// CNNѵ��
	//
	int test = 0;//����λ��Ϊ1ֻ����
	if (!test)
	{
		CNNOpts opts;
		opts.numepochs=10;//ѵ������,Ĭ��Ϊ1
		opts.alpha=1.0;//ѧϰ��
		int trainNum=60000;//��ʱδ֪,Ӧ��ָ����ѵ������ѵ������
						   //��������
		//importcnn(cnnarray[0], "E:\\minst.cnn");
		time2.startT();
		cnntrain(cnna,trainImg,trainLabel,opts,trainNum, fp,testImg, testLabel,10000);//ѵ��CNN����
		time2.endT();
		printf("traintotaltime:%f", time2.getDu());
		savecnn(cnnarray[0],"minst.cnn");//����CNN����
		// ����ѵ�����
		FILE  *fp2=NULL;
		fp2=fopen("E:\\cnnL.ma","wb");
		if(fp2==NULL)
			printf("write file failed\n");
		fwrite(cnnarray[0]->L,sizeof(float),trainNum,fp2);
		fclose(fp2);
	}
	// CNN����
	if (test)
		importcnn(cnnarray[0], "E:\\1196.cnn");
	cnncpy(cnnarray, fp);
	int testNum=10000;//���Լ��Ĳ�������
	float incorrectRatio=0.0;//������,Ĭ��Ϊ0
	incorrectRatio=cnntest(cnnarray[0],testImg,testLabel,testNum, fp);//����CNN����,���������
	cout <<"error:"<< incorrectRatio << endl;
	fprintf(fp, "ѵ��֮���������%f\n", incorrectRatio);
	fclose(fp);
	system("pause");
	return 0;
}