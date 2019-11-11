#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
using namespace std;

clock_t start, finish;//����ʱ���õ�
double  duration;

void cnnsetup(CNN** cnnarray,nSize inputSize,int outputSize,FILE *fp)
{
	start = clock();//��ʼ��ʱ
	for(int i=0;i<BATCHSIZE;i++)//��ʼ��BATCHSIZE��CNN����
	{
		cnnarray[i]->layerNum=5;//����cnn����Ϊ5
		nSize inSize;//����ͼ���С
		int mapSize=5;//�������˴�СΪ5
		inSize.c=inputSize.c;//����ͼ���СΪ28X28
		inSize.r=inputSize.r;//����ͼ���СΪ28X28
		cnnarray[i]->C1=initCovLayer(inSize.c,inSize.r,5,1,6);//������ͼ���СΪ28X28,����˴�СΪ5X5,����ͼ����Ϊ1,���MAP��Ϊ6��ʼ��C1��,�����ʼ�����̼�initCovLayer��������
		inSize.c=inSize.c-mapSize+1;//S2�������MAP�Ĵ�СΪ28-5+1=24,��24X24
		inSize.r=inSize.r-mapSize+1;//S2�������MAP�Ĵ�СΪ28-5+1=24,��24X24
		cnnarray[i]->S2=initPoolLayer(inSize.c,inSize.r,2,6,6,AvePool); //������ͼ���СΪ24X24, �ػ���СΪ2X2, ����ͼ����Ϊ6, ���MAP��Ϊ6,�ػ�����Ϊƽ���ػ���ʼ��S2��, �����ʼ�����̼�initPoolLayer��������
		inSize.c=inSize.c/2;//C3�������ͼ���СΪ24/2=12,��12X12
		inSize.r=inSize.r/2;//C3�������ͼ���СΪ24/2=12,��12X12
		cnnarray[i]->C3=initCovLayer(inSize.c,inSize.r,5,6,12);//������ͼ���СΪ12X12,����˴�СΪ5X5,����ͼ����Ϊ6,���MAP��Ϊ12��ʼ��C3��,�����ʼ�����̼�initCovLayer��������
		inSize.c=inSize.c-mapSize+1;//S4������ͼ���СΪ12-5+1=8,��8X8
		inSize.r=inSize.r-mapSize+1;//S4������ͼ���СΪ12-5+1=8,��8X8
		cnnarray[i]->S4=initPoolLayer(inSize.c,inSize.r,2,12,12,AvePool);//������ͼ���СΪ8X8, �ػ���СΪ2X2, ����ͼ����Ϊ12, ���MAP��Ϊ12,�ػ�����Ϊƽ���ػ���ʼ��S4��, �����ʼ�����̼�initPoolLayer��������
		inSize.c=inSize.c/2;//ȫ�������������ͼ���СΪ8/2=4,��4X4
		inSize.r=inSize.r/2;//ȫ�������������ͼ���СΪ8/2=4,��4X4
		cnnarray[i]->O5=initOutLayer(inSize.c*inSize.r*12,outputSize);//������ͼ���СΪ4*4*12=192,���ͼ��Ϊ10��ʼ�������,�����ʼ�����̼�initOutLayer��������
		cnnarray[i]->e=(float*)calloc(cnnarray[i]->O5->outputNum,sizeof(float));//��ѵ������ʼ��Ϊһ������Ϊcnn->O5->outputNum��,Ҳ����10��������������ÿ��ѵ��֮��ÿ�������ѵ�����,����ʼ��ֵΪ0
	}
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("setuptime:%f seconds\n", duration);
	fprintf(fp,"setuptime:%f seconds\n", duration);
}
//��ʼ�������,����Ϊ����ͼ��Ĵ�СinputWidth,inputHeight,����˴�СmapSize,����ͼ�����inChannels,���ͼ�����outChannels
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));//����һ�������ṹ�����

	covL->inputHeight=inputHeight;//����ͼ��߶�ΪinputHeight
	covL->inputWidth=inputWidth;//����ͼ����ΪinputWidth
	covL->mapSize=mapSize;//����˴�СΪmapSize

	covL->inChannels=inChannels;//����ͼ�����
	covL->outChannels=outChannels;//���MAP����

	covL->isFullConnect=true; // Ĭ��Ϊȫ����

	// Ȩ�ؿռ�ĳ�ʼ�����������е��ã�[r][c]
	int i,j,c,r;
	srand((unsigned)time(NULL));//�������ʼ�������Ա�ÿ�γ�ʼ���õ��������������ͬ
	covL->mapData=(float****)malloc(inChannels*sizeof(float***));//��������ͼ�����ά�������洢����˵�����,����һ����ά����ĵ���ά��ʾ��ǰ���ͼ���Ӧ���ͼ��ĸ���,����C3�����6X12�������,ÿ������˴�СΪ5X5,��C3��mapData��СΪ6X12X5X5,ͬ��C1��mapData��СΪ1X6X5X5
	for(i=0;i<inChannels;i++){//��һ������ͼ��
		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));//�������ͼ���������ά����
		for(j=0;j<outChannels;j++){//��һ�����ͼ��
			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));//�������˿�ȸ�����
			for(r=0;r<mapSize;r++){
				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));//һ����������������������
				for(c=0;c<mapSize;c++){
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; //�Ծ���˽��г�ʼ������,����0~1֮��������((float)rand()/(float)RAND_MAX),Ȼ���ȥ0.5�ٳ�2,Ҳ����˵�������ɵ���-1~1֮��������
					covL->mapData[i][j][r][c]=randnum*sqrt((float)6.0/(float)(mapSize*mapSize*(inChannels+outChannels)));//����Ѿ���˵�ֵ��ʼ��Ϊ��Ӧ�Ĺ�ʽ,ԭ�����Nguyen-widrow����,��������Ȩֵ��ʼ����sigmoid������[-0.7,0.7]������,�����ʽ��ô�����������ڻ���̫���
				}
			}
		}
	}
	// Ȩ���ݶȱ仯
	covL->dmapData=(float****)malloc(inChannels*sizeof(float***));//ͬ��,���ﱣ�����ÿ�η��򴫲���֮��,�����Ҫ���µ�Ȩֵ,��С��Ȼ���������ͬ,����C3��Ϊ6X12X5X5
	//����ķ��䷽��Ҳ��ͬ
	for(i=0;i<inChannels;i++){
		covL->dmapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->dmapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->dmapData[i][j][r]=(float*)calloc(mapSize,sizeof(float));//����һ������ΪmapSize,Ҳ����5�����鲢ȫ����ʼ��Ϊ0
			}
		}
	}

	covL->basicData=(float*)calloc(outChannels,sizeof(float));//����ÿһ�����MAP��ƫ��b,�ж��ٸ����MAP���ж��ٸ�b
	covL->dbasicData = (float*)calloc(outChannels, sizeof(float));//����ÿһ�����MAP��ƫ���ݶ�db,�ж��ٸ����MAP���ж��ٸ�db
	int outW=inputWidth-mapSize+1;//���MAP��С�Ŀ��
	int outH=inputHeight-mapSize+1;//���MAP��С�ĸ߶�


	covL->d=(float***)malloc(outChannels*sizeof(float**));// ����ľֲ��ݶ�,��ֵ,��������С�����MAP,���������y��ͬ
	covL->v=(float***)malloc(outChannels*sizeof(float**)); // ���뼤���������ֵ,��������洢������,MATLAB�汾����û�д洢���ֵ,������cnnff.m�а�������ʱ����z�洢��
	covL->y=(float***)malloc(outChannels*sizeof(float**));//���MAP
	//��������MAP��С��ΪoutChannels*outH*outW,��Ϊ�������MAP�й�,���漸�а�����ͬʱ��ʼ��Ϊ0.0,��C3��Ϊ12X8X8
	for(j=0;j<outChannels;j++){
		covL->d[j]=(float**)malloc(outH*sizeof(float*));
		covL->v[j]=(float**)malloc(outH*sizeof(float*));
		covL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			covL->d[j][r]=(float*)calloc(outW,sizeof(float));
			covL->v[j][r]=(float*)calloc(outW,sizeof(float));
			covL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return covL;//���س�ʼ���õľ����
}
//��ʼ���ػ���,����Ϊ����ͼ��Ĵ�СinputWidth,inputHeight,�ػ���СmapSize,����ͼ�����inChannels,���ͼ�����outChannels,�ػ�����poolType
PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));//����ػ���ռ�
	//���·ֱ�ֵ��Ӧ�Ĳ���,��������
	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;
	poolL->poolType=poolType; 

	poolL->basicData=(float*)calloc(outChannels,sizeof(float));//ÿһ�����map��ƫ��b,�����cnn������˵,ʵ���϶���0.0
	//���涨�����ͼ���С,��S2��Ϊ24/2=12
	int outW=inputWidth/mapSize;
	int outH=inputHeight/mapSize;

	int j,r;
	poolL->d=(float***)malloc(outChannels*sizeof(float**));// ����ľֲ��ݶ�,��ֵ
	poolL->y=(float***)malloc(outChannels*sizeof(float**));// ������������Ԫ�����,�޼����
	//��ʼ��ƫ�ú����MAPΪ0.0,ע���СΪoutChannels*outH*outW,��S4��Ϊ12X4X4
	for(j=0;j<outChannels;j++){
		poolL->d[j]=(float**)malloc(outH*sizeof(float*));
		poolL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			poolL->d[j][r]=(float*)calloc(outW,sizeof(float));
			poolL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return poolL;
}
//��ʼ�����һ��Ĳ���,������Ϊ��ͨ������,����Ϊ����ڵ���inputNum������ڵ���outputNum
OutLayer* initOutLayer(int inputNum,int outputNum)
{
	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));//�����ʼ�ռ�
	//��ֵ����ֵ
	outL->inputNum=inputNum;
	outL->outputNum=outputNum;
	

	outL->basicData=(float*)calloc(outputNum,sizeof(float));//10��ƫ��b,ÿ��ֵ��ʼΪ0.0
	outL->dbasicData = (float*)calloc(outputNum, sizeof(float));//10��ƫ���ݶ�db,ÿ��ֵ��ʼΪ0.0

	outL->d=(float*)calloc(outputNum,sizeof(float));// ����ľֲ��ݶ�,��ֵ,10��0.0
	outL->v=(float*)calloc(outputNum,sizeof(float));// ���뼤���������ֵ,10��0.0,��cnnff.m��ֱ�Ӽ����û�б���
	outL->y=(float*)calloc(outputNum,sizeof(float));// ���������Ԫ�����,10��0.0

	// Ȩ�صĳ�ʼ��
	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // �����У������,�����10X192��Ȩֵ����
	outL->dwData = (float**)malloc(outputNum * sizeof(float*)); // �����У������,�����10X192��Ȩֵ����
	int i,j;
	//���³�ʼ��Ȩֵ����
	srand((unsigned)time(NULL));
	for(i=0;i<outputNum;i++){
		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));
		outL->dwData[i] = (float*)calloc(inputNum,sizeof(float));//��ʼ��dw10��������ݶ�ֵ
		for(j=0;j<inputNum;j++){
			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // ����һ��-1��1�������
			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));//��ʽ��Ȼ�����Ϊʲô��������
		}
	}

	outL->isFullConnect=true;//����Ϊȫ����

	return outL;
}
//������,MATLAB�汾cnnsetup�����ݾ�ȫ������

// ������������������,ע������������,������һά����
int vecmaxIndex(float* vec, int veclength)
{
	//�������һ����򵥵���veclength�����������������Ԫ�ص��㷨,��󷵻ص��������Ǹ����
	//������������������Ƚϲ��Խ������ȷ�ı�ǩ����ǲ���ͬһ��ֵ,Ȼ��������
	int i;
	float maxnum=-1.0;
	int maxIndex=0;
	for(i=0;i<veclength;i++){
		if(maxnum<vec[i]){
			maxnum=vec[i];
			maxIndex=i;
		}
	}
	return maxIndex;//�������ƶ������Ǹ���Ԫ�����
}

// ����cnn����,����Ĳ���Ϊ:cnn����ѵ���õ�cnn����,inputDataΪ���Լ���ԭʼͼ������,outputDataΪ���Լ�ʵ�ʵ���ȷ���,testNumΪ���Լ�����,����Ϊ10000
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum, FILE *fp)
{
	start = clock();
	int n=0;
	int incorrectnum=0;  //����Ԥ�����Ŀ
	/*for(n=0;n<testNum;n++){*/
		clSetKernelArgSVMPointer(kernel, 0, cnn);
		clSetKernelArgSVMPointer(kernel, 1, inputData);
		clSetKernelArgSVMPointer(kernel, 2, outputData);
		clSetKernelArgSVMPointer(kernel, 3, &incorrectnum);
		size_t globalWorkSize = testNum;//ȫ�ֹ������С
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, &globalWorkSize, NULL, NULL, NULL, NULL);
		//cout << num << " ";
	//}
	clFinish(commandQueue);
	//cout << "wrongnum:" << incorrectnum << endl;
	//for (int i = 0; i < 1000; i++)
	//	cout << t2[i] << ' ';
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("testtime:%f seconds\n", duration);
	cnnclear(cnn);//�����������һ�ν��
	return (float)incorrectnum/(float)testNum;//���ش�����
}
//���ݵ�һ��CNN���Ƴ�����BATCHSIZE-1��CNN
void cnncpy(CNN** cnnarray,FILE *fp)
{
	//start = clock();//��ʼ��ʱ
	for (int k = 1; k < BATCHSIZE; k++)
	{
		int i, j, r,s;
		// ����C1������
		for (i = 0; i < cnnarray[0]->C1->inChannels; i++)
			for (j = 0; j < cnnarray[0]->C1->outChannels; j++)
				for (r = 0; r < cnnarray[0]->C1->mapSize; r++)
					for (s = 0; s < cnnarray[0]->C1->mapSize; s++)
					cnnarray[k]->C1->mapData[i][j][r][s] = cnnarray[0]->C1->mapData[i][j][r][s];
		for (i = 0; i < cnnarray[0]->C1->outChannels; i++)
			cnnarray[k]->C1->basicData[i] = cnnarray[0]->C1->basicData[i];
		//C3����Ϣ����
		for (i = 0; i < cnnarray[0]->C3->inChannels; i++)
			for (j = 0; j < cnnarray[0]->C3->outChannels; j++)
				for (r = 0; r < cnnarray[0]->C3->mapSize; r++)
					for (s = 0; s < cnnarray[0]->C3->mapSize; s++)
						cnnarray[k]->C3->mapData[i][j][r][s] = cnnarray[0]->C3->mapData[i][j][r][s];
		for (i = 0; i < cnnarray[0]->C3->outChannels; i++)
			cnnarray[k]->C3->basicData[i] = cnnarray[0]->C3->basicData[i];
		//�������Ϣ����
		for (i = 0; i<cnnarray[0]->O5->outputNum; i++)
			for (j = 0; j < cnnarray[0]->O5->inputNum; j++)
			cnnarray[k]->O5->wData[i][j] = cnnarray[0]->O5->wData[i][j];
		for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
			cnnarray[k]->O5->basicData = cnnarray[0]->O5->basicData;
	}
	//finish = clock();//������ʱ,��λ����
	//duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	//printf("copytime:%f seconds\n", duration);
	//fprintf(fp, "copytime:%f seconds\n", duration);
}
// ����cnn
void savecnn(CNN* cnn, const char* filename)//������CNN������ÿһ���Ȩֵ(�����)��ƫ�ô洢���ļ���
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,r;
	// C1������
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	// C3����
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	// O5�����
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}
// ����cnn������
void importcnn(CNN* cnn, const char* filename)//�������ļ��е���ÿһ���Ȩֵ(�����)��ƫ�õ�CNN����
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,c,r;
	// C1������
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					float* in=(float*)malloc(sizeof(float));//����һ������Ϊ1������?Ϊʲô������
					fread(in,sizeof(float),1,fp);
					cnn->C1->mapData[i][j][r][c]=*in;
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		fread(&cnn->C1->basicData[i],sizeof(float),1,fp);//��ȡƫ��ֵ,һ��6��

	// C3����
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				fread(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);//ͬ��,��ȡ����ֵ

	for(i=0;i<cnn->C3->outChannels;i++)
		fread(&cnn->C3->basicData[i],sizeof(float),1,fp);//��ȡƫ��ֵ,һ��12��

	// O5�����
	for(i=0;i<cnn->O5->outputNum;i++)
		for(j=0;j<cnn->O5->inputNum;j++)
			fread(&cnn->O5->wData[i][j],sizeof(float),1,fp);//��ȡ������Ȩֵ����

	for(i=0;i<cnn->O5->outputNum;i++)
		fread(&cnn->O5->basicData[i],sizeof(float),1,fp);//��ȡ������ƫ��ֵ,һ��10��

	fclose(fp);
}
void int2str(const int &int_temp, string &string_temp)
{
	stringstream stream;
	stream << int_temp;
	string_temp = stream.str();   //�˴�Ҳ������ stream>>string_temp  
}
//����ѵ��CNN������,���ݴ����ԭʼͼ��inputData,ͼ�����ȷֵ(��ǩ)outputData,ѵ���Ĳ���opts�Լ�ѵ����������trainNum��ѵ������,����trainNumΪ55000,inputDataΪ60000��ԭʼͼ��,outputDataΪ60000����ǩ
void cnntrain(CNNA *cnns,ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum)
{
	GTime time = GTime("setuptime:", fp), time2 = GTime("setuptime:", fp), time3 = GTime("setuptime:", fp);
	int testtime = 0;
	//���ﲢû�д���ԭʼ����,����˳��ѵ����,��������Ϊ���ҵĳɱ�̫��
	// ѧϰѵ���������,����Ϊ55000��
	cnns->cnn[0]->L=(float*)malloc(trainNum*sizeof(float));//��һ��cnn������ѧϰ���
	int e;
	if (trainNum % BATCHSIZE != 0)
	{
		cout << "�Բ���,�����������ܱ�ȫ������������,���ܽ���ѵ��!" << endl;
		exit(-1);
	}
	for(e=0;e<opts.numepochs;e++){//ѵ������
		float incorrectRatio = 0.0;//������,Ĭ��Ϊ0
		string t;
		//int2str(e, t);
		//time2.startT("test" + t + "time:");
		//incorrectRatio = cnntest(cnns->cnn[0], inputData1, outputData1, testNum, fp);//����CNN����,���������,�õ��ǵ�һ��CNN����,����ĺ͵�һ����һ����
		//time2.endT();
		//cout << "test" << e << "time:" << time2.getDu() << "ms" << endl;
		//cout << "test" << e << "error:" << incorrectRatio << endl;
		//fprintf(fp, "test%derror:%f\n", e, incorrectRatio);
		time.startT("��" + t + "��ѵ��ʱ��:");
		int train = trainNum / BATCHSIZE;//��ѵ������
		for(int n=0;n<train;n++){//��ѵ��
			//printf("%d\n",n);		
			//if (n == 0)//��һ��ѵ����ʼ��ʱ
			if (n == 0)
			{
				int2str(n, t);
				//time2.startT("trainset" + t + "time:");
			}
			cnncpy(cnns->cnn, fp);//�ѵ�һ��CNN����Ϣ���Ƹ�����BATCHSIZE-1��
				int bs = n*BATCHSIZE;
				//cout << bs << endl;
				int *pb = &bs;
				clSetKernelArgSVMPointer(kernel2, 0, cnns);
				clSetKernelArgSVMPointer(kernel2, 1, inputData);
				clSetKernelArgSVMPointer(kernel2, 2, outputData);
				clSetKernelArgSVMPointer(kernel2, 3, &opts);
				clSetKernelArgSVMPointer(kernel2, 4, pb);
				size_t globalWorkSize = BATCHSIZE;//ȫ�ֹ������С
				clEnqueueNDRangeKernel(commandQueue, kernel2, 1, 0, &globalWorkSize, NULL, NULL, NULL, NULL);
			//�Ŷӵȴ�ִ�����
			clFinish(commandQueue);
			//cout << bs << endl;
			//cout << b << endl;
			//system("pause");
			//for (int s = 0; s < BATCHSIZE; s++)//��ÿһ�����������ֵ
			//{
			//	if (n == 10)
			//	{
			//		cout << cnn[s]->O5->basicData[0] << " ";
			//		cout << cnn[s]->O5->y[0] << " ";
			//		cout << cnn[s]->O5->dbasicData[0] << " ";
			//		cout << cnn[s]->O5->dwData[0][0] << " ";
			//		cout << endl;
			//		
			//	}
			//}
			//if(n==10)
			//	system("pause");
			cnnupdategrad(cnns->cnn);//������������������ݶ�

			float l = 0.0;
			int i;
			for (i = 0; i<cnns->cnn[0]->O5->outputNum; i++)
				l = l + cnns->cnn[0]->e[i] * cnns->cnn[0]->e[i];//����������e[i]^2,�������2���������ľ������E,e[i] = t[i] - y[i],��cnnbp����
			if (n == 0)
				cnns->cnn[0]->L[n] = l / (float)2.0;//��һ�������ֵΪl(L)/2
			else
				cnns->cnn[0]->L[n] = cnns->cnn[0]->L[n - 1] * 0.99 + 0.01*l / (float)2.0;//�ڶ��ο�ʼ�����ֵ�����������
			if (n % 20 == 0)
			{
				char* filedir = "E:\\CNNData\\";//�Ȱ�cnnԭ����Ȩֵ���浽���Ŀ¼��
				const char* filename = combine_strings(filedir, combine_strings(intTochar(testtime++), ".cnn"));//�ļ�������n.cnn
				savecnn(cnns->cnn[0], filename);//�Ѿ�������籣������
				//time2.endT();
				//printf("trainset%dtime:%f ms\n", n, time2.getDu());
				//printf("error:%f\n", cnns->cnn[0]->L[n]);
				//fprintf(fp, "error:%f\n", cnns->cnn[0]->L[n]);
				//int2str(n, t);
				//time2.startT("tr		time2.startT("test" + t + "time:");
				incorrectRatio = cnntest(cnns->cnn[0], inputData1, outputData1, testNum, fp);//����CNN����,���������,�õ��ǵ�һ��CNN����,����ĺ͵�һ����һ����
				cout << "test" << "time:" << time2.getDu() << "ms" << endl;
				cout << "test" << "error:" << incorrectRatio << endl;
				fprintf(fp, "testerror:%f\n", incorrectRatio);
				//time2.endT();
				cout << "test" << e << "time:" << time2.getDu() << "ms" << endl;
				cout << "test" << e << "error:" << incorrectRatio << endl;
				//fprintf(fp, "test%derror:%f\n", e, incorrectRatio); ainset" + t + "time:");

			}
		}
		time.endT();
		printf("train(%d)time:%f ms\n", e, time.getDu());
	}
}

// ����InputData��ͼ�����ݣ�inputData[r][c],r��c�У��������Ȩ��ģ����һ�µ�
//ע��������õ�������ѧϰ,Ҳ����һ��ͼ��һ��ͼ���ѧϰ,ÿ��ͼ�񶼻�����һ��Ȩֵ,Ȼ�����ϸ���
void cnnff(CNN* cnn,float* inputData)
{
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn->S2->inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn->S2->inputHeight;//�����һ������MAP����Ĵ�С,������24X24
										// ��һ��Ĵ���
	int i, j, r, c, t, k, m, n;
	// ��һ���������
	nSize mapSize = { cnn->C1->mapSize,cnn->C1->mapSize };//����˴�С,5X5
	nSize inSize = { cnn->C1->inputWidth,cnn->C1->inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnn->S2->inputWidth,cnn->S2->inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnn->C1->outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnn->C1->inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
												   //�Ծ������ת180��
												   //��ʼ�����������
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->C1->v[i][t][k] = cnn->C1->v[i][t][k] + mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->basicData[i]);
			}
		}
	}

	// �ڶ�����������S2��������
	outSize.c = cnn->C3->inputWidth;//���ͼ���С,12X12
	outSize.r = cnn->C3->inputHeight;//���ͼ���С,12X12
	inSize.c = cnn->S2->inputWidth;//����ͼ���С,24X24
	inSize.r = cnn->S2->inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i<(cnn->S2->outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												//avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);//C1->y[i]����S2->mapSize��Сƽ���ػ����������S2->y[i]
												//�²����ػ�
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->C1->y[i][m][n];
					}
				}
				cnn->S2->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// �������������,������ȫ����
	outSize.c = cnn->S4->inputWidth;//���ͼ���С,8X8
	outSize.r = cnn->S4->inputHeight;//���ͼ���С,8X8
	inSize.c = cnn->C3->inputWidth;//����ͼ���С,12X12
	inSize.r = cnn->C3->inputHeight;//����ͼ���С,12X12
	mapSize.c = cnn->C3->mapSize;//����˴�С,5X5
	mapSize.r = cnn->C3->mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnn->C3->outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnn->C3->inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
												   //��ʼ�����������
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->S2->y[j][t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->C3->v[i][t][k] = cnn->C3->v[i][t][k] + mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);//�õ�C3���������MAP
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnn->S4->inputWidth;//����ͼ���С,8X8
	inSize.r = cnn->S4->inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnn->S4->mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnn->S4->mapSize;//���ͼ���С,4X4
	for (i = 0; i<(cnn->S4->outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->C3->y[i][m][n];
					}
				}
				cnn->S4->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->S4->y[i][r][c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
															//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->O5->wData[i][j];//�������֮�����,Ȼ�󷵻ؽ��
		cnn->O5->v[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//�����sigmoid����
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);//����sigmoid����,�����������ֵ
}

// sigmoid����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input,float bas) // sigma�����
{
	float temp=input+bas;
	return (float)1.0/((float)(1.0+exp(-temp)));
}
//��һ�����ƽ��ֵ�ĺ���,����S��ػ�,����:output������ĳػ�����,outputSize������ػ�����Ĵ�С.input���������,inputsize����������С,mapSize�ǳػ�����Ĵ�С
//��S2���������һ��24X24��С�ľ���,Ȼ����2X2��СΪһ��������ƽ��ֵ,������12X12��С�ľ���
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // ��ƽ��ֵ
{
	int outputW=inputSize.c/mapSize;//������
	int outputH=inputSize.r/mapSize;//����߶�
	if(outputSize.c!=outputW||outputSize.r!=outputH)//��������������С�͸����������С����ͬ��ʱ��,����
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	//���¼���ƽ��ֵ,��������ƽ��,�ܼ򵥲�������,ע���int���͵�mapsizeת����float������
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float sum=0.0;
			for(m=i*mapSize;m<i*mapSize+mapSize;m++)
				for(n=j*mapSize;n<j*mapSize+mapSize;n++)
					sum=sum+input[m][n];

			output[i][j]=sum/(float)(mapSize*mapSize);
		}
}

// ����ȫ�����������ǰ�򴫲�
// ���������,����λ�ö�ӦԪ�����Ȼ�����,ע������ĳ��ǵ�˲���,���Ǿ�����˲���
float vecMulti(float* vec1,float* vec2,int vecL)
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];//�������֮�����,Ȼ�󷵻ؽ��
	return m;
}
//�˺�������������ͨ�������ǰ�򴫲�����,�����һ������map�ļ��㷽��,����˵��:��input�����ÿһ�����ݺ�wdata�����ÿһ�����ݵ��Ȼ�����,���õ��Ľ������output������,nnSize��������˾���Ĵ�С,Ҫ��������˾���Ĵ�С��ͬ,��ΪnnSize
float sigma_derivation(float y){ // Logic��������Ա���΢��,��sigmoid�����ĵ���
	return y*(1-y); // ����y��ָ��������������ֵ���������Ա���
}
// ����ĺ��򴫲�
void cnnbp(CNN* cnn,float* outputData) 
{
	int i,j,c,r,m,n,t,k; // �����浽������
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->e[i] = cnn->O5->y[i] - outputData[i];//�����ʵ�������ȥ������ȷ�����,��Ӧ��ʽΪai-yi=-(yi-ai),ע�������y[i]��ai,��yi��outputData[i]
												 // �����O5��������
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);//��10����Ԫ��˵,ÿ����Ԫ�������������ȹ�ʽΪ-(yi-ai)(ai*(1-ai)),ע�������y[i]��ai,��yi��outputData[i]
																	// S4�㣬���ݵ�S4������
																	// ����û�м����
	nSize outSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4�����������С,������4X4

	for (i = 0; i < cnn->S4->outChannels; i++) {//��ÿһ���������,����һ�����������һ����С�����жȾ�����֮��Ӧ
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn->O5->outputNum; j++) {//�����Ӧ��ʽ����ͨ������������Ĳв���㹫ʽ,����MATLAB�汾������˵����ƪ����fvd������˵��
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt������λȨֵ,S4���i�����MAP��r�е�c�����j����Ԫ��ȨֵΪ[j][i*outSize.c*outSize.r + r*outSize.c + c],��Ϊ���Ƕ�ά�����ȴ洢����,��һά�����������ӵ������ĵ�j����Ԫ,�ڶ�ά��������������ϵ�Ȩֵ
					cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnn->S4->mapSize;//������Ҫ�����ϲ�������,�����Ҫ����mapSize��С���ϲ���,������2X2
	nSize S4dSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4������жȾ����С,������4X4,Ҳ����S4����������С
	float C3e[8][8];
	for (i = 0; i<cnn->C3->outChannels; i++) {//C3��ÿһ�����MAP����Ӧһ�����жȾ���
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn->S4->mapSize; j = j + cnn->S4->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S4dSize.c*cnn->S4->mapSize; t = t + cnn->S4->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn->S4->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C3e[j][t + m] = cnn->S4->d[i][j / cnn->S4->mapSize][t / cnn->S4->mapSize];//�����
				}
			}
			for (n = 1; n < cnn->S4->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S4dSize.c*cnn->S4->mapSize; t++) {//�з����л�
					C3e[j + n][t] = C3e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn->S4->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��8
			for (c = 0; c<cnn->S4->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��8
				cnn->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->C3->y[i][r][c]) / (float)(cnn->S4->mapSize*cnn->S4->mapSize);//ע��������Ҫ����(float)(cnn->S4->mapSize*cnn->S4->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C3������жȾ���
	}
	// S2�㣬S2��û�м����������ֻ�о�����м��������
	// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��
	outSize.c = cnn->C3->inputWidth;//S2�����жȾ����СΪ12X12
	outSize.r = cnn->C3->inputHeight;//S2�����жȾ����СΪ12X12
	nSize inSize = { cnn->S4->inputWidth,cnn->S4->inputHeight };//C3�����жȾ���Ĵ�С
	nSize mapSize = { cnn->C3->mapSize,cnn->C3->mapSize };//C3�����˴�С5X5
	float corr[12][12];//�洢��ؼ�����
	float exData[16][16];//�洢full֮�����ʱ����
	int addr, addc;
	
	addr = addc = mapSize.r - 1;//Ҫ��չ�ı߳�
	for (i = 0; i<cnn->S2->outChannels; i++) {//����S2��ÿһ�����MAP,6
		for (j = 0; j<cnn->C3->outChannels; j++) {//����C3��ÿһ�����MAP,����������ȫ���ӽṹ,���S2���ÿһ��ͼ����C3���ÿһ��ͼ���й�,12
												  //float** corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);//���ﱾ��Ҫ��C3���Ӧ�ľ����������ת180��Ȼ���ڽ��о������,��ʵ���Ͼ�������ְѾ������ת��180��,�������ֱ�ӾͲ���ת�����,����ֱ�Ӻ;�������,full�������
			int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����,12
			int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData��С,16
												   //��չ����
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//�������������ı�Ե��,����Ϊ0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn->C3->d[j][t - addr][k - addc]; // ��Ȼ,����ԭ����������
				}
			}
			//�������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							corr[t][k] = corr[t][k] + cnn->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->S2->d[i][t][k] = cnn->S2->d[i][t][k] + corr[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
	}
	// C1�㣬�����
	mapdata = cnn->S2->mapSize;//C1��������map�Ĵ�С,24X24
	nSize S2dSize = { cnn->S2->inputWidth / cnn->S2->mapSize,cnn->S2->inputHeight / cnn->S2->mapSize };//S2��������MAP�Ĵ�С,12X12���Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	float C1e[24][24];
	for (i = 0; i<cnn->C1->outChannels; i++) {//C1��ÿһ�����MAP����Ӧһ�����жȾ���
		for (j = 0; j<S2dSize.r*cnn->S2->mapSize; j = j + cnn->S2->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S2dSize.c*cnn->S2->mapSize; t = t + cnn->S2->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn->S2->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C1e[j][t + m] = cnn->S2->d[i][j / cnn->S2->mapSize][t / cnn->S2->mapSize];//�����
				}
			}
			for (n = 1; n < cnn->S2->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S2dSize.c*cnn->S2->mapSize; t++) {//�з����л�
					C1e[j + n][t] = C1e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn->S2->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��24
			for (c = 0; c<cnn->S2->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��24
				cnn->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->C1->y[i][r][c]) / (float)(cnn->S2->mapSize*cnn->S2->mapSize);//ע��������Ҫ����(float)(cnn->S2->mapSize*cnn->S2->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C1������жȾ���
	}
}
//���´�СΪBATCHSIZE����������ݶ�,��ѵ�����ݶȸ��¸���һ��CNN
void cnnupdategrad(CNN** cnnarray)
{
	int i, j;
	nSize mapSize = { cnnarray[0]->C1->mapSize,cnnarray[0]->C1->mapSize };//C1�����˴�С
	for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
		cnnarray[0]->e[i] *= cnnarray[0]->e[i];//�����������ƽ�������
	for (int s = 1; s < BATCHSIZE; s++)
	{
		//�ۼ����
		for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
			cnnarray[0]->e[i] += cnnarray[s]->e[i] * cnnarray[s]->e[i];
		//C1���ݶ��ۼ�
		for (i = 0; i < cnnarray[0]->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
			for (j = 0; j < cnnarray[0]->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			addmat(cnnarray[0]->C1->dmapData[j][i], cnnarray[0]->C1->dmapData[j][i],mapSize, cnnarray[s]->C1->dmapData[j][i],mapSize);//�ۼӾ�����ݶ�
			}
		}
		for (int j = 0; j < cnnarray[0]->C1->outChannels; j++) {//����ÿһ�����MAP,�ۼ�ƫ���ݶ�������6,��С24X24
			cnnarray[0]->C1->dbasicData[j] += cnnarray[s]->C1->dbasicData[j];
		}
		//C3���ݶ��ۼ�
		for (i = 0; i < cnnarray[0]->C3->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
			for (j = 0; j < cnnarray[0]->C3->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
				addmat(cnnarray[0]->C3->dmapData[j][i], cnnarray[0]->C3->dmapData[j][i], mapSize, cnnarray[s]->C3->dmapData[j][i], mapSize);//�ۼӾ�����ݶ�
			}
		}
		for (int j = 0; j < cnnarray[0]->C3->outChannels; j++) {//����ÿһ�����MAP,�ۼ�ƫ���ݶ�������6,��С24X24
			cnnarray[0]->C3->dbasicData[j] += cnnarray[s]->C3->dbasicData[j];
		}
		//������ݶ��ۼ�
		for (j = 0; j<cnnarray[0]->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
			for (i = 0; i<cnnarray[0]->O5->inputNum; i++)//��192����������ݶ�
				cnnarray[0]->O5->dwData[j][i] += cnnarray[s]->O5->dwData[j][i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
			cnnarray[0]->O5->dbasicData[j] += cnnarray[s]->O5->dbasicData[j];//��b�����ݶ�,b���ݶȾ������ж�delta
		}
	}
	//������Ȩ��ƽ��������Ȩ��
	for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
		cnnarray[0]->e[i] /= (float)BATCHSIZE;//����������ƽ��ֵ
	for (i = 0; i < cnnarray[0]->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnnarray[0]->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			multifactor(cnnarray[0]->C1->dmapData[j][i], cnnarray[0]->C1->dmapData[j][i], mapSize, 1.0/BATCHSIZE);//������ݶ���ƽ��
			addmat(cnnarray[0]->C1->mapData[j][i], cnnarray[0]->C1->mapData[j][i],mapSize, cnnarray[0]->C1->dmapData[j][i],mapSize);//�����ݶ�
		}
	}
	for (int j = 0; j < cnnarray[0]->C1->outChannels; j++) {
		cnnarray[0]->C1->dbasicData[j] /= (float)BATCHSIZE;//ƫ����ƽ��
		cnnarray[0]->C1->basicData[j] += cnnarray[0]->C1->dbasicData[j];
	}
	//C3���ݶ���ƽ��
	for (i = 0; i < cnnarray[0]->C3->outChannels; i++) {//����ÿһ�����MAP
		for (j = 0; j < cnnarray[0]->C3->inChannels; j++) {//����ÿһ������ͼ��
			multifactor(cnnarray[0]->C3->dmapData[j][i], cnnarray[0]->C3->dmapData[j][i], mapSize, 1.0 / (float)BATCHSIZE);//������ݶ���ƽ��
			addmat(cnnarray[0]->C3->mapData[j][i], cnnarray[0]->C3->mapData[j][i], mapSize, cnnarray[0]->C3->dmapData[j][i], mapSize);//�����ݶ�
		}
	}
	for (int j = 0; j < cnnarray[0]->C3->outChannels; j++) {
		cnnarray[0]->C3->dbasicData[j] /= (float)BATCHSIZE;
		cnnarray[0]->C3->basicData[j] += cnnarray[0]->C3->dbasicData[j];
	}
	//�������ƽ���ݶ�
	for (j = 0; j<cnnarray[0]->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i < cnnarray[0]->O5->inputNum; i++)//��192����������ݶ�
		{
			cnnarray[0]->O5->dwData[j][i] /= (float)BATCHSIZE;//��ƽ��
			cnnarray[0]->O5->wData[j][i] += cnnarray[0]->O5->dwData[j][i];//�����ݶ�
		}
		cnnarray[0]->O5->dbasicData[j] /= (float)BATCHSIZE;//��ƽ��
		cnnarray[0]->O5->basicData[j] += cnnarray[0]->O5->dbasicData[j];//�����ݶ�
	}
	
}
// ����Ȩ��
void cnnapplygrads(CNN* cnn,CNNOpts opts,float* inputData) 
{

}


void cnnclear(CNN* cnn)
{
	// ����Ԫ�Ĳ����������,��Ҫ��������м䱣�����v,ÿһ������y�Լ��������ֵd,�����ЩֵΪ0.0
	int i,t,k,j,c,r;
	// C1����
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				cnn->C1->d[j][r][c]=(float)0.0;
				cnn->C1->v[j][r][c]=(float)0.0;
				cnn->C1->y[j][r][c]=(float)0.0;
			}
		}
	}
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnn->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnn->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnn->C1->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnn->C1->mapSize; k++) {//�������MAP��ÿһ��
					cnn->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C1->dbasicData[i] = 0.0;
	}
	// S2����
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				cnn->S2->d[j][r][c]=(float)0.0;
				cnn->S2->y[j][r][c]=(float)0.0;
			}
		}
	}
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnn->C3->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnn->C3->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnn->C3->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnn->C3->mapSize; k++) {//�������MAP��ÿһ��
					cnn->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C3->dbasicData[i] = 0.0;
	}
	// C3����
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				cnn->C3->d[j][r][c]=(float)0.0;
				cnn->C3->v[j][r][c]=(float)0.0;
				cnn->C3->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S4����
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				cnn->S4->d[j][r][c]=(float)0.0;
				cnn->S4->y[j][r][c]=(float)0.0;
			}
		}
	}
	// O5���
	for(j=0;j<cnn->O5->outputNum;j++){
		cnn->O5->d[j]=(float)0.0;
		cnn->O5->v[j]=(float)0.0;
		cnn->O5->y[j]=(float)0.0;
	}
	for (j = 0; j<cnn->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i < cnn->O5->inputNum; i++)//��192����������ݶ�
			cnn->O5->dwData[j][i] = 0.0;
		cnn->O5->dbasicData[j] = 0.0;
	}
}

// �������ڲ��Եĺ���,�����Զ����Ƶķ�ʽ��ѵ���õ�CNN������������ݱ��浽�ļ���
void savecnndata(CNN* cnn,const char* filename,float** inputdata) // ����CNN�����е��������
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	// C1������
	int i,j,r;
	// C1����
	for(i=0;i<cnn->C1->inputHeight;i++)
		fwrite(inputdata[i],sizeof(float),cnn->C1->inputWidth,fp);
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->v[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->d[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
		for(r=0;r<cnn->S2->inputHeight;r++){
			fwrite(cnn->C1->y[j][r],sizeof(float),cnn->S2->inputWidth,fp);
		}
	}

	// S2����
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
	}
	// C3����
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->v[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->d[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
		for(r=0;r<cnn->S4->inputHeight;r++){
			fwrite(cnn->C3->y[j][r],sizeof(float),cnn->S4->inputWidth,fp);
		}
	}

	// S4����
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
	}

	// O5�����
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->v,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->d,sizeof(float),cnn->O5->outputNum,fp);
	fwrite(cnn->O5->y,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}
void cnntrain1(CNN* cnn, float* inputData, float* LabelData, CNNOpts opts)
{

	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn->S2->inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn->S2->inputHeight;//�����һ������MAP����Ĵ�С,������24X24
										// ��һ��Ĵ���
	int i, j, r, c, t, k, m, n;
	//clear
	for (j = 0; j<cnn->C1->outChannels; j++) {
		for (r = 0; r<cnn->S2->inputHeight; r++) {
			for (c = 0; c<cnn->S2->inputWidth; c++) {
				cnn->C1->d[j][r][c] = (float)0.0;
				cnn->C1->v[j][r][c] = (float)0.0;
				cnn->C1->y[j][r][c] = (float)0.0;
			}
		}
	}
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnn->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnn->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnn->C1->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnn->C1->mapSize; k++) {//�������MAP��ÿһ��
					cnn->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C1->dbasicData[i] = 0.0;
	}
	// S2����
	for (j = 0; j<cnn->S2->outChannels; j++) {
		for (r = 0; r<cnn->C3->inputHeight; r++) {
			for (c = 0; c<cnn->C3->inputWidth; c++) {
				cnn->S2->d[j][r][c] = (float)0.0;
				cnn->S2->y[j][r][c] = (float)0.0;
			}
		}
	}
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnn->C3->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnn->C3->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnn->C3->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnn->C3->mapSize; k++) {//�������MAP��ÿһ��
					cnn->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C3->dbasicData[i] = 0.0;
	}
	// C3����
	for (j = 0; j<cnn->C3->outChannels; j++) {
		for (r = 0; r<cnn->S4->inputHeight; r++) {
			for (c = 0; c<cnn->S4->inputWidth; c++) {
				cnn->C3->d[j][r][c] = (float)0.0;
				cnn->C3->v[j][r][c] = (float)0.0;
				cnn->C3->y[j][r][c] = (float)0.0;
			}
		}
	}
	// S4����
	for (j = 0; j<cnn->S4->outChannels; j++) {
		for (r = 0; r<cnn->S4->inputHeight / cnn->S4->mapSize; r++) {
			for (c = 0; c<cnn->S4->inputWidth / cnn->S4->mapSize; c++) {
				cnn->S4->d[j][r][c] = (float)0.0;
				cnn->S4->y[j][r][c] = (float)0.0;
			}
		}
	}
	// O5���
	for (j = 0; j<cnn->O5->outputNum; j++) {
		cnn->O5->d[j] = (float)0.0;
		cnn->O5->v[j] = (float)0.0;
		cnn->O5->y[j] = (float)0.0;
	}
	for (j = 0; j<cnn->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i < cnn->O5->inputNum; i++)//��192����������ݶ�
			cnn->O5->dwData[j][i] = 0.0;
		cnn->O5->dbasicData[j] = 0.0;
	}

	// ��һ���������
	nSize mapSize = { cnn->C1->mapSize,cnn->C1->mapSize };//����˴�С,5X5
	nSize inSize = { cnn->C1->inputWidth,cnn->C1->inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnn->S2->inputWidth,cnn->S2->inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnn->C1->outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnn->C1->inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
												   //�Ծ������ת180��
												   //��ʼ�����������
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->C1->v[i][t][k] = cnn->C1->v[i][t][k] + mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->basicData[i]);
			}
		}
	}

	// �ڶ�����������S2��������
	outSize.c = cnn->C3->inputWidth;//���ͼ���С,12X12
	outSize.r = cnn->C3->inputHeight;//���ͼ���С,12X12
	inSize.c = cnn->S2->inputWidth;//����ͼ���С,24X24
	inSize.r = cnn->S2->inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i<(cnn->S2->outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												//avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);//C1->y[i]����S2->mapSize��Сƽ���ػ����������S2->y[i]
												//�²����ػ�
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->C1->y[i][m][n];
					}
				}
				cnn->S2->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// �������������,������ȫ����
	outSize.c = cnn->S4->inputWidth;//���ͼ���С,8X8
	outSize.r = cnn->S4->inputHeight;//���ͼ���С,8X8
	inSize.c = cnn->C3->inputWidth;//����ͼ���С,12X12
	inSize.r = cnn->C3->inputHeight;//����ͼ���С,12X12
	mapSize.c = cnn->C3->mapSize;//����˴�С,5X5
	mapSize.r = cnn->C3->mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnn->C3->outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnn->C3->inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
												   //��ʼ�����������
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->S2->y[j][t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->C3->v[i][t][k] = cnn->C3->v[i][t][k] + mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);//�õ�C3���������MAP
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnn->S4->inputWidth;//����ͼ���С,8X8
	inSize.r = cnn->S4->inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnn->S4->mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnn->S4->mapSize;//���ͼ���С,4X4
	for (i = 0; i<(cnn->S4->outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->C3->y[i][m][n];
					}
				}
				cnn->S4->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->S4->y[i][r][c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
															//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->O5->wData[i][j];//�������֮�����,Ȼ�󷵻ؽ��
		cnn->O5->v[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//�����sigmoid����
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);//����sigmoid����,�����������ֵ

																			   //bp
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->e[i] = cnn->O5->y[i] - LabelData[i];//�����ʵ�������ȥ������ȷ�����,��Ӧ��ʽΪai-yi=-(yi-ai),ע�������y[i]��ai,��yi��outputData[i]
												 // �����O5��������
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);//��10����Ԫ��˵,ÿ����Ԫ�������������ȹ�ʽΪ-(yi-ai)(ai*(1-ai)),ע�������y[i]��ai,��yi��outputData[i]
																	// S4�㣬���ݵ�S4������
																	// ����û�м����
	outSize.r = cnn->S4->inputWidth / cnn->S4->mapSize;
	outSize.c = cnn->S4->inputHeight / cnn->S4->mapSize;//S4�����������С,������4X4
	for (i = 0; i < cnn->S4->outChannels; i++) {//��ÿһ���������,����һ�����������һ����С�����жȾ�����֮��Ӧ
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn->O5->outputNum; j++) {//�����Ӧ��ʽ����ͨ������������Ĳв���㹫ʽ,����MATLAB�汾������˵����ƪ����fvd������˵��
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt������λȨֵ,S4���i�����MAP��r�е�c�����j����Ԫ��ȨֵΪ[j][i*outSize.c*outSize.r + r*outSize.c + c],��Ϊ���Ƕ�ά�����ȴ洢����,��һά�����������ӵ������ĵ�j����Ԫ,�ڶ�ά��������������ϵ�Ȩֵ
					cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnn->S4->mapSize;//������Ҫ�����ϲ�������,�����Ҫ����mapSize��С���ϲ���,������2X2
	nSize S4dSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4������жȾ����С,������4X4,Ҳ����S4����������С
	float C3e[8][8];
	for (i = 0; i<cnn->C3->outChannels; i++) {//C3��ÿһ�����MAP����Ӧһ�����жȾ���
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn->S4->mapSize; j = j + cnn->S4->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S4dSize.c*cnn->S4->mapSize; t = t + cnn->S4->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn->S4->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C3e[j][t + m] = cnn->S4->d[i][j / cnn->S4->mapSize][t / cnn->S4->mapSize];//�����
				}
			}
			for (n = 1; n < cnn->S4->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S4dSize.c*cnn->S4->mapSize; t++) {//�з����л�
					C3e[j + n][t] = C3e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn->S4->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��8
			for (c = 0; c<cnn->S4->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��8
				cnn->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->C3->y[i][r][c]) / (float)(cnn->S4->mapSize*cnn->S4->mapSize);//ע��������Ҫ����(float)(cnn->S4->mapSize*cnn->S4->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C3������жȾ���
	}
	// S2�㣬S2��û�м����������ֻ�о�����м��������
	// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��
	outSize.c = cnn->C3->inputWidth;//S2�����жȾ����СΪ12X12
	outSize.r = cnn->C3->inputHeight;//S2�����жȾ����СΪ12X12
	inSize.r = cnn->S4->inputWidth;
	inSize.c = cnn->S4->inputHeight;//C3�����жȾ���Ĵ�С
	mapSize.r = cnn->C3->mapSize;
	mapSize.c = cnn->C3->mapSize;//C3�����˴�С5X5
	float corr[12][12];//�洢��ؼ�����
	float exData[16][16];//�洢full֮�����ʱ����
	int addr, addc;

	addr = addc = mapSize.r - 1;//Ҫ��չ�ı߳�
	for (i = 0; i<cnn->S2->outChannels; i++) {//����S2��ÿһ�����MAP,6
		for (j = 0; j<cnn->C3->outChannels; j++) {//����C3��ÿһ�����MAP,����������ȫ���ӽṹ,���S2���ÿһ��ͼ����C3���ÿһ��ͼ���й�,12
												  //float** corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);//���ﱾ��Ҫ��C3���Ӧ�ľ����������ת180��Ȼ���ڽ��о������,��ʵ���Ͼ�������ְѾ������ת��180��,�������ֱ�ӾͲ���ת�����,����ֱ�Ӻ;�������,full�������
			int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����,12
			int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData��С,16
												   //��չ����
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//�������������ı�Ե��,����Ϊ0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn->C3->d[j][t - addr][k - addc]; // ��Ȼ,����ԭ����������
				}
			}
			//�������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							corr[t][k] = corr[t][k] + cnn->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->S2->d[i][t][k] = cnn->S2->d[i][t][k] + corr[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
	}
	// C1�㣬�����
	mapdata = cnn->S2->mapSize;//C1��������map�Ĵ�С,24X24
	nSize S2dSize = { cnn->S2->inputWidth / cnn->S2->mapSize,cnn->S2->inputHeight / cnn->S2->mapSize };//S2��������MAP�Ĵ�С,12X12���Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	float C1e[24][24];
	for (i = 0; i<cnn->C1->outChannels; i++) {//C1��ÿһ�����MAP����Ӧһ�����жȾ���
		for (j = 0; j<S2dSize.r*cnn->S2->mapSize; j = j + cnn->S2->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S2dSize.c*cnn->S2->mapSize; t = t + cnn->S2->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn->S2->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C1e[j][t + m] = cnn->S2->d[i][j / cnn->S2->mapSize][t / cnn->S2->mapSize];//�����
				}
			}
			for (n = 1; n < cnn->S2->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S2dSize.c*cnn->S2->mapSize; t++) {//�з����л�
					C1e[j + n][t] = C1e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn->S2->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��24
			for (c = 0; c<cnn->S2->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��24
				cnn->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->C1->y[i][r][c]) / (float)(cnn->S2->mapSize*cnn->S2->mapSize);//ע��������Ҫ����(float)(cnn->S2->mapSize*cnn->S2->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C1������жȾ���
	}

	//apply
	// C1���Ȩ�ظ���
	nSize dSize = { cnn->S2->inputHeight,cnn->S2->inputWidth };//C1�������Ⱦ����С,24X24
	nSize ySize = { cnn->C1->inputHeight,cnn->C1->inputWidth };//C1����������С,28X28
	mapSize.r = cnn->C1->mapSize;
	mapSize.c = cnn->C1->mapSize;//C1�����˴�С
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnn->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j<cnn->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
												 //����,һάת��ά����,��ת180���ƺ�����
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = inputData[r*ySize.c + c];
				}
			}
			//����֮���Ի����,�����齻����򵥵�����,a=b,b=a����ֱ��д,Ҫ��C����ת!!!!
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
																	   //cout << tin[r][c] << " ";
				}
				//cout << endl;
			}
			//system("pause");
			//��ת�����
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov[r][c] = cnn->C1->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}

			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnn->C1->dmapData[j][i][t][k] = cnn->C1->dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnn->C1->dmapData[j][i][t][k] = cnn->C1->dmapData[j][i][t][k] * -1 * opts.alpha;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnn->C1->d[i][t][j];
		cnn->C1->dbasicData[i] = -1 * opts.alpha*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// C3���Ȩ�ظ���
	dSize.c = cnn->S4->inputWidth;//C3�������Ⱦ����С,8X8
	dSize.r = cnn->S4->inputHeight;//C3�������Ⱦ����С,8X8
	ySize.c = cnn->C3->inputWidth;//C3����������С,12X12
	ySize.r = cnn->C3->inputHeight;//C3����������С,12X12
	mapSize.c = cnn->C3->mapSize;//C3�����˴�С,5X5
	mapSize.r = cnn->C3->mapSize;//C3�����˴�С,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnn->C3->outChannels; i++) {//����ÿһ�����MAP,������12,��С8X8
		for (j = 0; j<cnn->C3->inChannels; j++) {//����ÿһ������ͼ��,������8,��С12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnn->S2->y[j][ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//��ת�����
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnn->C3->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnn->C3->dmapData[j][i][t][k] = cnn->C3->dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnn->C3->dmapData[j][i][t][k] = cnn->C3->dmapData[j][i][t][k] * -1 * opts.alpha;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnn->C3->d[i][t][j];
		cnn->C3->dbasicData[i] = -1 * opts.alpha*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// �����
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	for (j = 0; j<cnn->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i<cnn->O5->inputNum; i++)//��192����������ݶ�
			cnn->O5->dwData[j][i] = -1 * opts.alpha*cnn->O5->d[j] * O5inData[i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
		cnn->O5->dbasicData[j] = -1 * opts.alpha*cnn->O5->d[j];//��b�����ݶ�,b���ݶȾ������ж�delta
	}
}