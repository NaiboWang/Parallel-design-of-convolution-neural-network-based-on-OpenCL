#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
using namespace std;

clock_t start, finish;//计算时间用的
double  duration;

void cnnsetup(CNN** cnnarray,nSize inputSize,int outputSize,FILE *fp)
{
	start = clock();//开始计时
	for(int i=0;i<BATCHSIZE;i++)//初始化BATCHSIZE组CNN网络
	{
		cnnarray[i]->layerNum=5;//设置cnn层数为5
		nSize inSize;//输入图像大小
		int mapSize=5;//定义卷积核大小为5
		inSize.c=inputSize.c;//输入图像大小为28X28
		inSize.r=inputSize.r;//输入图像大小为28X28
		cnnarray[i]->C1=initCovLayer(inSize.c,inSize.r,5,1,6);//以输入图像大小为28X28,卷积核大小为5X5,输入图像数为1,输出MAP数为6初始化C1层,具体初始化过程见initCovLayer函数定义
		inSize.c=inSize.c-mapSize+1;//S2层的输入MAP的大小为28-5+1=24,即24X24
		inSize.r=inSize.r-mapSize+1;//S2层的输入MAP的大小为28-5+1=24,即24X24
		cnnarray[i]->S2=initPoolLayer(inSize.c,inSize.r,2,6,6,AvePool); //以输入图像大小为24X24, 池化大小为2X2, 输入图像数为6, 输出MAP数为6,池化方法为平均池化初始化S2层, 具体初始化过程见initPoolLayer函数定义
		inSize.c=inSize.c/2;//C3层的输入图像大小为24/2=12,即12X12
		inSize.r=inSize.r/2;//C3层的输入图像大小为24/2=12,即12X12
		cnnarray[i]->C3=initCovLayer(inSize.c,inSize.r,5,6,12);//以输入图像大小为12X12,卷积核大小为5X5,输入图像数为6,输出MAP数为12初始化C3层,具体初始化过程见initCovLayer函数定义
		inSize.c=inSize.c-mapSize+1;//S4层输入图像大小为12-5+1=8,即8X8
		inSize.r=inSize.r-mapSize+1;//S4层输入图像大小为12-5+1=8,即8X8
		cnnarray[i]->S4=initPoolLayer(inSize.c,inSize.r,2,12,12,AvePool);//以输入图像大小为8X8, 池化大小为2X2, 输入图像数为12, 输出MAP数为12,池化方法为平均池化初始化S4层, 具体初始化过程见initPoolLayer函数定义
		inSize.c=inSize.c/2;//全连接输出层输入图像大小为8/2=4,即4X4
		inSize.r=inSize.r/2;//全连接输出层输入图像大小为8/2=4,即4X4
		cnnarray[i]->O5=initOutLayer(inSize.c*inSize.r*12,outputSize);//以输入图像大小为4*4*12=192,输出图像为10初始化输出层,具体初始化过程见initOutLayer函数定义
		cnnarray[i]->e=(float*)calloc(cnnarray[i]->O5->outputNum,sizeof(float));//给训练误差初始化为一个长度为cnn->O5->outputNum个,也就是10的数组用来保存每次训练之后每个分类的训练误差,并初始赋值为0
	}
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("setuptime:%f seconds\n", duration);
	fprintf(fp,"setuptime:%f seconds\n", duration);
}
//初始化卷积层,参数为输入图像的大小inputWidth,inputHeight,卷积核大小mapSize,输入图像个数inChannels,输出图像个数outChannels
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));//分配一个卷积层结构体出来

	covL->inputHeight=inputHeight;//输入图像高度为inputHeight
	covL->inputWidth=inputWidth;//输入图像宽度为inputWidth
	covL->mapSize=mapSize;//卷积核大小为mapSize

	covL->inChannels=inChannels;//输入图像个数
	covL->outChannels=outChannels;//输出MAP个数

	covL->isFullConnect=true; // 默认为全连接

	// 权重空间的初始化，先行再列调用，[r][c]
	int i,j,c,r;
	srand((unsigned)time(NULL));//随机化初始化种子以便每次初始化得到的随机数都不相同
	covL->mapData=(float****)malloc(inChannels*sizeof(float***));//分配输入图像个三维数组来存储卷积核的数据,其中一个三维数组的第三维表示当前这幅图像对应输出图像的个数,例如C3层就有6X12个卷积核,每个卷积核大小为5X5,故C3层mapData大小为6X12X5X5,同理C1层mapData大小为1X6X5X5
	for(i=0;i<inChannels;i++){//对一副输入图像
		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));//分配输出图像个数个二维数组
		for(j=0;j<outChannels;j++){//对一副输出图像
			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));//分配卷积核宽度个数组
			for(r=0;r<mapSize;r++){
				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));//一个数组包含卷积核列数个数
				for(c=0;c<mapSize;c++){
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; //对卷积核进行初始化操作,生成0~1之间的随机数((float)rand()/(float)RAND_MAX),然后减去0.5再乘2,也就是说这里生成的是-1~1之间的随机数
					covL->mapData[i][j][r][c]=randnum*sqrt((float)6.0/(float)(mapSize*mapSize*(inChannels+outChannels)));//这里把卷积核的值初始化为对应的公式,原因详见Nguyen-widrow方法,尽可能让权值初始化到sigmoid函数的[-0.7,0.7]区间内,这个公式怎么得来的我现在还不太清楚
				}
			}
		}
	}
	// 权重梯度变化
	covL->dmapData=(float****)malloc(inChannels*sizeof(float***));//同理,这里保存的是每次反向传播完之后,卷积核要更新的权值,大小自然和上面的相同,例如C3层为6X12X5X5
	//下面的分配方法也相同
	for(i=0;i<inChannels;i++){
		covL->dmapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->dmapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->dmapData[i][j][r]=(float*)calloc(mapSize,sizeof(float));//分配一个长度为mapSize,也就是5个数组并全部初始化为0
			}
		}
	}

	covL->basicData=(float*)calloc(outChannels,sizeof(float));//这是每一个输出MAP的偏置b,有多少个输出MAP就有多少个b
	covL->dbasicData = (float*)calloc(outChannels, sizeof(float));//这是每一个输出MAP的偏置梯度db,有多少个输出MAP就有多少个db
	int outW=inputWidth-mapSize+1;//输出MAP大小的宽度
	int outH=inputHeight-mapSize+1;//输出MAP大小的高度


	covL->d=(float***)malloc(outChannels*sizeof(float**));// 网络的局部梯度,δ值,因此这个大小和输出MAP,就是下面的y相同
	covL->v=(float***)malloc(outChannels*sizeof(float**)); // 进入激活函数的输入值,这里把他存储起来了,MATLAB版本里面没有存储这个值,而是在cnnff.m中把他用临时变量z存储了
	covL->y=(float***)malloc(outChannels*sizeof(float**));//输出MAP
	//以上三个MAP大小均为outChannels*outH*outW,因为均和输出MAP有关,下面几行把他们同时初始化为0.0,如C3层为12X8X8
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

	return covL;//返回初始化好的卷积层
}
//初始化池化层,参数为输入图像的大小inputWidth,inputHeight,池化大小mapSize,输入图像个数inChannels,输出图像个数outChannels,池化类型poolType
PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));//分配池化层空间
	//以下分别赋值相应的参数,不做解释
	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;
	poolL->poolType=poolType; 

	poolL->basicData=(float*)calloc(outChannels,sizeof(float));//每一张输出map的偏置b,对这个cnn网络来说,实际上都是0.0
	//下面定义输出图像大小,如S2层为24/2=12
	int outW=inputWidth/mapSize;
	int outH=inputHeight/mapSize;

	int j,r;
	poolL->d=(float***)malloc(outChannels*sizeof(float**));// 网络的局部梯度,δ值
	poolL->y=(float***)malloc(outChannels*sizeof(float**));// 采样函数后神经元的输出,无激活函数
	//初始化偏置和输出MAP为0.0,注意大小为outChannels*outH*outW,如S4层为12X4X4
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
//初始化最后一层的参数,把他视为普通神经网络,参数为输入节点数inputNum和输出节点数outputNum
OutLayer* initOutLayer(int inputNum,int outputNum)
{
	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));//分配初始空间
	//赋值参数值
	outL->inputNum=inputNum;
	outL->outputNum=outputNum;
	

	outL->basicData=(float*)calloc(outputNum,sizeof(float));//10个偏置b,每个值初始为0.0
	outL->dbasicData = (float*)calloc(outputNum, sizeof(float));//10个偏置梯度db,每个值初始为0.0

	outL->d=(float*)calloc(outputNum,sizeof(float));// 网络的局部梯度,δ值,10个0.0
	outL->v=(float*)calloc(outputNum,sizeof(float));// 进入激活函数的输入值,10个0.0,在cnnff.m中直接计算的没有保存
	outL->y=(float*)calloc(outputNum,sizeof(float));// 激活函数后神经元的输出,10个0.0

	// 权重的初始化
	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // 输入行，输出列,存的是10X192个权值矩阵
	outL->dwData = (float**)malloc(outputNum * sizeof(float*)); // 输入行，输出列,存的是10X192个权值矩阵
	int i,j;
	//以下初始化权值矩阵
	srand((unsigned)time(NULL));
	for(i=0;i<outputNum;i++){
		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));
		outL->dwData[i] = (float*)calloc(inputNum,sizeof(float));//初始化dw10个矩阵的梯度值
		for(j=0;j<inputNum;j++){
			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数
			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));//公式仍然不清楚为什么这样定义
		}
	}

	outL->isFullConnect=true;//设置为全连接

	return outL;
}
//到这里,MATLAB版本cnnsetup的内容就全部结束

// 返回向量最大数的序号,注意这里是向量,所以是一维数组
int vecmaxIndex(float* vec, int veclength)
{
	//下面就是一个最简单的在veclength个长度数组中找最大元素的算法,最后返回的是最大的那个序号
	//这个函数的作用是最后比较测试结果和正确的标签结果是不是同一个值,然后计算误差
	int i;
	float maxnum=-1.0;
	int maxIndex=0;
	for(i=0;i<veclength;i++){
		if(maxnum<vec[i]){
			maxnum=vec[i];
			maxIndex=i;
		}
	}
	return maxIndex;//返回相似度最大的那个神经元的序号
}

// 测试cnn函数,这里的参数为:cnn代表训练好的cnn网络,inputData为测试集的原始图像数据,outputData为测试集实际的正确结果,testNum为测试集数量,这里为10000
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum, FILE *fp)
{
	start = clock();
	int n=0;
	int incorrectnum=0;  //错误预测的数目
	/*for(n=0;n<testNum;n++){*/
		clSetKernelArgSVMPointer(kernel, 0, cnn);
		clSetKernelArgSVMPointer(kernel, 1, inputData);
		clSetKernelArgSVMPointer(kernel, 2, outputData);
		clSetKernelArgSVMPointer(kernel, 3, &incorrectnum);
		size_t globalWorkSize = testNum;//全局工作项大小
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, 0, &globalWorkSize, NULL, NULL, NULL, NULL);
		//cout << num << " ";
	//}
	clFinish(commandQueue);
	//cout << "wrongnum:" << incorrectnum << endl;
	//for (int i = 0; i < 1000; i++)
	//	cout << t2[i] << ' ';
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("testtime:%f seconds\n", duration);
	cnnclear(cnn);//最后别忘了清空一次结果
	return (float)incorrectnum/(float)testNum;//返回错误率
}
//根据第一个CNN复制出后面BATCHSIZE-1个CNN
void cnncpy(CNN** cnnarray,FILE *fp)
{
	//start = clock();//开始计时
	for (int k = 1; k < BATCHSIZE; k++)
	{
		int i, j, r,s;
		// 复制C1的数据
		for (i = 0; i < cnnarray[0]->C1->inChannels; i++)
			for (j = 0; j < cnnarray[0]->C1->outChannels; j++)
				for (r = 0; r < cnnarray[0]->C1->mapSize; r++)
					for (s = 0; s < cnnarray[0]->C1->mapSize; s++)
					cnnarray[k]->C1->mapData[i][j][r][s] = cnnarray[0]->C1->mapData[i][j][r][s];
		for (i = 0; i < cnnarray[0]->C1->outChannels; i++)
			cnnarray[k]->C1->basicData[i] = cnnarray[0]->C1->basicData[i];
		//C3层信息复制
		for (i = 0; i < cnnarray[0]->C3->inChannels; i++)
			for (j = 0; j < cnnarray[0]->C3->outChannels; j++)
				for (r = 0; r < cnnarray[0]->C3->mapSize; r++)
					for (s = 0; s < cnnarray[0]->C3->mapSize; s++)
						cnnarray[k]->C3->mapData[i][j][r][s] = cnnarray[0]->C3->mapData[i][j][r][s];
		for (i = 0; i < cnnarray[0]->C3->outChannels; i++)
			cnnarray[k]->C3->basicData[i] = cnnarray[0]->C3->basicData[i];
		//输出层信息复制
		for (i = 0; i<cnnarray[0]->O5->outputNum; i++)
			for (j = 0; j < cnnarray[0]->O5->inputNum; j++)
			cnnarray[k]->O5->wData[i][j] = cnnarray[0]->O5->wData[i][j];
		for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
			cnnarray[k]->O5->basicData = cnnarray[0]->O5->basicData;
	}
	//finish = clock();//结束计时,单位毫秒
	//duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	//printf("copytime:%f seconds\n", duration);
	//fprintf(fp, "copytime:%f seconds\n", duration);
}
// 保存cnn
void savecnn(CNN* cnn, const char* filename)//用来把CNN网络中每一层的权值(卷积核)和偏置存储到文件中
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);

	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);

	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);

	fclose(fp);
}
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename)//用来从文件中导入每一层的权值(卷积核)和偏置到CNN网络
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("write file failed\n");

	int i,j,c,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
		for(j=0;j<cnn->C1->outChannels;j++)
			for(r=0;r<cnn->C1->mapSize;r++)
				for(c=0;c<cnn->C1->mapSize;c++){
					float* in=(float*)malloc(sizeof(float));//分配一个长度为1的数组?为什么这样做
					fread(in,sizeof(float),1,fp);
					cnn->C1->mapData[i][j][r][c]=*in;
				}

	for(i=0;i<cnn->C1->outChannels;i++)
		fread(&cnn->C1->basicData[i],sizeof(float),1,fp);//读取偏置值,一共6个

	// C3网络
	for(i=0;i<cnn->C3->inChannels;i++)
		for(j=0;j<cnn->C3->outChannels;j++)
			for(r=0;r<cnn->C3->mapSize;r++)
				for(c=0;c<cnn->C3->mapSize;c++)
				fread(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);//同上,读取参数值

	for(i=0;i<cnn->C3->outChannels;i++)
		fread(&cnn->C3->basicData[i],sizeof(float),1,fp);//读取偏置值,一共12个

	// O5输出层
	for(i=0;i<cnn->O5->outputNum;i++)
		for(j=0;j<cnn->O5->inputNum;j++)
			fread(&cnn->O5->wData[i][j],sizeof(float),1,fp);//读取输出层的权值矩阵

	for(i=0;i<cnn->O5->outputNum;i++)
		fread(&cnn->O5->basicData[i],sizeof(float),1,fp);//读取输出层的偏置值,一共10个

	fclose(fp);
}
void int2str(const int &int_temp, string &string_temp)
{
	stringstream stream;
	stream << int_temp;
	string_temp = stream.str();   //此处也可以用 stream>>string_temp  
}
//用来训练CNN的网络,根据传入的原始图像inputData,图像的正确值(标签)outputData,训练的参数opts以及训练集的数量trainNum来训练网络,这里trainNum为55000,inputData为60000幅原始图像,outputData为60000幅标签
void cnntrain(CNNA *cnns,ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum)
{
	GTime time = GTime("setuptime:", fp), time2 = GTime("setuptime:", fp), time3 = GTime("setuptime:", fp);
	int testtime = 0;
	//这里并没有打乱原始数据,而是顺序训练的,可能是因为打乱的成本太高
	// 学习训练误差曲线,个数为55000个
	cnns->cnn[0]->L=(float*)malloc(trainNum*sizeof(float));//第一个cnn来保存学习误差
	int e;
	if (trainNum % BATCHSIZE != 0)
	{
		cout << "对不起,批次数量不能被全样本个数整除,不能进行训练!" << endl;
		exit(-1);
	}
	for(e=0;e<opts.numepochs;e++){//训练次数
		float incorrectRatio = 0.0;//错误率,默认为0
		string t;
		//int2str(e, t);
		//time2.startT("test" + t + "time:");
		//incorrectRatio = cnntest(cnns->cnn[0], inputData1, outputData1, testNum, fp);//测试CNN网络,输出错误率,用的是第一个CNN网络,后面的和第一个是一样的
		//time2.endT();
		//cout << "test" << e << "time:" << time2.getDu() << "ms" << endl;
		//cout << "test" << e << "error:" << incorrectRatio << endl;
		//fprintf(fp, "test%derror:%f\n", e, incorrectRatio);
		time.startT("第" + t + "次训练时间:");
		int train = trainNum / BATCHSIZE;//批训练次数
		for(int n=0;n<train;n++){//批训练
			//printf("%d\n",n);		
			//if (n == 0)//第一次训练开始计时
			if (n == 0)
			{
				int2str(n, t);
				//time2.startT("trainset" + t + "time:");
			}
			cnncpy(cnns->cnn, fp);//把第一个CNN的信息复制给后面BATCHSIZE-1个
				int bs = n*BATCHSIZE;
				//cout << bs << endl;
				int *pb = &bs;
				clSetKernelArgSVMPointer(kernel2, 0, cnns);
				clSetKernelArgSVMPointer(kernel2, 1, inputData);
				clSetKernelArgSVMPointer(kernel2, 2, outputData);
				clSetKernelArgSVMPointer(kernel2, 3, &opts);
				clSetKernelArgSVMPointer(kernel2, 4, pb);
				size_t globalWorkSize = BATCHSIZE;//全局工作项大小
				clEnqueueNDRangeKernel(commandQueue, kernel2, 1, 0, &globalWorkSize, NULL, NULL, NULL, NULL);
			//排队等待执行完成
			clFinish(commandQueue);
			//cout << bs << endl;
			//cout << b << endl;
			//system("pause");
			//for (int s = 0; s < BATCHSIZE; s++)//对每一个输入变量的值
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
			cnnupdategrad(cnns->cnn);//批量更新整个网络的梯度

			float l = 0.0;
			int i;
			for (i = 0; i<cnns->cnn[0]->O5->outputNum; i++)
				l = l + cnns->cnn[0]->e[i] * cnns->cnn[0]->e[i];//计算均方误差e[i]^2,下面除以2才是真正的均方误差E,e[i] = t[i] - y[i],见cnnbp函数
			if (n == 0)
				cnns->cnn[0]->L[n] = l / (float)2.0;//第一次让误差值为l(L)/2
			else
				cnns->cnn[0]->L[n] = cnns->cnn[0]->L[n - 1] * 0.99 + 0.01*l / (float)2.0;//第二次开始让误差值等于这个函数
			if (n % 20 == 0)
			{
				char* filedir = "E:\\CNNData\\";//先把cnn原来的权值保存到这个目录下
				const char* filename = combine_strings(filedir, combine_strings(intTochar(testtime++), ".cnn"));//文件名字是n.cnn
				savecnn(cnns->cnn[0], filename);//把卷积神经网络保存下来
				//time2.endT();
				//printf("trainset%dtime:%f ms\n", n, time2.getDu());
				//printf("error:%f\n", cnns->cnn[0]->L[n]);
				//fprintf(fp, "error:%f\n", cnns->cnn[0]->L[n]);
				//int2str(n, t);
				//time2.startT("tr		time2.startT("test" + t + "time:");
				incorrectRatio = cnntest(cnns->cnn[0], inputData1, outputData1, testNum, fp);//测试CNN网络,输出错误率,用的是第一个CNN网络,后面的和第一个是一样的
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

// 这里InputData是图像数据，inputData[r][c],r行c列，这里跟各权重模板是一致的
//注意这里采用的是在线学习,也就是一个图像一个图像的学习,每个图像都会生成一堆权值,然后马上更新
void cnnff(CNN* cnn,float* inputData)
{
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn->S2->inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn->S2->inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
										// 第一层的传播
	int i, j, r, c, t, k, m, n;
	// 第一层输出数据
	nSize mapSize = { cnn->C1->mapSize,cnn->C1->mapSize };//卷积核大小,5X5
	nSize inSize = { cnn->C1->inputWidth,cnn->C1->inputHeight };//输入图像大小,28X28
	nSize outSize = { cnn->S2->inputWidth,cnn->S2->inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnn->C1->outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnn->C1->inChannels); j++) {//对C1层的每一个输入MAP,这里为1
												   //对卷积核旋转180度
												   //初始化卷积用数组
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->C1->v[i][t][k] = cnn->C1->v[i][t][k] + mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->basicData[i]);
			}
		}
	}

	// 第二层的输出传播S2，采样层
	outSize.c = cnn->C3->inputWidth;//输出图像大小,12X12
	outSize.r = cnn->C3->inputHeight;//输出图像大小,12X12
	inSize.c = cnn->S2->inputWidth;//输入图像大小,24X24
	inSize.r = cnn->S2->inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i<(cnn->S2->outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												//avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);//C1->y[i]经过S2->mapSize大小平均池化后结果输出到S2->y[i]
												//下采样池化
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

	// 第三层输出传播,这里是全连接
	outSize.c = cnn->S4->inputWidth;//输出图像大小,8X8
	outSize.r = cnn->S4->inputHeight;//输出图像大小,8X8
	inSize.c = cnn->C3->inputWidth;//输入图像大小,12X12
	inSize.r = cnn->C3->inputHeight;//输入图像大小,12X12
	mapSize.c = cnn->C3->mapSize;//卷积核大小,5X5
	mapSize.r = cnn->C3->mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnn->C3->outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnn->C3->inChannels); j++) {//对C3层的每一个输入MAP,这里为6
												   //初始化卷积用数组
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->S2->y[j][t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->C3->v[i][t][k] = cnn->C3->v[i][t][k] + mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);//得到C3层最后的输出MAP
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnn->S4->inputWidth;//输入图像大小,8X8
	inSize.r = cnn->S4->inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnn->S4->mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnn->S4->mapSize;//输出图像大小,4X4
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

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->S4->y[i][r][c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
															//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->O5->wData[i][j];//向量相乘之后相加,然后返回结果
		cnn->O5->v[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//最后用sigmoid函数
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);//计算sigmoid函数,即输出层的输出值
}

// sigmoid激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas) // sigma激活函数
{
	float temp=input+bas;
	return (float)1.0/((float)(1.0+exp(-temp)));
}
//求一块矩阵平均值的函数,用于S层池化,参数:output是输出的池化矩阵,outputSize是输出池化矩阵的大小.input是输入矩阵,inputsize是输入矩阵大小,mapSize是池化区域的大小
//如S2层就是输入一个24X24大小的矩阵,然后以2X2大小为一个区域求平均值,最后输出12X12大小的矩阵
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // 求平均值
{
	int outputW=inputSize.c/mapSize;//输出宽度
	int outputH=inputSize.r/mapSize;//输出高度
	if(outputSize.c!=outputW||outputSize.r!=outputH)//计算出来的输出大小和给定的输出大小不相同的时候,报错
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	//以下计算平均值,加起来求平均,很简单不做解释,注意把int类型的mapsize转化成float来计算
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

// 单层全连接神经网络的前向传播
// 两向量相乘,即各位置对应元素相乘然后求和,注意这里的乘是点乘操作,不是矩阵相乘操作
float vecMulti(float* vec1,float* vec2,int vecL)
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];//向量相乘之后相加,然后返回结果
	return m;
}
//此函数用来定义普通神经网络的前向传播过程,即最后一层的输出map的计算方法,参数说明:把input矩阵的每一行数据和wdata矩阵的每一行数据点乘然后求和,最后得到的结果放入output数组里,nnSize是两个相乘矩阵的大小,要求两个相乘矩阵的大小相同,均为nnSize
float sigma_derivation(float y){ // Logic激活函数的自变量微分,即sigmoid函数的导数
	return y*(1-y); // 这里y是指经过激活函数的输出值，而不是自变量
}
// 网络的后向传播
void cnnbp(CNN* cnn,float* outputData) 
{
	int i,j,c,r,m,n,t,k; // 将误差保存到网络中
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->e[i] = cnn->O5->y[i] - outputData[i];//误差是实际输出减去真正正确的输出,对应公式为ai-yi=-(yi-ai),注意这里的y[i]是ai,而yi是outputData[i]
												 // 输出层O5的灵敏度
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);//对10个神经元来说,每个神经元的输出层的灵敏度公式为-(yi-ai)(ai*(1-ai)),注意这里的y[i]是ai,而yi是outputData[i]
																	// S4层，传递到S4层的误差
																	// 这里没有激活函数
	nSize outSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4层的输出矩阵大小,这里是4X4

	for (i = 0; i < cnn->S4->outChannels; i++) {//对每一个输出矩阵,都有一个和输出矩阵一样大小的敏感度矩阵与之对应
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn->O5->outputNum; j++) {//这里对应公式是普通神经网络非输出层的残差计算公式,详解见MATLAB版本各变量说明那篇文章fvd变量的说明
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt用来定位权值,S4层第i个输出MAP第r行第c列与第j个神经元的权值为[j][i*outSize.c*outSize.r + r*outSize.c + c],因为他是二维行优先存储矩阵,第一维代表了他链接的输出层的第j个神经元,第二维代表的是那条边上的权值
					cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnn->S4->mapSize;//这里需要进行上采样操作,因此需要扩充mapSize大小的上采样,这里是2X2
	nSize S4dSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4层的敏感度矩阵大小,这里是4X4,也就是S4层输出矩阵大小
	float C3e[8][8];
	for (i = 0; i<cnn->C3->outChannels; i++) {//C3层每一个输出MAP都对应一个敏感度矩阵
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn->S4->mapSize; j = j + cnn->S4->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S4dSize.c*cnn->S4->mapSize; t = t + cnn->S4->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn->S4->mapSize; m++) {//每次对连续的upc个元素赋值
					C3e[j][t + m] = cnn->S4->d[i][j / cnn->S4->mapSize][t / cnn->S4->mapSize];//填充行
				}
			}
			for (n = 1; n < cnn->S4->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S4dSize.c*cnn->S4->mapSize; t++) {//列方向切换
					C3e[j + n][t] = C3e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn->S4->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是8
			for (c = 0; c<cnn->S4->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是8
				cnn->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->C3->y[i][r][c]) / (float)(cnn->S4->mapSize*cnn->S4->mapSize);//注意这里需要除以(float)(cnn->S4->mapSize*cnn->S4->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C3层的敏感度矩阵
	}
	// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c = cnn->C3->inputWidth;//S2层敏感度矩阵大小为12X12
	outSize.r = cnn->C3->inputHeight;//S2层敏感度矩阵大小为12X12
	nSize inSize = { cnn->S4->inputWidth,cnn->S4->inputHeight };//C3层敏感度矩阵的大小
	nSize mapSize = { cnn->C3->mapSize,cnn->C3->mapSize };//C3层卷积核大小5X5
	float corr[12][12];//存储相关计算结果
	float exData[16][16];//存储full之后的临时变量
	int addr, addc;
	
	addr = addc = mapSize.r - 1;//要扩展的边长
	for (i = 0; i<cnn->S2->outChannels; i++) {//对于S2层每一个输出MAP,6
		for (j = 0; j<cnn->C3->outChannels; j++) {//对于C3层每一个输出MAP,由于这里是全连接结构,因此S2层的每一副图像与C3层的每一副图像都有关,12
												  //float** corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);//这里本来要把C3层对应的卷积核在先旋转180度然后在进行卷积操作,而实际上卷积操作又把卷积核旋转了180度,因此这里直接就不旋转卷积核,而是直接和卷积核相乘,full类型相乘
			int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数,12
			int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData大小,16
												   //扩展矩阵
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//如果是在新扩充的边缘处,设置为0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn->C3->d[j][t - addr][k - addc]; // 不然,复制原向量的数据
				}
			}
			//卷积操作
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							corr[t][k] = corr[t][k] + cnn->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->S2->d[i][t][k] = cnn->S2->d[i][t][k] + corr[t][k];//相加然后返回给res
				}
			}
		}
	}
	// C1层，卷积层
	mapdata = cnn->S2->mapSize;//C1层灵敏度map的大小,24X24
	nSize S2dSize = { cnn->S2->inputWidth / cnn->S2->mapSize,cnn->S2->inputHeight / cnn->S2->mapSize };//S2层灵敏度MAP的大小,12X12里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	float C1e[24][24];
	for (i = 0; i<cnn->C1->outChannels; i++) {//C1层每一个输出MAP都对应一个敏感度矩阵
		for (j = 0; j<S2dSize.r*cnn->S2->mapSize; j = j + cnn->S2->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S2dSize.c*cnn->S2->mapSize; t = t + cnn->S2->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn->S2->mapSize; m++) {//每次对连续的upc个元素赋值
					C1e[j][t + m] = cnn->S2->d[i][j / cnn->S2->mapSize][t / cnn->S2->mapSize];//填充行
				}
			}
			for (n = 1; n < cnn->S2->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S2dSize.c*cnn->S2->mapSize; t++) {//列方向切换
					C1e[j + n][t] = C1e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn->S2->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是24
			for (c = 0; c<cnn->S2->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是24
				cnn->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->C1->y[i][r][c]) / (float)(cnn->S2->mapSize*cnn->S2->mapSize);//注意这里需要除以(float)(cnn->S2->mapSize*cnn->S2->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C1层的敏感度矩阵
	}
}
//更新大小为BATCHSIZE个神经网络的梯度,批训练把梯度更新给第一个CNN
void cnnupdategrad(CNN** cnnarray)
{
	int i, j;
	nSize mapSize = { cnnarray[0]->C1->mapSize,cnnarray[0]->C1->mapSize };//C1层卷积核大小
	for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
		cnnarray[0]->e[i] *= cnnarray[0]->e[i];//均方误差先求平均再求和
	for (int s = 1; s < BATCHSIZE; s++)
	{
		//累加误差
		for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
			cnnarray[0]->e[i] += cnnarray[s]->e[i] * cnnarray[s]->e[i];
		//C1层梯度累加
		for (i = 0; i < cnnarray[0]->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
			for (j = 0; j < cnnarray[0]->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			addmat(cnnarray[0]->C1->dmapData[j][i], cnnarray[0]->C1->dmapData[j][i],mapSize, cnnarray[s]->C1->dmapData[j][i],mapSize);//累加卷积核梯度
			}
		}
		for (int j = 0; j < cnnarray[0]->C1->outChannels; j++) {//对于每一副输出MAP,累加偏置梯度这里是6,大小24X24
			cnnarray[0]->C1->dbasicData[j] += cnnarray[s]->C1->dbasicData[j];
		}
		//C3层梯度累加
		for (i = 0; i < cnnarray[0]->C3->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
			for (j = 0; j < cnnarray[0]->C3->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
				addmat(cnnarray[0]->C3->dmapData[j][i], cnnarray[0]->C3->dmapData[j][i], mapSize, cnnarray[s]->C3->dmapData[j][i], mapSize);//累加卷积核梯度
			}
		}
		for (int j = 0; j < cnnarray[0]->C3->outChannels; j++) {//对于每一副输出MAP,累加偏置梯度这里是6,大小24X24
			cnnarray[0]->C3->dbasicData[j] += cnnarray[s]->C3->dbasicData[j];
		}
		//输出层梯度累加
		for (j = 0; j<cnnarray[0]->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
			for (i = 0; i<cnnarray[0]->O5->inputNum; i++)//对192个输入更新梯度
				cnnarray[0]->O5->dwData[j][i] += cnnarray[s]->O5->dwData[j][i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
			cnnarray[0]->O5->dbasicData[j] += cnnarray[s]->O5->dbasicData[j];//对b更新梯度,b的梯度就是敏感度delta
		}
	}
	//以下求权重平均并更新权重
	for (i = 0; i < cnnarray[0]->O5->outputNum; i++)
		cnnarray[0]->e[i] /= (float)BATCHSIZE;//计算均方误差平均值
	for (i = 0; i < cnnarray[0]->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnnarray[0]->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			multifactor(cnnarray[0]->C1->dmapData[j][i], cnnarray[0]->C1->dmapData[j][i], mapSize, 1.0/BATCHSIZE);//卷积核梯度求平均
			addmat(cnnarray[0]->C1->mapData[j][i], cnnarray[0]->C1->mapData[j][i],mapSize, cnnarray[0]->C1->dmapData[j][i],mapSize);//更新梯度
		}
	}
	for (int j = 0; j < cnnarray[0]->C1->outChannels; j++) {
		cnnarray[0]->C1->dbasicData[j] /= (float)BATCHSIZE;//偏置求平均
		cnnarray[0]->C1->basicData[j] += cnnarray[0]->C1->dbasicData[j];
	}
	//C3层梯度求平均
	for (i = 0; i < cnnarray[0]->C3->outChannels; i++) {//对于每一副输出MAP
		for (j = 0; j < cnnarray[0]->C3->inChannels; j++) {//对于每一副输入图像
			multifactor(cnnarray[0]->C3->dmapData[j][i], cnnarray[0]->C3->dmapData[j][i], mapSize, 1.0 / (float)BATCHSIZE);//卷积核梯度求平均
			addmat(cnnarray[0]->C3->mapData[j][i], cnnarray[0]->C3->mapData[j][i], mapSize, cnnarray[0]->C3->dmapData[j][i], mapSize);//更新梯度
		}
	}
	for (int j = 0; j < cnnarray[0]->C3->outChannels; j++) {
		cnnarray[0]->C3->dbasicData[j] /= (float)BATCHSIZE;
		cnnarray[0]->C3->basicData[j] += cnnarray[0]->C3->dbasicData[j];
	}
	//输出层求平均梯度
	for (j = 0; j<cnnarray[0]->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i < cnnarray[0]->O5->inputNum; i++)//对192个输入更新梯度
		{
			cnnarray[0]->O5->dwData[j][i] /= (float)BATCHSIZE;//求平均
			cnnarray[0]->O5->wData[j][i] += cnnarray[0]->O5->dwData[j][i];//更新梯度
		}
		cnnarray[0]->O5->dbasicData[j] /= (float)BATCHSIZE;//求平均
		cnnarray[0]->O5->basicData[j] += cnnarray[0]->O5->dbasicData[j];//更新梯度
	}
	
}
// 更新权重
void cnnapplygrads(CNN* cnn,CNNOpts opts,float* inputData) 
{

}


void cnnclear(CNN* cnn)
{
	// 将神经元的部分数据清除,主要清楚的是中间保存变量v,每一层的输出y以及敏感误差值d,清空这些值为0.0
	int i,t,k,j,c,r;
	// C1网络
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				cnn->C1->d[j][r][c]=(float)0.0;
				cnn->C1->v[j][r][c]=(float)0.0;
				cnn->C1->y[j][r][c]=(float)0.0;
			}
		}
	}
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnn->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnn->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnn->C1->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnn->C1->mapSize; k++) {//对于输出MAP的每一列
					cnn->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C1->dbasicData[i] = 0.0;
	}
	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				cnn->S2->d[j][r][c]=(float)0.0;
				cnn->S2->y[j][r][c]=(float)0.0;
			}
		}
	}
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnn->C3->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnn->C3->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnn->C3->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnn->C3->mapSize; k++) {//对于输出MAP的每一列
					cnn->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C3->dbasicData[i] = 0.0;
	}
	// C3网络
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				cnn->C3->d[j][r][c]=(float)0.0;
				cnn->C3->v[j][r][c]=(float)0.0;
				cnn->C3->y[j][r][c]=(float)0.0;
			}
		}
	}
	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				cnn->S4->d[j][r][c]=(float)0.0;
				cnn->S4->y[j][r][c]=(float)0.0;
			}
		}
	}
	// O5输出
	for(j=0;j<cnn->O5->outputNum;j++){
		cnn->O5->d[j]=(float)0.0;
		cnn->O5->v[j]=(float)0.0;
		cnn->O5->y[j]=(float)0.0;
	}
	for (j = 0; j<cnn->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i < cnn->O5->inputNum; i++)//对192个输入更新梯度
			cnn->O5->dwData[j][i] = 0.0;
		cnn->O5->dbasicData[j] = 0.0;
	}
}

// 这是用于测试的函数,用来以二进制的方式把训练好的CNN网络的所有数据保存到文件中
void savecnndata(CNN* cnn,const char* filename,float** inputdata) // 保存CNN网络中的相关数据
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	// C1的数据
	int i,j,r;
	// C1网络
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

	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
		for(r=0;r<cnn->C3->inputHeight;r++){
			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
		}
	}
	// C3网络
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

	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
		}
	}

	// O5输出层
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

	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn->S2->inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn->S2->inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
										// 第一层的传播
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
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnn->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnn->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnn->C1->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnn->C1->mapSize; k++) {//对于输出MAP的每一列
					cnn->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C1->dbasicData[i] = 0.0;
	}
	// S2网络
	for (j = 0; j<cnn->S2->outChannels; j++) {
		for (r = 0; r<cnn->C3->inputHeight; r++) {
			for (c = 0; c<cnn->C3->inputWidth; c++) {
				cnn->S2->d[j][r][c] = (float)0.0;
				cnn->S2->y[j][r][c] = (float)0.0;
			}
		}
	}
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnn->C3->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnn->C3->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnn->C3->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnn->C3->mapSize; k++) {//对于输出MAP的每一列
					cnn->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnn->C3->dbasicData[i] = 0.0;
	}
	// C3网络
	for (j = 0; j<cnn->C3->outChannels; j++) {
		for (r = 0; r<cnn->S4->inputHeight; r++) {
			for (c = 0; c<cnn->S4->inputWidth; c++) {
				cnn->C3->d[j][r][c] = (float)0.0;
				cnn->C3->v[j][r][c] = (float)0.0;
				cnn->C3->y[j][r][c] = (float)0.0;
			}
		}
	}
	// S4网络
	for (j = 0; j<cnn->S4->outChannels; j++) {
		for (r = 0; r<cnn->S4->inputHeight / cnn->S4->mapSize; r++) {
			for (c = 0; c<cnn->S4->inputWidth / cnn->S4->mapSize; c++) {
				cnn->S4->d[j][r][c] = (float)0.0;
				cnn->S4->y[j][r][c] = (float)0.0;
			}
		}
	}
	// O5输出
	for (j = 0; j<cnn->O5->outputNum; j++) {
		cnn->O5->d[j] = (float)0.0;
		cnn->O5->v[j] = (float)0.0;
		cnn->O5->y[j] = (float)0.0;
	}
	for (j = 0; j<cnn->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i < cnn->O5->inputNum; i++)//对192个输入更新梯度
			cnn->O5->dwData[j][i] = 0.0;
		cnn->O5->dbasicData[j] = 0.0;
	}

	// 第一层输出数据
	nSize mapSize = { cnn->C1->mapSize,cnn->C1->mapSize };//卷积核大小,5X5
	nSize inSize = { cnn->C1->inputWidth,cnn->C1->inputHeight };//输入图像大小,28X28
	nSize outSize = { cnn->S2->inputWidth,cnn->S2->inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnn->C1->outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnn->C1->inChannels); j++) {//对C1层的每一个输入MAP,这里为1
												   //对卷积核旋转180度
												   //初始化卷积用数组
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->C1->v[i][t][k] = cnn->C1->v[i][t][k] + mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->basicData[i]);
			}
		}
	}

	// 第二层的输出传播S2，采样层
	outSize.c = cnn->C3->inputWidth;//输出图像大小,12X12
	outSize.r = cnn->C3->inputHeight;//输出图像大小,12X12
	inSize.c = cnn->S2->inputWidth;//输入图像大小,24X24
	inSize.r = cnn->S2->inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i<(cnn->S2->outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												//avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);//C1->y[i]经过S2->mapSize大小平均池化后结果输出到S2->y[i]
												//下采样池化
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

	// 第三层输出传播,这里是全连接
	outSize.c = cnn->S4->inputWidth;//输出图像大小,8X8
	outSize.r = cnn->S4->inputHeight;//输出图像大小,8X8
	inSize.c = cnn->C3->inputWidth;//输入图像大小,12X12
	inSize.r = cnn->C3->inputHeight;//输入图像大小,12X12
	mapSize.c = cnn->C3->mapSize;//卷积核大小,5X5
	mapSize.r = cnn->C3->mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnn->C3->outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnn->C3->inChannels); j++) {//对C3层的每一个输入MAP,这里为6
												   //初始化卷积用数组
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->S2->y[j][t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->C3->v[i][t][k] = cnn->C3->v[i][t][k] + mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);//得到C3层最后的输出MAP
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnn->S4->inputWidth;//输入图像大小,8X8
	inSize.r = cnn->S4->inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnn->S4->mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnn->S4->mapSize;//输出图像大小,4X4
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

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->S4->y[i][r][c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
															//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->O5->wData[i][j];//向量相乘之后相加,然后返回结果
		cnn->O5->v[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//最后用sigmoid函数
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);//计算sigmoid函数,即输出层的输出值

																			   //bp
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->e[i] = cnn->O5->y[i] - LabelData[i];//误差是实际输出减去真正正确的输出,对应公式为ai-yi=-(yi-ai),注意这里的y[i]是ai,而yi是outputData[i]
												 // 输出层O5的灵敏度
	for (i = 0; i<cnn->O5->outputNum; i++)
		cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);//对10个神经元来说,每个神经元的输出层的灵敏度公式为-(yi-ai)(ai*(1-ai)),注意这里的y[i]是ai,而yi是outputData[i]
																	// S4层，传递到S4层的误差
																	// 这里没有激活函数
	outSize.r = cnn->S4->inputWidth / cnn->S4->mapSize;
	outSize.c = cnn->S4->inputHeight / cnn->S4->mapSize;//S4层的输出矩阵大小,这里是4X4
	for (i = 0; i < cnn->S4->outChannels; i++) {//对每一个输出矩阵,都有一个和输出矩阵一样大小的敏感度矩阵与之对应
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn->O5->outputNum; j++) {//这里对应公式是普通神经网络非输出层的残差计算公式,详解见MATLAB版本各变量说明那篇文章fvd变量的说明
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt用来定位权值,S4层第i个输出MAP第r行第c列与第j个神经元的权值为[j][i*outSize.c*outSize.r + r*outSize.c + c],因为他是二维行优先存储矩阵,第一维代表了他链接的输出层的第j个神经元,第二维代表的是那条边上的权值
					cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnn->S4->mapSize;//这里需要进行上采样操作,因此需要扩充mapSize大小的上采样,这里是2X2
	nSize S4dSize = { cnn->S4->inputWidth / cnn->S4->mapSize,cnn->S4->inputHeight / cnn->S4->mapSize };//S4层的敏感度矩阵大小,这里是4X4,也就是S4层输出矩阵大小
	float C3e[8][8];
	for (i = 0; i<cnn->C3->outChannels; i++) {//C3层每一个输出MAP都对应一个敏感度矩阵
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn->S4->mapSize; j = j + cnn->S4->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S4dSize.c*cnn->S4->mapSize; t = t + cnn->S4->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn->S4->mapSize; m++) {//每次对连续的upc个元素赋值
					C3e[j][t + m] = cnn->S4->d[i][j / cnn->S4->mapSize][t / cnn->S4->mapSize];//填充行
				}
			}
			for (n = 1; n < cnn->S4->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S4dSize.c*cnn->S4->mapSize; t++) {//列方向切换
					C3e[j + n][t] = C3e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn->S4->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是8
			for (c = 0; c<cnn->S4->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是8
				cnn->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->C3->y[i][r][c]) / (float)(cnn->S4->mapSize*cnn->S4->mapSize);//注意这里需要除以(float)(cnn->S4->mapSize*cnn->S4->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C3层的敏感度矩阵
	}
	// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c = cnn->C3->inputWidth;//S2层敏感度矩阵大小为12X12
	outSize.r = cnn->C3->inputHeight;//S2层敏感度矩阵大小为12X12
	inSize.r = cnn->S4->inputWidth;
	inSize.c = cnn->S4->inputHeight;//C3层敏感度矩阵的大小
	mapSize.r = cnn->C3->mapSize;
	mapSize.c = cnn->C3->mapSize;//C3层卷积核大小5X5
	float corr[12][12];//存储相关计算结果
	float exData[16][16];//存储full之后的临时变量
	int addr, addc;

	addr = addc = mapSize.r - 1;//要扩展的边长
	for (i = 0; i<cnn->S2->outChannels; i++) {//对于S2层每一个输出MAP,6
		for (j = 0; j<cnn->C3->outChannels; j++) {//对于C3层每一个输出MAP,由于这里是全连接结构,因此S2层的每一副图像与C3层的每一副图像都有关,12
												  //float** corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);//这里本来要把C3层对应的卷积核在先旋转180度然后在进行卷积操作,而实际上卷积操作又把卷积核旋转了180度,因此这里直接就不旋转卷积核,而是直接和卷积核相乘,full类型相乘
			int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数,12
			int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData大小,16
												   //扩展矩阵
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//如果是在新扩充的边缘处,设置为0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn->C3->d[j][t - addr][k - addc]; // 不然,复制原向量的数据
				}
			}
			//卷积操作
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							corr[t][k] = corr[t][k] + cnn->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->S2->d[i][t][k] = cnn->S2->d[i][t][k] + corr[t][k];//相加然后返回给res
				}
			}
		}
	}
	// C1层，卷积层
	mapdata = cnn->S2->mapSize;//C1层灵敏度map的大小,24X24
	nSize S2dSize = { cnn->S2->inputWidth / cnn->S2->mapSize,cnn->S2->inputHeight / cnn->S2->mapSize };//S2层灵敏度MAP的大小,12X12里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	float C1e[24][24];
	for (i = 0; i<cnn->C1->outChannels; i++) {//C1层每一个输出MAP都对应一个敏感度矩阵
		for (j = 0; j<S2dSize.r*cnn->S2->mapSize; j = j + cnn->S2->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S2dSize.c*cnn->S2->mapSize; t = t + cnn->S2->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn->S2->mapSize; m++) {//每次对连续的upc个元素赋值
					C1e[j][t + m] = cnn->S2->d[i][j / cnn->S2->mapSize][t / cnn->S2->mapSize];//填充行
				}
			}
			for (n = 1; n < cnn->S2->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S2dSize.c*cnn->S2->mapSize; t++) {//列方向切换
					C1e[j + n][t] = C1e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn->S2->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是24
			for (c = 0; c<cnn->S2->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是24
				cnn->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->C1->y[i][r][c]) / (float)(cnn->S2->mapSize*cnn->S2->mapSize);//注意这里需要除以(float)(cnn->S2->mapSize*cnn->S2->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C1层的敏感度矩阵
	}

	//apply
	// C1层的权重更新
	nSize dSize = { cnn->S2->inputHeight,cnn->S2->inputWidth };//C1层灵敏度矩阵大小,24X24
	nSize ySize = { cnn->C1->inputHeight,cnn->C1->inputWidth };//C1层输入矩阵大小,28X28
	mapSize.r = cnn->C1->mapSize;
	mapSize.c = cnn->C1->mapSize;//C1层卷积核大小
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnn->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j<cnn->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
												 //首先,一维转二维计算,旋转180度似乎不对
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = inputData[r*ySize.c + c];
				}
			}
			//这里之所以会出错,是数组交换最简单的问题,a=b,b=a不能直接写,要用C做中转!!!!
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
																	   //cout << tin[r][c] << " ";
				}
				//cout << endl;
			}
			//system("pause");
			//旋转卷积核
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov[r][c] = cnn->C1->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
				}
			}

			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnn->C1->dmapData[j][i][t][k] = cnn->C1->dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
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
		cnn->C1->dbasicData[i] = -1 * opts.alpha*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// C3层的权重更新
	dSize.c = cnn->S4->inputWidth;//C3层灵敏度矩阵大小,8X8
	dSize.r = cnn->S4->inputHeight;//C3层灵敏度矩阵大小,8X8
	ySize.c = cnn->C3->inputWidth;//C3层输入矩阵大小,12X12
	ySize.r = cnn->C3->inputHeight;//C3层输入矩阵大小,12X12
	mapSize.c = cnn->C3->mapSize;//C3层卷积核大小,5X5
	mapSize.r = cnn->C3->mapSize;//C3层卷积核大小,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnn->C3->outChannels; i++) {//对于每一副输出MAP,这里是12,大小8X8
		for (j = 0; j<cnn->C3->inChannels; j++) {//对于每一副输入图像,这里是8,大小12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnn->S2->y[j][ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//旋转卷积核
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnn->C3->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnn->C3->dmapData[j][i][t][k] = cnn->C3->dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
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
		cnn->C3->dbasicData[i] = -1 * opts.alpha*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// 输出层
	// 首先需要将前面的多维输出展开成一维向量
	for (j = 0; j<cnn->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i<cnn->O5->inputNum; i++)//对192个输入更新梯度
			cnn->O5->dwData[j][i] = -1 * opts.alpha*cnn->O5->d[j] * O5inData[i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
		cnn->O5->dbasicData[j] = -1 * opts.alpha*cnn->O5->d[j];//对b更新梯度,b的梯度就是敏感度delta
	}
}