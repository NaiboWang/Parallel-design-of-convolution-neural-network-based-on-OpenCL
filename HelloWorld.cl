//首先定义结构体
typedef struct Mat2DSize {//定义矩阵大小的结构体,c和r表示列数和行数
	int c; // 列数（宽度）
	int r; // 行数（高度）
}nSize;
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
	global float**** mapData;     //存放特征模块的数据
	global float**** dmapData;    //存放特征模块的数据的局部梯度

	global float* basicData;   //偏置，偏置的大小，为outChannels
	global float* dbasicData;   //偏置的梯度，偏置的大小，为outChannels
	bool isFullConnect; //是否为全连接
	global bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	global float*** v; // 进入激活函数的输入值
	global float*** y; // 激活函数后神经元的输出

	// 输出像素的局部梯度
	global float*** d; // 网络的局部梯度,δ值  
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	global float* basicData;   //偏置,实际上没有用到

	global float*** y; // 采样函数后神经元的输出,无激活函数
	global float*** d; // 网络的局部梯度,δ值
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct nn_layer{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	global float** wData; // 权重数据，为一个inputNum*outputNum大小
	global float* basicData;   //偏置，大小为outputNum大小

	global float** dwData; // 权重数据梯度，为一个inputNum*outputNum大小
	global float* dbasicData;   //偏置梯度，大小为outputNum大小
	// 下面三者的大小同输出的维度相同
	global float* v; // 进入激活函数的输入值
	global float* y; // 激活函数后神经元的输出
	global float* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network{//整个CNN的最外面那一层,包括五个层和误差,层的数目
	int layerNum;//层数目
	global CovLayer* C1;
	global PoolLayer* S2;
	global CovLayer* C3;
	global PoolLayer* S4;
	global OutLayer* O5;
	global float* e; // 训练误差
	global float* L; // 瞬时误差能量
}CNN;
typedef struct cnn_arr
{
	global CNN** cnn;
}CNNA;
typedef struct train_opts {
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;
typedef struct MinstImg{
	int c;           // 图像宽,这里是28
	int r;           // 图像高,这里是28
	global float* ImgData; // 图像数据二维动态数组,28X28的一维原始图像
}MinstImg;
//60000张原始图像
typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目,这里训练集是60000,测试集为10000
	global MinstImg* ImgPtr;  // 存储图像指针数组,每一张就是上面的28X28的结构体
}ImgArr;              // 存储图像数据的数组,注意是指针类型
//用来存标签的结构体
typedef struct MinstLabel{
	int l;            // 输出标记的长,这里是10
	global float* LabelData; // 输出标记数据,这里是10个元素,分别代表0到9,初始化的时候全部为0,这个图像对应的数字几就让相应位置的值为1.0
}MinstLabel;
//60000个标签
typedef struct MinstLabelArr{
	int LabelNum;//存储标签数目,这里训练集是60000,测试集为10000
	global MinstLabel* LabelPtr;// 存储标签指针数组,每一张就是上面1个标签结构体
}LabelArr;              // 存储图像标记的数组

float activation_Sigma(float input,float bas) // sigma激活函数
{
	float temp=input+bas;
	return (float)1.0/((float)(1.0+exp(-temp)));
}
float sigma_derivation(float y) { // Logic激活函数的自变量微分,即sigmoid函数的导数
	return y*(1 - y); // 这里y是指经过激活函数的输出值，而不是自变量
}
//opencl训练单网络优化实现
kernel void traincnn(global CNNA* cnns,global ImgArr* IData,global LabelArr* LData,global CNNOpts* opts,global int *bs){
    int x = get_global_id(0);
	int py = *bs + x;
	//if(py>58999)
	//printf("%d %d\n",py,*bs);
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnns->cnn[x]->S2->inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnns->cnn[x]->S2->inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
										// 第一层的传播
	int i, j, r, c, t, k, m, n;
	//clear
	for (j = 0; j<cnns->cnn[x]->C1->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->S2->inputHeight; r++) {
			for (c = 0; c<cnns->cnn[x]->S2->inputWidth; c++) {
				cnns->cnn[x]->C1->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->C1->v[j][r][c] = (float)0.0;
				cnns->cnn[x]->C1->y[j][r][c] = (float)0.0;
			}
		}
	}
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnns->cnn[x]->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnns->cnn[x]->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnns->cnn[x]->C1->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnns->cnn[x]->C1->mapSize; k++) {//对于输出MAP的每一列
					cnns->cnn[x]->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnns->cnn[x]->C1->dbasicData[i] = 0.0;
	}
	// S2网络
	for (j = 0; j<cnns->cnn[x]->S2->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->C3->inputHeight; r++) {
			for (c = 0; c<cnns->cnn[x]->C3->inputWidth; c++) {
				cnns->cnn[x]->S2->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->S2->y[j][r][c] = (float)0.0;
			}
		}
	}
	//先清空原来dmapData的值,不让他累加,类似于cnnclear对v的操作一样!!!!
	for (i = 0; i < cnns->cnn[x]->C3->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j < cnns->cnn[x]->C3->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
			for (t = 0; t < cnns->cnn[x]->C3->mapSize; t++) {//对于输出MAP的每一行
				for (k = 0; k < cnns->cnn[x]->C3->mapSize; k++) {//对于输出MAP的每一列
					cnns->cnn[x]->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnns->cnn[x]->C3->dbasicData[i] = 0.0;
	}
	// C3网络
	for (j = 0; j<cnns->cnn[x]->C3->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight; r++) {
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth; c++) {
				cnns->cnn[x]->C3->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->C3->v[j][r][c] = (float)0.0;
				cnns->cnn[x]->C3->y[j][r][c] = (float)0.0;
			}
		}
	}
	// S4网络
	for (j = 0; j<cnns->cnn[x]->S4->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize; r++) {
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize; c++) {
				cnns->cnn[x]->S4->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->S4->y[j][r][c] = (float)0.0;
			}
		}
	}
	// O5输出
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {
		cnns->cnn[x]->O5->d[j] = (float)0.0;
		cnns->cnn[x]->O5->v[j] = (float)0.0;
		cnns->cnn[x]->O5->y[j] = (float)0.0;
	}
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i < cnns->cnn[x]->O5->inputNum; i++)//对192个输入更新梯度
			cnns->cnn[x]->O5->dwData[j][i] = 0.0;
		cnns->cnn[x]->O5->dbasicData[j] = 0.0;
	}

	// 第一层输出数据
	nSize mapSize = { cnns->cnn[x]->C1->mapSize,cnns->cnn[x]->C1->mapSize };//卷积核大小,5X5
	nSize inSize = { cnns->cnn[x]->C1->inputWidth,cnns->cnn[x]->C1->inputHeight };//输入图像大小,28X28
	nSize outSize = { cnns->cnn[x]->S2->inputWidth,cnns->cnn[x]->S2->inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnns->cnn[x]->C1->outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnns->cnn[x]->C1->inChannels); j++) {//对C1层的每一个输入MAP,这里为1
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
					tempconv[r][c] = cnns->cnn[x]->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * IData->ImgPtr[py].ImgData[(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnns->cnn[x]->C1->v[i][t][k] = cnns->cnn[x]->C1->v[i][t][k] + mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnns->cnn[x]->C1->y[i][r][c] = activation_Sigma(cnns->cnn[x]->C1->v[i][r][c], cnns->cnn[x]->C1->basicData[i]);
			}
		}
	}

	// 第二层的输出传播S2，采样层
	outSize.c = cnns->cnn[x]->C3->inputWidth;//输出图像大小,12X12
	outSize.r = cnns->cnn[x]->C3->inputHeight;//输出图像大小,12X12
	inSize.c = cnns->cnn[x]->S2->inputWidth;//输入图像大小,24X24
	inSize.r = cnns->cnn[x]->S2->inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i<(cnns->cnn[x]->S2->outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												//avgPooling(cnns->cnn[x]->S2->y[i], outSize, cnns->cnn[x]->C1->y[i], inSize, cnns->cnn[x]->S2->mapSize);//C1->y[i]经过S2->mapSize大小平均池化后结果输出到S2->y[i]
												//下采样池化
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnns->cnn[x]->C1->y[i][m][n];
					}
				}
				cnns->cnn[x]->S2->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// 第三层输出传播,这里是全连接
	outSize.c = cnns->cnn[x]->S4->inputWidth;//输出图像大小,8X8
	outSize.r = cnns->cnn[x]->S4->inputHeight;//输出图像大小,8X8
	inSize.c = cnns->cnn[x]->C3->inputWidth;//输入图像大小,12X12
	inSize.r = cnns->cnn[x]->C3->inputHeight;//输入图像大小,12X12
	mapSize.c = cnns->cnn[x]->C3->mapSize;//卷积核大小,5X5
	mapSize.r = cnns->cnn[x]->C3->mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnns->cnn[x]->C3->outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnns->cnn[x]->C3->inChannels); j++) {//对C3层的每一个输入MAP,这里为6
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
					tempconv[r][c] = cnns->cnn[x]->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnns->cnn[x]->S2->y[j][t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnns->cnn[x]->C3->v[i][t][k] = cnns->cnn[x]->C3->v[i][t][k] + mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnns->cnn[x]->C3->y[i][r][c] = activation_Sigma(cnns->cnn[x]->C3->v[i][r][c], cnns->cnn[x]->C3->basicData[i]);//得到C3层最后的输出MAP
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnns->cnn[x]->S4->inputWidth;//输入图像大小,8X8
	inSize.r = cnns->cnn[x]->S4->inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnns->cnn[x]->S4->mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnns->cnn[x]->S4->mapSize;//输出图像大小,4X4
	for (i = 0; i<(cnns->cnn[x]->S4->outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnns->cnn[x]->C3->y[i][m][n];
					}
				}
				cnns->cnn[x]->S4->y[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnns->cnn[x]->S4->outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnns->cnn[x]->S4->y[i][r][c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
			}
		}
	}
	nSize nnSize = { cnns->cnn[x]->O5->inputNum,cnns->cnn[x]->O5->outputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
															//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnns->cnn[x]->O5->wData[i][j];//向量相乘之后相加,然后返回结果
		cnns->cnn[x]->O5->v[i] = o;
	}
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)//最后用sigmoid函数
		cnns->cnn[x]->O5->y[i] = activation_Sigma(cnns->cnn[x]->O5->v[i], cnns->cnn[x]->O5->basicData[i]);//计算sigmoid函数,即输出层的输出值
	
	//bp
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)
		cnns->cnn[x]->e[i] = cnns->cnn[x]->O5->y[i] - LData->LabelPtr[py].LabelData[i];//误差是实际输出减去真正正确的输出,对应公式为ai-yi=-(yi-ai),注意这里的y[i]是ai,而yi是outputData[i]
												  // 输出层O5的灵敏度
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)
		cnns->cnn[x]->O5->d[i] = cnns->cnn[x]->e[i] * sigma_derivation(cnns->cnn[x]->O5->y[i]);//对10个神经元来说,每个神经元的输出层的灵敏度公式为-(yi-ai)(ai*(1-ai)),注意这里的y[i]是ai,而yi是outputData[i]
																	// S4层，传递到S4层的误差
																	// 这里没有激活函数
	outSize.r = cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize;
	outSize.c= cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize;//S4层的输出矩阵大小,这里是4X4
	for (i = 0; i < cnns->cnn[x]->S4->outChannels; i++) {//对每一个输出矩阵,都有一个和输出矩阵一样大小的敏感度矩阵与之对应
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnns->cnn[x]->O5->outputNum; j++) {//这里对应公式是普通神经网络非输出层的残差计算公式,详解见MATLAB版本各变量说明那篇文章fvd变量的说明
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt用来定位权值,S4层第i个输出MAP第r行第c列与第j个神经元的权值为[j][i*outSize.c*outSize.r + r*outSize.c + c],因为他是二维行优先存储矩阵,第一维代表了他链接的输出层的第j个神经元,第二维代表的是那条边上的权值
					cnns->cnn[x]->S4->d[i][r][c] = cnns->cnn[x]->S4->d[i][r][c] + cnns->cnn[x]->O5->d[j] * cnns->cnn[x]->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnns->cnn[x]->S4->mapSize;//这里需要进行上采样操作,因此需要扩充mapSize大小的上采样,这里是2X2
	nSize S4dSize = { cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize,cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize };//S4层的敏感度矩阵大小,这里是4X4,也就是S4层输出矩阵大小
	float C3e[8][8];
	for (i = 0; i<cnns->cnn[x]->C3->outChannels; i++) {//C3层每一个输出MAP都对应一个敏感度矩阵
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnns->cnn[x]->S4->mapSize; j = j + cnns->cnn[x]->S4->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S4dSize.c*cnns->cnn[x]->S4->mapSize; t = t + cnns->cnn[x]->S4->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnns->cnn[x]->S4->mapSize; m++) {//每次对连续的upc个元素赋值
					C3e[j][t + m] = cnns->cnn[x]->S4->d[i][j / cnns->cnn[x]->S4->mapSize][t / cnns->cnn[x]->S4->mapSize];//填充行
				}
			}
			for (n = 1; n < cnns->cnn[x]->S4->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S4dSize.c*cnns->cnn[x]->S4->mapSize; t++) {//列方向切换
					C3e[j + n][t] = C3e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是8
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是8
				cnns->cnn[x]->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnns->cnn[x]->C3->y[i][r][c]) / (float)(cnns->cnn[x]->S4->mapSize*cnns->cnn[x]->S4->mapSize);//注意这里需要除以(float)(cnns->cnn[x]->S4->mapSize*cnns->cnn[x]->S4->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C3层的敏感度矩阵
	}
	// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c = cnns->cnn[x]->C3->inputWidth;//S2层敏感度矩阵大小为12X12
	outSize.r = cnns->cnn[x]->C3->inputHeight;//S2层敏感度矩阵大小为12X12
	inSize.r = cnns->cnn[x]->S4->inputWidth;
	inSize.c = cnns->cnn[x]->S4->inputHeight;//C3层敏感度矩阵的大小
	mapSize.r = cnns->cnn[x]->C3->mapSize;
	mapSize.c = cnns->cnn[x]->C3->mapSize;//C3层卷积核大小5X5
	float corr[12][12];//存储相关计算结果
	float exData[16][16];//存储full之后的临时变量
	int addr, addc;

	addr = addc = mapSize.r - 1;//要扩展的边长
	for (i = 0; i<cnns->cnn[x]->S2->outChannels; i++) {//对于S2层每一个输出MAP,6
		for (j = 0; j<cnns->cnn[x]->C3->outChannels; j++) {//对于C3层每一个输出MAP,由于这里是全连接结构,因此S2层的每一副图像与C3层的每一副图像都有关,12
												  //float** corr = correlation(cnns->cnn[x]->C3->mapData[i][j], mapSize, cnns->cnn[x]->C3->d[j], inSize, full);//这里本来要把C3层对应的卷积核在先旋转180度然后在进行卷积操作,而实际上卷积操作又把卷积核旋转了180度,因此这里直接就不旋转卷积核,而是直接和卷积核相乘,full类型相乘
			int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数,12
			int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData大小,16
												   //扩展矩阵
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//如果是在新扩充的边缘处,设置为0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnns->cnn[x]->C3->d[j][t - addr][k - addc]; // 不然,复制原向量的数据
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
							corr[t][k] = corr[t][k] + cnns->cnn[x]->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnns->cnn[x]->S2->d[i][t][k] = cnns->cnn[x]->S2->d[i][t][k] + corr[t][k];//相加然后返回给res
				}
			}
		}
	}
	// C1层，卷积层
	mapdata = cnns->cnn[x]->S2->mapSize;//C1层灵敏度map的大小,24X24
	nSize S2dSize = { cnns->cnn[x]->S2->inputWidth / cnns->cnn[x]->S2->mapSize,cnns->cnn[x]->S2->inputHeight / cnns->cnn[x]->S2->mapSize };//S2层灵敏度MAP的大小,12X12里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	float C1e[24][24];
	for (i = 0; i<cnns->cnn[x]->C1->outChannels; i++) {//C1层每一个输出MAP都对应一个敏感度矩阵
		for (j = 0; j<S2dSize.r*cnns->cnn[x]->S2->mapSize; j = j + cnns->cnn[x]->S2->mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S2dSize.c*cnns->cnn[x]->S2->mapSize; t = t + cnns->cnn[x]->S2->mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnns->cnn[x]->S2->mapSize; m++) {//每次对连续的upc个元素赋值
					C1e[j][t + m] = cnns->cnn[x]->S2->d[i][j / cnns->cnn[x]->S2->mapSize][t / cnns->cnn[x]->S2->mapSize];//填充行
				}
			}
			for (n = 1; n < cnns->cnn[x]->S2->mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S2dSize.c*cnns->cnn[x]->S2->mapSize; t++) {//列方向切换
					C1e[j + n][t] = C1e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnns->cnn[x]->S2->inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是24
			for (c = 0; c<cnns->cnn[x]->S2->inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是24
				cnns->cnn[x]->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnns->cnn[x]->C1->y[i][r][c]) / (float)(cnns->cnn[x]->S2->mapSize*cnns->cnn[x]->S2->mapSize);//注意这里需要除以(float)(cnns->cnn[x]->S2->mapSize*cnns->cnn[x]->S2->mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C1层的敏感度矩阵
	}

	//apply
	// C1层的权重更新
	nSize dSize = { cnns->cnn[x]->S2->inputHeight,cnns->cnn[x]->S2->inputWidth };//C1层灵敏度矩阵大小,24X24
	nSize ySize = { cnns->cnn[x]->C1->inputHeight,cnns->cnn[x]->C1->inputWidth };//C1层输入矩阵大小,28X28
	mapSize.r = cnns->cnn[x]->C1->mapSize;
	mapSize.c = cnns->cnn[x]->C1->mapSize;//C1层卷积核大小
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnns->cnn[x]->C1->outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j<cnns->cnn[x]->C1->inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
												 //首先,一维转二维计算,旋转180度似乎不对
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = IData->ImgPtr[py].ImgData[r*ySize.c + c];
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
					cov[r][c] = cnns->cnn[x]->C1->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
				}
			}

			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnns->cnn[x]->C1->dmapData[j][i][t][k] = cnns->cnn[x]->C1->dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnns->cnn[x]->C1->dmapData[j][i][t][k] = cnns->cnn[x]->C1->dmapData[j][i][t][k] * -1 * opts->alpha;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnns->cnn[x]->C1->d[i][t][j];
		cnns->cnn[x]->C1->dbasicData[i] = -1 * opts->alpha*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// C3层的权重更新
	dSize.c = cnns->cnn[x]->S4->inputWidth;//C3层灵敏度矩阵大小,8X8
	dSize.r = cnns->cnn[x]->S4->inputHeight;//C3层灵敏度矩阵大小,8X8
	ySize.c = cnns->cnn[x]->C3->inputWidth;//C3层输入矩阵大小,12X12
	ySize.r = cnns->cnn[x]->C3->inputHeight;//C3层输入矩阵大小,12X12
	mapSize.c = cnns->cnn[x]->C3->mapSize;//C3层卷积核大小,5X5
	mapSize.r = cnns->cnn[x]->C3->mapSize;//C3层卷积核大小,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnns->cnn[x]->C3->outChannels; i++) {//对于每一副输出MAP,这里是12,大小8X8
		for (j = 0; j<cnns->cnn[x]->C3->inChannels; j++) {//对于每一副输入图像,这里是8,大小12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnns->cnn[x]->S2->y[j][ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//旋转卷积核
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnns->cnn[x]->C3->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnns->cnn[x]->C3->dmapData[j][i][t][k] = cnns->cnn[x]->C3->dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnns->cnn[x]->C3->dmapData[j][i][t][k] = cnns->cnn[x]->C3->dmapData[j][i][t][k] * -1 * opts->alpha;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnns->cnn[x]->C3->d[i][t][j];
		cnns->cnn[x]->C3->dbasicData[i] = -1 * opts->alpha*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// 输出层
	// 首先需要将前面的多维输出展开成一维向量
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i<cnns->cnn[x]->O5->inputNum; i++)//对192个输入更新梯度
			cnns->cnn[x]->O5->dwData[j][i] = -1 * opts->alpha*cnns->cnn[x]->O5->d[j] * O5inData[i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
		cnns->cnn[x]->O5->dbasicData[j] = -1 * opts->alpha*cnns->cnn[x]->O5->d[j];//对b更新梯度,b的梯度就是敏感度delta
	}
}



//opencl卷积测试优化实现
kernel void cnntest(global CNN* cnn,global ImgArr* IData,global LabelArr* LData,global atomic_int *wrongnum){
	int x = get_global_id(0);
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn->S2->inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn->S2->inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
	// 第一层的传播
	int i, j, r, c, t, k,m,n;
	float v1[6][24][24];
	float y1[6][24][24];
	float y2[6][12][12];
	float v3[12][8][8];
	float y3[12][8][8];
	float y4[12][4][4];
	float v5[10];
	float y5[10];
	//清空网络原有参数
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				v1[j][r][c]=(float)0.0;
				y1[j][r][c]=(float)0.0;
			}
		}
	}
	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				y2[j][r][c]=(float)0.0;
			}
		}
	}
	// C3网络
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				v3[j][r][c]=(float)0.0;
				y3[j][r][c]=(float)0.0;
			}
		}
	}
	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				y4[j][r][c]=(float)0.0;
			}
		}
	}
	// O5输出
	for(j=0;j<cnn->O5->outputNum;j++){
		v5[j]=(float)0.0;
		y5[j]=(float)0.0;
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
			for (r = 0; r<mapSize.r; r++){
				for (c = 0; c<mapSize.c; c++){
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r -1 - r][mapSize.c -1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * IData->ImgPtr[x].ImgData[(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++){
				for (k = 0; k<outSize.c; k++){
					v1[i][t][k] = v1[i][t][k] + mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				y1[i][r][c] = activation_Sigma(v1[i][r][c], cnn->C1->basicData[i]);
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
			//avgPooling(y2[i], outSize, y1[i], inSize, cnn->S2->mapSize);//C1->y[i]经过S2->mapSize大小平均池化后结果输出到S2->y[i]
		//下采样池化
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + y1[i][m][n];
					}
				}
				y2[i][t][j] = sum / (float)(mSize*mSize);
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
			//计算卷积s
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * y2[j][t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					v3[i][t][k] = v3[i][t][k] + mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				y3[i][r][c] = activation_Sigma(v3[i][r][c], cnn->C3->basicData[i]);//得到C3层最后的输出MAP
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
						sum = sum + y3[i][m][n];
					}
				}
				y4[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = y4[i][r][c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
	//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j]* cnn->O5->wData[i][j];//向量相乘之后相加,然后返回结果
		v5[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//最后用sigmoid函数
		y5[i] = activation_Sigma(v5[i], cnn->O5->basicData[i]);//计算sigmoid函数,即输出层的输出值

	float maxnum = -1.0;
	int maxIndex = 0;
	for (i = 0; i<cnn->O5->outputNum; i++) {
		if (maxnum<y5[i]) {
			maxnum = y5[i];
			maxIndex = i;
		}
	}
	maxnum = -1.0;
	int maxIndex2 = 0;
	for (i = 0; i<cnn->O5->outputNum; i++) {
		if (maxnum<LData->LabelPtr[x].LabelData[i]) {
			maxnum = LData->LabelPtr[x].LabelData[i];
			maxIndex2 = i;
		}
	}
	if(maxIndex != maxIndex2)
		atomic_fetch_add(wrongnum,1);//原子操作算是否相同
}
