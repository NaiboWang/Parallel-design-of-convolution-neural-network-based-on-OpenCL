//���ȶ���ṹ��
typedef struct Mat2DSize {//��������С�Ľṹ��,c��r��ʾ����������
	int c; // ��������ȣ�
	int r; // �������߶ȣ�
}nSize;
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
	global float**** mapData;     //�������ģ�������
	global float**** dmapData;    //�������ģ������ݵľֲ��ݶ�

	global float* basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	global float* dbasicData;   //ƫ�õ��ݶȣ�ƫ�õĴ�С��ΪoutChannels
	bool isFullConnect; //�Ƿ�Ϊȫ����
	global bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	// �������ߵĴ�Сͬ�����ά����ͬ
	global float*** v; // ���뼤���������ֵ
	global float*** y; // ���������Ԫ�����

	// ������صľֲ��ݶ�
	global float*** d; // ����ľֲ��ݶ�,��ֵ  
}CovLayer;

// ������ pooling
typedef struct pooling_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int poolType;     //Pooling�ķ���
	global float* basicData;   //ƫ��,ʵ����û���õ�

	global float*** y; // ������������Ԫ�����,�޼����
	global float*** d; // ����ľֲ��ݶ�,��ֵ
}PoolLayer;

// ����� ȫ���ӵ�������
typedef struct nn_layer{
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	global float** wData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	global float* basicData;   //ƫ�ã���СΪoutputNum��С

	global float** dwData; // Ȩ�������ݶȣ�Ϊһ��inputNum*outputNum��С
	global float* dbasicData;   //ƫ���ݶȣ���СΪoutputNum��С
	// �������ߵĴ�Сͬ�����ά����ͬ
	global float* v; // ���뼤���������ֵ
	global float* y; // ���������Ԫ�����
	global float* d; // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
}OutLayer;

typedef struct cnn_network{//����CNN����������һ��,�������������,�����Ŀ
	int layerNum;//����Ŀ
	global CovLayer* C1;
	global PoolLayer* S2;
	global CovLayer* C3;
	global PoolLayer* S4;
	global OutLayer* O5;
	global float* e; // ѵ�����
	global float* L; // ˲ʱ�������
}CNN;
typedef struct cnn_arr
{
	global CNN** cnn;
}CNNA;
typedef struct train_opts {
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;
typedef struct MinstImg{
	int c;           // ͼ���,������28
	int r;           // ͼ���,������28
	global float* ImgData; // ͼ�����ݶ�ά��̬����,28X28��һάԭʼͼ��
}MinstImg;
//60000��ԭʼͼ��
typedef struct MinstImgArr{
	int ImgNum;        // �洢ͼ�����Ŀ,����ѵ������60000,���Լ�Ϊ10000
	global MinstImg* ImgPtr;  // �洢ͼ��ָ������,ÿһ�ž��������28X28�Ľṹ��
}ImgArr;              // �洢ͼ�����ݵ�����,ע����ָ������
//�������ǩ�Ľṹ��
typedef struct MinstLabel{
	int l;            // �����ǵĳ�,������10
	global float* LabelData; // ����������,������10��Ԫ��,�ֱ����0��9,��ʼ����ʱ��ȫ��Ϊ0,���ͼ���Ӧ�����ּ�������Ӧλ�õ�ֵΪ1.0
}MinstLabel;
//60000����ǩ
typedef struct MinstLabelArr{
	int LabelNum;//�洢��ǩ��Ŀ,����ѵ������60000,���Լ�Ϊ10000
	global MinstLabel* LabelPtr;// �洢��ǩָ������,ÿһ�ž�������1����ǩ�ṹ��
}LabelArr;              // �洢ͼ���ǵ�����

float activation_Sigma(float input,float bas) // sigma�����
{
	float temp=input+bas;
	return (float)1.0/((float)(1.0+exp(-temp)));
}
float sigma_derivation(float y) { // Logic��������Ա���΢��,��sigmoid�����ĵ���
	return y*(1 - y); // ����y��ָ��������������ֵ���������Ա���
}
//openclѵ���������Ż�ʵ��
kernel void traincnn(global CNNA* cnns,global ImgArr* IData,global LabelArr* LData,global CNNOpts* opts,global int *bs){
    int x = get_global_id(0);
	int py = *bs + x;
	//if(py>58999)
	//printf("%d %d\n",py,*bs);
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnns->cnn[x]->S2->inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnns->cnn[x]->S2->inputHeight;//�����һ������MAP����Ĵ�С,������24X24
										// ��һ��Ĵ���
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
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnns->cnn[x]->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnns->cnn[x]->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnns->cnn[x]->C1->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnns->cnn[x]->C1->mapSize; k++) {//�������MAP��ÿһ��
					cnns->cnn[x]->C1->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnns->cnn[x]->C1->dbasicData[i] = 0.0;
	}
	// S2����
	for (j = 0; j<cnns->cnn[x]->S2->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->C3->inputHeight; r++) {
			for (c = 0; c<cnns->cnn[x]->C3->inputWidth; c++) {
				cnns->cnn[x]->S2->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->S2->y[j][r][c] = (float)0.0;
			}
		}
	}
	//�����ԭ��dmapData��ֵ,�������ۼ�,������cnnclear��v�Ĳ���һ��!!!!
	for (i = 0; i < cnns->cnn[x]->C3->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j < cnns->cnn[x]->C3->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
			for (t = 0; t < cnns->cnn[x]->C3->mapSize; t++) {//�������MAP��ÿһ��
				for (k = 0; k < cnns->cnn[x]->C3->mapSize; k++) {//�������MAP��ÿһ��
					cnns->cnn[x]->C3->dmapData[j][i][t][k] = 0.0;
				}
			}
		}
		cnns->cnn[x]->C3->dbasicData[i] = 0.0;
	}
	// C3����
	for (j = 0; j<cnns->cnn[x]->C3->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight; r++) {
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth; c++) {
				cnns->cnn[x]->C3->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->C3->v[j][r][c] = (float)0.0;
				cnns->cnn[x]->C3->y[j][r][c] = (float)0.0;
			}
		}
	}
	// S4����
	for (j = 0; j<cnns->cnn[x]->S4->outChannels; j++) {
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize; r++) {
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize; c++) {
				cnns->cnn[x]->S4->d[j][r][c] = (float)0.0;
				cnns->cnn[x]->S4->y[j][r][c] = (float)0.0;
			}
		}
	}
	// O5���
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {
		cnns->cnn[x]->O5->d[j] = (float)0.0;
		cnns->cnn[x]->O5->v[j] = (float)0.0;
		cnns->cnn[x]->O5->y[j] = (float)0.0;
	}
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i < cnns->cnn[x]->O5->inputNum; i++)//��192����������ݶ�
			cnns->cnn[x]->O5->dwData[j][i] = 0.0;
		cnns->cnn[x]->O5->dbasicData[j] = 0.0;
	}

	// ��һ���������
	nSize mapSize = { cnns->cnn[x]->C1->mapSize,cnns->cnn[x]->C1->mapSize };//����˴�С,5X5
	nSize inSize = { cnns->cnn[x]->C1->inputWidth,cnns->cnn[x]->C1->inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnns->cnn[x]->S2->inputWidth,cnns->cnn[x]->S2->inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnns->cnn[x]->C1->outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnns->cnn[x]->C1->inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
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
					tempconv[r][c] = cnns->cnn[x]->C1->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * IData->ImgPtr[py].ImgData[(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnns->cnn[x]->C1->v[i][t][k] = cnns->cnn[x]->C1->v[i][t][k] + mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnns->cnn[x]->C1->y[i][r][c] = activation_Sigma(cnns->cnn[x]->C1->v[i][r][c], cnns->cnn[x]->C1->basicData[i]);
			}
		}
	}

	// �ڶ�����������S2��������
	outSize.c = cnns->cnn[x]->C3->inputWidth;//���ͼ���С,12X12
	outSize.r = cnns->cnn[x]->C3->inputHeight;//���ͼ���С,12X12
	inSize.c = cnns->cnn[x]->S2->inputWidth;//����ͼ���С,24X24
	inSize.r = cnns->cnn[x]->S2->inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i<(cnns->cnn[x]->S2->outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												//avgPooling(cnns->cnn[x]->S2->y[i], outSize, cnns->cnn[x]->C1->y[i], inSize, cnns->cnn[x]->S2->mapSize);//C1->y[i]����S2->mapSize��Сƽ���ػ����������S2->y[i]
												//�²����ػ�
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

	// �������������,������ȫ����
	outSize.c = cnns->cnn[x]->S4->inputWidth;//���ͼ���С,8X8
	outSize.r = cnns->cnn[x]->S4->inputHeight;//���ͼ���С,8X8
	inSize.c = cnns->cnn[x]->C3->inputWidth;//����ͼ���С,12X12
	inSize.r = cnns->cnn[x]->C3->inputHeight;//����ͼ���С,12X12
	mapSize.c = cnns->cnn[x]->C3->mapSize;//����˴�С,5X5
	mapSize.r = cnns->cnn[x]->C3->mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnns->cnn[x]->C3->outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnns->cnn[x]->C3->inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
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
					tempconv[r][c] = cnns->cnn[x]->C3->mapData[j][i][mapSize.r - 1 - r][mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnns->cnn[x]->S2->y[j][t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnns->cnn[x]->C3->v[i][t][k] = cnns->cnn[x]->C3->v[i][t][k] + mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnns->cnn[x]->C3->y[i][r][c] = activation_Sigma(cnns->cnn[x]->C3->v[i][r][c], cnns->cnn[x]->C3->basicData[i]);//�õ�C3���������MAP
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnns->cnn[x]->S4->inputWidth;//����ͼ���С,8X8
	inSize.r = cnns->cnn[x]->S4->inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnns->cnn[x]->S4->mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnns->cnn[x]->S4->mapSize;//���ͼ���С,4X4
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

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnns->cnn[x]->S4->outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnns->cnn[x]->S4->y[i][r][c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
			}
		}
	}
	nSize nnSize = { cnns->cnn[x]->O5->inputNum,cnns->cnn[x]->O5->outputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
															//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnns->cnn[x]->O5->wData[i][j];//�������֮�����,Ȼ�󷵻ؽ��
		cnns->cnn[x]->O5->v[i] = o;
	}
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)//�����sigmoid����
		cnns->cnn[x]->O5->y[i] = activation_Sigma(cnns->cnn[x]->O5->v[i], cnns->cnn[x]->O5->basicData[i]);//����sigmoid����,�����������ֵ
	
	//bp
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)
		cnns->cnn[x]->e[i] = cnns->cnn[x]->O5->y[i] - LData->LabelPtr[py].LabelData[i];//�����ʵ�������ȥ������ȷ�����,��Ӧ��ʽΪai-yi=-(yi-ai),ע�������y[i]��ai,��yi��outputData[i]
												  // �����O5��������
	for (i = 0; i<cnns->cnn[x]->O5->outputNum; i++)
		cnns->cnn[x]->O5->d[i] = cnns->cnn[x]->e[i] * sigma_derivation(cnns->cnn[x]->O5->y[i]);//��10����Ԫ��˵,ÿ����Ԫ�������������ȹ�ʽΪ-(yi-ai)(ai*(1-ai)),ע�������y[i]��ai,��yi��outputData[i]
																	// S4�㣬���ݵ�S4������
																	// ����û�м����
	outSize.r = cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize;
	outSize.c= cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize;//S4�����������С,������4X4
	for (i = 0; i < cnns->cnn[x]->S4->outChannels; i++) {//��ÿһ���������,����һ�����������һ����С�����жȾ�����֮��Ӧ
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnns->cnn[x]->O5->outputNum; j++) {//�����Ӧ��ʽ����ͨ������������Ĳв���㹫ʽ,����MATLAB�汾������˵����ƪ����fvd������˵��
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt������λȨֵ,S4���i�����MAP��r�е�c�����j����Ԫ��ȨֵΪ[j][i*outSize.c*outSize.r + r*outSize.c + c],��Ϊ���Ƕ�ά�����ȴ洢����,��һά�����������ӵ������ĵ�j����Ԫ,�ڶ�ά��������������ϵ�Ȩֵ
					cnns->cnn[x]->S4->d[i][r][c] = cnns->cnn[x]->S4->d[i][r][c] + cnns->cnn[x]->O5->d[j] * cnns->cnn[x]->O5->wData[j][wInt];
				}
			}
		}
	}
	int mapdata = cnns->cnn[x]->S4->mapSize;//������Ҫ�����ϲ�������,�����Ҫ����mapSize��С���ϲ���,������2X2
	nSize S4dSize = { cnns->cnn[x]->S4->inputWidth / cnns->cnn[x]->S4->mapSize,cnns->cnn[x]->S4->inputHeight / cnns->cnn[x]->S4->mapSize };//S4������жȾ����С,������4X4,Ҳ����S4����������С
	float C3e[8][8];
	for (i = 0; i<cnns->cnn[x]->C3->outChannels; i++) {//C3��ÿһ�����MAP����Ӧһ�����жȾ���
											  //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnns->cnn[x]->S4->mapSize; j = j + cnns->cnn[x]->S4->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S4dSize.c*cnns->cnn[x]->S4->mapSize; t = t + cnns->cnn[x]->S4->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnns->cnn[x]->S4->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C3e[j][t + m] = cnns->cnn[x]->S4->d[i][j / cnns->cnn[x]->S4->mapSize][t / cnns->cnn[x]->S4->mapSize];//�����
				}
			}
			for (n = 1; n < cnns->cnn[x]->S4->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S4dSize.c*cnns->cnn[x]->S4->mapSize; t++) {//�з����л�
					C3e[j + n][t] = C3e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnns->cnn[x]->S4->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��8
			for (c = 0; c<cnns->cnn[x]->S4->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��8
				cnns->cnn[x]->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnns->cnn[x]->C3->y[i][r][c]) / (float)(cnns->cnn[x]->S4->mapSize*cnns->cnn[x]->S4->mapSize);//ע��������Ҫ����(float)(cnns->cnn[x]->S4->mapSize*cnns->cnn[x]->S4->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C3������жȾ���
	}
	// S2�㣬S2��û�м����������ֻ�о�����м��������
	// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��
	outSize.c = cnns->cnn[x]->C3->inputWidth;//S2�����жȾ����СΪ12X12
	outSize.r = cnns->cnn[x]->C3->inputHeight;//S2�����жȾ����СΪ12X12
	inSize.r = cnns->cnn[x]->S4->inputWidth;
	inSize.c = cnns->cnn[x]->S4->inputHeight;//C3�����жȾ���Ĵ�С
	mapSize.r = cnns->cnn[x]->C3->mapSize;
	mapSize.c = cnns->cnn[x]->C3->mapSize;//C3�����˴�С5X5
	float corr[12][12];//�洢��ؼ�����
	float exData[16][16];//�洢full֮�����ʱ����
	int addr, addc;

	addr = addc = mapSize.r - 1;//Ҫ��չ�ı߳�
	for (i = 0; i<cnns->cnn[x]->S2->outChannels; i++) {//����S2��ÿһ�����MAP,6
		for (j = 0; j<cnns->cnn[x]->C3->outChannels; j++) {//����C3��ÿһ�����MAP,����������ȫ���ӽṹ,���S2���ÿһ��ͼ����C3���ÿһ��ͼ���й�,12
												  //float** corr = correlation(cnns->cnn[x]->C3->mapData[i][j], mapSize, cnns->cnn[x]->C3->d[j], inSize, full);//���ﱾ��Ҫ��C3���Ӧ�ľ����������ת180��Ȼ���ڽ��о������,��ʵ���Ͼ�������ְѾ������ת��180��,�������ֱ�ӾͲ���ת�����,����ֱ�Ӻ;�������,full�������
			int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����,12
			int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData��С,16
												   //��չ����
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//�������������ı�Ե��,����Ϊ0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnns->cnn[x]->C3->d[j][t - addr][k - addc]; // ��Ȼ,����ԭ����������
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
							corr[t][k] = corr[t][k] + cnns->cnn[x]->C3->mapData[i][j][r][c] * exData[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnns->cnn[x]->S2->d[i][t][k] = cnns->cnn[x]->S2->d[i][t][k] + corr[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
	}
	// C1�㣬�����
	mapdata = cnns->cnn[x]->S2->mapSize;//C1��������map�Ĵ�С,24X24
	nSize S2dSize = { cnns->cnn[x]->S2->inputWidth / cnns->cnn[x]->S2->mapSize,cnns->cnn[x]->S2->inputHeight / cnns->cnn[x]->S2->mapSize };//S2��������MAP�Ĵ�С,12X12���Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	float C1e[24][24];
	for (i = 0; i<cnns->cnn[x]->C1->outChannels; i++) {//C1��ÿһ�����MAP����Ӧһ�����жȾ���
		for (j = 0; j<S2dSize.r*cnns->cnn[x]->S2->mapSize; j = j + cnns->cnn[x]->S2->mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S2dSize.c*cnns->cnn[x]->S2->mapSize; t = t + cnns->cnn[x]->S2->mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnns->cnn[x]->S2->mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C1e[j][t + m] = cnns->cnn[x]->S2->d[i][j / cnns->cnn[x]->S2->mapSize][t / cnns->cnn[x]->S2->mapSize];//�����
				}
			}
			for (n = 1; n < cnns->cnn[x]->S2->mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S2dSize.c*cnns->cnn[x]->S2->mapSize; t++) {//�з����л�
					C1e[j + n][t] = C1e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnns->cnn[x]->S2->inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��24
			for (c = 0; c<cnns->cnn[x]->S2->inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��24
				cnns->cnn[x]->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnns->cnn[x]->C1->y[i][r][c]) / (float)(cnns->cnn[x]->S2->mapSize*cnns->cnn[x]->S2->mapSize);//ע��������Ҫ����(float)(cnns->cnn[x]->S2->mapSize*cnns->cnn[x]->S2->mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C1������жȾ���
	}

	//apply
	// C1���Ȩ�ظ���
	nSize dSize = { cnns->cnn[x]->S2->inputHeight,cnns->cnn[x]->S2->inputWidth };//C1�������Ⱦ����С,24X24
	nSize ySize = { cnns->cnn[x]->C1->inputHeight,cnns->cnn[x]->C1->inputWidth };//C1����������С,28X28
	mapSize.r = cnns->cnn[x]->C1->mapSize;
	mapSize.c = cnns->cnn[x]->C1->mapSize;//C1�����˴�С
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnns->cnn[x]->C1->outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j<cnns->cnn[x]->C1->inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
												 //����,һάת��ά����,��ת180���ƺ�����
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = IData->ImgPtr[py].ImgData[r*ySize.c + c];
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
					cov[r][c] = cnns->cnn[x]->C1->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}

			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnns->cnn[x]->C1->dmapData[j][i][t][k] = cnns->cnn[x]->C1->dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
		cnns->cnn[x]->C1->dbasicData[i] = -1 * opts->alpha*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// C3���Ȩ�ظ���
	dSize.c = cnns->cnn[x]->S4->inputWidth;//C3�������Ⱦ����С,8X8
	dSize.r = cnns->cnn[x]->S4->inputHeight;//C3�������Ⱦ����С,8X8
	ySize.c = cnns->cnn[x]->C3->inputWidth;//C3����������С,12X12
	ySize.r = cnns->cnn[x]->C3->inputHeight;//C3����������С,12X12
	mapSize.c = cnns->cnn[x]->C3->mapSize;//C3�����˴�С,5X5
	mapSize.r = cnns->cnn[x]->C3->mapSize;//C3�����˴�С,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnns->cnn[x]->C3->outChannels; i++) {//����ÿһ�����MAP,������12,��С8X8
		for (j = 0; j<cnns->cnn[x]->C3->inChannels; j++) {//����ÿһ������ͼ��,������8,��С12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnns->cnn[x]->S2->y[j][ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//��ת�����
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnns->cnn[x]->C3->d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnns->cnn[x]->C3->dmapData[j][i][t][k] = cnns->cnn[x]->C3->dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
		cnns->cnn[x]->C3->dbasicData[i] = -1 * opts->alpha*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// �����
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	for (j = 0; j<cnns->cnn[x]->O5->outputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i<cnns->cnn[x]->O5->inputNum; i++)//��192����������ݶ�
			cnns->cnn[x]->O5->dwData[j][i] = -1 * opts->alpha*cnns->cnn[x]->O5->d[j] * O5inData[i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
		cnns->cnn[x]->O5->dbasicData[j] = -1 * opts->alpha*cnns->cnn[x]->O5->d[j];//��b�����ݶ�,b���ݶȾ������ж�delta
	}
}



//opencl��������Ż�ʵ��
kernel void cnntest(global CNN* cnn,global ImgArr* IData,global LabelArr* LData,global atomic_int *wrongnum){
	int x = get_global_id(0);
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn->S2->inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn->S2->inputHeight;//�����һ������MAP����Ĵ�С,������24X24
	// ��һ��Ĵ���
	int i, j, r, c, t, k,m,n;
	float v1[6][24][24];
	float y1[6][24][24];
	float y2[6][12][12];
	float v3[12][8][8];
	float y3[12][8][8];
	float y4[12][4][4];
	float v5[10];
	float y5[10];
	//�������ԭ�в���
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				v1[j][r][c]=(float)0.0;
				y1[j][r][c]=(float)0.0;
			}
		}
	}
	// S2����
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				y2[j][r][c]=(float)0.0;
			}
		}
	}
	// C3����
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				v3[j][r][c]=(float)0.0;
				y3[j][r][c]=(float)0.0;
			}
		}
	}
	// S4����
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				y4[j][r][c]=(float)0.0;
			}
		}
	}
	// O5���
	for(j=0;j<cnn->O5->outputNum;j++){
		v5[j]=(float)0.0;
		y5[j]=(float)0.0;
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
			for (r = 0; r<mapSize.r; r++){
				for (c = 0; c<mapSize.c; c++){
					tempconv[r][c] = cnn->C1->mapData[j][i][mapSize.r -1 - r][mapSize.c -1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] = mapout[t][k] + tempconv[r][c] * IData->ImgPtr[x].ImgData[(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++){
				for (k = 0; k<outSize.c; k++){
					v1[i][t][k] = v1[i][t][k] + mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				y1[i][r][c] = activation_Sigma(v1[i][r][c], cnn->C1->basicData[i]);
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
			//avgPooling(y2[i], outSize, y1[i], inSize, cnn->S2->mapSize);//C1->y[i]����S2->mapSize��Сƽ���ػ����������S2->y[i]
		//�²����ػ�
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
			//������s
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * y2[j][t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					v3[i][t][k] = v3[i][t][k] + mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				y3[i][r][c] = activation_Sigma(v3[i][r][c], cnn->C3->basicData[i]);//�õ�C3���������MAP
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
						sum = sum + y3[i][m][n];
					}
				}
				y4[i][t][j] = sum / (float)(mSize*mSize);
			}
		}
	}

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn->S4->outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = y4[i][r][c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
			}
		}
	}
	nSize nnSize = { cnn->O5->inputNum,cnn->O5->outputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
	//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j]* cnn->O5->wData[i][j];//�������֮�����,Ȼ�󷵻ؽ��
		v5[i] = o;
	}
	for (i = 0; i<cnn->O5->outputNum; i++)//�����sigmoid����
		y5[i] = activation_Sigma(v5[i], cnn->O5->basicData[i]);//����sigmoid����,�����������ֵ

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
		atomic_fetch_add(wrongnum,1);//ԭ�Ӳ������Ƿ���ͬ
}
