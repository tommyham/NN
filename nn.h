#include <stdio.h>
#include<stdlib.h>
#include<fstream>
#include<string>
#include<sstream>
#define _USE_MATH_DEFINES
#include <iostream>
#include<complex>
#include<math.h>
#include<vector>
#include<time.h>
using namespace std;

//�t�@�C�����͂́u,�v����������֐�
vector<string> split(string& input, char delimiter) {
	istringstream stream(input);
	string field;
	vector<string> result;
	while (getline(stream, field, delimiter)) {
		result.push_back(field);
	}
	return result;
}

//
void display_answer(vector<vector<double>>input_data, vector<vector<double>>label_data) {
	cout << "input" << endl;
	for (int i = 0; i < input_data.size(); i++) {
		for (int j = 0; j < input_data[i].size(); j++) {
			cout << input_data[i][j] << " ";
		}
		cout << endl;
	}
	cout << "label" << endl;
	for (int i = 0; i < label_data.size(); i++) {
		for (int j = 0; j < label_data[i].size(); j++) {
			cout << label_data[i][j] << " ";
		}
		cout << endl;
	}
}

//�e�w�̒l��\������֐�
void display_layer(vector<double> input, vector <vector<double>> middle, vector<double> output,vector<double> error) {
	cout << "input" << endl;
	for (int i = 0; i < input.size(); i++) {
		cout << input[i] << " ";
	}
	cout << endl;
	cout << "middle" << endl;
	for (int i = 0; i < middle.size(); i++) {
		for (int j = 0; j < middle[i].size(); j++) {
			cout << middle[i][j] << " ";
		}
		cout << endl;
	}
	cout << "output" << endl;
	for (int i = 0; i < output.size(); i++) {
		cout << output[i] << " ";
	}
	cout << endl;
	cout << "error" << endl;
	for (int i = 0; i < error.size(); i++) {
		cout << error[i] << " ";
	}
	cout << endl;
}

//�e�w�Ԃ̏d�݂ƌ덷��\������֐�
void display_interval(vector<vector<vector<double>>>weights, vector<vector<vector<double>>>update) {
	cout << "weight" << endl;
	for (int i = 0; i < weights.size(); i++) {
		cout << i << endl;
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				cout << weights[i][j][k]<<" ";
			}
			cout << endl;
		}
	}
	cout << "update" << endl;
	for (int i = 0; i < update.size(); i++) {
		cout << i << endl;
		for (int j = 0; j < update[i].size(); j++) {
			for (int k = 0; k < update[i][j].size(); k++) {
				cout << update[i][j][k]<<" ";
			}
			cout << endl;
		}
	}
}

//�d�݂ƌ덷��ۑ����邽�߂̔��𐶐����銪�q
void make_weight(vector<vector<vector<double>>>& weights,vector<vector<vector<double>>>& loss,double input_element,double interval_element,double layer,double output_element ) {
	srand(time(NULL));
	vector<double> input(0, 0);
	vector<vector<double>>weight((0, 0));
	vector<double>layers(0, 0);
	//���͑w���璆�ԑw�̈���͂ւ̏d�݂̐���
	for (int i = 0; i < input_element + 1; i++) {
		input.push_back((double)rand() / RAND_MAX);
	}
	//���ԑw���玟�̒��ԑw�̈���͂ւ̏d�݂̐�������ѓ��͑w����̏d��
	for (int i = 0; i < interval_element; i++) {
		layers.push_back((double)rand() / RAND_MAX);
		weight.push_back(input);
	}
	layers.push_back((double)rand() / RAND_MAX);//���ԑw�̏d�݂Ƀo�C�A�X�̍���ǉ�
	weights.push_back(weight);
	weight.resize(0);//�d�݂�������
	//���ԑw���m�̏d�݂̐���
	for (int i = 0; i < interval_element; i++) {
		weight.push_back(layers);
	}
	//���ԑw�S�̂̏d�݂̐���
	for (int i = 0; i < layer; i++) {
		weights.push_back(weight);
	}
	weights.pop_back();//�d�݂���w�����̂ŏ���
	//�o�͂ƒ��ԑw�̒[�q���̍��������d�݂̐������炷
	for (int i = 0; i < interval_element - output_element; i++) {
		weight.pop_back();
	}
	//�o�͂ƒ��ԑw�̒[�q���̍��������d�݂̐��𑝂₷
	for (int i = output_element - interval_element; i > 0; i--) {
		weight.push_back(layers);
	}
	weights.push_back(weight);
	loss = weights;
}

//���ԑw�Əo�͑w�𐶐�����֐�
void make_box(vector<vector<double>>& middle, vector<double>& output, double interval_element, double layer, double output_element) {
	vector<double>layers(0, 0);
	//���ԑw�̐���
	for (int i = 0; i < interval_element; i++) {
		layers.push_back(0);
	}
	layers.push_back(1);//�o�C�A�X����ǉ�
	for (int i = 0; i < layer; i++) {
		middle.push_back(layers);
	}
	//�o�͑w�̐���
	for (int i = 0; i < output_element; i++) {
		output.push_back(0);
	}

}

//�V�O���C�h�֐�
void sigmoid(vector<double>& input,vector<vector<double>>weight) {
	for (int i = 0; i < weight.size(); i++) {
		input[i] = 1 / (1 + exp(-input[i]));
	}
}

//���������v�Z����֐�
void forward_caliculation(vector<double> input,vector<vector<double>>& middle, vector<double>& output,vector<vector<vector<double>>>weights,vector<double>label,vector<double>& error) {
	int layer = middle.size();
	int weights_size = weights.size();
	double sum = 0;
	//���͑w���璆�ԑw�ւ̏������v�Z
	for (int i = 0; i < weights[0].size(); i++) {
		for (int j = 0; j < input.size(); j++) {
			sum += input[j] * weights[0][i][j];
		}
		middle[0][i] = sum;
		sum = 0;
	}
	//�V�O���C�h�֐��̌v�Z
	sigmoid(middle[0], weights[0]);
	//���ԑw���璆�ԑw�ւ̏������v�Z
	for (int i = 1; i < layer; i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < middle[i - 1].size(); k++) {
				sum += middle[i-1][k] * weights[i][j][k];
			}
			middle[i][j] = sum;
			sum = 0;
		}
		//�V�O���C�h�֐��̌v�Z
		sigmoid(middle[i], weights[i]);
	}
	//���ԑw����o�͑w�ւ̏������v�Z
	for (int i = 0; i < weights[weights_size-1].size(); i++) {
		for (int j = 0; j < middle[layer-1].size(); j++) {
			sum+=middle[layer-1][j] * weights[weights_size-1][i][j];
		}
		output[i] = sum;
		sum = 0;
	}
	//�V�O���C�h�֐��̌v�Z
	sigmoid(output, weights[weights_size - 1]);

	//�덷�v�Z
	for (int i = 0; i < error.size(); i++) {
		error[i] = output[i] - label[i];
	}
}

//�덷�t�`���̍X�V�l���v�Z����֐�
void backward(vector<double>input,vector<vector<double>>middle, vector<double>output,vector<vector<vector<double>>> weights, vector<vector<vector<double>>>& update,vector<double>error) {
	int middle_size = middle.size();
	int weights_size = weights.size();
	double byas_loss = 0;
	//�o�͑w�ւ̏d�݂̌덷�t�`��
	for (int i = 0; i < weights[weights_size - 1].size(); i++) {
		byas_loss = 2 * error[i] * output[i] * (1 - output[i]);//�o�C�A�X���̌덷�v�Z
		update[weights_size - 1][i][update[weights_size - 1][i].size()-1] = byas_loss;
		//�d�݂̌덷�v�Z
		for (int j = update[weights_size - 1][i].size()-2; j>=0; j--) {
			update[weights_size - 1][i][j] = middle[middle_size-1][j] * byas_loss;
		}
	}
	byas_loss = 0;
	//���ԑw���m�̏d�݂̌덷�t�`��
	for (int i = middle_size-1; i >0; i--) {
		for (int j = 0; j < weights[i].size(); j++) {
			int weight_size = weights[i][j].size();
			for (int k = 0; k < update[i + 1].size(); k++) {
				byas_loss += update[i+1][k][update[i+1][k].size()-1]*weights[i+1][k][j];
			}
			byas_loss= byas_loss* middle[i][j] * (1 - middle[i][j]);
			update[i][j][update[i][j].size()-1] =byas_loss;
			for (int k = weights[i][j].size()-2; k >=0; k--) {
				update[i][j][k] =  middle[i-1][k] * byas_loss;
			}
			byas_loss = 0;
		}
		byas_loss = 0;
	}
	byas_loss = 0;
	//���͑w����̏d�݂̌덷�`��
	for (int j = 0; j < weights[0].size(); j++) {
		int weight_size = weights[0][j].size();
		for (int k = 0; k <update[1].size(); k++) {
			byas_loss += update[1][k][update[1][k].size()-1]*weights[1][k][j];
		}
		byas_loss= byas_loss * middle[0][j] * (1 - middle[0][j]);
		update[0][j][update[0][j].size() - 1] =byas_loss;
		for (int k = update[0][j].size() - 2; k >= 0; k--) {
			update[0][j][k] =input[k] * byas_loss;
		}
		byas_loss = 0;
	}
}

//�d�݂��X�V����֐�
void change_weight(vector<vector<vector<double>>>& weights, vector<vector<vector<double>>>update,double rate,double size) {
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				weights[i][j][k] -= rate*update[i][j][k]/size;
			}
		}
	}
}

//�����w�K���s���֐�
void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label) {
	int input_element = input_data[0].size()-1;
	int output_element = 0;
	int layer, interval_element = 0;
	int learning_times = 0;
	double rate = 0;
	double sum_error = 1;
	double boader = 0;
	vector<vector<double>>input = input_data;
	vector<vector<double>> middle((0, 0));
	vector<vector<vector<double>>> weights((0, 0));
	vector<vector<vector<double>>> update((0,0));
	vector<double> output(0, 0);

	cout << "output_element=";
	scanf_s("%d", &output_element);
	cout << "layer=";
	scanf_s("%d", &layer);
	cout << "interval_element=";
	scanf_s("%d", &interval_element);
	cout << "learning rate=";
	scanf_s("%lf", &rate);
	cout << "learning times=";
	scanf_s("%d", &learning_times);
	cout << "boader rate=";
	scanf_s("%lf", &boader);
	boader = pow(0.1, boader);

	vector<double>error(output_element, 0);
	make_weight(weights, update, input_element, interval_element, layer, output_element);
	make_box(middle, output, interval_element, layer, output_element);
	/*
	for (int i = 0; i < input_label.size(); i++) {
		int label_size = input_label[i].size();
		//�w�肵���o�͂�葽���v�f�������폜
		for (int i = 0; i < label_size - output_element; i++) {
			input_label[i].pop_back();
		}
	}
	*/
	vector<vector<double>>label = input_label;
	int input_size = input_data.size();
	int count = 0;
	srand(time(NULL));
	//�����_���ȓ��͏��̌���
	for (int i = 0; i < input_size; i++) {
		int unit = (int)rand() % input.size();
		input_data[i] = input[unit];
		input_label[i] = label[unit];
		input.erase(input.begin() + unit);
		label.erase(label.begin() + unit);
	}
	//���͂̏��Ԃ������_���Ɉ�񌈂߁A�ȍ~�͂��̏��Ԃœ��͂����ꍇ
	while (learning_times > count) {
		sum_error = 0;
		//�����w�K��1�G�|�b�N
		for (int i = 0; i < input_size; i++) {
			forward_caliculation(input_data[i], middle, output, weights, input_label[i], error);
			//display_layer(input[i], middle, output, error);

			backward(input_data[i], middle, output, weights, update, error);
			change_weight(weights, update, rate,1);
			//display_interval(weights, update);
		}
		//display_layer(input, middle, output, error);
		//display_interval(weights, update);
		for (int j = 0; j < error.size(); j++) {
			sum_error += error[j] * error[j];
		}
		cout << sum_error << endl;
		count++;

	}
	/*
	//���͂̏��Ԃ𖈉񃉃��_���ɂ����ꍇ
	while (learning_times>count) {
		input = input_data;
		label = input_label;
		sum_error = 0;
		//�����w�K��1�G�|�b�N
		for (int i = 0; i < input_size; i++) {
			int unit = (int)rand() % input.size();
			//display_layer(input[unit], middle, output, error);
			//display_interval(weights, loss);
			forward_caliculation(input[unit], middle, output, weights, label[unit], error);
			//display_layer(input[unit], middle, output, error);

			backward(input[unit], middle, output, weights,update, error);
			change_weight(weights, update,rate);
			//display_interval(weights, loss);
			input.erase(input.begin() + unit);
			label.erase(label.begin() + unit);
		}
		//display_layer(input, middle, output, error);
		//display_interval(weights, loss);
		for (int j = 0; j < error.size(); j++) {
			sum_error += error[j] * error[j];
		}
		cout << sum_error << endl;
		count++;
	}
	*/
	cout << "count" << endl;
	cout << count << endl;

	input = { {0, 1, 0}, {0, 1, 1} };
	label = { {1},{0} };
	cout << "result" << endl;
	for (int i = 0; i < input.size(); i++) {
		forward_caliculation(input[i], middle, output, weights, label[i], error);
		for (int j = 0; j < output.size(); j++) {
			cout << output[j] << " ";
		}
		cout << endl;
	}
}

//�ꊇ�w�K���s���֐�
void learn_patch(vector<vector<double>>& input_data, vector<vector<double>>& input_label) {
	int input_element = input_data[0].size()-1;
	int output_element = 0;
	int layer, interval_element = 0;
	int learning_times = 0;
	double rate = 0;
	double sum_error = 0;
	double boader = 0;
	vector<vector<double>> middle((0, 0));
	vector<vector<vector<double>>> weights((0, 0));
	vector<vector<vector<double>>> update((0,0));
	vector<double> output(0, 0);

	cout << "output_element=";
	scanf_s("%d", &output_element);
	cout << "layer=";
	scanf_s("%d", &layer);
	cout << "interval_element=";
	scanf_s("%d", &interval_element);
	cout << "learning rate=";
	scanf_s("%lf", &rate);
	cout << "learning times=";
	scanf_s("%d", &learning_times);
	//cout << "boader rate=";
	//scanf_s("%lf", &boader);
	//boader = pow(0.1, boader);

	make_weight(weights, update, input_element, interval_element, layer, output_element);
	make_box(middle, output, interval_element, layer, output_element);
	vector<vector<vector<double>>> copy = weights;
	vector<double>error(output_element, 0);
	display_layer(input_data[0], middle, output, error);
	display_interval(weights, update);

	for (int i = 0; i < input_label.size(); i++) {
		int label_size = input_label[i].size();
		//�w�肵���o�͂�葽���v�f�������폜
		for (int i = 0; i < label_size - output_element; i++) {
			input_label[i].pop_back();
		}
	}
	int input_size = input_data.size();
	int count = 0;
	//�w�肵���w�K�񐔕��w�K������
	while (learning_times>count) {
		sum_error = 0;
		//�X�V�l��ۑ����锠�̏�����
		for (int p = 0; p < copy.size(); p++) {
			for (int q = 0; q < copy[p].size(); q++) {
				for (int r = 0; r < copy[p][q].size(); r++) {
					update[p][q][r] =0;
				}
			}
		}
		//�ꊇ�w�K��1�G�|�b�N
		for (int i = 0; i < input_size; i++) {
			//display_layer(input[unit], middle, output, error);
			//display_interval(weights, loss);
			forward_caliculation(input_data[i], middle, output, weights, input_label[i], error);
			//display_layer(input[unit], middle, output, error);
			backward(input_data[i], middle, output,weights, copy,error);
			//display_interval(weights, loss);
			for (int p = 0; p < copy.size(); p++) {
				for (int q = 0; q < copy[p].size(); q++) {
					for (int r = 0; r < copy[p][q].size(); r++) {
						update[p][q][r] += copy[p][q][r];
					}
				}
			}
		}
		double size = input_size;
		for (int p = 0; p < copy.size(); p++) {
			for (int q = 0; q < copy[p].size(); q++) {
				for (int r = 0; r < copy[p][q].size(); r++) {
					update[p][q][r] =update[p][q][r]/size;
				}
			}
		}
		change_weight(weights, update, rate, 1);
		//change_weight(weights, update,rate,size);
		//display_layer(input, middle, output, error);
		//display_interval(weights, loss);
		for (int j = 0; j < error.size(); j++) {
			sum_error += error[j] * error[j];
		}
		cout << sum_error << endl;
		count++;
	}
	vector<vector<double>>input = { {0, 1, 0}, {0, 1, 1} };
	vector<vector<double>>label = { {1},{0} };
	cout << "result" << endl;
	for (int i = 0; i < input.size(); i++) {
		forward_caliculation(input[i], middle, output, weights, label[i], error);
		for (int j = 0; j < output.size(); j++) {
			cout << output[j] << " ";
		}
		cout << endl;
	}
}

void element_rand(vector<vector<double>>input_data) {
	vector<vector<double>>input = input_data;
	int learning_times = 0;
	int input_size = input_data.size();
	int count = 0;
	cout << "learning times=";
	scanf_s("%d", &learning_times);
	srand(time(NULL));
	while (learning_times > count) {
		input = input_data;
		//�����w�K��1�G�|�b�N
		for (int i = 0; i < input_size; i++) {
			cout << "input size=";
			cout << input.size() << endl;
			int unit =(int) rand() % input.size();
			cout << "unit="; 
			cout << unit << endl;
			input.erase(input.begin() + unit);
		}
		cout << endl;
		count++;
	}

}