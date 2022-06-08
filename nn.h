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

//ファイル入力の「,」を除去する関数
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

//各層の値を表示する関数
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

//各層間の重みと誤差を表示する関数
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

//重みと誤差を保存するための箱を生成する巻子
void make_weight(vector<vector<vector<double>>>& weights,vector<vector<vector<double>>>& loss,double input_element,double interval_element,double layer,double output_element ) {
	srand(time(NULL));
	vector<double> input(0, 0);
	vector<vector<double>>weight((0, 0));
	vector<double>layers(0, 0);
	//入力層から中間層の一入力への重みの生成
	for (int i = 0; i < input_element + 1; i++) {
		input.push_back((double)rand() / RAND_MAX);
	}
	//中間層から次の中間層の一入力への重みの生成および入力層からの重み
	for (int i = 0; i < interval_element; i++) {
		layers.push_back((double)rand() / RAND_MAX);
		weight.push_back(input);
	}
	layers.push_back((double)rand() / RAND_MAX);//中間層の重みにバイアスの項を追加
	weights.push_back(weight);
	weight.resize(0);//重みを初期化
	//中間層同士の重みの生成
	for (int i = 0; i < interval_element; i++) {
		weight.push_back(layers);
	}
	//中間層全体の重みの生成
	for (int i = 0; i < layer; i++) {
		weights.push_back(weight);
	}
	weights.pop_back();//重みが一層多いので除去
	//出力と中間層の端子数の差分だけ重みの数を減らす
	for (int i = 0; i < interval_element - output_element; i++) {
		weight.pop_back();
	}
	//出力と中間層の端子数の差分だけ重みの数を増やす
	for (int i = output_element - interval_element; i > 0; i--) {
		weight.push_back(layers);
	}
	weights.push_back(weight);
	loss = weights;
}

//中間層と出力層を生成する関数
void make_box(vector<vector<double>>& middle, vector<double>& output, double interval_element, double layer, double output_element) {
	vector<double>layers(0, 0);
	//中間層の生成
	for (int i = 0; i < interval_element; i++) {
		layers.push_back(0);
	}
	layers.push_back(1);//バイアス項を追加
	for (int i = 0; i < layer; i++) {
		middle.push_back(layers);
	}
	//出力層の生成
	for (int i = 0; i < output_element; i++) {
		output.push_back(0);
	}

}

//シグモイド関数
void sigmoid(vector<double>& input,vector<vector<double>>weight) {
	for (int i = 0; i < weight.size(); i++) {
		input[i] = 1 / (1 + exp(-input[i]));
	}
}

//順方向を計算する関数
void forward_caliculation(vector<double> input,vector<vector<double>>& middle, vector<double>& output,vector<vector<vector<double>>>weights,vector<double>label,vector<double>& error) {
	int layer = middle.size();
	int weights_size = weights.size();
	double sum = 0;
	//入力層から中間層への順方向計算
	for (int i = 0; i < weights[0].size(); i++) {
		for (int j = 0; j < input.size(); j++) {
			sum += input[j] * weights[0][i][j];
		}
		middle[0][i] = sum;
		sum = 0;
	}
	//シグモイド関数の計算
	sigmoid(middle[0], weights[0]);
	//中間層から中間層への順方向計算
	for (int i = 1; i < layer; i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < middle[i - 1].size(); k++) {
				sum += middle[i-1][k] * weights[i][j][k];
			}
			middle[i][j] = sum;
			sum = 0;
		}
		//シグモイド関数の計算
		sigmoid(middle[i], weights[i]);
	}
	//中間層から出力層への順方向計算
	for (int i = 0; i < weights[weights_size-1].size(); i++) {
		for (int j = 0; j < middle[layer-1].size(); j++) {
			sum+=middle[layer-1][j] * weights[weights_size-1][i][j];
		}
		output[i] = sum;
		sum = 0;
	}
	//シグモイド関数の計算
	sigmoid(output, weights[weights_size - 1]);

	//誤差計算
	for (int i = 0; i < error.size(); i++) {
		error[i] = output[i] - label[i];
	}
}

//誤差逆伝搬の更新値を計算する関数
void backward(vector<double>input,vector<vector<double>>middle, vector<double>output,vector<vector<vector<double>>> weights, vector<vector<vector<double>>>& update,vector<double>error) {
	int middle_size = middle.size();
	int weights_size = weights.size();
	double byas_loss = 0;
	//出力層への重みの誤差逆伝搬
	for (int i = 0; i < weights[weights_size - 1].size(); i++) {
		byas_loss = 2 * error[i] * output[i] * (1 - output[i]);//バイアス項の誤差計算
		update[weights_size - 1][i][update[weights_size - 1][i].size()-1] = byas_loss;
		//重みの誤差計算
		for (int j = update[weights_size - 1][i].size()-2; j>=0; j--) {
			update[weights_size - 1][i][j] = middle[middle_size-1][j] * byas_loss;
		}
	}
	byas_loss = 0;
	//中間層同士の重みの誤差逆伝搬
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
	//入力層からの重みの誤差伝搬
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

//重みを更新する関数
void change_weight(vector<vector<vector<double>>>& weights, vector<vector<vector<double>>>update,double rate,double size) {
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				weights[i][j][k] -= rate*update[i][j][k]/size;
			}
		}
	}
}

//逐次学習を行う関数
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
		//指定した出力より多い要素分だけ削除
		for (int i = 0; i < label_size - output_element; i++) {
			input_label[i].pop_back();
		}
	}
	*/
	vector<vector<double>>label = input_label;
	int input_size = input_data.size();
	int count = 0;
	srand(time(NULL));
	//ランダムな入力順の決定
	for (int i = 0; i < input_size; i++) {
		int unit = (int)rand() % input.size();
		input_data[i] = input[unit];
		input_label[i] = label[unit];
		input.erase(input.begin() + unit);
		label.erase(label.begin() + unit);
	}
	//入力の順番をランダムに一回決め、以降はその順番で入力した場合
	while (learning_times > count) {
		sum_error = 0;
		//逐次学習の1エポック
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
	//入力の順番を毎回ランダムにした場合
	while (learning_times>count) {
		input = input_data;
		label = input_label;
		sum_error = 0;
		//逐次学習の1エポック
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

//一括学習を行う関数
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
		//指定した出力より多い要素分だけ削除
		for (int i = 0; i < label_size - output_element; i++) {
			input_label[i].pop_back();
		}
	}
	int input_size = input_data.size();
	int count = 0;
	//指定した学習回数分学習させる
	while (learning_times>count) {
		sum_error = 0;
		//更新値を保存する箱の初期化
		for (int p = 0; p < copy.size(); p++) {
			for (int q = 0; q < copy[p].size(); q++) {
				for (int r = 0; r < copy[p][q].size(); r++) {
					update[p][q][r] =0;
				}
			}
		}
		//一括学習の1エポック
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
		//逐次学習の1エポック
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