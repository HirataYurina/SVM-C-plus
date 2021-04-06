/*
* My SVM codes.
* author: 栗山未来ii
* data:   2077/3/31/22:14
*/

#include <iostream>
#include <math.h>
#include <C:\Users\Administrator\Desktop\SVM\svm-techi.h>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

template <class T>

int length(T &arr)
{
    return sizeof(arr) / sizeof(arr[0]);
}

double my_max(double a, double b)
{
    if (a >= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

double my_min(double a, double b)
{
    if (a >= b)
    {
        return b;
    }
    else
    {
        return a;
    }
}

SVM::~SVM()
{
    // free memory
}

double **SVM::calculate_kernel()
{
    // train_data: (num, shape_feature)
    double **k = new double *[num_data];
    for (int i = 0; i < num_data; i++)
    {   
        k[i] = new double[num_data];
        for (int j = 0; j < num_data; j++)
        {   
            double res = 0.0;
            for (int m = 0; m < num_feature; m++)
            {
                res += (x_train[i][m] - x_train[j][m]) * (x_train[i][m] - x_train[j][m]);
            }
            k[i][j] = exp(-res / (2 * pow(sigma, 2)));
        }
    }
    return k;
}

// choose index that does not satisfy kkt condition
bool SVM::is_satisfy_kkt(int i, double **k)
{
    double gxi = this->calc_gxi(i, k);
    double label_i = this->y_train[i];
    double alpha_i = this->alpha[i];

    if ((fabs(alpha_i) < toler) && (label_i * gxi >= 1))
    {
        return true;
    }

    if ((fabs(alpha_i - C) < toler) && (label_i * gxi) >= 1)
    {
        return true;
    }

    if ((alpha_i > toler) && (alpha_i < (C + toler)) && (fabs(label_i * gxi) < toler))
    {
        return true;
    }

    return false;
}

double SVM::calc_gxi(int i, double **k)
{
    double gxi = 0.0;
    for (int j = 0; j < num_data; j++)
    {
        // calculate gxi
        gxi += alpha[j] * y_train[j] * k[j][i];
    }

    gxi += b;

    return gxi;
}

double SVM::calc_ei(int i, double **k)
{
    double gxi = calc_gxi(i, k);
    return gxi - y_train[i];
}

// get alpha j
// get_alpha_j is a important function in SVM
E2AndJ *SVM::get_alpha_j(double e1, int i, double **k)
{
    double e2 = 0.0;
    double maxE1_E2 = -1.0;
    int max_index = -1;

    for (int j = 0; j < num_data; j++)
    {
        double E2_temp = calc_ei(j, k);
        if (fabs(e1 - E2_temp) > maxE1_E2)
        {
            maxE1_E2 = fabs(e1 - E2_temp);
            e2 = E2_temp;
            max_index = j;
        }
    }

    if (max_index == -1)
    {
        max_index = i;
        while (max_index == -1)
        {
            max_index = (int)rand() / RAND_MAX * num_data;
        }
        e2 = calc_ei(max_index, k);
    }

    E2AndJ *e2_j = new E2AndJ();
    e2_j->e2 = e2;
    e2_j->j = max_index;

    return e2_j;
}

void SVM::train(int iter)
// train SVM by SMO
{
    int iter_step = 0;
    int param_changed = 1;
    cout << "start train" << endl;

    double **k = calculate_kernel();

    cout << "start train" << endl;

    while ((iter_step < iter) && (param_changed) > 0)
    {
        iter_step += 1;
        param_changed = 0;

        for (int i = 0; i < num_data; i++)
        {
            if (!is_satisfy_kkt(i, k))
            {
                double e1 = calc_ei(i, k);
                E2AndJ *e2_j = get_alpha_j(e1, i, k);
                double e2 = e2_j->e2;
                int j = e2_j->j;
                double y1 = y_train[i];
                double y2 = y_train[j];
                double alpha_old1 = alpha[i];
                double alpha_old2 = alpha[j];

                double L = 0.0;
                double H = 0.0;

                if (y1 != y2)
                {
                    L = my_max(0.0, alpha_old2 - alpha_old1);
                    H = my_min(C, C + alpha_old2 - alpha_old1);
                }
                else
                {
                    L = my_max(0.0, alpha_old2 + alpha_old1 - C);
                    H = my_min(C, alpha_old1 + alpha_old2);
                }

                if (L == H)
                {
                    continue;
                }

                double k11 = k[i][i];
                double k22 = k[j][j];
                double k12 = k[i][j];
                double k21 = k[j][i];

                double alpha_new_2 = alpha_old2 + y2 * (e1 - e2) / (k11 + k22 - 2 * k12);

                // crop alpha_new_2
                if (alpha_new_2 < L)
                {
                    alpha_new_2 = L;
                }
                else if (alpha_new_2 > H)
                {
                    alpha_new_2 = H;
                }
                double alpha_new_1 = alpha_old1 + y1 * y2 * (alpha_old2 - alpha_new_2);

                double b1_new = -1 * e1 - y1 * k11 * (alpha_new_1 - alpha_old1) - y2 * k21 * (alpha_new_2 - alpha_old2) + b;
                double b2_new = -1 * e2 - y1 * k12 * (alpha_new_1 - alpha_old1) - y2 * k22 * (alpha_new_2 - alpha_old2) + b;

                // ======================================================
                double b_new;
                if ((alpha_new_1 > 0) && (alpha_new_1 < C))
                {
                    b_new = b1_new;
                }
                else if ((alpha_new_2 > 0) && (alpha_new_2 < C))
                {
                    b_new = b2_new;
                }
                else
                {
                    b_new = (b1_new + b2_new) / 2;
                }
                // ======================================================

                alpha[i] = alpha_new_1;
                alpha[j] = alpha_new_2;
                b = b_new;
                E[i] = calc_ei(i, k);
                E[j] = calc_ei(j, k);

                if (fabs(alpha_new_2 - alpha_old2) > 0.00001)
                {
                    param_changed += 1;
                }
            }

            cout << "iter step:" << iter_step << "  i:" << i << "  param changed:" << param_changed << endl;

        }
    }
}

void SVM::initiate(double *data, double *label, int num_data, int num_feature)
{   

    cout << "start initiate" << endl;

    this->num_data = num_data;
    this->num_feature = num_feature;
    this->x_train = new double *[num_data];
    this->y_train = new double[num_data];

    // store data into x_train
    // data is [num_data * num_feature,]
    // x_train is [num_data, num_feature]
    for (int i = 0; i < num_data; i++)
    {
        this->x_train[i] = new double[num_feature];
        for (int j = 0; j < num_feature; j++)
        {
            this->x_train[i][j] = data[i * num_feature + j];
        }
    }

    // store label into y_train
    for (int k = 0; k < num_data; k++)
    {   
        // binary
        if (label[k] == 1)
        {
            this->y_train[k] = 1;
        }
        else
        {
            this->y_train[k] = -1;
        }
    }

    // initial alpha by zero
    this->alpha = new double[num_data]();
    // initiate E by zero
    this->E = new double[num_data]();
    this->sigma = 10;
    this->toler = 0.001;
    this->b = 0.0;
    this->C = 200.0;
}

double SVM::predict(double *data)
{
    double result = 0;

    for (int i = 0; i < num_data; i++)
    {   
        // x_train(178, 13)
        double *data_temp = x_train[i];
        double res = 0;
        // calculate kernel by data and data_temp
        for (int j = 0; j < num_feature; j++)
        {
            res += (data[j] - data_temp[j]) * (data[j] - data_temp[j]);
        }
        double single_kernel = exp(res / (-2 * pow(sigma, 2)));
        result += alpha[i] * y_train[i] * single_kernel;
    }
    // sign function
    if (result < 0)
    {
        return -1;
    }
    else if (result > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

double SVM::accuracy(double **x_test, double *y_test, int num_test)
{
    double num_correct = 0;
    for (int i = 0; i < num_test; i++)
    {
        double prediction = predict(x_test[i]);
        if (prediction == y_test[i]) num_correct += 1;
    }
    cout << "number of correct samples are: " << num_correct << endl;
    return num_correct / num_test;
}

int main()
{
    // test my codes
    // double x_train_dummy[12] = {1.0, 2.0, 3.0,
    //                             4.0, 1.2, 3.3,
    //                             4.4, 7.6, 9.8,
    //                             7.7, 8.0, 4.8};
    // double y_train_dummy[4] = {1.0, -1.0, 1.0, -1.0};
    // double *x_train_p = x_train_dummy;
    // double *y_train_p = y_train_dummy;
    // int num_data = 4;
    // int num_feat = 3;
    // SVM svm;
    // // initiate SVM    
    // svm.initiate(x_train_p, y_train_p, num_data, num_feat);
    // svm.train(10);

    // wine dataset
    ifstream file("C://Users//Administrator//Desktop//SVM//wine.csv", ios::in);
    string line_str;
    int num_data_ = 178;
    int num_feature_ = 13;
    double inputs[num_data_ * num_feature_];
    double labels[num_data_];
    int j = 0;

    while (getline(file, line_str))
    {
    //   cout << line_str << endl;
        stringstream ss(line_str);
        string str;
        while (getline(ss, str, ','))
        {
            if (j % 14 == 0)
            {
                labels[j / 14] = atof(str.c_str());
            }
            else
            {
                inputs[j / 14 * 13 + j % 14 - 1] = atof(str.c_str());
            }
            j++;
        }
    }

    // check out x_train and y_train
    // for(int m = 0; m < num_data * num_feature; m++)
    // {   
    //     if ( m % 13 == 0)
    //     {
    //         cout << x_train[m] << endl;
    //     }
    // }

    // for (int n = 0; n < num_data; n++)
    // {
    //     cout << y_train[n] << endl;
    // }

    // start train svm
    double *x_train_p = inputs;
    double *y_train_p = labels;
    SVM svm;
    svm.initiate(x_train_p, y_train_p, num_data_, num_feature_);
    svm.train(100);
    double accuracy = svm.accuracy(svm.x_train, svm.y_train, num_data_);
    cout << "accuracy of train dataset is: " << accuracy << endl;

    return 0;
}
