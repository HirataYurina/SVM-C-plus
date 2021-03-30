#include <iostream>
#include <math.h>
#include <svm_techi.h>
#include <cmath>
#include <stdlib.h>

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
    for (int i; i < num_data; i++)
    {
        for (int j; j < num_data; j++)
        {
            double res = 0.0;
            for (int k; k < num_feature; k++)
            {
                res += (x_train[i][k] - x_train[j][k]) * (x_train[i][k] - x_train[j][k]);
            }
            k[i][i] = exp(-res / (2 * pow(sigma, 2)));
        }
    }

    return k;
}

// choose i that does not satisfy kkt condition
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
    for (int j; j < num_data; j++)
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
// 参考<<统计学习方法>>
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
// train SVM bu SMO
{
    int iter_step = 0;
    int param_changed = -1;

    double **k = calculate_kernel();

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
                    double L = my_max(0.0, alpha_old2 - alpha_old1);
                    double H = my_min(C, C + alpha_old2 - alpha_old1);
                }
                else
                {
                    double L = my_max(0.0, alpha_old2 + alpha_old1 - C);
                    double H = my_min(C, alpha_old1 + alpha_old2);
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
        }
    }
}

void SVM::initiate(double *data, double *label, int num_data, int num_feature)
{
    this->num_data = num_data;
    this->num_feature = num_feature;
    this->x_train = new double *[num_data];
    this->y_train = new double[num_feature];

    // store data into x_train
    // data is [num_data * num_feature,]
    // x_train is [num_data, num_feature]
    for (int i; i < num_data; i++)
    {
        this->x_train[i] = new double[num_feature];
        for (int j; j < num_feature; j++)
        {
            this->x_train[i][j] = data[i * num_feature + j];
        }
    }

    // store label into y_train
    for (int k; k < num_data; k++)
    {
        this->y_train[k] = label[k];
    }

    // initial alpha by zero
    this->alpha = new double[num_data]{0};
    // initiate E by zero
    this->E = new double[num_data]{0};
}

int main()
{

    return 0;
}
