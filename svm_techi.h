#include <iostream>

using namespace std;

struct E2AndJ
{
    double e2;
    int j;
};


class SVM
{
private:
    double *alpha;
    double *E;

    int num_data;
    int num_feature;
    int sigma;
    double C;
    double toler;
    double b;
    double ** calculate_kernel();
    bool is_satisfy_kkt(int, double **);
    double calc_gxi(int, double **);
    double calc_ei(int, double **);
    E2AndJ * get_alpha_j(double e1, int i, double **k);

public:
    double **x_train;
    double *y_train;
    ~SVM();
    void train(int iter);
    void initiate(double *, double *, int, int);
    double predict(double *);
    double accuracy(double **, double *, int);
};
