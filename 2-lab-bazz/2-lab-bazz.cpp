#include <iostream>
#include <complex>
#include <ctime>
#include "mkl.h"
using namespace std;

const int n = 1024;

float rand_float(float a, float b) 
{
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}

complex<float> rand_complex(float a, float b) 
{
    return { rand_float(a, b), rand_float(a, b) };
}

void generate_matrix(complex<float>* A) 
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < n * n; i++) {
        A[i] = rand_complex(-10.0f, 10.0f);
    }
}

void print_matrix(complex<float>* A) 
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            cout << A[i * n + j] << " ";
        }
        cout << endl;
    }
}

// Перемножение по формуле из линейной алгебры
void mult1(complex<float>* A, complex<float>* B, complex<float>* C)
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            C[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) 
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Перемножение с помощью функции cblas_cgemm из BLAS
void mult2(complex<float>* A, complex<float>* B, complex<float>* C) 
{
    complex<float> alpha(1.0f, 0.0f);
    complex<float> beta(0.0f, 0.0f);
    int lda = n;
    int ldb = n;
    int ldc = n;
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// Оптимизированное перемножение
void mult3(complex<float>* A, complex<float>* B, complex<float>* C) {
    const int block_size = 32; 
    for (int i = 0; i < n; i += block_size) 
    {
        for (int j = 0; j < n; j += block_size) 
        {
            for (int k = 0; k < n; k += block_size) 
            {
                for (int ii = i; ii < i + block_size && ii < n; ++ii)
                {
                    for (int jj = j; jj < j + block_size && jj < n; ++jj) 
                    {
                        complex<float> sum(0.0f, 0.0f);
                        for (int kk = k; kk < k + block_size && kk < n; ++kk) 
                        {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

int main() 
{
    setlocale(LC_ALL, "rus");
    cout << "Работу выполнил Ершов Владислав Олегович РПИа-о22 " << endl;
    complex<float>* A = new complex<float>[n * n];
    complex<float>* B = new complex<float>[n * n];
    complex<float>* C1 = new complex<float>[n * n];
    complex<float>* C2 = new complex<float>[n * n];
    complex<float>* C3 = new complex<float>[n * n];

    generate_matrix(A);
    generate_matrix(B);

    cout << "Matrix A: " << endl;
    print_matrix(A);
    cout << endl;
    cout << "Matrix B: " << endl;
    print_matrix(B);
    cout << endl;

    // Перемножение по формуле из линейной алгебры
    clock_t t1 = clock();
    mult1(A, B, C1);
    t1 = clock() - t1;

    cout << "Result (method 1): " << endl;
    print_matrix(C1);
    cout << endl;

    // Перемножение с помощью функции cblas_cgemm из BLAS
    clock_t t2 = clock();
    mult2(A, B, C2);
    t2 = clock() - t2;

    cout << "Result (method 2): " << endl;
    print_matrix(C2);
    cout << endl;

    // Оптимизированное перемножение
    clock_t t3 = clock();
    mult3(A, B, C3);
    t3 = clock() - t3;

    cout << "Result (method 3): " << endl;
    print_matrix(C3);
    cout << endl;

    
    float c = 2.0f * n * n * n;
    double mflops1 = c / (t1 * 1e-6);
    double mflops2 = c / (t2 * 1e-6);
    double mflops3 = c / (t3 * 1e-6);

    cout << "Время затраченное на 1 способ: " << ((float)t1) / CLOCKS_PER_SEC << "s" << endl;
    cout << "MFLOPS первый способ " << mflops1 << endl;
    cout << endl;
    cout << "Время затраченное на 2 способ: " << ((float)t2) / CLOCKS_PER_SEC << "s" << endl;
    cout << "MFLOPS второй способ: " << mflops2 << endl;
    cout << endl;
    cout << "Время затраченное на 3 способ: " << ((float)t3) / CLOCKS_PER_SEC << "s" << endl;
    cout << "MFLOPS третий способ: " << mflops3 << endl;
    cout << endl;

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
