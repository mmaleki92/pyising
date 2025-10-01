// src/fft.hpp
#pragma once
#include <vector>
#include <complex>
#include <cmath>

const double PI = 3.14159265358979323846;

// 1D FFT implementation
void fft(std::vector<std::complex<double>>& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;

    std::vector<std::complex<double>> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(ang), sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

// 2D FFT implementation
void fft2d(std::vector<std::complex<double>>& a, int L, bool invert) {
    // FFT on rows
    for (int i = 0; i < L; ++i) {
        std::vector<std::complex<double>> row(L);
        for (int j = 0; j < L; ++j) {
            row[j] = a[i * L + j];
        }
        fft(row, invert);
        for (int j = 0; j < L; ++j) {
            a[i * L + j] = row[j];
        }
    }

    // FFT on columns
    for (int j = 0; j < L; ++j) {
        std::vector<std::complex<double>> col(L);
        for (int i = 0; i < L; ++i) {
            col[i] = a[i * L + j];
        }
        fft(col, invert);
        for (int i = 0; i < L; ++i) {
            a[i * L + j] = col[i];
        }
    }
}