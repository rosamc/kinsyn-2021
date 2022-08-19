
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "pos_stp_fromGRF.h"

using namespace std;
using namespace Eigen;
namespace py=pybind11;

typedef double T;
typedef Eigen::Matrix< T, Eigen::Dynamic, 1 > VectorXd;
typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
typedef Eigen::SparseMatrix<T> SM;

void pre_laplacianpars(py::array_t<double> parsar, SM& L , std::vector<Eigen::Triplet<T>>& LxA, std::vector<Eigen::Triplet<T>>& LxB ){
//laplacian matrix with only parameter values, no variables. Also no values for those labels that depend upon variables.

    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;
    int n=12;

    T a1_0_1=pars[0];
    T k_1_0=pars[1];
    T kr_1_0=pars[2];
    T a1_0_2=pars[3];
    T k_2_0=pars[4];
    T a1_0_3=pars[5];
    T k_3_0=pars[6];
    T b1_1_1=pars[7];
    T k_1_1=pars[8];
    T kr_1_1=pars[9];
    T b1_1_2=pars[10];
    T k_2_1=pars[11];
    T b1_1_3=pars[12];
    T k_3_1=pars[13];
    T a2_0_1=pars[14];
    T a2_0_2=pars[15];
    T a2_0_3=pars[16];
    T b2_2_1=pars[17];
    T k_1_2=pars[18];
    T kr_1_2=pars[19];
    T b2_2_2=pars[20];
    T k_2_2=pars[21];
    T b2_2_3=pars[22];
    T k_3_2=pars[23];
    T a2_1_1=pars[24];
    T a2_1_2=pars[25];
    T a2_1_3=pars[26];
    T b2_1U2_1=pars[27];
    T k_1_1U2=pars[28];
    T kr_1_1U2=pars[29];
    T b2_1U2_2=pars[30];
    T k_2_1U2=pars[31];
    T b2_1U2_3=pars[32];
    T k_3_1U2=pars[33];
    T a1_2_1=pars[34];
    T a1_2_2=pars[35];
    T a1_2_3=pars[36];
    T b1_1U2_1=pars[37];
    T b1_1U2_2=pars[38];
    T b1_1U2_3=pars[39];
    std::vector<Eigen::Triplet<T>> clist;
    clist.push_back(Eigen::Triplet<T>(0,1,b1_1_1));
    clist.push_back(Eigen::Triplet<T>(0,2,b2_2_1));
    clist.push_back(Eigen::Triplet<T>(0,4,kr_1_0));
    clist.push_back(Eigen::Triplet<T>(0,8,k_3_0));
    clist.push_back(Eigen::Triplet<T>(1,3,b2_1U2_1));
    clist.push_back(Eigen::Triplet<T>(1,5,kr_1_1));
    clist.push_back(Eigen::Triplet<T>(1,9,k_3_1));
    clist.push_back(Eigen::Triplet<T>(2,3,b1_1U2_1));
    clist.push_back(Eigen::Triplet<T>(2,6,kr_1_2));
    clist.push_back(Eigen::Triplet<T>(2,10,k_3_2));
    clist.push_back(Eigen::Triplet<T>(3,7,kr_1_1U2));
    clist.push_back(Eigen::Triplet<T>(3,11,k_3_1U2));
    clist.push_back(Eigen::Triplet<T>(4,0,k_1_0));
    clist.push_back(Eigen::Triplet<T>(4,5,b1_1_2));
    clist.push_back(Eigen::Triplet<T>(4,6,b2_2_2));
    clist.push_back(Eigen::Triplet<T>(5,1,k_1_1));
    clist.push_back(Eigen::Triplet<T>(5,7,b2_1U2_2));
    clist.push_back(Eigen::Triplet<T>(6,2,k_1_2));
    clist.push_back(Eigen::Triplet<T>(6,7,b1_1U2_2));
    clist.push_back(Eigen::Triplet<T>(7,3,k_1_1U2));
    clist.push_back(Eigen::Triplet<T>(8,4,k_2_0));
    clist.push_back(Eigen::Triplet<T>(8,9,b1_1_3));
    clist.push_back(Eigen::Triplet<T>(8,10,b2_2_3));
    clist.push_back(Eigen::Triplet<T>(9,5,k_2_1));
    clist.push_back(Eigen::Triplet<T>(9,11,b2_1U2_3));
    clist.push_back(Eigen::Triplet<T>(10,6,k_2_2));
    clist.push_back(Eigen::Triplet<T>(10,11,b1_1U2_3));
    clist.push_back(Eigen::Triplet<T>(11,7,k_2_1U2));

    Eigen::Triplet<double> trd;
    for (int j=0;j<clist.size();j++){
        trd=clist[j];
        L.insert(trd.row(),trd.col())=trd.value();
    }
    L.makeCompressed();
    LxA.push_back(Eigen::Triplet<T>(1,0,a1_0_1));
    LxA.push_back(Eigen::Triplet<T>(3,2,a1_2_1));
    LxA.push_back(Eigen::Triplet<T>(5,4,a1_0_2));
    LxA.push_back(Eigen::Triplet<T>(7,6,a1_2_2));
    LxA.push_back(Eigen::Triplet<T>(9,8,a1_0_3));
    LxA.push_back(Eigen::Triplet<T>(11,10,a1_2_3));
    LxB.push_back(Eigen::Triplet<T>(2,0,a2_0_1));
    LxB.push_back(Eigen::Triplet<T>(3,1,a2_1_1));
    LxB.push_back(Eigen::Triplet<T>(6,4,a2_0_2));
    LxB.push_back(Eigen::Triplet<T>(7,5,a2_1_2));
    LxB.push_back(Eigen::Triplet<T>(10,8,a2_0_3));
    LxB.push_back(Eigen::Triplet<T>(11,9,a2_1_3));
    return;
}

double interfacess(py::array_t<double> parsar, py::array_t<double> varvals, bool doublecheck, string method){
    const int n=12;
    SM L(n,n);
    L.reserve(VectorXi::Constant(n,3));    auto varsbuf=varvals.request();
    double *vars=(double *) varsbuf.ptr;
    double Aval=vars[0];
    std::vector<Eigen::Triplet<T>> LxA;
    double Bval=vars[1];
    std::vector<Eigen::Triplet<T>> LxB;
    pre_laplacianpars(parsar,L , LxA, LxB);
    insert_L_Lx_atval(L, LxA, Aval);
    insert_L_Lx_atval(L, LxB, Bval);

    T cs;
    for (int k=0; k<L.outerSize(); ++k) {
        cs=0;
        for(typename Eigen::SparseMatrix<T>::InnerIterator it (L,k); it; ++it){
            cs+=it.value();
        }
        L.insert(k,k)=-cs;
        }
        double ssval;
    vector<int>indicesC={8,9,10,11};
    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;
    vector<double>coeffsC={pars[6],pars[13],pars[23],pars[33]};
    if (method=="svd"){
    MatrixXd Ld=MatrixXd(L);
    ssval=ssfromnullspace(Ld,indicesC,coeffsC,doublecheck);
    }else{
    ssval=ssfromnullspace(L,indicesC,coeffsC,doublecheck);
    }
    
    return  ssval;
    }

py::array_t<double> interfacerhos(py::array_t<double> parsar, py::array_t<double> varvals, bool doublecheck, string method){
    const int n=12;
    SM L(n,n);
    L.reserve(VectorXi::Constant(n,3));    auto varsbuf=varvals.request();
    double *vars=(double *) varsbuf.ptr;
    double Aval=vars[0];
    std::vector<Eigen::Triplet<T>> LxA;
    double Bval=vars[1];
    std::vector<Eigen::Triplet<T>> LxB;
    pre_laplacianpars(parsar,L, LxA, LxB);
    insert_L_Lx_atval(L,LxA,Aval);
    insert_L_Lx_atval(L,LxB,Bval);

    T cs;
    for (int k=0; k<L.outerSize(); ++k) {
         cs=0;
            for(typename Eigen::SparseMatrix<T>::InnerIterator it (L,k); it; ++it){
                cs+=it.value();
            }
            L.insert(k,k)=-cs;
            }
        VectorXd N;
    N.resize(n,1);
    int i;
    if (method=="svd"){
    MatrixXd Ld=MatrixXd(L);
    nullspace(Ld,N,doublecheck);
    }else{
    nullspace(L,N,doublecheck);
    }
    py::array_t<double> resultpy = py::array_t<double>(n);
    py::buffer_info bufresultpy = resultpy.request();
    double *ptrresultpy=(double *) bufresultpy.ptr;
    for (i=0;i<n;i++){
        ptrresultpy[i]=N[i];
    }

    return  resultpy;
    }

PYBIND11_MODULE(bPcycle_2bs_3states_rev1_A_B_ssonly,m){

      m.def("interfacess", &interfacess, "A function which returns ss. Method should be qr (done on sparse matrix) or svd (done on dense matrix).",
            py::arg("parsar"), py::arg("varvals"),py::arg("doublecheck")=false, py::arg("method")="qr");

    m.def("interfacerhos", &interfacerhos, "A function which returns normalised nullspace (sums to 1 already). Method should be qr (done on sparse matrix) or svd (done on dense matrix).",
            py::arg("parsar"), py::arg("varvals"),py::arg("doublecheck")=false, py::arg("method")="qr");
    }

