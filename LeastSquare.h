#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>

#ifndef LEAST_SQUARE_FITTING
#define LEAST_SQUARE_FITTING

extern "C" 
{
	// BLAS subroutine
	void dcopy_(const int* N, const double* x, const int* incx, double* y, const int* incy);

	void dscal_(const int* N, const double* ALPHA, double* x, const int* INC);

	void dsyrk_(const char* UPLO, const char* TRANS, const int* N, const int* K, 
		const double* ALPHA, const double* A, const int* LDA, const double* BETA,
		double* C, const int* LDC);

	void dgemv_(const char* TRANS, const int* M, const int* N, const double* ALPHA, const double* A,
		const int* LDA, const double* X, const int* INCX, const double* BETA, double* Y,
		const int* INCY); 

	double ddot_(const int* N, const double* X, const int* incx, const double* Y, const int* incy);

	void dsymv_(const char*, const int*, const double* ALPHA, const double* A,
		const int*, const double* X, const int*, const double* BETA,
		double* Y,const int* );

	// LAPACK subroutine
	void dsytrf_(const char* UPLO, const int* N, double* A, const int* LDA, int* IPIV,
		double* WORK, const int* LWORK, int* INFO);

	void dsytri_(const char* UPLO, const int* N, double* A, const int* LDA, const int* IPIV,
		double* WORK, int* INFO);
}

namespace lapack
{

inline void invSymMatrix(const int N, const double* A, double* inv_A, const char UPLO='U')
{
	if(N == 1)
		inv_A[0] = 1./A[0];
	else if(N == 2)
	{
		const double det = A[0]*A[3] - A[1]*A[2];
		inv_A[0] =  1./det*A[3];
		inv_A[1] = -1./det*A[1];
		inv_A[2] = -1./det*A[2];
		inv_A[3] =  1./det*A[3];
	}
	else if(N == 3)
	{
                const double det=1.0/(A[0]*(A[8]*A[4]-A[5]*A[7])
                                -A[1]*(A[8]*A[3]-A[5]*A[6])
                                +A[2]*(A[7]*A[3]-A[4]*A[6]));
                inv_A[0]=(A[8]*A[4]-A[5]*A[7])*det;
                inv_A[1]=-(A[8]*A[1]-A[2]*A[7])*det;
                inv_A[2]=(A[5]*A[1]-A[2]*A[4])*det;
                inv_A[3]=-(A[8]*A[3]-A[5]*A[6])*det;
                inv_A[4]=(A[8]*A[0]-A[2]*A[6])*det;
                inv_A[5]=-(A[5]*A[0]-A[2]*A[3])*det;
                inv_A[6]=(A[7]*A[3]-A[4]*A[6])*det;
                inv_A[7]=-(A[7]*A[0]-A[1]*A[6])*det;
                inv_A[8]=(A[4]*A[0]-A[1]*A[3])*det;
	}
	else
	{
		std::vector<int> IPIV(N);
		int INFO, LWORK;
		std::vector<double> WORK;
		double temp_WORK;

		std::memcpy(inv_A, A, sizeof(double)*N*N);

		LWORK = -1;

		// DSYTRF(1)
		dsytrf_(&UPLO, &N, inv_A, &N, &IPIV[0], &temp_WORK, &LWORK, &INFO);

		if(INFO != 0)
		{
			std::cout<<"ERROR IN DSYTRF(1)"<<std::endl;
			std::abort();
		}

		LWORK = (int)temp_WORK;

		WORK.resize(LWORK);

		// DSYTRF(2)
		dsytrf_(&UPLO, &N, inv_A, &N, &IPIV[0], &WORK[0], &LWORK, &INFO);

		if(INFO != 0)
		{
			std::cout<<"ERROR IN DSYTRF(2)"<<std::endl;
			std::abort();
		}

		std::vector<double>().swap(WORK);
		WORK.resize(N);

		dsytri_(&UPLO, &N, inv_A, &N, &IPIV[0], &WORK[0], &INFO);

		if(INFO != 0)
		{
			std::cout<<"ERROR IN DSYTRI"<<std::endl;
			std::abort();
		}
	}
}

} // namespace lapack

namespace LeastSquare
{

template <typename Tx, typename Ty>
class BaseModel
{
public:
	BaseModel(const int NumDomain_, const int NumParameter_, const int NumDimension_ = 1)
	: NumDomain(NumDomain_), NumParameter(NumParameter_), NumDimension(NumDimension_) 
	{
		assert(NumDomain > 0);
		assert(NumParameter > 0);
		assert(NumDimension > 0);
		xi = new Tx [NumDomain * NumDimension];
		yi = new Ty [NumDomain * NumDimension];
	}

	enum dataList{ num_domain, num_parameter, num_dimension };
	int getData(const dataList Data) const
	{
		int result;
		switch(Data)
		{
			case num_domain:
				result = NumDomain;
				break;
			case num_parameter:
				result = NumParameter;
				break;
			case num_dimension:
				result = NumDimension;
				break;
			default:
				std::cout<<"error! check your arguments"<<std::endl;
				std::abort();
		}

		return result;
	}

	virtual ~BaseModel()
	{
		if(xi != nullptr) delete [] xi;
		if(yi != nullptr) delete [] yi;
	}

	virtual void inData(const std::vector<Tx>& v_xi, const std::vector<Ty>& v_yi)
	{
		assert(v_xi.size() == NumDomain*NumDimension);
		assert(v_yi.size() == NumDomain*NumDimension);

		std::memcpy(xi, &v_xi[0], sizeof(Tx)*NumDomain*NumDimension);
		std::memcpy(yi, &v_yi[0], sizeof(Ty)*NumDomain*NumDimension);
	}

	virtual double cost(const double* parameter) const = 0;

	virtual double cost(const double* parameter, const int Index) const = 0;

	virtual void get_jacobian(const double* parameter, double* jacobian) const = 0;

protected:
	const int NumDomain;
	const int NumParameter;
	const int NumDimension;
	Tx* xi = nullptr;
	Ty* yi = nullptr;
};


template<typename Tx, typename Ty>
void Levenberg_Marquardt(const BaseModel<Tx,Ty>* model, double* parameter, const int Iter = (int)1e4, const double Tol = 1e-4)
{
	const char UPLO='U', TRANS='N';
	const double ALPHA = 1., mALPHA = -1., BETA = 0, oBETA = 1., nu = 10.;
	const int INC = 1, 
		   Np = model -> getData(BaseModel<Tx,Ty>::dataList::num_parameter),
		   Nd = model -> getData(BaseModel<Tx,Ty>::dataList::num_domain) * 
		        model -> getData(BaseModel<Tx,Ty>::dataList::num_dimension);
	double lambda = 1., temp = 0, before = 0, after = 0, initial = 0;
	int iter = 0;
	std::vector<double> p(Np), grad(Np), r_array(Nd), Jacobian(Nd*Np), JT_J(Np*Np),
			tot_M(Np*Np), inv_M(Np*Np), p_before(Np), p_after(Np);


	std::memcpy(&p[0], parameter, sizeof(double)*Np);
	std::memcpy(&p_before[0], parameter, sizeof(double)*Np);
	std::memcpy(&p_after[0], parameter, sizeof(double)*Np);

	initial = model -> cost(&p[0]);

	while(iter < Iter)
	{
		if( lambda > 1e20) 
		{
			std::cout<<"lambda is too huge."<<
			"(iter:"<<iter<<",cost:"<<initial<<")"<<std::endl;
			break;
		}
		/*
		std::cout<<iter<<"'th step."<<std::endl;
		for(int i=0;i<Np;i++) 
			std::cout<<"p["<<i<<"]="<<p[i]<<std::endl;
		*/
		iter += 1;

		model -> get_jacobian(&p[0] ,&Jacobian[0]);
		for(int i=0;i<Nd;i++) r_array[i] = model -> cost(&p[0], i);

		dgemv_(&TRANS, &Np, &Nd, &ALPHA, &Jacobian[0], &Np, &r_array[0], &INC, &BETA, &grad[0], &INC);
		temp = ddot_(&Np, &grad[0], &INC, &grad[0], &INC);
		temp = std::sqrt(temp);

		if( temp < Tol)
		{
			std::cout<<"converge! (0.5*|f(x) - y|^2 : "<<temp<<")"<<std::endl;
			break;
		}

		dsyrk_(&UPLO, &TRANS, &Np, &Nd, &ALPHA, &Jacobian[0], &Np, &BETA, &JT_J[0], &Np);


		for(int i=0;i<Np*Np;i++) tot_M[i] = JT_J[i];
		for(int i=0;i<Np;i++) tot_M[i*Np + i] += lambda*JT_J[Np*i + i];

		lapack::invSymMatrix(Np, &tot_M[0], &inv_M[0], UPLO);

		dsymv_(&UPLO, &Np, &mALPHA, &inv_M[0], &Np, &grad[0], &INC, &oBETA, &p_before[0], &INC);
 
		before = model -> cost(&p_before[0]);


		for(int i=0;i<Np*Np;i++) tot_M[i] = JT_J[i];
		for(int i=0;i<Np;i++) tot_M[i*Np + i] += lambda/nu*JT_J[Np*i + i];

		lapack::invSymMatrix(Np, &tot_M[0], &inv_M[0], UPLO);

		dsymv_(&UPLO, &Np, &mALPHA, &inv_M[0], &Np, &grad[0], &INC, &oBETA, &p_after[0], &INC);

		after = model -> cost(&p_after[0]);


		if(before >= initial and after >= initial)
		{
			std::memcpy(&p_before[0], &p[0], sizeof(double)*Np);
			std::memcpy(&p_after[0], &p[0], sizeof(double)*Np);
			lambda *= nu;
		}
		else if(before < initial and after > initial) 
		{
			std::memcpy(&p[0], &p_before[0], sizeof(double)*Np);
			std::memcpy(&p_after[0], &p_before[0], sizeof(double)*Np);
			initial = before;
		}
		else if(before > initial and after < initial) 
		{
			std::memcpy(&p[0], &p_after[0], sizeof(double)*Np);
			std::memcpy(&p_before[0], &p_after[0], sizeof(double)*Np);
			lambda /= nu;
			initial = after;
		}
		else if(before < initial and after < initial) 
		{
			if(before > after) 
			{
				std::memcpy(&p[0], &p_after[0], sizeof(double)*Np);
				std::memcpy(&p_before[0], &p_after[0], sizeof(double)*Np);
				lambda /= nu;
				initial = after;
			}
			else 
			{
				std::memcpy(&p[0], &p_before[0], sizeof(double)*Np);
				std::memcpy(&p_after[0], &p_before[0], sizeof(double)*Np);
				initial = before;
			}
		}
	}

	std::memcpy(parameter, &p[0], sizeof(double)*Np);

	for(int i=0;i<Np;i++) 
		std::cout<<"p["<<i<<"]="<<p[i]<<" ";
	std::cout<<std::endl;
}

} // namespace LeastSquare

#endif
