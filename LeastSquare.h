#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <functional>

#ifndef LEAST_SQUARE_FITTING
#define LEAST_SQUARE_FITTING

#define PRINT_RESULT

// Ref : https://github.com/gon1332/fort320/blob/master/include/Utils/colors.h
#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif  /* _COLORS_ */


extern "C" 
{
	// BLAS subroutine
	void dgemv_(const char* TRANS, const int* M, const int* N, const double* ALPHA, const double* A,
		const int* LDA, const double* X, const int* INCX, const double* BETA, double* Y,
		const int* INCY); 

	double ddot_(const int* N, const double* X, const int* incx, const double* Y, const int* incy);

        void dgemm_(const char* transa, const char* transb, const int* m, const int* n, 
		const int* k, const double* alpha, const double* a, const int* lda, 
		const double* b, const int* ldb, const double* beta, double* c, const int* ldc);


	// LAPACK subroutine
	void dgetrf_(const int* M, const int* N, const double* A, const int* LDA, 
			const int* IPIV, const int* INFO);

	void dgetri_(const int* N, const double* A, const int* LDA, const int* IPIV, 
			double* WORK, const int* LWORK, const int* INFO);

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

		dgetrf_(&N, &N, inv_A, &N, &IPIV[0], &INFO);

		if(INFO != 0)
		{
			std::cout<<FRED(BOLD("ERROR IN DGETRF"))<<std::endl;
			std::abort();
		}

		LWORK = -1;

		dgetri_(&N, inv_A, &N, &IPIV[0], &temp_WORK, &LWORK, &INFO);

		LWORK = temp_WORK;
		WORK.resize(LWORK);

		dgetri_(&N, inv_A, &N, &IPIV[0], &WORK[0], &LWORK, &INFO);

		if(INFO != 0)
		{
			std::cout<<FRED(BOLD("ERROR IN DGETRI"))<<std::endl;
			std::abort();
		}
	}
}

} // namespace lapack

namespace LeastSquare
{

template <typename Tx>
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
		yi = new double [NumDomain];
	}

	virtual ~BaseModel()
	{
		if(xi != nullptr) delete [] xi;
		if(yi != nullptr) delete [] yi;
	}

	enum dataList{ num_domain, num_parameter};
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
			default:
				std::cout<<FRED(BOLD("error! check your arguments"))<<std::endl;
				std::abort();
		}

		return result;
	}

	double cost(const double* parameter) const
	{
		double accum = 0;
		for(int i=0;i<NumDomain;++i)
			accum += std::abs(yi[i] - modelFunc(parameter, i));
		return accum;
	}

	double yDelFunc(const double* parameter, const int Index) const
	{
		return yi[Index] - modelFunc(parameter, Index);
	}

	virtual void inData(const std::vector<Tx>& v_xi, const std::vector<double>& v_yi)
	{
		assert(v_xi.size() == NumDomain*NumDimension);
		assert(v_yi.size() == NumDomain);

		std::memcpy(xi, &v_xi[0], sizeof(Tx)*NumDomain*NumDimension);
		std::memcpy(yi, &v_yi[0], sizeof(double)*NumDomain);
	}

	virtual void get_jacobian(const double* parameter, double* J) const
	{
		const double h = 1e-5;
		std::vector<double> p_pdh(NumParameter);
		std::memcpy(&p_pdh[0], parameter, sizeof(double)*NumParameter);
		std::vector<double> p_mdh(NumParameter);
		std::memcpy(&p_mdh[0], parameter, sizeof(double)*NumParameter);

		for(int i=0;i<NumDomain;++i)
		{
			for(int j=0;j<NumParameter;++j)
			{
				p_pdh[j] += h;
				p_mdh[j] -= h;

				J[i*NumParameter + j] = (this -> modelFunc(&p_pdh[0],i) - this -> modelFunc(&p_mdh[0],i))/(2.*h);

				p_pdh[j] -= h;
				p_mdh[j] += h;
			}
		}
	}

	virtual double modelFunc(const double* parameter, const int Index) const = 0;

protected:
	const int NumDomain;
	const int NumParameter;
	const int NumDimension;
	Tx* xi = nullptr;
	double* yi = nullptr;
};


template <typename Tx = double>
class FunctionalModel : public BaseModel<Tx>
{
        using BaseModel<Tx>::xi;
        using LeastSquare::BaseModel<Tx>::NumDimension;

public:
	FunctionalModel(const std::function<double(const double*, const Tx*)> model_f_,
	const int NumDomain_, const int NumParameter_, const int NumDimension_ = 1)
	: BaseModel<Tx>(NumDomain_, NumParameter_, NumDimension_), model_f(model_f_),
	_xi_copy(new Tx [NumDimension_])
	{}

	~FunctionalModel()
	{
		if(_xi_copy != nullptr) delete [] _xi_copy;
	}

	virtual double modelFunc(const double* parameter, const int Index) const
	{
		std::memcpy(_xi_copy, &xi[NumDimension*Index], sizeof(Tx)*NumDimension);
		return model_f(parameter, _xi_copy);
	}

private:
	Tx* _xi_copy = nullptr;
	std::function<double(const double*, const Tx*)> model_f;
};


void loadtxt(const char fileName[], const int numDimX, 
	std::vector<double>& xi, std::vector<double>& yi)
{
	std::vector<double> xi_, yi_;
	double temp;
	std::ifstream file(fileName);

	assert(file.is_open());

	while(!file.eof())
	{
		for(int i=0;i<numDimX;++i)
		{
			file >> temp;
			xi_.push_back(temp);
		}
		file >> temp;
		yi_.push_back(temp);
	}

	const int length = yi_.size() - 1;

	xi_.erase(xi_.begin() + numDimX*length, xi_.begin() + numDimX*(length+1));
	yi_.erase(yi_.begin() + length, yi_.begin() + length + 1);

	xi_.swap(xi);
	yi_.swap(yi);

	file.close();
}


template<typename Tx>
void Levenberg_Marquardt(const BaseModel<Tx>& model, double* parameter, const int Iter = (int)1e4, const double Tol = 1e-4)
{
	const char TRANS = 'N', CMODE = 'T';
	const double ALPHA = 1., BETA = 0, nu = 2.;
	const int INC = 1, 
		   Np = model.getData(BaseModel<Tx>::dataList::num_parameter),
		   Nd = model.getData(BaseModel<Tx>::dataList::num_domain); 
	double lambda = 1., before = 0, after = 0, initial = 0;
	int iter = 0;
	std::vector<double> p(Np), dp(Np), r_array(Nd), Jacobian(Nd*Np), JT_J(Np*Np),
			p_before(Np), p_after(Np), lhs(Np*Np), rhs(Np), 
			inv_lhs(Np*Np), i_lhs_rhs(Np);


	std::memcpy(&p[0], parameter, sizeof(double)*Np);

	initial = model.cost(&p[0]);

	while(iter < Iter)
	{
		iter += 1;
	
		//std::cout<<"(iter:"<<iter<<", cost:"<<initial<<")"<<std::endl;

		model.get_jacobian(&p[0] ,&Jacobian[0]);

        	dgemm_(&TRANS, &CMODE, &Np, &Np, &Nd, &ALPHA, &Jacobian[0], &Np, 
		&Jacobian[0], &Np, &BETA, &JT_J[0], &Np);

		for(int i=0;i<Nd;i++) r_array[i] = model.yDelFunc(&p[0], i);

		std::memcpy(&lhs[0], &JT_J[0], sizeof(double)*Np*Np);
		for(int i=0;i<Np;++i) lhs[i*Np + i] = JT_J[i*Np + i]*(1. + lambda);
		dgemv_(&TRANS, &Np, &Nd, &ALPHA, &Jacobian[0], &Np, &r_array[0], 
		&INC, &BETA, &rhs[0], &INC);
		
		lapack::invSymMatrix(Np, &lhs[0], &inv_lhs[0]);

		dgemv_(&CMODE, &Np, &Np, &ALPHA, &inv_lhs[0], &Np, &rhs[0], 
		&INC, &BETA, &i_lhs_rhs[0], &INC);

		for(int i=0;i<Np;++i)
			p_before[i] = p[i] + i_lhs_rhs[i];
		before = model.cost(&p_before[0]);

		for(int i=0;i<Np;++i) lhs[i*Np + i] = JT_J[i*Np + i]*(1 + lambda/nu);

		lapack::invSymMatrix(Np, &lhs[0], &inv_lhs[0]);
		
		dgemv_(&CMODE, &Np, &Np, &ALPHA, &inv_lhs[0], &Np, &rhs[0], 
		&INC, &BETA, &i_lhs_rhs[0], &INC);

		for(int i=0;i<Np;++i)
			p_after[i] = p[i] + i_lhs_rhs[i];
		after = model.cost(&p_after[0]);


		if(before >= initial and after >= initial)
		{
			lambda *= nu;
			if(lambda > 1e30)
			{
#ifdef PRINT_RESULT
				std::cout<<BOLD("lambda is too big!");
#endif
				break;
			}
		}
		else if(before > after ) 
		{
			lambda /= nu;
			initial = after;
			for(int ip=0;ip<Np;++ip)
				dp[ip] = p[ip] - p_after[ip];
			if(std::sqrt(ddot_(&Np, &dp[0], &INC, &dp[0], &INC)) < Tol)
			{
				std::memcpy(&p[0], &p_after[0], sizeof(double)*Np);
#ifdef PRINT_RESULT
				std::cout<<FGRN(BOLD("converge!"));
#endif

				break;
			}
			else
				std::memcpy(&p[0], &p_after[0], sizeof(double)*Np);
		}
		else
		{
			initial = before;
			for(int ip=0;ip<Np;++ip)
				dp[ip] = p[ip] - p_before[ip];
			if(std::sqrt(ddot_(&Np, &dp[0], &INC, &dp[0], &INC)) < Tol)
			{
				std::memcpy(&p[0], &p_before[0], sizeof(double)*Np);
#ifdef PRINT_RESULT
				std::cout<<FGRN(BOLD("converge!"));
#endif
				break;
			}
			else
				std::memcpy(&p[0], &p_before[0], sizeof(double)*Np);
		}
	}

	std::memcpy(parameter, &p[0], sizeof(double)*Np);

#ifdef PRINT_RESULT
	std::cout<<"(iter:"<<iter<<", cost:"<<initial<<")"<<std::endl;

	for(int i=0;i<Np;i++) 
		std::cout<<FCYN("p[")<<i<<FCYN("] = ")<<p[i]<<"  ";
	std::cout<<std::endl;
#endif
}

} // namespace LeastSquare

#endif
