#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//device parameter//
#define  ndim	    2
#define  ip         2
#define  kp         3
#define  DPN        3  //device per note
#define  stepall    100
#define  iprint     10
#define  idata_3d   100000
//droplet parameter//
#define  nx         120
#define  ny         120
#define  nz         120
#define  radd       30.0 //1不用調
#define  thick      5.0
#define  tau_h      0.5
#define  tau_l      0.05
#define  tau_g      0.5
#define  rho_l      1.0
#define  rho_g      0.001
#define  sigma      0.001
#define  bo         100.0

//0 one bubble rising ,1 two bubble rising
#define  condition  0

//condition 1
#define  distance_x 0.0
#define  distance_z 10.0
#define  radd_t     30.0
#define  radd_b     30.0

//constant parameter//
#define  thita      10
#define  dx         1.0
#define  dt         1.0
#define  q          19

__constant__ double eex[q];
__constant__ double eey[q];
__constant__ double eez[q];
__constant__ double wwt[q];
__constant__ int    eet[q];

void parameter (double *beta,double *zeta,double *mobi,double *kappa,double *phic,double *gravity,double *ex_h,double *ey_h,double *ez_h,double *wt_h,int *et_h)
{
	*zeta =(double)thick*dx;
	*beta =(double)12.0*sigma/(*zeta);
	*kappa=(double)(*beta)*(*zeta)*(*zeta)/8.0;
	*mobi =(double)0.02/(*beta);
	double omega=-cos(thita*M_PI/180.0);
	*phic =omega*pow(2.0*(*kappa)*(*beta),0.5);
	*gravity = bo*sigma/(rho_l-rho_g)/(2*radd)/(2*radd);
	//ex
	ex_h[ 0]= 0.0;
	ex_h[ 1]= 1.0;
	ex_h[ 2]=-1.0;
	ex_h[ 3]= 0.0;
	ex_h[ 4]= 0.0;
	ex_h[ 5]= 0.0;
	ex_h[ 6]= 0.0;
	ex_h[ 7]= 1.0;
	ex_h[ 8]=-1.0;
	ex_h[ 9]= 1.0;
	ex_h[10]=-1.0;
	ex_h[11]= 1.0;
	ex_h[12]=-1.0;
	ex_h[13]=-1.0;
	ex_h[14]= 1.0;
	ex_h[15]= 0.0;
	ex_h[16]= 0.0;
	ex_h[17]= 0.0;
	ex_h[18]= 0.0;
	//ey
	ey_h[ 0]= 0.0;
	ey_h[ 1]= 0.0;
	ey_h[ 2]= 0.0;
	ey_h[ 3]= 1.0;
	ey_h[ 4]=-1.0;
	ey_h[ 5]= 0.0;
	ey_h[ 6]= 0.0;
	ey_h[ 7]= 1.0;
	ey_h[ 8]=-1.0;
	ey_h[ 9]=-1.0;
	ey_h[10]= 1.0;
	ey_h[11]= 0.0;
	ey_h[12]= 0.0;
	ey_h[13]= 0.0;
	ey_h[14]= 0.0;
	ey_h[15]= 1.0;
	ey_h[16]=-1.0;
	ey_h[17]= 1.0;
	ey_h[18]=-1.0;
	//ez
	ez_h[ 0]= 0.0;
	ez_h[ 1]= 0.0;
	ez_h[ 2]= 0.0;
	ez_h[ 3]= 0.0;
	ez_h[ 4]= 0.0;
	ez_h[ 5]= 1.0;
	ez_h[ 6]=-1.0;
	ez_h[ 7]= 0.0;
	ez_h[ 8]= 0.0;
	ez_h[ 9]= 0.0;
	ez_h[10]= 0.0;
	ez_h[11]= 1.0;
	ez_h[12]=-1.0;
	ez_h[13]= 1.0;
	ez_h[14]=-1.0;
	ez_h[15]= 1.0;
	ez_h[16]=-1.0;
	ez_h[17]=-1.0;
	ez_h[18]= 1.0;
	//wt
	wt_h[ 0]=1.0/ 3.0;
	wt_h[ 1]=1.0/18.0;
	wt_h[ 2]=1.0/18.0;
	wt_h[ 3]=1.0/18.0;
	wt_h[ 4]=1.0/18.0;
	wt_h[ 5]=1.0/18.0;
	wt_h[ 6]=1.0/18.0;
	wt_h[ 7]=1.0/36.0;
	wt_h[ 8]=1.0/36.0;
	wt_h[ 9]=1.0/36.0;
	wt_h[10]=1.0/36.0;
	wt_h[11]=1.0/36.0;
	wt_h[12]=1.0/36.0;
	wt_h[13]=1.0/36.0;
	wt_h[14]=1.0/36.0;
	wt_h[15]=1.0/36.0;
	wt_h[16]=1.0/36.0;
	wt_h[17]=1.0/36.0;
	wt_h[18]=1.0/36.0;
	int l;
	for(l=0;l<q;l++)
	{
	et_h[l]=(nx/ip+4)*((ny+4)*(int)ez_h[l]+(int)ey_h[l])+(int)ex_h[l];
	}
}

void initial_macro (double *c,double *m,double *b,double *p,double *u,double *v,double *w)
{
	int i,j,k,index;
	double icent,jcent,kcent;
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<nz;k++){
	index=nx*(k*ny+j)+i;
	c[index]=0.0;
	m[index]=0.0;
	b[index]=0.0;
	p[index]=0.0;
	u[index]=0.0;
	v[index]=0.0;
	w[index]=0.0;
	}}}
	
	icent=(double)(nx-1.0)/2.0;
	jcent=(double)(ny-1.0)/2.0;
	kcent=(double)(nz-1.0)/2.0;
	
	if(condition==1){
	double icent_r=icent+0.5*distance_x;
	double icent_l=icent-0.5*distance_x;
	double kcent_b=50;
	double kcent_t=kcent_b+thick+(radd_t+radd_b)+distance_z;
	int    mid    =0.5*(distance_z+thick)+50+radd_b;
	double raddd=radd+thick/2.0+1.0;
	
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<mid;k++){
	double rad=sqrt( (i-icent_l)*(i-icent_l)+(j-jcent)*(j-jcent)+(k-kcent_b)*(k-kcent_b));
	index=nx*(k*ny+j)+i;
	c[index]=(double)0.5-(double)0.5*tanh(2.0*(radd_b-rad)/thick);
	}}}
	
 	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=mid;k<nz;k++){
	double rad=sqrt( (i-icent_r)*(i-icent_r)+(j-jcent)*(j-jcent)+(k-kcent_t)*(k-kcent_t));
	index=nx*(k*ny+j)+i;
	c[index]=(double)0.5-(double)0.5*tanh(2.0*(radd_t-rad)/thick);
	}}}
	}
	
	else{
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<nz;k++){
	double rad=sqrt( (i-icent)*(i-icent)+(j-jcent)*(j-jcent)+(k-kcent)*(k-kcent));
	index=nx*(k*ny+j)+i;
	c[index]=(double)0.5-(double)0.5*tanh(2.0*(radd-rad)/thick);
	}}}}
}

void array_2D_do (double *phi,double *phi_do)
{
	int i,j,k,index;
	int ii,jj,kk,iindex;
	int iside;
	int xd=nx/ip;
	jj=-1;
	kk=0;
	iside=0;
	
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	ii=i%xd;
	if(ii == 0){
	jj=jj+1;
	}
	if(jj == ny){
	kk=kk+1;
	jj=0;
	}
	if(kk == nz){
	iside=iside+1;
	kk=0;
	}
	
	ii=ii+xd*iside;
	iindex=nx*(kk*ny+jj)+ii;
	phi_do[index]=phi[iindex];
}
}
}
}

void array_2D_undo (double *phi,double *phi_do)
{
	int i,j,k,index;
	int ii,jj,kk,iindex;
	int iside;
	int xd=nx/ip;
	jj=-1;
	kk=0;
	iside=0;
	
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	ii=i%xd;
	if(ii == 0){
	jj=jj+1;
	}
	if(jj == ny){
	kk=kk+1;
	jj=0;
	}
	if(kk == nz){
	iside=iside+1;
	kk=0;
	}
	
	ii=ii+xd*iside;
	iindex=nx*(kk*ny+jj)+ii;
	phi[iindex]=phi_do[index];
}
}
}
}

void array_1D_undo (double *phi,double *phi_do)
{
	int i,k,index;
	int ii,kk,iindex;
	int iside;
	int xd=nx/ip;
	kk=0;
	iside=0;
	
	for(k=0;k<nz;k++){
	for(i=0;i<nx;i++){
	index=nx*k+i;
	ii=i%xd;
	if(ii == 0){
	kk=kk+1;
	}
	if(kk == nz){
	iside=iside+1;
	kk=0;
	}
	
	ii=ii+xd*iside;
	iindex=nx*kk+ii;
	phi[iindex]=phi_do[index];
}
}
}

__device__ int index_3d (int i, int j,int k)
{
	int ans=(nx/ip+4)*((ny+4)*k+j)+i;
	return ans;
}

__device__ int index_3d_x (int i, int j,int k)
{
	int ans=(ny+4)*((nz/kp+4)*i+k)+j;
	return ans;
}

__device__ int index_4d (int i, int j,int k,int l)
{
	int ans=(nx/ip+4)*((ny+4)*((nz/kp+4)*l+k)+j)+i;
	return ans;
}

__global__ void array_do( double *phi_d, double *phi)
{
	int ii=threadIdx.x;
	int jj= blockIdx.x%ny;
	int kk= blockIdx.x/ny;
	int iindex	=(nx/ip)*(kk*ny+jj)+ii;
	
	int i=threadIdx.x+2;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+2;
	int index=index_3d(i,j,k);
	phi[index]=phi_d[iindex];
}
__global__ void array_undo( double *phi_d, double *phi)
{
	int ii=threadIdx.x;
	int jj= blockIdx.x%ny;
	int kk= blockIdx.x/ny;
	int iindex	=(nx/ip)*(kk*ny+jj)+ii;
	
	int i=threadIdx.x+2;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+2;
	int index=index_3d(i,j,k);
	phi_d[iindex]=phi[index];
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                boundary                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void boundary_ym( double *phi)
{
	int i= blockIdx.x;
	int k=threadIdx.x;
	int distance=(ny)*(nx/ip+4);
	
	for(int j=0;j<2;j++){
	int index	=index_3d(i,j,k);
	phi[index]=phi[index+distance];
	}
	for(int j=ny+2;j<ny+4;j++){
	int index	=index_3d(i,j,k);
	phi[index]=phi[index-distance];
	}
}

__global__ void boundary_zm1( double *phi, double *t_phi )
{
	int k,index,index_t;
	int i= blockIdx.x;
	int j=threadIdx.x;
	k=2;
	index	=index_3d(i,j,k);
	index_t	=index_3d(i,j,1);
	t_phi[index_t]=phi[index];

	k=nz/kp+1;
	index	=index_3d(i,j,k);
	index_t	=index_3d(i,j,2);
	t_phi[index_t]=phi[index];
}

__global__ void boundary_xm1( double *phi, double *t_phi  )
{
	int i,index,index_t;
	int j= blockIdx.x;
	int k=threadIdx.x;
	i=2;
	index	=index_3d(i,j,k);
	index_t	=index_3d_x(1,j,k);
	t_phi[index_t]=phi[index];

	i=nx/ip+1;
	index	=index_3d(i,j,k);
	index_t	=index_3d_x(2,j,k);
	t_phi[index_t]=phi[index];
}

__global__ void boundary_zm1_undo( double *phi, double *t_phi)
{
	int k,index,index_t;
	int i= blockIdx.x;
	int j=threadIdx.x;
	k=1;
	index	=index_3d(i,j,k);
	index_t	=index_3d(i,j,0);
	phi[index]=t_phi[index_t];

	k=nz/kp+2;
	index	=index_3d(i,j,k);
	index_t	=index_3d(i,j,3);
	phi[index]=t_phi[index_t];
}

__global__ void boundary_xm1_undo( double *phi, double *t_phi)
{
	int i,index,index_t;
	int j= blockIdx.x;
	int k=threadIdx.x;
	i=1;
	index	=index_3d(i,j,k);
	index_t	=index_3d_x(0,j,k);
	phi[index]=t_phi[index_t];

	i=nx/ip+2;
	index	=index_3d(i,j,k);
	index_t	=index_3d_x(3,j,k);
	phi[index]=t_phi[index_t];
}

__global__ void boundary_zm2( double *phi, double *t_phi )
{
	int k,l,index,index_t;
	int i= blockIdx.x;
	int j=threadIdx.x;
	
	for(l=0;l<2;l++){
	k=2;
	index	=index_3d(i,j,k+l);
	index_t	=index_3d(i,j,2+l);
	t_phi[index_t]=phi[index];
	
	k=nz/kp;
	index	=index_3d(i,j,k+l);
	index_t	=index_3d(i,j,4+l);
	t_phi[index_t]=phi[index];
	}
}

__global__ void boundary_xm2( double *phi, double *t_phi )
{
	int i,l,index,index_t;
	int j= blockIdx.x;
	int k=threadIdx.x;
	
	for(l=0;l<2;l++){
	i=2;
	index	=index_3d(i+l,j,k);
	index_t	=index_3d_x(2+l,j,k);
	t_phi[index_t]=phi[index];
	
	i=nx/ip;
	index	=index_3d(i+l,j,k);
	index_t	=index_3d_x(4+l,j,k);
	t_phi[index_t]=phi[index];
	}
}

__global__ void boundary_zm2_undo( double *phi, double *t_phi)
{
	int k,l,index,index_t;
	int i= blockIdx.x;
	int j=threadIdx.x;
	
	for(l=0;l<2;l++){
	k=0;
	index	=index_3d(i,j,k+l);
	index_t	=index_3d(i,j,0+l);
	phi[index]=t_phi[index_t];
	
	k=nz/kp+2;
	index	=index_3d(i,j,k+l);
	index_t	=index_3d(i,j,6+l);
	phi[index]=t_phi[index_t];
	}
}

__global__ void boundary_xm2_undo( double *phi, double *t_phi)
{
	int i,l,index,index_t;
	int j= blockIdx.x;
	int k=threadIdx.x;
	
	for(l=0;l<2;l++){
	i=0;
	index	=index_3d(i+l,j,k);
	index_t	=index_3d_x(0+l,j,k);
	phi[index]=t_phi[index_t];
	
	i=nx/ip+2;
	index	=index_3d(i+l,j,k);
	index_t	=index_3d_x(6+l,j,k);
	phi[index]=t_phi[index_t];
	}
}

__global__ void boundary_yd_bc( double *g,double *h)
{
	int i= blockIdx.x+2;
	int j,index_l;
	int zd=nz/kp;
	int l=threadIdx.x;
	int distance=(ny)*(nx/ip+4);

	for(int k=2;k<zd+2;k=k+zd-1){
	j=1;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l+distance];
	h[index_l]=h[index_l+distance];
	j=ny+2;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l-distance];
	h[index_l]=h[index_l-distance];
	}
}

__global__ void boundary_yd_bc_x( double *g,double *h)
{
	int k= blockIdx.x+2;
	int j,index_l;
	int xd=nx/ip;
	int l=threadIdx.x;
	int distance=(ny)*(nx/ip+4);
	
	for(int i=2;i<xd+2;i=i+xd-1){
	j=1;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l+distance];
	h[index_l]=h[index_l+distance];
	j=ny+2;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l-distance];
	h[index_l]=h[index_l-distance];
	}
}

__global__ void boundary_zd( double *phi,double *t_phi )
{
	int i= blockIdx.x+1;
	int j=threadIdx.x+1;
	int k,index_l,index_l_t;
	int xd=nx/ip;
	int l_top[5]={5,11,13,15,18};
	int l_bot[5]={6,12,14,16,17};

	for(int l=0;l<5;l++){
	k=2;
	index_l  =index_4d(i,j,k,l_bot[l]);
	index_l_t=((xd+4)*(1*(ny+4)+j)+i)*5+l;//k=1;q=5
	t_phi[index_l_t]=phi[index_l];
	
	k=nz/kp+1;
	index_l  =index_4d(i,j,k,l_top[l]);
	index_l_t=((xd+4)*(2*(ny+4)+j)+i)*5+l;//k=2;q=5
	t_phi[index_l_t]=phi[index_l];
	}
}

__global__ void boundary_xd( double *phi,double *t_phi )
{
	int j= blockIdx.x+1;
	int k=threadIdx.x+1;
	int i,index_l,index_l_t;
	int zd=nz/kp;
	int l_right[5]={1,7, 9,11,14};
	int l_left[5] ={2,8,10,12,13};

	for(int l=0;l<5;l++){
	i=2;
	index_l  =index_4d(i,j,k,l_left[l]);
	index_l_t=((ny+4)*(1*(zd+4)+k)+j)*5+l;//k=1;q=5
	t_phi[index_l_t]=phi[index_l];
	
	i=nx/ip+1;
	index_l  =index_4d(i,j,k,l_right[l]);
	index_l_t=((ny+4)*(2*(zd+4)+k)+j)*5+l;//k=2;q=5
	t_phi[index_l_t]=phi[index_l];
	}
}

__global__ void boundary_zd_undo( double *phi,double *t_phi)
{
	int i= blockIdx.x+1;
	int j=threadIdx.x+1;
	int k,index_l,index_l_t;
	int xd=nx/ip;
	int l_top[5]={5,11,13,15,18};
	int l_bot[5]={6,12,14,16,17};
	for(int l=0;l<5;l++){
	k=1;
	index_l  =index_4d(i,j,k,l_top[l]);
	index_l_t=((xd+4)*(0*(ny+4)+j)+i)*5+l;
	phi[index_l]=t_phi[index_l_t];
	
	k=nz/kp+2;
	index_l  =index_4d(i,j,k,l_bot[l]);
	index_l_t=((xd+4)*(3*(ny+4)+j)+i)*5+l;
	phi[index_l]=t_phi[index_l_t];
	}
}

__global__ void boundary_xd_undo( double *phi,double *t_phi)
{
	int j= blockIdx.x+1;
	int k=threadIdx.x+1;
	int i,index_l,index_l_t;
	int zd=nz/kp;
	int l_right[5]={1,7, 9,11,14};
	int l_left[5] ={2,8,10,12,13};
	for(int l=0;l<5;l++){
	i=1;
	index_l  =index_4d(i,j,k,l_right[l]);
	index_l_t=((ny+4)*(0*(zd+4)+k)+j)*5+l;
	phi[index_l]=t_phi[index_l_t];
	
	i=nx/ip+2;
	index_l  =index_4d(i,j,k,l_left[l]);
	index_l_t=((ny+4)*(3*(zd+4)+k)+j)*5+l;
	phi[index_l]=t_phi[index_l_t];
	}
}

__global__ void boundary_yd_in( double *g,double *h)
{
	int i= blockIdx.x+3;
	int k=threadIdx.x+3;
	int j,index_l;
	int distance=(ny)*(nx/ip+4);
	for(int l=0;l<q;l++){
	j=1;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l+distance];
	h[index_l]=h[index_l+distance];
	j=ny+2;
	index_l=index_4d(i,j,k,l);
	g[index_l]=g[index_l-distance];
	h[index_l]=h[index_l-distance];
	}
}

__global__ void boundary_ym_bc( double *phi)
{
	int i =threadIdx.x+2;
	int zd=nz/kp;
	int kk[4]= {2,3,zd,zd+1};
	int distance=(ny)*(nx/ip+4);
	for (int t=0;t<4;t++){
	int k=kk[t];
	for (int j=0;j<2;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index+distance];
	}
	for (int j=ny+2;j<ny+4;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index-distance];
	}}
}

__global__ void boundary_ym_bc_x( double *phi)
{
	int k =threadIdx.x+2;
	int xd=nx/ip;
	int ii[4]= {2,3,xd,xd+1};
	int distance=(ny)*(xd+4);
	for (int t=0;t<4;t++){
	int i=ii[t];
	for (int j=0;j<2;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index+distance];
	}
	for (int j=ny+2;j<ny+4;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index-distance];
	}}
}

__global__ void boundary_ym_in( double *phi)
{
	int i= blockIdx.x+4;
	int k=threadIdx.x+4;
	int distance=(ny)*(nx/ip+4);

	for(int j=0;j<2;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index+distance];
	}
	for(int j=ny+2;j<ny+4;j++){
	int index=index_3d(i,j,k);
	phi[index]=phi[index-distance];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                 gradient                                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void gradient_cen (	double *gra_phi, double *phi)
{
	const int i=threadIdx.x+2;
	const int j= blockIdx.x%ny+2;
	const int k= blockIdx.x/ny+2;
	const int index=index_3d(i,j,k);
	const double cs2_inv=3.0;
	
	double temp  =0.0;
	double temp_x=0.0;
	double temp_y=0.0;
	double temp_z=0.0;

	for(int l=1;l<q;l=l+2){
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	temp=2.0*wt*(phi[index+et]-phi[index-et]);
	temp_x=ex*temp+temp_x;
	temp_y=ey*temp+temp_y;
	temp_z=ez*temp+temp_z;
	}
	
	gra_phi[index_4d(i,j,k,0)]=temp_x*0.5*cs2_inv;
	gra_phi[index_4d(i,j,k,1)]=temp_y*0.5*cs2_inv;
	gra_phi[index_4d(i,j,k,2)]=temp_z*0.5*cs2_inv;
}

__device__ double grad_phie_c(double *phi,int index,int et)
{
	double ans;
	ans=(phi[index+et]-phi[index-et])*0.5;
	return ans;
}

__device__ double grad_phie_m(double *phi,int index,int et)
{
	double ans;
	ans=(-phi[index+2*et]+5.0*phi[index+et]-3.0*phi[index]-phi[index-et])*0.25;
	return ans;
}

__device__ double gradient_cen_x (	double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ex=eex[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ex*2.0*wt*(phi[index+et]-phi[index-et])+ans;
	}
	ans=ans*0.5*cs2_inv;
	return ans;
}

__device__ double gradient_cen_y (	double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ey=eey[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ey*2.0*wt*(phi[index+et]-phi[index-et])+ans;
	}
	ans=ans*0.5*cs2_inv;
	return ans;
}

__device__ double gradient_cen_z (	double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ez*2.0*wt*(phi[index+et]-phi[index-et])+ans;
	}
	ans=ans*0.5*cs2_inv;
	return ans;
}

__device__ double gradient_mix_x ( double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ex=eex[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ex*wt*(-phi[index+2*et]+6.0*phi[index+et]-6.0*phi[index-et]+phi[index-2*et])+ans;
	}
	ans=ans*0.25*cs2_inv;
	return ans;
}

__device__ double gradient_mix_y ( double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ey=eey[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ey*wt*(-phi[index+2*et]+6.0*phi[index+et]-6.0*phi[index-et]+phi[index-2*et])+ans;
	}
	ans=ans*0.25*cs2_inv;
	return ans;
}

__device__ double gradient_mix_z ( double *phi, int index )
{
	double ans=0.0;
	double cs2_inv=3.0;

#pragma unroll 9
	for(int l=1;l<q;l=l+2){
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	ans=ez*wt*(-phi[index+2*et]+6.0*phi[index+et]-6.0*phi[index-et]+phi[index-2*et])+ans;
	}
	ans=ans*0.25*cs2_inv;
	return ans;
}

__device__ double laplace_phi (double *phi,int index)
{
	double ans=0.0;
	double phi_index=phi[index];
	double cs2_inv  =3.0;
	double dt_inv=1./dt;
	for(int l=1;l<q;l=l+2)
	{
	double wt=wwt[l];
	int	   et=eet[l];
	ans=2.0*wt*(phi[index+et]-2.0*phi_index+phi[index-et])+ans;
	}
	ans=ans*cs2_inv*dt_inv;
	return ans;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                chemical mu                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void	chemical(double *c,double *m,double kappa,double beta )
{
	int i=  threadIdx.x+2;
	int j=blockIdx.x%ny+2;
	int k=blockIdx.x/ny+2;
	int index=index_3d(i,j,k);
	double cl=c[index];
	m[index]=beta*(4.0*cl*cl*cl-6.0*cl*cl+2.0*cl)-kappa*laplace_phi( c,index );
}

__global__ void  chemical_bc( double *c,double *m,double kappa,double beta )
{
	int i=threadIdx.x+2;
	int j=blockIdx.x +2;
	int zd=nz/kp;
	int kk[4]= {2,3,zd,zd+1};
	for (int t=0;t<4;t++){
	int k=kk[t];
	int index=index_3d(i,j,k);
	double cl=c[index];
	m[index]=beta*(4.0*cl*cl*cl-6.0*cl*cl+2.0*cl)-kappa*laplace_phi( c,index );
	}
}

__global__ void  chemical_bc_x( double *c,double *m,double kappa,double beta )
{
	int k=threadIdx.x+4;
	int j=blockIdx.x +2;
	int xd=nx/ip;
	int ii[4]= {2,3,xd,xd+1};
	for (int t=0;t<4;t++){
	int i=ii[t];
	int index=index_3d(i,j,k);
	double cl=c[index];
	m[index]=beta*(4.0*cl*cl*cl-6.0*cl*cl+2.0*cl)-kappa*laplace_phi( c,index );
	}
}

__global__ void  chemical_in( double *c,double *m,double kappa,double beta )
{
	int i=threadIdx.x+4;
	int j=blockIdx.x%ny+2;
	int k=blockIdx.x/ny+4;
	int index=index_3d(i,j,k);
	double cl=c[index];

	m[index]=beta*(4.0*cl*cl*cl-6.0*cl*cl+2.0*cl)-kappa*laplace_phi( c,index );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                 eq collision                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void	eq_collision(double *g,double *h,double *c,double *m,double *p,double gravity,double *gra_c,
							 double *gra_m,double *u,double *v,double *w,double mobi)
							 
{
	int i=threadIdx.x+2;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+2;
	int index=index_3d(i,j,k);
	
	double cs2_inv  =3.0;
	const double cs2=1.0/cs2_inv;
	
	double uu=u[index];
	double vv=v[index];
	double ww=w[index];
	double cc=c[index];
	double rr=cc*rho_l+(1.0-cc)*rho_g;
//	double tt=cc*tau_l+(1.0-cc)*tau_g;
	const double rr_inv=1.0/rr;
	double pp=p[index];
	double dr = rho_l-rho_g;
	
	double gr_cx_c=gra_c[index_4d(i,j,k,0)];
	double gr_cy_c=gra_c[index_4d(i,j,k,1)];
	double gr_cz_c=gra_c[index_4d(i,j,k,2)];
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	double gr_px_c=gradient_cen_x ( p,index );
	double gr_py_c=gradient_cen_y ( p,index );
	double gr_pz_c=gradient_cen_z ( p,index );
	
	double gr_cx_m=gradient_mix_x ( c,index );
	double gr_cy_m=gradient_mix_y ( c,index );
	double gr_cz_m=gradient_mix_z ( c,index );
	double gr_mx_m=gradient_mix_x ( m,index );
	double gr_my_m=gradient_mix_y ( m,index );
	double gr_mz_m=gradient_mix_z ( m,index );
	double gr_px_m=gradient_mix_x ( p,index );
	double gr_py_m=gradient_mix_y ( p,index );
	double gr_pz_m=gradient_mix_z ( p,index );
	
	double lap_mu   =laplace_phi( m,index );
	double udotu=uu*uu+vv*vv+ww*ww;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	
	double edotu=ex*uu+ey*vv+ez*ww;
	double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
	double gamma=wt*(1.0+uugly);
	
	double u_et=u[index+et];
	double v_et=v[index+et];
	double w_et=w[index+et];
	double lap_mu_et=laplace_phi( m,index+et );
	double udotu_et=u_et*u_et+v_et*v_et+w_et*w_et;
	
	double edotu_et=ex*u_et+ey*v_et+ez*w_et;
	double uugly_et=edotu_et*cs2_inv+edotu_et*edotu_et*0.5*cs2_inv*cs2_inv-udotu_et*0.5*cs2_inv;
	double gamma_et=wt*(1.0+uugly_et);
	///////////////////////////////////////////////////////
	double geq_t=wt*(pp+rr*cs2*uugly);//geq
	double heq_t=cc*gamma;//heq
	///////////////////////////////////////////////////////
	double temp_cc = grad_phie_c( c,index,et ) - ( uu * gr_cx_c + vv * gr_cy_c + ww * gr_cz_c );
	double temp_mc = grad_phie_c( m,index,et ) - ( uu * gr_mx_c + vv * gr_my_c + ww * gr_mz_c );
	double temp_pc = grad_phie_c( p,index,et ) - ( uu * gr_px_c + vv * gr_py_c + ww * gr_pz_c );
	
	double temp_cm = grad_phie_m( c,index,et ) - ( uu * gr_cx_m + vv * gr_cy_m + ww * gr_cz_m );
	double temp_mm = grad_phie_m( m,index,et ) - ( uu * gr_mx_m + vv * gr_my_m + ww * gr_mz_m );
	double temp_pm = grad_phie_m( p,index,et ) - ( uu * gr_px_m + vv * gr_py_m + ww * gr_pz_m );
	
	double temp_z = ez*gravity-ww*gravity;
	///////////////////////////////////////////////////////
	double temp_gc = cs2*wt*uugly*temp_cc*dr-(cc*temp_mc+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hc = temp_cc-cc*rr_inv*cs2_inv*(temp_pc+cc*temp_mc+(rr-rho_l)*temp_z);
	
	geq_t=geq_t-0.5*temp_gc;//geq_bar
	heq_t=heq_t-0.5*gamma*temp_hc;//heq_bar
	///////////////////////////////////////////////////////
	double temp_gm = cs2*wt*uugly*temp_cm*dr-(cc*temp_mm+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hm = temp_cm-cc*rr_inv*cs2_inv*(temp_pm+cc*temp_mm+(rr-rho_l)*temp_z);
	temp_hm = 0.5*dt*mobi*( gamma*lap_mu + gamma_et*lap_mu_et )+temp_hm*gamma;
	////////////////////////collision//////////////////////////////
	g[index_l] = geq_t+temp_gm;
	h[index_l] = heq_t+temp_hm;
	}
}

__global__ void	eq_collision_bc(double *g,double *h,double *c,double *m,double *p,double gravity,double *gra_c,
							    double *gra_m,double *u,double *v,double *w,double mobi)
{
	int i=threadIdx.x+2;
	int j= blockIdx.x+2;
	int zd=nz/kp;
	double cs2_inv  =3.0;
	double cs2=1.0/cs2_inv;
	double dr = rho_l-rho_g;
	
	for(int k=2;k<zd+2;k=k+zd-1)
	{
	int index=index_3d(i,j,k);
	double uu=u[index];
	double vv=v[index];
	double ww=w[index];
	double cc=c[index];
	double ceq=cc;
  	if     (cc < 0)ceq=0;
	else if(cc > 1)ceq=1;
	else           ceq=cc;
	double rr=cc*rho_l+(1.0-cc)*rho_g;
	double tt=cc*tau_l+(1.0-cc)*tau_g;
	double rr_inv=1.0/rr;
	double pp=p[index];
	
	double gr_cx_c=gra_c[index_4d(i,j,k,0)];
	double gr_cy_c=gra_c[index_4d(i,j,k,1)];
	double gr_cz_c=gra_c[index_4d(i,j,k,2)];
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	double gr_px_c=gradient_cen_x ( p,index );
	double gr_py_c=gradient_cen_y ( p,index );
	double gr_pz_c=gradient_cen_z ( p,index );
	
	double gr_cx_m=gradient_mix_x ( c,index );
	double gr_cy_m=gradient_mix_y ( c,index );
	double gr_cz_m=gradient_mix_z ( c,index );
	double gr_mx_m=gradient_mix_x ( m,index );
	double gr_my_m=gradient_mix_y ( m,index );
	double gr_mz_m=gradient_mix_z ( m,index );
	double gr_px_m=gradient_mix_x ( p,index );
	double gr_py_m=gradient_mix_y ( p,index );
	double gr_pz_m=gradient_mix_z ( p,index );
	
	double lap_mu =laplace_phi( m,index );
	double udotu=uu*uu+vv*vv+ww*ww;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	
	double edotu=ex*uu+ey*vv+ez*ww;
	double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
	double gamma=wt*(1.0+uugly);
	
	double u_et=u[index+et];
	double v_et=v[index+et];
	double w_et=w[index+et];
	double lap_mu_et=laplace_phi( m,index+et );
	double udotu_et=u_et*u_et+v_et*v_et+w_et*w_et;
	
	double edotu_et=ex*u_et+ey*v_et+ez*w_et;
	double uugly_et=edotu_et*cs2_inv+edotu_et*edotu_et*0.5*cs2_inv*cs2_inv-udotu_et*0.5*cs2_inv;
	double gamma_et=wt*(1.0+uugly_et);
	///////////////////////////////////////////////////////
	double geq_t=wt*(pp+rr*cs2*uugly);//geq
	double heq_t=ceq*gamma;//heq
	///////////////////////////////////////////////////////
	double temp_cc = grad_phie_c( c,index,et ) - ( uu * gr_cx_c + vv * gr_cy_c + ww * gr_cz_c );
	double temp_mc = grad_phie_c( m,index,et ) - ( uu * gr_mx_c + vv * gr_my_c + ww * gr_mz_c );
	double temp_pc = grad_phie_c( p,index,et ) - ( uu * gr_px_c + vv * gr_py_c + ww * gr_pz_c );
	
	double temp_cm = grad_phie_m( c,index,et ) - ( uu * gr_cx_m + vv * gr_cy_m + ww * gr_cz_m );
	double temp_mm = grad_phie_m( m,index,et ) - ( uu * gr_mx_m + vv * gr_my_m + ww * gr_mz_m );
	double temp_pm = grad_phie_m( p,index,et ) - ( uu * gr_px_m + vv * gr_py_m + ww * gr_pz_m );
	
	double temp_z = ez*gravity-ww*gravity;
	///////////////////////////////////////////////////////
	double temp_gc = cs2*wt*uugly*temp_cc*dr-(cc*temp_mc+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hc = temp_cc-cc*rr_inv*cs2_inv*(temp_pc+cc*temp_mc+rr*temp_z);
	
	geq_t=geq_t-0.5*temp_gc;//geq_bar
	heq_t=heq_t-0.5*temp_hc*gamma;//heq_bar
	///////////////////////////////////////////////////////
	double temp_gm = cs2*wt*uugly*temp_cm*dr-(cc*temp_mm+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hm = temp_cm-cc*rr_inv*cs2_inv*(temp_pm+cc*temp_mm+rr*temp_z);
	temp_hm = 0.5*dt*mobi*( gamma*lap_mu + gamma_et*lap_mu_et )+temp_hm*gamma;
	////////////////////////collision//////////////////////////////
	g[index_l] = g[index_l]*(1.0-1.0/(tt    +0.5))+geq_t/(tt    +0.5)+temp_gm;
	h[index_l] = h[index_l]*(1.0-1.0/(tau_h +0.5))+heq_t/(tau_h +0.5)+temp_hm;
	}
	}
}

__global__ void	eq_collision_bc_x(double *g,double *h,double *c,double *m,double *p,double gravity,double *gra_c,
							      double *gra_m,double *u,double *v,double *w,double mobi)
{
	int k=threadIdx.x+3;
	int j= blockIdx.x+2;
	int xd=nx/ip;
	double cs2_inv  =3.0;
	double cs2      =1.0/cs2_inv;
	double dr = rho_l-rho_g;
	
	for(int i=2;i<xd+2;i=i+xd-1)
	{
	int index=index_3d(i,j,k);
	double uu=u[index];
	double vv=v[index];
	double ww=w[index];
	double cc=c[index];
	double ceq=cc;
  	if     (cc < 0)ceq=0;
	else if(cc > 1)ceq=1;
	else           ceq=cc;
	double rr=cc*rho_l+(1.0-cc)*rho_g;
	double tt=cc*tau_l+(1.0-cc)*tau_g;
	double rr_inv=1.0/rr;
	double pp=p[index];
	
	double gr_cx_c=gra_c[index_4d(i,j,k,0)];
	double gr_cy_c=gra_c[index_4d(i,j,k,1)];
	double gr_cz_c=gra_c[index_4d(i,j,k,2)];
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	double gr_px_c=gradient_cen_x ( p,index );
	double gr_py_c=gradient_cen_y ( p,index );
	double gr_pz_c=gradient_cen_z ( p,index );
	
	double gr_cx_m=gradient_mix_x ( c,index );
	double gr_cy_m=gradient_mix_y ( c,index );
	double gr_cz_m=gradient_mix_z ( c,index );
	double gr_mx_m=gradient_mix_x ( m,index );
	double gr_my_m=gradient_mix_y ( m,index );
	double gr_mz_m=gradient_mix_z ( m,index );
	double gr_px_m=gradient_mix_x ( p,index );
	double gr_py_m=gradient_mix_y ( p,index );
	double gr_pz_m=gradient_mix_z ( p,index );
	
	double lap_mu =laplace_phi( m,index );
	double udotu=uu*uu+vv*vv+ww*ww;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	
	double edotu=ex*uu+ey*vv+ez*ww;
	double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
	double gamma=wt*(1.0+uugly);
	
	double u_et=u[index+et];
	double v_et=v[index+et];
	double w_et=w[index+et];
	double lap_mu_et=laplace_phi( m,index+et );
	double udotu_et=u_et*u_et+v_et*v_et+w_et*w_et;
	
	double edotu_et=ex*u_et+ey*v_et+ez*w_et;
	double uugly_et=edotu_et*cs2_inv+edotu_et*edotu_et*0.5*cs2_inv*cs2_inv-udotu_et*0.5*cs2_inv;
	double gamma_et=wt*(1.0+uugly_et);
	///////////////////////////////////////////////////////
	double geq_t=wt*(pp+rr*cs2*uugly);//geq
	double heq_t=ceq*gamma;//heq
	///////////////////////////////////////////////////////
	double temp_cc = grad_phie_c( c,index,et ) - ( uu * gr_cx_c + vv * gr_cy_c + ww * gr_cz_c );
	double temp_mc = grad_phie_c( m,index,et ) - ( uu * gr_mx_c + vv * gr_my_c + ww * gr_mz_c );
	double temp_pc = grad_phie_c( p,index,et ) - ( uu * gr_px_c + vv * gr_py_c + ww * gr_pz_c );
	
	double temp_cm = grad_phie_m( c,index,et ) - ( uu * gr_cx_m + vv * gr_cy_m + ww * gr_cz_m );
	double temp_mm = grad_phie_m( m,index,et ) - ( uu * gr_mx_m + vv * gr_my_m + ww * gr_mz_m );
	double temp_pm = grad_phie_m( p,index,et ) - ( uu * gr_px_m + vv * gr_py_m + ww * gr_pz_m );
	
	double temp_z = ez*gravity-ww*gravity;
	///////////////////////////////////////////////////////
	double temp_gc = cs2*wt*uugly*temp_cc*dr-(cc*temp_mc+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hc = temp_cc-cc*rr_inv*cs2_inv*(temp_pc+cc*temp_mc+rr*temp_z);
	
	geq_t=geq_t-0.5*temp_gc;//geq_bar
	heq_t=heq_t-0.5*temp_hc*gamma;//heq_bar
	///////////////////////////////////////////////////////
	double temp_gm = cs2*wt*uugly*temp_cm*dr-(cc*temp_mm+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hm = temp_cm-cc*rr_inv*cs2_inv*(temp_pm+cc*temp_mm+rr*temp_z);
	temp_hm = 0.5*dt*mobi*( gamma*lap_mu + gamma_et*lap_mu_et )+temp_hm*gamma;
	////////////////////////collision//////////////////////////////
	g[index_l] = g[index_l]*(1.0-1.0/(tt    +0.5))+geq_t/(tt    +0.5)+temp_gm;
	h[index_l] = h[index_l]*(1.0-1.0/(tau_h +0.5))+heq_t/(tau_h +0.5)+temp_hm;
	}
	}
}

__global__ void	eq_collision_in(double *g,double *h,double *c,double *m,double *p,double gravity,double *gra_c,
							    double *gra_m,double *u,double *v,double *w,double mobi)
{
	int i=threadIdx.x+3;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+3;
	int index=index_3d(i,j,k);
	
	double cs2_inv  =3.0;
	double cs2=1.0/cs2_inv;
	double uu=u[index];
	double vv=v[index];
	double ww=w[index];
	double cc=c[index];
	double ceq=cc;
  	if     (cc < 0)ceq=0;
	else if(cc > 1)ceq=1;
	else           ceq=cc;
	double rr=cc*rho_l+(1.0-cc)*rho_g;
	double tt=cc*tau_l+(1.0-cc)*tau_g;
	double rr_inv=1.0/rr;
	double pp=p[index];
	double dr = rho_l-rho_g;
	
	double gr_cx_c=gra_c[index_4d(i,j,k,0)];
	double gr_cy_c=gra_c[index_4d(i,j,k,1)];
	double gr_cz_c=gra_c[index_4d(i,j,k,2)];
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	double gr_px_c=gradient_cen_x ( p,index );
	double gr_py_c=gradient_cen_y ( p,index );
	double gr_pz_c=gradient_cen_z ( p,index );
	
	double gr_cx_m=gradient_mix_x ( c,index );
	double gr_cy_m=gradient_mix_y ( c,index );
	double gr_cz_m=gradient_mix_z ( c,index );
	double gr_mx_m=gradient_mix_x ( m,index );
	double gr_my_m=gradient_mix_y ( m,index );
	double gr_mz_m=gradient_mix_z ( m,index );
	double gr_px_m=gradient_mix_x ( p,index );
	double gr_py_m=gradient_mix_y ( p,index );
	double gr_pz_m=gradient_mix_z ( p,index );
	
	double lap_mu =laplace_phi( m,index );
	double udotu=uu*uu+vv*vv+ww*ww;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	double wt=wwt[l];
	int	   et=eet[l];
	
	double edotu=ex*uu+ey*vv+ez*ww;
	double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
	double gamma=wt*(1.0+uugly);
	
	double u_et=u[index+et];
	double v_et=v[index+et];
	double w_et=w[index+et];
	double lap_mu_et=laplace_phi( m,index+et );
	double udotu_et=u_et*u_et+v_et*v_et+w_et*w_et;
	
	double edotu_et=ex*u_et+ey*v_et+ez*w_et;
	double uugly_et=edotu_et*cs2_inv+edotu_et*edotu_et*0.5*cs2_inv*cs2_inv-udotu_et*0.5*cs2_inv;
	double gamma_et=wt*(1.0+uugly_et);
	///////////////////////////////////////////////////////
	double geq_t=wt*(pp+rr*cs2*uugly);//geq
	double heq_t=ceq*gamma;//heq
	///////////////////////////////////////////////////////
	double temp_cc = grad_phie_c( c,index,et ) - ( uu * gr_cx_c + vv * gr_cy_c + ww * gr_cz_c );
	double temp_mc = grad_phie_c( m,index,et ) - ( uu * gr_mx_c + vv * gr_my_c + ww * gr_mz_c );
	double temp_pc = grad_phie_c( p,index,et ) - ( uu * gr_px_c + vv * gr_py_c + ww * gr_pz_c );
	
	double temp_cm = grad_phie_m( c,index,et ) - ( uu * gr_cx_m + vv * gr_cy_m + ww * gr_cz_m );
	double temp_mm = grad_phie_m( m,index,et ) - ( uu * gr_mx_m + vv * gr_my_m + ww * gr_mz_m );
	double temp_pm = grad_phie_m( p,index,et ) - ( uu * gr_px_m + vv * gr_py_m + ww * gr_pz_m );
	
	double temp_z = ez*gravity-ww*gravity;
	///////////////////////////////////////////////////////
	double temp_gc = cs2*wt*uugly*temp_cc*dr-(cc*temp_mc+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hc = temp_cc-cc*rr_inv*cs2_inv*(temp_pc+cc*temp_mc+rr*temp_z);
	
	geq_t=geq_t-0.5*temp_gc;//geq_bar
	heq_t=heq_t-0.5*temp_hc*gamma;//heq_bar
	///////////////////////////////////////////////////////
	double temp_gm = cs2*wt*uugly*temp_cm*dr-(cc*temp_mm+rr*temp_z)*gamma+ez*rho_l*gravity*wt;
	double temp_hm = temp_cm-cc*rr_inv*cs2_inv*(temp_pm+cc*temp_mm+rr*temp_z);
	temp_hm = 0.5*dt*mobi*( gamma*lap_mu + gamma_et*lap_mu_et )+temp_hm*gamma;
	////////////////////////collision//////////////////////////////
	g[index_l] = g[index_l]*(1.0-1.0/(tt    +0.5))+geq_t/(tt    +0.5)+temp_gm;
	h[index_l] = h[index_l]*(1.0-1.0/(tau_h +0.5))+heq_t/(tau_h +0.5)+temp_hm;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                    macro                                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void macro_h_bc(double *h,double *h_next,double *c)
{
	int i=threadIdx.x+2;
	int j= blockIdx.x+2;
	int zd=nz/kp;
	int kk[4]= {2,3,zd,zd+1};
	for (int t=0;t<4;t++){
	int k=kk[t];
	int index=index_3d(i,j,k);
	double sum_c=0.0;
	for(int l=0;l<q;l++){
	int index_l =index_4d(i,j,k,l);
	int et=eet[l];
	sum_c=h[index_l-et]+sum_c;
	h_next[index_l]=h[index_l-et];
	}
	c[index]=sum_c;
	}
}

__global__ void macro_h_bc_x(double *h,double *h_next,double *c)
{
	int k=threadIdx.x+4;
	int j= blockIdx.x+2;
	int xd=nx/ip;
	int ii[4]= {2,3,xd,xd+1};
	for (int t=0;t<4;t++){
	int i=ii[t];
	int index=index_3d(i,j,k);
	double sum_c=0.0;
	for(int l=0;l<q;l++){
	int index_l =index_4d(i,j,k,l);
	int et=eet[l];
	sum_c=h[index_l-et]+sum_c;
	h_next[index_l]=h[index_l-et];
	}
	c[index]=sum_c;
	}
}

__global__ void macro_h_in(double *h,double *h_next,double *c)
{
	int i=threadIdx.x+4;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+4;
	int index=index_3d(i,j,k);
	double sum_c=0.0;

	for(int l=0;l<q;l++){
	int index_l =index_4d(i,j,k,l);
	int et=eet[l];
	sum_c=h[index_l-et]+sum_c;
	h_next[index_l]=h[index_l-et];
	}
	c[index]=sum_c;
}

__global__ void	macro_g_bc(double *g,double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w)
{
	int i=threadIdx.x+2;
	int j= blockIdx.x+2;
	int zd=nz/kp;
	int kk[4]= {2,3,zd,zd+1};
	double dr=rho_l-rho_g;
	double cs2_inv=3.0;
	double cs2=1.0/cs2_inv;
	for (int t=0;t<4;t++){
	int k=kk[t];
	int index=index_3d(i,j,k);
	double cc=c[index];
	double rr=cc*rho_l+((double)1.0-cc)*rho_g;
 	double gr_rx_c=gra_c[index_4d(i,j,k,0)]*dr;
	double gr_ry_c=gra_c[index_4d(i,j,k,1)]*dr;
	double gr_rz_c=gra_c[index_4d(i,j,k,2)]*dr;
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	
	double sum_u=0.0;
	double sum_v=0.0;
	double sum_w=0.0;
	double sum_p=0.0;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	int	   et=eet[l];
	
	double temp_g=g[index_l-et];
	sum_u=ex*temp_g+sum_u;
	sum_v=ey*temp_g+sum_v;
	sum_w=ez*temp_g+sum_w;
	sum_p=   temp_g+sum_p;
	g_next[index_l]=temp_g;
	}
	double uu=(sum_u*cs2_inv-0.5*dt*cc*gr_mx_c)/rr;
	double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/rr;
	double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/rr;
	u[index]=uu;
	v[index]=vv;
	w[index]=ww;
	p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
	}
}

__global__ void	macro_g_bc_x(double *g,double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w)
{
	int k=threadIdx.x+4;
	int j= blockIdx.x+2;
	int xd=nx/ip;
	int ii[4]= {2,3,xd,xd+1};
	double cs2_inv=3.0;
	double cs2=1.0/cs2_inv;
	double dr=rho_l-rho_g;

	for (int t=0;t<4;t++){
	int i=ii[t];
	int index=index_3d(i,j,k);
	double cc=c[index];
	double rr=cc*rho_l+((double)1.0-cc)*rho_g;
 	double gr_rx_c=gra_c[index_4d(i,j,k,0)]*dr;
	double gr_ry_c=gra_c[index_4d(i,j,k,1)]*dr;
	double gr_rz_c=gra_c[index_4d(i,j,k,2)]*dr;
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	
	double sum_u=0.0;
	double sum_v=0.0;
	double sum_w=0.0;
	double sum_p=0.0;
	
	for(int l=0;l<q;l++)
	{
	int index_l=index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	int	   et=eet[l];
	
	double temp_g=g[index_l-et];
	sum_u=ex*temp_g+sum_u;
	sum_v=ey*temp_g+sum_v;
	sum_w=ez*temp_g+sum_w;
	sum_p=   temp_g+sum_p;
	g_next[index_l]=temp_g;
	}
	double uu=(sum_u*cs2_inv-0.5*dt*cc*gr_mx_c)/rr;
	double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/rr;
	double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/rr;
	u[index]=uu;
	v[index]=vv;
	w[index]=ww;
	p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
	}
}

__global__ void	macro_g_in( double *g, double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w)
{
	int i=threadIdx.x+4;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+4;
	int index=index_3d(i,j,k);
	double cs2_inv=3.0;
	double cs2=1.0/cs2_inv;
	double cc=c[index];
	double rr=cc*rho_l+((double)1.0-cc)*rho_g;
	double dr=rho_l-rho_g;
	
	double gr_rx_c=gra_c[index_4d(i,j,k,0)]*dr;
	double gr_ry_c=gra_c[index_4d(i,j,k,1)]*dr;
	double gr_rz_c=gra_c[index_4d(i,j,k,2)]*dr;
	double gr_mx_c=gra_m[index_4d(i,j,k,0)];
	double gr_my_c=gra_m[index_4d(i,j,k,1)];
	double gr_mz_c=gra_m[index_4d(i,j,k,2)];
	
	double sum_u=0.0;
	double sum_v=0.0;
	double sum_w=0.0;
	double sum_p=0.0;
	
	for(int l=0;l<q;l++){
	int index_l =index_4d(i,j,k,l);
	double ex=eex[l];
	double ey=eey[l];
	double ez=eez[l];
	int	   et=eet[l];
	
	double temp_g=g[index_l-et];
	sum_u=ex*temp_g+sum_u;
	sum_v=ey*temp_g+sum_v;
	sum_w=ez*temp_g+sum_w;
	sum_p=   temp_g+sum_p;
	g_next[index_l]=temp_g;
	}
	
	double uu=(sum_u*cs2_inv-0.5*dt*cc*gr_mx_c)/rr;
	double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/rr;
	double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/rr;
	u[index]=uu;
	v[index]=vv;
	w[index]=ww;
	p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                      post                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void	p_real(double *c,double *p,double *a,double beta,double kappa,double *gra_c)
{
	int i=threadIdx.x+2;
	int j= blockIdx.x%ny+2;
	int k= blockIdx.x/ny+2;
	int index=index_3d(i,j,k);
	
	double gr_cx_c=gra_c[index_4d(i,j,k,0)];
	double gr_cy_c=gra_c[index_4d(i,j,k,1)];
	double gr_cz_c=gra_c[index_4d(i,j,k,2)];
	double la_c   =laplace_phi(c,index );
	double cc=c[index];
	double pp=p[index];
	
	
	double th,cu,e0;
	e0=beta*cc*cc*(cc-1)*(cc-1);
	th=cc*beta*(4*cc*cc*cc-6*cc*cc+2*cc)-e0;
	cu=-kappa*cc*la_c+0.5*kappa*(gr_cx_c*gr_cx_c+gr_cy_c*gr_cy_c+gr_cz_c*gr_cz_c);
	a[index]=pp+th+cu;
}

double maxvalue(double *phi, int* indexx)
{
	double max=0.0;
	int i,j,k;
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<nz;k++){
	int index=nx*(k*ny+j)+i;
	if(max < phi[index]){
		max=phi[index];
		*indexx=index;
	}}}}
	return max;
}

void max_w(double *c,double *w,double *max)
{
	*max=0.0;
	int i,j,k;
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<nz;k++){
	int index=nx*(k*ny+j)+i;
	if(*max < w[index]){
	   *max = w[index];
	}}}}
}

double minvalue(double *phi, int* indexx)
{
	double min=100.0;
	int i,j,k;
	for(i=0;i<nx;i++){
	for(j=0;j<ny;j++){
	for(k=0;k<nz;k++){
	int index=nx*(k*ny+j)+i;
	if(min > phi[index]){
		min=phi[index];
		*indexx=index;
	}}}}
	return min;
}

void Reynolds_Time(double w, double *Re, int step)
{
	Re[step/2-1]=2*radd*3/tau_l*w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                      main                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	int i,j,k,index;
	//define matrix(會切割的)
	double *c_d_h,*c_f_h,*c_fdo_h,*c_d,*c; // dicom & final & transfered on host/ orifinal & transfered on device
	double *m_d_h,*m_f_h,*m_fdo_h,*m_d,*m;
	double *b_d_h,*b_f_h,*b_fdo_h,*b_d,*b; // wettability
	double *p_d_h,*p_f_h,*p_fdo_h,*p_d,*p;
	double *u_d_h,*u_f_h,*u_fdo_h,*u_d,*u;
	double *v_d_h,*v_f_h,*v_fdo_h,*v_d,*v;
	double *w_d_h,*w_f_h,*w_fdo_h,*w_d,*w;
	double *a_d_h,*a_f_h,*a_fdo_h,*a_d,*a; //total pressure
	
	double *xz_d_h,*xz_f_h,*xz_fdo_h,*xz_d;
	
	
	//define matrix(不會切割的)
	int    *et_h;//方向
	double *ex_h,*ey_h,*ez_h,*wt_h;
	double *h,*h_t;
	double *g,*g_t;
	//gradient matrix
	double *gra_c;
	double *gra_m;
	//define matrix(邊界交換的小矩陣)
	double *t_c_h,*t_c;
	double *t_m_h,*t_m;
	double *t_b_h,*t_b;
	double *t_p_h,*t_p;
	double *t_u_h,*t_u;
	double *t_v_h,*t_v;
	double *t_w_h,*t_w;
	double *t_g_h,*t_g;
	double *t_h_h,*t_h;
	
	double *t_c_x_h,*t_c_x;
	double *t_m_x_h,*t_m_x;
	double *t_b_x_h,*t_b_x;
	double *t_p_x_h,*t_p_x;
	double *t_u_x_h,*t_u_x;
	double *t_v_x_h,*t_v_x;
	double *t_w_x_h,*t_w_x;
	double *t_g_x_h,*t_g_x;
	double *t_h_x_h,*t_h_x;
	
	double *lx,*lz;
	double *Re;

////mpi
	int nproc,myid;
	int l_nbr, b_nbr, r_nbr, t_nbr, my_coord[ndim], iroot, itag;
	int ipart[ndim],periods[ndim],sideways,updown,right,up,reorder;
	int n_f;
	MPI_Status istat[8];
	MPI_Comm comm;
	
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	comm = MPI_COMM_WORLD;
	
	ipart[0]=ip;
	ipart[1]=kp;
	periods[0]=1;
	periods[1]=1;
	reorder=1;
	
	MPI_Cart_create(MPI_COMM_WORLD,ndim,ipart,periods,reorder,&comm);
	MPI_Comm_rank(comm,&myid);
	MPI_Cart_coords(comm,myid,ndim,my_coord);
	
	sideways=0;
	updown=1;
	right=1;
	up=1;
	
	MPI_Cart_shift(comm,sideways,right,&l_nbr,&r_nbr);
	MPI_Cart_shift(comm,updown  ,up   ,&b_nbr,&t_nbr);
	
	n_f=nx/ip*ny*nz/kp;
	if(myid==0){
	printf("===============================================================\n");
	printf("Checking devices...\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("NPROC,MYID,i,k=%d\t%d\t%d\t%d\t\n",nproc,myid,my_coord[0],my_coord[1]);
	MPI_Barrier(MPI_COMM_WORLD);
	
	cudaSetDevice(myid%DPN);
	
////memory allocate on cpu
	
	int size_final = nx*ny*nz;
	int size_dicom = (nx/ip+4)*(ny+4)*(nz/kp+4);
	int size_difun = (nx/ip+4)*(ny+4)*(nz/kp+4)*q;
	int size_allgr = (nx/ip+4)*(ny+4)*(nz/kp+4)*3;//(x+y+z)
	
	int tran_mac_1 = (nx/ip+4)*(ny+4)*4*1; //u,v,w
	int tran_mac_2 = (nx/ip+4)*(ny+4)*4*2; //c,m,b,p
	int tran_difun = (nx/ip+4)*(ny+4)*4*5;//5個方向
	
	int tran_mac_1_x = (nz/kp+4)*(ny+4)*4*1; //u,v,w x face
	int tran_mac_2_x = (nz/kp+4)*(ny+4)*4*2; //c,m,b,p x face
	int tran_difun_x = (nz/kp+4)*(ny+4)*4*5;//5個方向 x face
	
	cudaMallocHost((void**)&c_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&m_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&b_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&p_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&u_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&v_d_h ,sizeof(double)*size_dicom);
	cudaMallocHost((void**)&w_d_h ,sizeof(double)*size_dicom); 
	cudaMallocHost((void**)&a_d_h ,sizeof(double)*size_dicom); 
	
	cudaMallocHost((void**)&et_h ,sizeof(double)* q ); 
	cudaMallocHost((void**)&ex_h ,sizeof(double)* q ); 
	cudaMallocHost((void**)&ey_h ,sizeof(double)* q ); 
	cudaMallocHost((void**)&ez_h ,sizeof(double)* q );
	cudaMallocHost((void**)&wt_h ,sizeof(double)* q );
	
	cudaMallocHost((void**)&t_c_h	,sizeof(double)* tran_mac_2 );
	cudaMallocHost((void**)&t_m_h	,sizeof(double)* tran_mac_2 );
	cudaMallocHost((void**)&t_b_h	,sizeof(double)* tran_mac_2 );
	cudaMallocHost((void**)&t_p_h	,sizeof(double)* tran_mac_2 );
	cudaMallocHost((void**)&t_u_h	,sizeof(double)* tran_mac_1 );
	cudaMallocHost((void**)&t_v_h	,sizeof(double)* tran_mac_1 );
	cudaMallocHost((void**)&t_w_h	,sizeof(double)* tran_mac_1 );
	cudaMallocHost((void**)&t_g_h	,sizeof(double)* tran_difun ); 
	cudaMallocHost((void**)&t_h_h	,sizeof(double)* tran_difun );
	
	cudaMallocHost((void**)&t_c_x_h	,sizeof(double)* tran_mac_2_x );
	cudaMallocHost((void**)&t_m_x_h	,sizeof(double)* tran_mac_2_x );
	cudaMallocHost((void**)&t_b_x_h	,sizeof(double)* tran_mac_2_x );
	cudaMallocHost((void**)&t_p_x_h	,sizeof(double)* tran_mac_2_x );
	cudaMallocHost((void**)&t_u_x_h	,sizeof(double)* tran_mac_1_x );
	cudaMallocHost((void**)&t_v_x_h	,sizeof(double)* tran_mac_1_x );
	cudaMallocHost((void**)&t_w_x_h	,sizeof(double)* tran_mac_1_x );
	cudaMallocHost((void**)&t_g_x_h	,sizeof(double)* tran_difun_x ); 
	cudaMallocHost((void**)&t_h_x_h	,sizeof(double)* tran_difun_x );	

	cudaMallocHost((void**)&xz_d_h  ,sizeof(double)*(nx/ip+4)*(nz/kp+4)); 
	
///////////////////////////////////////////////////////////////////////////////////////////
//                                         zz                                            //
///////////////////////////////////////////////////////////////////////////////////////////
	int step=0;
	double beta,zeta,mobi,kappa,phic,gravity;
	parameter (&beta,&zeta,&mobi,&kappa,&phic,&gravity,ex_h,ey_h,ez_h,wt_h,et_h);
	
	FILE *data_2d_t;
	FILE *data_3d_t;
	FILE *data_2d;
	FILE *data_3d;
	FILE *properties;
	FILE *final_2d;
	FILE *final_3d;

	if(myid == 0){
	cudaMallocHost((void**)&c_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&m_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&b_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&p_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&u_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&v_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&w_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&a_f_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&xz_f_h ,sizeof(double)*  nx*nz   );
	
	cudaMallocHost((void**)&c_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&m_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&b_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&p_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&u_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&v_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&w_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&a_fdo_h  ,sizeof(double)*size_final);
	cudaMallocHost((void**)&xz_fdo_h ,sizeof(double)*  nx*nz   );	
	
	cudaMallocHost((void**)&lx ,sizeof(double)*   stepall/2   );
	cudaMallocHost((void**)&lz ,sizeof(double)*   stepall/2   );
	cudaMallocHost((void**)&Re ,sizeof(double)*   stepall/2   );
	initial_macro(c_f_h,m_f_h,b_f_h,p_f_h,u_f_h,v_f_h,w_f_h);
	
	array_2D_do(c_f_h,c_fdo_h);
	array_2D_do(m_f_h,m_fdo_h);
	array_2D_do(b_f_h,b_fdo_h);
	array_2D_do(p_f_h,p_fdo_h);
	array_2D_do(u_f_h,u_fdo_h);
	array_2D_do(v_f_h,v_fdo_h);
	array_2D_do(w_f_h,w_fdo_h);
	array_2D_do(a_f_h,a_fdo_h);
	
	//writing data
	properties = fopen("properties.txt","w");
	if(condition==0){
	double mo=gravity*(rho_l-rho_g)*pow(tau_l,4)*rho_l*rho_l/81.0/pow(sigma,3);
	printf("===============================================================\n");
	fprintf( properties, "Three dimensional droplets - Bubble rising\n");
	fprintf( properties, "Grid size nx=%d, ny=%d, nz=%d\n",nx,ny,nz);
	fprintf( properties, "Radius=%f, Thickness=%f\n",radd, thick);
	fprintf( properties, "Bo=%f\n",bo);
	fprintf( properties, "Mo=%f\n",mo);
	printf ("Bo=%f\n",bo);
	printf ("Mo=%f\n",mo);
	printf("Three dimensional droplets - One Bubble rising\n");
	printf("===============================================================\n");
	}
	else if(condition==1){
	double mo=gravity*(rho_l-rho_g)*pow(tau_l,4)*rho_l*rho_l/81.0/pow(sigma,3);
	printf("===============================================================\n");
	fprintf( properties, "Three dimensional droplets - Bubble rising\n");
	fprintf( properties, "Grid size nx=%d, ny=%d, nz=%d\n",nx,ny,nz);
	fprintf( properties, "Radius=%f, Thickness=%f\n",radd, thick);
	fprintf( properties, "Bo=%f\n",bo);
	fprintf( properties, "Mo=%f\n",mo);
	printf ("Bo=%f\n",bo);
	printf ("Mo=%f\n",mo);
	printf("Three dimensional droplets - Two Bubble rising\n");
	printf("===============================================================\n");
	}
	
	printf("Initializing...");
	fprintf( properties, "Tau_h =%f, Tau_g=%f, Tau_l=%f\n", tau_h,tau_g,tau_l);
	fprintf( properties, "rho_l =%f, rho_g=%f, sigma=%f\n", rho_l,rho_g,sigma);
	fclose(properties);

	data_2d = fopen("data_2d.dat","w");
	fprintf( data_2d, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"p\"\n");
	fprintf( data_2d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_2d, "I=%d, J=%d\n", nx,nz);
	j=ny/2;
	for(k=0;k<nz;k++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	fprintf( data_2d, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t\n",
	i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],p_f_h[index]);
	}}
	fclose(data_2d);
	
	data_3d = fopen("data_3d.dat","w");
	fprintf( data_3d, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\"\n");
	fprintf( data_3d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_3d, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=(nx)*(k*(ny)+j)+i;
	fprintf( data_3d, "%d\t%d\t%d\t%e\t\n",
	i,j,k,c_f_h[index]);
	}}}
	fclose(data_3d);
	printf("done\n");
	printf("===============================================================\n");
	printf("Iterating...\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	//scatter
	iroot = 0;
	MPI_Scatter((void *)&c_fdo_h[0],n_f, MPI_DOUBLE,(void *)&c_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&m_fdo_h[0],n_f, MPI_DOUBLE,(void *)&m_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&b_fdo_h[0],n_f, MPI_DOUBLE,(void *)&b_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&p_fdo_h[0],n_f, MPI_DOUBLE,(void *)&p_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&u_fdo_h[0],n_f, MPI_DOUBLE,(void *)&u_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&v_fdo_h[0],n_f, MPI_DOUBLE,(void *)&v_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&w_fdo_h[0],n_f, MPI_DOUBLE,(void *)&w_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Scatter((void *)&a_fdo_h[0],n_f, MPI_DOUBLE,(void *)&a_d_h[0],n_f, MPI_DOUBLE,iroot,comm);
	MPI_Barrier(MPI_COMM_WORLD);
	
	//memory allocation on gpu
	cudaMalloc((void**)&c_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&m_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&b_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&p_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&u_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&v_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&w_d ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&a_d ,sizeof(double)* size_dicom );

	cudaMalloc((void**)&h   ,sizeof(double)* size_difun );
	cudaMalloc((void**)&g   ,sizeof(double)* size_difun );
	cudaMalloc((void**)&h_t ,sizeof(double)* size_difun );
	cudaMalloc((void**)&g_t ,sizeof(double)* size_difun );
	
	cudaMalloc((void**)&t_c ,sizeof(double)* tran_mac_2 );
	cudaMalloc((void**)&t_m ,sizeof(double)* tran_mac_2 );
	cudaMalloc((void**)&t_b ,sizeof(double)* tran_mac_2 );
	cudaMalloc((void**)&t_p ,sizeof(double)* tran_mac_2 );
	cudaMalloc((void**)&t_u ,sizeof(double)* tran_mac_1 );
	cudaMalloc((void**)&t_v ,sizeof(double)* tran_mac_1 );
	cudaMalloc((void**)&t_w ,sizeof(double)* tran_mac_1 );
	cudaMalloc((void**)&t_g ,sizeof(double)* tran_difun );
	cudaMalloc((void**)&t_h ,sizeof(double)* tran_difun );
	
	cudaMalloc((void**)&t_c_x ,sizeof(double)* tran_mac_2_x );
	cudaMalloc((void**)&t_m_x ,sizeof(double)* tran_mac_2_x );
	cudaMalloc((void**)&t_b_x ,sizeof(double)* tran_mac_2_x );
	cudaMalloc((void**)&t_p_x ,sizeof(double)* tran_mac_2_x );
	cudaMalloc((void**)&t_u_x ,sizeof(double)* tran_mac_1_x );
	cudaMalloc((void**)&t_v_x ,sizeof(double)* tran_mac_1_x );
	cudaMalloc((void**)&t_w_x ,sizeof(double)* tran_mac_1_x );
	cudaMalloc((void**)&t_g_x ,sizeof(double)* tran_difun_x );
	cudaMalloc((void**)&t_h_x ,sizeof(double)* tran_difun_x );
	
	cudaMalloc((void**)&gra_c ,sizeof(double)* size_allgr );
	cudaMalloc((void**)&gra_m ,sizeof(double)* size_allgr );
	
	cudaMalloc((void**)&xz_d,sizeof(double)*(nx/ip+4)*(nz/kp+4));	
	
	MPI_Barrier(MPI_COMM_WORLD);
	//cpu to gpu
	cudaMemcpy(c_d, c_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, m_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(p_d, p_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(u_d, u_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);
	cudaMemcpy(a_d, a_d_h, sizeof(double)* size_dicom , cudaMemcpyHostToDevice);

	cudaMemcpy(t_c, t_c_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_m, t_m_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_b, t_b_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_p, t_p_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_u, t_u_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_v, t_v_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_w, t_w_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	cudaMemcpy(t_g, t_g_h, sizeof(double)* tran_difun , cudaMemcpyHostToDevice);
	cudaMemcpy(t_h, t_h_h, sizeof(double)* tran_difun , cudaMemcpyHostToDevice);
	
	cudaMemcpy(t_c_x, t_c_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_m_x, t_m_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_b_x, t_b_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_p_x, t_p_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_u_x, t_u_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_v_x, t_v_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_w_x, t_w_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_g_x, t_g_x_h, sizeof(double)* tran_difun_x , cudaMemcpyHostToDevice);
	cudaMemcpy(t_h_x, t_h_x_h, sizeof(double)* tran_difun_x , cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol (  eex ,  ex_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  eey ,  ey_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  eez ,  ez_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  wwt ,  wt_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  eet ,  et_h,   sizeof(int   )*q  );
	
	MPI_Barrier(MPI_COMM_WORLD);

	int xd=nx/ip; //x decomposition
	int zd=nz/kp; //z decomposition
	
	int grid_t0		=ny*zd;
	int block_t0	=xd;
	int grid_bc		=ny;
	int block_t0_x	=zd-2;
	int grid_in		=ny*(zd-2);
	int grid_in2	=ny*(zd-4);
	
	cudaMalloc((void**)&c   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&m   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&b   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&p   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&u   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&v   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&w   ,sizeof(double)* size_dicom );
	cudaMalloc((void**)&a   ,sizeof(double)* size_dicom );

	array_do <<<grid_t0 , block_t0>>>( c_d,c );
	array_do <<<grid_t0 , block_t0>>>( m_d,m );
	array_do <<<grid_t0 , block_t0>>>( b_d,b );
	array_do <<<grid_t0 , block_t0>>>( p_d,p );
	array_do <<<grid_t0 , block_t0>>>( u_d,u );
	array_do <<<grid_t0 , block_t0>>>( v_d,v );
	array_do <<<grid_t0 , block_t0>>>( w_d,w );
	array_do <<<grid_t0 , block_t0>>>( a_d,a );
	MPI_Barrier(MPI_COMM_WORLD);

///////////////////////////////////////////////////////////////////////////////////////////
	int num_trans_m_2	=(xd+4)*(ny+4)*2;
	int num_trans_m_1	=(xd+4)*(ny+4)*1;
	int startb			=(xd+4)*( 0 *(ny+4)+0)+0;
	int start			=(xd+4)*( 2 *(ny+4)+0)+0;
	int end				=(xd+4)*( 4 *(ny+4)+0)+0;
	int endb			=(xd+4)*( 6 *(ny+4)+0)+0;
	int startb_1		=(xd+4)*( 0 *(ny+4)+0)+0;
	int start_1			=(xd+4)*( 1 *(ny+4)+0)+0;
	int end_1			=(xd+4)*( 2 *(ny+4)+0)+0;
	int endb_1			=(xd+4)*( 3 *(ny+4)+0)+0;
	int num_trans_d		=(xd+4)*(ny+4)*5;
	int startb_d		=((xd+4)*( 0 *(ny+4)+0)+0)*5;
	int start_d			=((xd+4)*( 1 *(ny+4)+0)+0)*5;
	int end_d			=((xd+4)*( 2 *(ny+4)+0)+0)*5;
	int endb_d			=((xd+4)*( 3 *(ny+4)+0)+0)*5;

	int num_trans_m_2_x	=(ny+4)*(zd+4)*2;
	int num_trans_m_1_x	=(ny+4)*(zd+4)*1;
	int startb_x		=(ny+4)*( 0 *(zd+4)+0)+0;
	int start_x			=(ny+4)*( 2 *(zd+4)+0)+0;
	int end_x			=(ny+4)*( 4 *(zd+4)+0)+0;
	int endb_x			=(ny+4)*( 6 *(zd+4)+0)+0;
	int startb_1_x		=(ny+4)*( 0 *(zd+4)+0)+0;
	int start_1_x		=(ny+4)*( 1 *(zd+4)+0)+0;
	int end_1_x			=(ny+4)*( 2 *(zd+4)+0)+0;
	int endb_1_x		=(ny+4)*( 3 *(zd+4)+0)+0;
	int num_trans_d_x	=(ny+4)*(zd+4)*5;
	int startb_d_x		=((ny+4)*( 0 *(zd+4)+0)+0)*5;
	int start_d_x		=((ny+4)*( 1 *(zd+4)+0)+0)*5;
	int end_d_x			=((ny+4)*( 2 *(zd+4)+0)+0)*5;
	int endb_d_x		=((ny+4)*( 3 *(zd+4)+0)+0)*5;
///////////////////////////////////////////////////////////////////////////////////////////
/* 	checkk <<<grid_t2 , block_t2>>>( c_d,c ); 
	cudaMemcpy(c_d_h,c_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	FILE *check;
	if(myid==1){
	check = fopen("check.dat","w");
	fprintf( check, "VARIABLES=\"X\",\"Z\",\"c\"\n");
	fprintf( check, "ZONE T=\"gpu\" F=POINT\n");
	fprintf( check, "I=%d, J=%d\n", nx+4,zd+4);
	j=ny/2;
	for(k=0;k<zd+4;k++){
	for(i=0;i<nx+4;i++){
	index_3d(i,j,k);
	fprintf( check, "%d\t%d\t%e\t\n",
	i,k,c_d_h[index]);
	}}
	fclose(check);
	} */

///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( c );
////z
	boundary_zm2	<<< xd+4 , ny+4  >>>( c,t_c );
	cudaMemcpy(t_c_h, t_c, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=110;
	MPI_Sendrecv	((void *)&t_c_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_c_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_c_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_c_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat);

	cudaMemcpy(t_c, t_c_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	boundary_zm2_undo	<<< xd+4 , ny+4  >>>( c,t_c );
////x
	boundary_xm2	<<< ny+4 , zd+4  >>>( c,t_c_x );
	cudaMemcpy(t_c_x_h, t_c_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=11;
	MPI_Sendrecv	((void *)&t_c_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_c_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_c_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_c_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat);

	cudaMemcpy(t_c_x, t_c_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	boundary_xm2_undo	<<< ny+4 , zd+4  >>>( c,t_c_x );

///////////////////////////////////////////////////////////////////////////////////////////
	chemical   <<<grid_t0, block_t0>>>( c,m,kappa,beta );
//	chemical_b <<<grid_t0, block_t0>>>( c,m,b,kappa,beta,phic );//wettability
///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym <<< xd+4 , zd+4  >>>( m );
////z
	boundary_zm2<<< xd+4 , ny+4  >>>( m,t_m );
	cudaMemcpy(t_m_h, t_m, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=110;
	MPI_Sendrecv	((void *)&t_m_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_m_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_m_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_m_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_m, t_m_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	boundary_zm2_undo	<<< xd+4 , ny+4  >>>( m,t_m );
////x
	boundary_xm2<<< ny+4 , zd+4  >>>( m,t_m_x );
	cudaMemcpy(t_m_x_h, t_m_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=11;
	MPI_Sendrecv	((void *)&t_m_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_m_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_m_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_m_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_m_x, t_m_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	boundary_xm2_undo	<<< ny+4 , zd+4  >>>( m,t_m_x );

	///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( b );
////z
	boundary_zm2	<<< xd+4 , ny+4  >>>( b,t_b );
	cudaMemcpy(t_b_h, t_b, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=130;
	MPI_Sendrecv	((void *)&t_b_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_b_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=140;
	MPI_Sendrecv	((void *)&t_b_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_b_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_b, t_b_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	boundary_zm2_undo	<<< xd+4 , ny+4  >>>( b,t_b );
////x
	boundary_xm2	<<< ny+4 , zd+4  >>>( b,t_b_x );
	cudaMemcpy(t_b_x_h, t_b_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=13;
	MPI_Sendrecv	((void *)&t_b_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_b_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=14;
	MPI_Sendrecv	((void *)&t_b_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_b_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_b_x, t_b_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	boundary_xm2_undo	<<< ny+4 , zd+4  >>>( b,t_b_x );

	///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( p );
////z
	boundary_zm2	<<< xd+4 , ny+4  >>>( p,t_p );
	cudaMemcpy(t_p_h, t_p, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=150;
	MPI_Sendrecv	((void *)&t_p_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_p_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=160;
	MPI_Sendrecv	((void *)&t_p_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_p_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_p, t_p_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice);
	boundary_zm2_undo	<<< xd+4 , ny+4  >>>( p,t_p );
////x
	boundary_xm2	<<< ny+4 , zd+4  >>>( p,t_p_x );
	cudaMemcpy(t_p_x_h, t_p_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=15;
	MPI_Sendrecv	((void *)&t_p_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_p_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=16;
	MPI_Sendrecv	((void *)&t_p_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_p_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_p_x, t_p_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice);
	boundary_xm2_undo	<<< ny+4 , zd+4  >>>( p,t_p_x );

	///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( u );
////z
	boundary_zm1	<<< xd+4 , ny+4  >>>( u,t_u );
	cudaMemcpy(t_u_h, t_u, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=170;
	MPI_Sendrecv	((void *)&t_u_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_u_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=180;
	MPI_Sendrecv	((void *)&t_u_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_u_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_u, t_u_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	boundary_zm1_undo	<<< xd+4 , ny+4  >>>( u,t_u );
////x
	boundary_xm1	<<< ny+4 , zd+4  >>>( u,t_u_x );
	cudaMemcpy(t_u_x_h, t_u_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=17;
	MPI_Sendrecv	((void *)&t_u_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_u_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=18;
	MPI_Sendrecv	((void *)&t_u_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_u_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_u_x, t_u_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	boundary_xm1_undo	<<< ny+4 , zd+4  >>>( u,t_u_x );

	///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( v );
////z
	boundary_zm1	<<< xd+4 , ny+4  >>>( v,t_v );
	cudaMemcpy(t_v_h, t_v, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=190;
	MPI_Sendrecv	((void *)&t_v_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_v_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=200;
	MPI_Sendrecv	((void *)&t_v_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_v_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_v, t_v_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	boundary_zm1_undo	<<< xd+4 , ny+4  >>>( v,t_v );
////x
	boundary_xm1	<<< ny+4 , zd+4  >>>( v,t_v_x );
	cudaMemcpy(t_v_x_h, t_v_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=19;
	MPI_Sendrecv	((void *)&t_v_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_v_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=20;
	MPI_Sendrecv	((void *)&t_v_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_v_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_v_x, t_v_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	boundary_xm1_undo	<<< ny+4 , zd+4  >>>( v,t_v_x );

	///////////////////////////////////////////////////////////////////////////////////////////

////y
	boundary_ym		<<< xd+4 , zd+4  >>>( w );
////z
	boundary_zm1	<<< xd+4 , ny+4  >>>( w,t_w );
	cudaMemcpy(t_w_h, t_w, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=210;
	MPI_Sendrecv	((void *)&t_w_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_w_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=220;
	MPI_Sendrecv	((void *)&t_w_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_w_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_w, t_w_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice);
	boundary_zm1_undo	<<< xd+4 , ny+4  >>>( w,t_w );
////x
	boundary_xm1	<<< ny+4 , zd+4  >>>( w,t_w_x );
	cudaMemcpy(t_w_x_h, t_w_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	itag=21;
	MPI_Sendrecv	((void *)&t_w_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_w_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=22;
	MPI_Sendrecv	((void *)&t_w_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_w_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpy(t_w_x, t_w_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice);
	boundary_xm1_undo	<<< ny+4 , zd+4  >>>( w,t_w_x );
	
	MPI_Barrier(MPI_COMM_WORLD);
	
///////////////////////////////////////////////////////////////////////////////////////////
	gradient_cen   <<<grid_t0, block_t0,0>>>(gra_c,c);
	gradient_cen   <<<grid_t0, block_t0,0>>>(gra_m,m);
	cudaThreadSynchronize();
	eq_collision   <<<grid_t0, block_t0  >>>( g,h,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi);
	cudaThreadSynchronize();
	
	cudaStream_t  stream0,stream1;
	int leastPriority;
	int greatestPriority;
	cudaDeviceGetStreamPriorityRange (&leastPriority,&greatestPriority);
	int priority=greatestPriority;
	cudaStreamCreateWithPriority(&stream0,0,priority);
	cudaStreamCreate(&stream1);
	//time
	cudaEvent_t gpu_start,gpu_start_temp,gpu_stop,gpu_stop_temp;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventCreate(&gpu_start_temp);
	cudaEventCreate(&gpu_stop_temp);
	cudaEventRecord(gpu_start_temp,0);
	cudaEventRecord(gpu_start,0);

///////////////////////////////////////////////////////////////////////////////////////////
//                                        sstart                                         //
///////////////////////////////////////////////////////////////////////////////////////////
	for(step=1;step<=stepall;step++){

	eq_collision_bc    <<< grid_bc    , block_t0   , 0, stream0 >>>( g,h,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );
	eq_collision_bc_x  <<< grid_bc    , block_t0_x , 0, stream0 >>>( g,h,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );	
	boundary_yd_bc     <<< xd         , q          , 0, stream0 >>>( g,h );
	boundary_yd_bc_x   <<< zd         , q          , 0, stream0 >>>( g,h );
////z...
	boundary_zd        <<< xd+2       , ny+2       , 0, stream0 >>>( g,t_g );
	boundary_zd        <<< xd+2       , ny+2       , 0, stream0 >>>( h,t_h );
	eq_collision_in    <<< grid_in    , xd-2       , 0, stream1 >>>( g,h,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );
////...z	
	cudaMemcpyAsync(t_g_h, t_g, sizeof(double)*tran_difun , cudaMemcpyDeviceToHost,stream0);
	cudaMemcpyAsync(t_h_h, t_h, sizeof(double)*tran_difun , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	
	itag=230;
	MPI_Sendrecv	((void *)&t_g_h[end_d   ], num_trans_d, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_g_h[startb_d], num_trans_d, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=231;
	MPI_Sendrecv	((void *)&t_g_h[start_d ], num_trans_d, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_g_h[endb_d  ], num_trans_d, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	itag=232;
	MPI_Sendrecv	((void *)&t_h_h[end_d   ], num_trans_d, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_h_h[startb_d], num_trans_d, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=233;
	MPI_Sendrecv	((void *)&t_h_h[start_d ], num_trans_d, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_h_h[endb_d  ], num_trans_d, MPI_DOUBLE, t_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_g, t_g_h, sizeof(double)*tran_difun , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_h, t_h_h, sizeof(double)*tran_difun , cudaMemcpyHostToDevice,stream0);
	boundary_zd_undo   <<< xd+2       , ny+2       , 0, stream0 >>>( g,t_g );
	boundary_zd_undo   <<< xd+2       , ny+2       , 0, stream0 >>>( h,t_h );
////x...
	boundary_xd        <<< ny+2       , zd+2       , 0, stream0 >>>( g,t_g_x );
	boundary_xd        <<< ny+2       , zd+2       , 0, stream0 >>>( h,t_h_x );
	boundary_yd_in     <<< xd-2       , zd-2       , 0, stream1 >>>( g,h );
////...x
	cudaMemcpyAsync(t_g_x_h, t_g_x, sizeof(double)*tran_difun_x , cudaMemcpyDeviceToHost,stream0);
	cudaMemcpyAsync(t_h_x_h, t_h_x, sizeof(double)*tran_difun_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	
	itag=23;
	MPI_Sendrecv	((void *)&t_g_x_h[end_d_x   ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_g_x_h[startb_d_x], num_trans_d_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=24;
	MPI_Sendrecv	((void *)&t_g_x_h[start_d_x ], num_trans_d_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_g_x_h[endb_d_x  ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	itag=25;
	MPI_Sendrecv	((void *)&t_h_x_h[end_d_x   ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_h_x_h[startb_d_x], num_trans_d_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=26;
	MPI_Sendrecv	((void *)&t_h_x_h[start_d_x ], num_trans_d_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_h_x_h[endb_d_x  ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_g_x, t_g_x_h, sizeof(double)*tran_difun_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_h_x, t_h_x_h, sizeof(double)*tran_difun_x , cudaMemcpyHostToDevice,stream0);
	boundary_xd_undo   <<< ny+2       , zd+2       , 0, stream0 >>>( g,t_g_x );
	boundary_xd_undo   <<< ny+2       , zd+2       , 0, stream0 >>>( h,t_h_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	macro_h_bc		   <<< grid_bc	  , block_t0   , 0, stream0 >>>( h,h_t,c );
	macro_h_bc_x	   <<< grid_bc    , zd-4	   , 0, stream0 >>>( h,h_t,c );
	boundary_ym_bc	   <<< 1		  , xd		   , 0, stream0 >>>( c );
	boundary_ym_bc_x   <<< 1		  , zd		   , 0, stream0 >>>( c );
////z...
	boundary_zm2       <<< xd+4	      , ny+4       , 0, stream0 >>>( c,t_c );
	macro_h_in		   <<< grid_in2	  , xd-4       , 0, stream1 >>>( h,h_t,c );
////...z	
	cudaMemcpyAsync(t_c_h, t_c, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=110;
	MPI_Sendrecv	((void *)&t_c_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_c_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_c_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_c_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_c, t_c_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( c,t_c );
////x...
	boundary_xm2       <<< ny+4       , zd+4	   , 0, stream0 >>>( c,t_c_x );
	boundary_ym_in     <<< xd-4       ,	zd-4       , 0, stream1 >>>( c );
////...x
	cudaMemcpyAsync(t_c_x_h, t_c_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=11;
	MPI_Sendrecv	((void *)&t_c_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_c_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_c_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_c_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_c_x, t_c_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( c,t_c_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	chemical_bc		   <<< grid_bc	  , block_t0   , 0, stream0 >>>( c,m,kappa,beta );
	chemical_bc_x	   <<< grid_bc	  , zd-4  	   , 0, stream0 >>>( c,m,kappa,beta );
	boundary_ym_bc	   <<< 1		  , xd		   , 0, stream0 >>>( m );
	boundary_ym_bc_x   <<< 1		  , zd		   , 0, stream0 >>>( m );
////z...
	boundary_zm2	   <<< xd+4	      , ny+4	   , 0, stream0 >>>( m,t_m );
	chemical_in		   <<< grid_in2	  , xd-4  	   , 0, stream1 >>>( c,m,kappa,beta );
////...z	
	cudaMemcpyAsync(t_m_h, t_m, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=110;
	MPI_Sendrecv	((void *)&t_m_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_m_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_m_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_m_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat);
					
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_m, t_m_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( m,t_m );
////x...	
	boundary_xm2	   <<< ny+4       , zd+4	   , 0, stream0 >>>( m,t_m_x );
	boundary_ym_in	   <<< xd-4		  , zd-4	   , 0, stream1 >>>( m );	
	gradient_cen	   <<< grid_t0	  , block_t0   , 0, stream1 >>>( gra_c,c );
////...x
	cudaMemcpyAsync(t_m_x_h, t_m_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=11;
	MPI_Sendrecv	((void *)&t_m_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_m_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_m_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_m_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
					
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_m_x, t_m_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4	   , 0, stream0 >>>( m,t_m_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	gradient_cen	   <<< grid_t0	  , block_t0                >>>( gra_m,m );
	macro_g_bc		   <<< grid_bc 	  , block_t0   , 0, stream0 >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w );
	macro_g_bc_x	   <<< grid_bc	  , zd-4       , 0, stream0 >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w );
////y bc	
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( u );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( v );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( w );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( p );

	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( u );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( v );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( w );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( p );
////z...
	boundary_zm2	   <<< xd+4       , ny+4	   , 0, stream0 >>>( p,t_p );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( u,t_u );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( v,t_v );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( w,t_w );
	macro_g_in		   <<< grid_in2	  , xd-4       , 0, stream1 >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
////...z
	cudaMemcpyAsync(t_p_h, t_p, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=150;
	MPI_Sendrecv	((void *)&t_p_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_p_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=160;
	MPI_Sendrecv	((void *)&t_p_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_p_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat); 

	cudaMemcpyAsync(t_u_h, t_u, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=170;
	MPI_Sendrecv	((void *)&t_u_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_u_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=180;
	MPI_Sendrecv	((void *)&t_u_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_u_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_v_h, t_v, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=190;
	MPI_Sendrecv	((void *)&t_v_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_v_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=200;
	MPI_Sendrecv	((void *)&t_v_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_v_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_w_h, t_w, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=210;
	MPI_Sendrecv	((void *)&t_w_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_w_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=220;
	MPI_Sendrecv	((void *)&t_w_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_w_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_p, t_p_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_u, t_u_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_v, t_v_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_w, t_w_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	
	cudaStreamSynchronize(stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( p,t_p );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( u,t_u );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( v,t_v );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( w,t_w );
////x...
	boundary_xm2	   <<< ny+4       , zd+4       , 0, stream0 >>>( p,t_p_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( u,t_u_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( v,t_v_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( w,t_w_x );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( p );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( u );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( v );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( w );
////...x
	cudaMemcpyAsync(t_p_x_h, t_p_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=15;
	MPI_Sendrecv	((void *)&t_p_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_p_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=16;
	MPI_Sendrecv	((void *)&t_p_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_p_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 

	cudaMemcpyAsync(t_u_x_h, t_u_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=17;
	MPI_Sendrecv	((void *)&t_u_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_u_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=18;
	MPI_Sendrecv	((void *)&t_u_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_u_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_v_x_h, t_v_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=19;
	MPI_Sendrecv	((void *)&t_v_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_v_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=20;
	MPI_Sendrecv	((void *)&t_v_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_v_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_w_x_h, t_w_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=21;
	MPI_Sendrecv	((void *)&t_w_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_w_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=22;
	MPI_Sendrecv	((void *)&t_w_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_w_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_p_x, t_p_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_u_x, t_u_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_v_x, t_v_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_w_x, t_w_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	
	cudaStreamSynchronize(stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( p,t_p_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( u,t_u_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( v,t_v_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( w,t_w_x );
///////////////////////////////////////////////////////////////////////////////////////////
//                                    nnext time step                                    //
///////////////////////////////////////////////////////////////////////////////////////////
	step=step+1;
	cudaDeviceSynchronize();
		
	eq_collision_bc    <<< grid_bc    , block_t0   , 0, stream0 >>>( g_t,h_t,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );
	eq_collision_bc_x  <<< grid_bc    , block_t0_x , 0, stream0 >>>( g_t,h_t,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );	
	boundary_yd_bc     <<< xd         , q          , 0, stream0 >>>( g_t,h_t );
	boundary_yd_bc_x   <<< zd         , q          , 0, stream0 >>>( g_t,h_t );
////z...
	boundary_zd        <<< xd+2       , ny+2       , 0, stream0 >>>( g_t,t_g );
	boundary_zd        <<< xd+2       , ny+2       , 0, stream0 >>>( h_t,t_h );
	eq_collision_in    <<< grid_in    , xd-2       , 0, stream1 >>>( g_t,h_t,c,m,p,gravity,gra_c,gra_m,u,v,w,mobi );
////...z	
	cudaMemcpyAsync(t_g_h, t_g, sizeof(double)*tran_difun , cudaMemcpyDeviceToHost,stream0);
	cudaMemcpyAsync(t_h_h, t_h, sizeof(double)*tran_difun , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	
	itag=230;
	MPI_Sendrecv	((void *)&t_g_h[end_d   ], num_trans_d, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_g_h[startb_d], num_trans_d, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=231;
	MPI_Sendrecv	((void *)&t_g_h[start_d ], num_trans_d, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_g_h[endb_d  ], num_trans_d, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	itag=232;
	MPI_Sendrecv	((void *)&t_h_h[end_d   ], num_trans_d, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_h_h[startb_d], num_trans_d, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=233;
	MPI_Sendrecv	((void *)&t_h_h[start_d ], num_trans_d, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_h_h[endb_d  ], num_trans_d, MPI_DOUBLE, t_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_g, t_g_h, sizeof(double)*tran_difun , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_h, t_h_h, sizeof(double)*tran_difun , cudaMemcpyHostToDevice,stream0);
	boundary_zd_undo   <<< xd+2       , ny+2       , 0, stream0 >>>( g_t,t_g );
	boundary_zd_undo   <<< xd+2       , ny+2       , 0, stream0 >>>( h_t,t_h );
////x...
	boundary_xd        <<< ny+2       , zd+2       , 0, stream0 >>>( g_t,t_g_x );
	boundary_xd        <<< ny+2       , zd+2       , 0, stream0 >>>( h_t,t_h_x );
	boundary_yd_in     <<< xd-2       , zd-2       , 0, stream1 >>>( g_t,h_t );
////...x
	cudaMemcpyAsync(t_g_x_h, t_g_x, sizeof(double)*tran_difun_x , cudaMemcpyDeviceToHost,stream0);
	cudaMemcpyAsync(t_h_x_h, t_h_x, sizeof(double)*tran_difun_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	
	itag=23;
	MPI_Sendrecv	((void *)&t_g_x_h[end_d_x   ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_g_x_h[startb_d_x], num_trans_d_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=24;
	MPI_Sendrecv	((void *)&t_g_x_h[start_d_x ], num_trans_d_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_g_x_h[endb_d_x  ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	itag=25;
	MPI_Sendrecv	((void *)&t_h_x_h[end_d_x   ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_h_x_h[startb_d_x], num_trans_d_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=26;
	MPI_Sendrecv	((void *)&t_h_x_h[start_d_x ], num_trans_d_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_h_x_h[endb_d_x  ], num_trans_d_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_g_x, t_g_x_h, sizeof(double)*tran_difun_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_h_x, t_h_x_h, sizeof(double)*tran_difun_x , cudaMemcpyHostToDevice,stream0);
	boundary_xd_undo   <<< ny+2       , zd+2       , 0, stream0 >>>( g_t,t_g_x );
	boundary_xd_undo   <<< ny+2       , zd+2       , 0, stream0 >>>( h_t,t_h_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	macro_h_bc		   <<< grid_bc	  , block_t0   , 0, stream0 >>>( h_t,h,c );
	macro_h_bc_x	   <<< grid_bc    , zd-4	   , 0, stream0 >>>( h_t,h,c );
	boundary_ym_bc	   <<< 1		  , xd		   , 0, stream0 >>>( c );
	boundary_ym_bc_x   <<< 1		  , zd		   , 0, stream0 >>>( c );
////z...
	boundary_zm2       <<< xd+4	      , ny+4       , 0, stream0 >>>( c,t_c );
	macro_h_in		   <<< grid_in2	  , xd-4       , 0, stream1 >>>( h_t,h,c );
////...z	
	cudaMemcpyAsync(t_c_h, t_c, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=110;
	MPI_Sendrecv	((void *)&t_c_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_c_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_c_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_c_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_c, t_c_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( c,t_c );
////x...
	boundary_xm2       <<< ny+4       , zd+4	   , 0, stream0 >>>( c,t_c_x );
	boundary_ym_in     <<< xd-4       ,	zd-4       , 0, stream1 >>>( c );
////...x
	cudaMemcpyAsync(t_c_x_h, t_c_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=11;
	MPI_Sendrecv	((void *)&t_c_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_c_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_c_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_c_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_c_x, t_c_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( c,t_c_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	chemical_bc		   <<< grid_bc	  , block_t0   , 0, stream0 >>>( c,m,kappa,beta );
	chemical_bc_x	   <<< grid_bc	  , zd-4  	   , 0, stream0 >>>( c,m,kappa,beta );
	boundary_ym_bc	   <<< 1		  , xd		   , 0, stream0 >>>( m );
	boundary_ym_bc_x   <<< 1		  , zd		   , 0, stream0 >>>( m );
////z...
	boundary_zm2	   <<< xd+4	      , ny+4	   , 0, stream0 >>>( m,t_m );
	chemical_in		   <<< grid_in2	  , xd-4  	   , 0, stream1 >>>( c,m,kappa,beta );
////...z	
	cudaMemcpyAsync(t_m_h, t_m, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=110;
	MPI_Sendrecv	((void *)&t_m_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_m_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=120;
	MPI_Sendrecv	((void *)&t_m_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_m_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat);
					
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_m, t_m_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( m,t_m );
////x...	
	boundary_xm2	   <<< ny+4       , zd+4	   , 0, stream0 >>>( m,t_m_x );
	boundary_ym_in	   <<< xd-4		  , zd-4	   , 0, stream1 >>>( m );	
	gradient_cen	   <<< grid_t0	  , block_t0   , 0, stream1 >>>( gra_c,c );
////...x
	cudaMemcpyAsync(t_m_x_h, t_m_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=11;
	MPI_Sendrecv	((void *)&t_m_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_m_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=12;
	MPI_Sendrecv	((void *)&t_m_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_m_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat);
					
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_m_x, t_m_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4	   , 0, stream0 >>>( m,t_m_x );
///////////////////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();

	gradient_cen	   <<< grid_t0	  , block_t0                >>>( gra_m,m );
	macro_g_bc		   <<< grid_bc 	  , block_t0   , 0, stream0 >>>( g_t,g,c,m,p,gra_c,gra_m,u,v,w );
	macro_g_bc_x	   <<< grid_bc	  , zd-4       , 0, stream0 >>>( g_t,g,c,m,p,gra_c,gra_m,u,v,w );
////y bc	
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( u );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( v );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( w );
	boundary_ym_bc	   <<< 1	      , xd         , 0, stream0 >>>( p );

	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( u );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( v );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( w );
	boundary_ym_bc_x   <<< 1	      , zd         , 0, stream0 >>>( p );
////z...
	boundary_zm2	   <<< xd+4       , ny+4	   , 0, stream0 >>>( p,t_p );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( u,t_u );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( v,t_v );
	boundary_zm1	   <<< xd+4       , ny+4	   , 0, stream0 >>>( w,t_w );
	macro_g_in		   <<< grid_in2	  , xd-4       , 0, stream1 >>>( g_t,g,c,m,p,gra_c,gra_m,u,v,w);
////...z
	cudaMemcpyAsync(t_p_h, t_p, sizeof(double)* tran_mac_2 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=150;
	MPI_Sendrecv	((void *)&t_p_h[end   ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_p_h[startb], num_trans_m_2, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=160;
	MPI_Sendrecv	((void *)&t_p_h[start ], num_trans_m_2, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_p_h[endb  ], num_trans_m_2, MPI_DOUBLE, t_nbr, itag, comm, istat); 

	cudaMemcpyAsync(t_u_h, t_u, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=170;
	MPI_Sendrecv	((void *)&t_u_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_u_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=180;
	MPI_Sendrecv	((void *)&t_u_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_u_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_v_h, t_v, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=190;
	MPI_Sendrecv	((void *)&t_v_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_v_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=200;
	MPI_Sendrecv	((void *)&t_v_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_v_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_w_h, t_w, sizeof(double)* tran_mac_1 , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=210;
	MPI_Sendrecv	((void *)&t_w_h[end_1   ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag,
					( void *)&t_w_h[startb_1], num_trans_m_1, MPI_DOUBLE, b_nbr, itag, comm, istat);
	itag=220;
	MPI_Sendrecv	((void *)&t_w_h[start_1 ], num_trans_m_1, MPI_DOUBLE, b_nbr, itag,
					( void *)&t_w_h[endb_1  ], num_trans_m_1, MPI_DOUBLE, t_nbr, itag, comm, istat); 
	
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_p, t_p_h, sizeof(double)* tran_mac_2 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_u, t_u_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_v, t_v_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_w, t_w_h, sizeof(double)* tran_mac_1 , cudaMemcpyHostToDevice,stream0);
	
	cudaStreamSynchronize(stream0);
	boundary_zm2_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( p,t_p );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( u,t_u );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( v,t_v );
	boundary_zm1_undo  <<< xd+4       , ny+4       , 0, stream0 >>>( w,t_w );
////x...
	boundary_xm2	   <<< ny+4       , zd+4       , 0, stream0 >>>( p,t_p_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( u,t_u_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( v,t_v_x );
	boundary_xm1	   <<< ny+4       , zd+4       , 0, stream0 >>>( w,t_w_x );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( p );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( u );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( v );
	boundary_ym_in	   <<< xd-4       ,	zd-4       , 0, stream1 >>>( w );
////...x
	cudaMemcpyAsync(t_p_x_h, t_p_x, sizeof(double)* tran_mac_2_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=15;
	MPI_Sendrecv	((void *)&t_p_x_h[end_x   ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_p_x_h[startb_x], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=16;
	MPI_Sendrecv	((void *)&t_p_x_h[start_x ], num_trans_m_2_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_p_x_h[endb_x  ], num_trans_m_2_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 

	cudaMemcpyAsync(t_u_x_h, t_u_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=17;
	MPI_Sendrecv	((void *)&t_u_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_u_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=18;
	MPI_Sendrecv	((void *)&t_u_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_u_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_v_x_h, t_v_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=19;
	MPI_Sendrecv	((void *)&t_v_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_v_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=20;
	MPI_Sendrecv	((void *)&t_v_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_v_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaMemcpyAsync(t_w_x_h, t_w_x, sizeof(double)* tran_mac_1_x , cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	itag=21;
	MPI_Sendrecv	((void *)&t_w_x_h[end_1_x   ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag,
					( void *)&t_w_x_h[startb_1_x], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag, comm, istat);
	itag=22;
	MPI_Sendrecv	((void *)&t_w_x_h[start_1_x ], num_trans_m_1_x, MPI_DOUBLE, l_nbr, itag,
					( void *)&t_w_x_h[endb_1_x  ], num_trans_m_1_x, MPI_DOUBLE, r_nbr, itag, comm, istat); 
	
	cudaStreamSynchronize(stream0);
	cudaMemcpyAsync(t_p_x, t_p_x_h, sizeof(double)* tran_mac_2_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_u_x, t_u_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_v_x, t_v_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(t_w_x, t_w_x_h, sizeof(double)* tran_mac_1_x , cudaMemcpyHostToDevice,stream0);
	
	cudaStreamSynchronize(stream0);
	boundary_xm2_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( p,t_p_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( u,t_u_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( v,t_v_x );
	boundary_xm1_undo  <<< ny+4       , zd+4       , 0, stream0 >>>( w,t_w_x );
	
	if(condition == 0){
	array_undo <<<grid_t0 , block_t0>>>( c_d,c );
	array_undo <<<grid_t0 , block_t0>>>( w_d,w );
	MPI_Barrier(MPI_COMM_WORLD);
	cudaMemcpy(c_d_h,c_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(w_d_h,w_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather((void *)&c_d_h[0], n_f, MPI_DOUBLE,(void *)&c_f_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&w_d_h[0], n_f, MPI_DOUBLE,(void *)&w_f_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myid==0){
	double	maxw;
	max_w(c_f_h,w_f_h,&maxw);
	Reynolds_Time( maxw, Re, step );
	}}
	
	if(step%iprint ==0){
	
	p_real	   <<<grid_t0 , block_t0>>>(c,p,a,beta,kappa,gra_c);
	
	array_undo <<<grid_t0 , block_t0>>>( c_d,c );
	array_undo <<<grid_t0 , block_t0>>>( m_d,m );
	array_undo <<<grid_t0 , block_t0>>>( b_d,b );
	array_undo <<<grid_t0 , block_t0>>>( p_d,p );
	array_undo <<<grid_t0 , block_t0>>>( u_d,u );
	array_undo <<<grid_t0 , block_t0>>>( v_d,v );
	array_undo <<<grid_t0 , block_t0>>>( w_d,w );
	array_undo <<<grid_t0 , block_t0>>>( a_d,a );
	MPI_Barrier(MPI_COMM_WORLD);
	
	cudaMemcpy(c_d_h,c_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_d_h,m_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(b_d_h,b_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(p_d_h,p_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(u_d_h,u_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(v_d_h,v_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(w_d_h,w_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(a_d_h,a_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Gather((void *)&c_d_h[0], n_f, MPI_DOUBLE,(void *)&c_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&m_d_h[0], n_f, MPI_DOUBLE,(void *)&m_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&b_d_h[0], n_f, MPI_DOUBLE,(void *)&b_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&p_d_h[0], n_f, MPI_DOUBLE,(void *)&p_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&u_d_h[0], n_f, MPI_DOUBLE,(void *)&u_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&v_d_h[0], n_f, MPI_DOUBLE,(void *)&v_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&w_d_h[0], n_f, MPI_DOUBLE,(void *)&w_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&a_d_h[0], n_f, MPI_DOUBLE,(void *)&a_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(myid==0){
		
	array_2D_undo(c_f_h,c_fdo_h);
	array_2D_undo(m_f_h,m_fdo_h);
	array_2D_undo(b_f_h,b_fdo_h);
	array_2D_undo(p_f_h,p_fdo_h);
	array_2D_undo(u_f_h,u_fdo_h);
	array_2D_undo(v_f_h,v_fdo_h);
	array_2D_undo(w_f_h,w_fdo_h);
	array_2D_undo(a_f_h,a_fdo_h);
	
	printf("step=%d\n",step);
	cudaEventRecord(gpu_stop_temp,0);
	cudaEventSynchronize(gpu_stop_temp);
	float cudatime_temp;
	cudaEventElapsedTime(&cudatime_temp,gpu_start_temp,gpu_stop_temp);
	cudatime_temp=cudatime_temp/1000.0;//unit sec
	int remain_time=(int)(cudatime_temp/iprint*(stepall-step));
	printf("time remaining: %d hr,%d min,%d sec\n",(int)remain_time/3600,(int)(remain_time%3600)/60,(int)remain_time%60);
	int indexx;
	printf("c max=%lf\n",maxvalue(c_f_h,&indexx));
	printf("c min=%lf\n",minvalue(c_f_h,&indexx));
	printf("p max=%e\n" ,maxvalue(p_f_h,&indexx));
	printf("u max=%e\n" ,maxvalue(u_f_h,&indexx));
	printf("v max=%e\n" ,maxvalue(v_f_h,&indexx));
	printf("w max=%e\n" ,maxvalue(w_f_h,&indexx));

	data_2d = fopen("data_2d.dat","a");
	fprintf( data_2d, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"p\"\n");
	fprintf( data_2d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_2d, "I=%d, J=%d\n", nx,nz);
	j=ny/2;
	for(k=0;k<nz;k++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	fprintf( data_2d, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t\n",
	i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],p_f_h[index]);
	}}
	fclose(data_2d);
	
	data_2d_t = fopen("data_2d_t.dat","w");
	fprintf( data_2d_t, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"p\"\n");
	fprintf( data_2d_t, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_2d_t, "I=%d, J=%d\n", nx,nz);
	j=ny/2;
	for(k=0;k<nz;k++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	fprintf( data_2d_t, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t\n",
	i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],p_f_h[index]);
	}}
	fclose(data_2d_t);
	
	
	if(step%idata_3d ==0){
	data_3d = fopen("data_3d.dat","a");
	fprintf( data_3d, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\"\n");
	fprintf( data_3d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_3d, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=(nx)*(k*(ny)+j)+i;
	fprintf( data_3d, "%d\t%d\t%d\t%e\t\n",
	i,j,k,c_f_h[index]);
	}}}
	fclose(data_3d);
	
	data_3d_t = fopen("data_3d_t.dat","w");
	fprintf( data_3d_t, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\"\n");
	fprintf( data_3d_t, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_3d_t, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=(nx)*(k*(ny)+j)+i;
	fprintf( data_3d_t, "%d\t%d\t%d\t%e\t\n",
	i,j,k,c_f_h[index]);
	}}}
	fclose(data_3d_t);	
	}
	printf("===============================================================\n");
	}
	cudaEventRecord(gpu_start_temp,0);
	}
	} 
///////////////////////////////////////////////////////////////////////////////////////////
//                                        eend                                           //
///////////////////////////////////////////////////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	cudaEventRecord(gpu_stop,0);
	cudaEventSynchronize(gpu_stop);
	float cudatime;
	if(myid==0){
	printf("===============================================================\n");
	printf("Iteration terminated!\n");
	cudaEventElapsedTime(&cudatime,gpu_start,gpu_stop);
	printf("GPU total time = %f ms\n",cudatime); //unit = ms
	printf("mlups=%lf \n",(double)(nx*ny*nz)*stepall*pow(10.0,-6.0)/(cudatime/1000.0));
	printf("===============================================================\n");
	}
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	
	array_undo <<<grid_t0 , block_t0>>>( c_d,c );
	array_undo <<<grid_t0 , block_t0>>>( m_d,m );
	array_undo <<<grid_t0 , block_t0>>>( b_d,b );
	array_undo <<<grid_t0 , block_t0>>>( p_d,p );
	array_undo <<<grid_t0 , block_t0>>>( u_d,u );
	array_undo <<<grid_t0 , block_t0>>>( v_d,v );
	array_undo <<<grid_t0 , block_t0>>>( w_d,w );
	array_undo <<<grid_t0 , block_t0>>>( a_d,a );
	MPI_Barrier(MPI_COMM_WORLD);
	
	cudaMemcpy(c_d_h,c_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(m_d_h,m_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(b_d_h,b_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(p_d_h,p_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(u_d_h,u_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(v_d_h,v_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(w_d_h,w_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	cudaMemcpy(a_d_h,a_d,sizeof(double)*size_dicom,cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	MPI_Gather((void *)&c_d_h[0], n_f, MPI_DOUBLE,(void *)&c_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&m_d_h[0], n_f, MPI_DOUBLE,(void *)&m_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&b_d_h[0], n_f, MPI_DOUBLE,(void *)&b_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&p_d_h[0], n_f, MPI_DOUBLE,(void *)&p_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&u_d_h[0], n_f, MPI_DOUBLE,(void *)&u_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&v_d_h[0], n_f, MPI_DOUBLE,(void *)&v_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&w_d_h[0], n_f, MPI_DOUBLE,(void *)&w_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Gather((void *)&a_d_h[0], n_f, MPI_DOUBLE,(void *)&a_fdo_h[0],   n_f, MPI_DOUBLE,iroot,comm);
	MPI_Barrier(MPI_COMM_WORLD);
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if(myid==0){
		
	array_2D_undo(c_f_h,c_fdo_h);
	array_2D_undo(m_f_h,m_fdo_h);
	array_2D_undo(b_f_h,b_fdo_h);
	array_2D_undo(p_f_h,p_fdo_h);
	array_2D_undo(u_f_h,u_fdo_h);
	array_2D_undo(v_f_h,v_fdo_h);
	array_2D_undo(w_f_h,w_fdo_h);
	array_2D_undo(a_f_h,a_fdo_h);
	
	final_2d = fopen("final_2d.dat","w");
	fprintf( final_2d, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"p\",\"p_real\"\n");
	fprintf( final_2d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( final_2d, "I=%d, J=%d\n", nx,nz);
	j=ny/2;
	for(k=0;k<nz;k++){
	for(i=0;i<nx;i++){
	index=nx*(k*ny+j)+i;
	fprintf( final_2d, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t\n",
	i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],p_f_h[index],a_f_h[index]);
	}}
	fclose(final_2d);
	
	final_3d = fopen("final_3d.dat","w");
	fprintf( final_3d, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"p\",\"p_real\"\n");
	fprintf( final_3d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( final_3d, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(k=0;k<nz;k++){
	for(j=0;j<ny;j++){
	for(i=0;i<nx;i++){
	index=(nx)*(k*(ny)+j)+i;
	fprintf( final_3d, "%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t\n",
	i,j,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],p_f_h[index],a_f_h[index]);
	}}}
	fclose(final_3d);

	properties = fopen("properties.txt","a");
	fprintf( properties,"MLUPS =%e\n",(double)(nx*ny*nz)*stepall*pow(10.0,-6.0)/(cudatime/1000.0));
 	if(condition == 0){
	FILE *Reynolds;
	double T=sqrt(radd*2/gravity);
	Reynolds = fopen("Reynolds.dat","w");
	fprintf( Reynolds, "VARIABLES=\"T\",\"Reynolds\"\n");
	fprintf( Reynolds, "ZONE T=\"Reynolds\" F=POINT\n");
	fprintf( Reynolds, "I=%d\n", stepall/2);
 	for(i=0;i<stepall/2;i++){
	fprintf( Reynolds, "%e\t%e\n",(double)2*(i+1)/T,Re[i]);}
	fclose ( Reynolds);
	}
	}
	// Free memory
	cudaFreeHost(  c_d_h  );
	cudaFreeHost(  m_d_h  );
	cudaFreeHost(  b_d_h  );
	cudaFreeHost(  p_d_h  );
	cudaFreeHost(  u_d_h  );
	cudaFreeHost(  v_d_h  );
	cudaFreeHost(  w_d_h  );
	cudaFreeHost(  a_d_h  );
	cudaFreeHost(   et_h  );
	cudaFreeHost(   ex_h  );
	cudaFreeHost(   ey_h  );
	cudaFreeHost(   ez_h  );
	cudaFreeHost(   wt_h  );
	cudaFreeHost( t_c_h );
	cudaFreeHost( t_m_h );
	cudaFreeHost( t_b_h );
	cudaFreeHost( t_p_h );
	cudaFreeHost( t_u_h );
	cudaFreeHost( t_v_h );
	cudaFreeHost( t_w_h );
	cudaFreeHost( t_g_h );
	cudaFreeHost( t_h_h );
	if(myid==0){
	cudaFreeHost( c_f_h );
	cudaFreeHost( m_f_h );
	cudaFreeHost( b_f_h );
	cudaFreeHost( p_f_h );
	cudaFreeHost( u_f_h );
	cudaFreeHost( v_f_h );
	cudaFreeHost( w_f_h );
	cudaFreeHost( a_f_h );
	cudaFreeHost( xz_f_h );
	cudaFreeHost( lx );
	cudaFreeHost( lz );
	}
	cudaFreeHost( xz_d_h );

	cudaFree( xz_d  );
	cudaFree(  c_d  );
	cudaFree(  m_d  );
	cudaFree(  b_d  );
	cudaFree(  p_d  );
	cudaFree(  u_d  );
	cudaFree(  v_d  );
	cudaFree(  w_d  );
	cudaFree(  a_d  );
	cudaFree(  h  );
	cudaFree(  g  );
	cudaFree( h_t  );
	cudaFree( g_t  );
	cudaFree( gra_c );
	cudaFree( gra_m );
	cudaFree( t_c );
	cudaFree( t_m );
	cudaFree( t_b );
	cudaFree( t_p );
	cudaFree( t_u );
	cudaFree( t_v );
	cudaFree( t_w );
	cudaFree( t_g );
	cudaFree( t_h );
	cudaFree( c );
	cudaFree( m );
	cudaFree( b );
	cudaFree( p );
	cudaFree( u );
	cudaFree( v );
	cudaFree( w );
	cudaFree( a );

	MPI_Finalize();
	return 0;
	}
