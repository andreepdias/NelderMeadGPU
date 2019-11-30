#ifndef NELMIN_H
#define NELMIN_H

#include "util.h"

__constant__ char aminoacid_sequence[150];


__global__ void nelderMead_initialize(int dimension, float * p_simplex, float step, float * start){

    const int blockId = blockIdx.x;
    const int threadId = threadIdx.x;
    const int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

__global__ void nelderMead_calculate(int protein_length, int dimension, float * p_simplex, float * p_objective_function){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;
	const int threadsMax = protein_length - 2;
	const int stride = blockId * dimension;
	
	__shared__ float aminoacid_position[150 * 3];

	if(threadId == 0){
		aminoacid_position[0] = 0.0f;
		aminoacid_position[0 + protein_length] = 0.0f;
		aminoacid_position[0 + protein_length * 2] = 0.0f;
	
		aminoacid_position[1] = 0.0f;
		aminoacid_position[1 + protein_length] = 1.0f; 
		aminoacid_position[1 + protein_length * 2] = 0.0f;
	
		aminoacid_position[2] = cosf(p_simplex[stride + 0]);
		aminoacid_position[2 + protein_length] = sinf(p_simplex[stride + 0]) + 1.0f;
		aminoacid_position[2 + protein_length * 2] = 0.0f;
	
		for(int i = 3; i < protein_length; i++){
			aminoacid_position[i] = aminoacid_position[i - 1] + cosf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]); // i - 3 + protein_length - 2
			aminoacid_position[i + protein_length] = aminoacid_position[i - 1 + protein_length] + sinf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]);
			aminoacid_position[i + protein_length * 2] = aminoacid_position[i - 1 + protein_length * 2] + sinf(p_simplex[stride + i + protein_length - 5]);
		}
	}

	__syncthreads();

	float sum = 0.0f, c, d, dx, dy, dz;
	sum += (1.0f - cosf(p_simplex[stride + threadId])) / 4.0f;

	for(unsigned int i = threadId + 2; i < protein_length; i++){

		if(aminoacid_sequence[threadId] == 'A' && aminoacid_sequence[i] == 'A')
			c = 1.0;
		else if(aminoacid_sequence[threadId] == 'B' && aminoacid_sequence[i] == 'B')
			c = 0.5;
		else
			c = -0.5;

		dx = aminoacid_position[threadId] - aminoacid_position[i];
		dy = aminoacid_position[threadId + protein_length] - aminoacid_position[i + protein_length];
		dz = aminoacid_position[threadId + protein_length * 2] - aminoacid_position[i + protein_length * 2];
		d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
		
		sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );		
	}
	__syncthreads();

	__shared__ float threads_sum [128];
	threads_sum[threadId] = sum;
  
	if(threadId < 64 && threadId + 64 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 64];
	}  
	__syncthreads();

	if(threadId < 32 && threadId + 32 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 32];
	}  
	__syncthreads();
  
	if(threadId < 16 && threadId + 16 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 16];
	}  
	__syncthreads();
  
	if(threadId < 8 && threadId + 8 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 8];
	}  
	__syncthreads();
  
	if(threadId < 4 && threadId + 4 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 4];
	}  
	__syncthreads();
  
	if(threadId < 2 && threadId + 2 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 2];
	}  
	__syncthreads();
  
	if(threadId == 0){
		threads_sum[threadId] += threads_sum[threadId + 1];
  
		p_objective_function[blockId] = threads_sum[0];
	}
}

__global__ void nelderMead_centroid(float * p_centroid, float * p_simplex, uint * p_indexes, const int dimension, const int p){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;
	const int threadsMax = dimension - p + 1;

	const int index = p_indexes[dimension - (dimension - p) + threadId];
	const int stride = index * dimension;

	float value = p_simplex[stride + blockId];

	__syncthreads();

	__shared__ float threads_sum [256];
	threads_sum[threadId] = value;
  
	if(threadId < 128 && threadId + 128 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 128];
	}  
	__syncthreads();
	if(threadId < 64 && threadId + 64 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 64];
	}  
	__syncthreads();

	if(threadId < 32 && threadId + 32 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 32];
	}  
	__syncthreads();
  
	if(threadId < 16 && threadId + 16 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 16];
	}  
	__syncthreads();
  
	if(threadId < 8 && threadId + 8 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 8];
	}  
	__syncthreads();
  
	if(threadId < 4 && threadId + 4 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 4];
	}  
	__syncthreads();
  
	if(threadId < 2 && threadId + 2 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 2];
	}  
	__syncthreads();
	
	if(threadId == 0){
		threads_sum[threadId] += threads_sum[threadId + 1];
  
	  	p_centroid[blockId] = threads_sum[0] / (threadsMax);
	}
}

__global__ void nelderMead_reflection(float * p_simplex_reflected, float * p_centroid, float * p_simplex, uint * p_indexes, int dimension, int p, float reflection_coef){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;

	const int index = p_indexes[blockId + p];
	const int stride = index * dimension;

	p_simplex_reflected[blockId * dimension + threadId] = p_centroid[threadId] + reflection_coef * (p_centroid[threadId] - p_simplex[stride + threadId]);
}



__global__ void nelderMead_update(float * p_simplex_reflected, float * p_centroid, float * p_simplex, uint * p_indexes, float * p_objective_function, float * p_objective_function_reflected, int dimension, int p, float reflection_coef){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;

	const int index = blockId;
	const int stride = index * dimension;

	const float best = p_objective_function[0];

	if(p_objective_function_reflected[index] > best){
		nelderMead_updateExtension<<< 1, dimension >>>();
		cudaDeviceSynchronize();
	}else{
		const int next_index = (index + 1) % p;

		if(p_objective_function_reflected[index] < p_objective_function[next_index + p]){
			nelderMead_updateReplace<<< >>>();
			cudaDeviceSynchronize();

		}else{
			const bool use_reflected = false;

			if(p_objective_function_reflected[index] < p_objective_function[index + p]){
				use_reflected = true;
			}
			nelderMead_updateContract<<< >>>();
			cudaDeviceSynchronize();

		}
	}
}

void nelderMead(int dimension, int protein_length, float start[]){

	const int n = dimension;
	const int psl = protein_length;
	const int p = 2;
	
	const float step = 1.0f;
	const float reflection_coef = 1.0f;

    thrust::device_vector<float> d_objective_function(n + 1);
	thrust::device_vector<float> d_simplex(n * (n + 1));
	thrust::device_vector<float> d_start(n);
	thrust::device_vector<float> d_centroid(n);
	thrust::device_vector<uint>  d_indexes(n + 1);
	thrust::device_vector<float> d_simplex_reflected(n * p);
    thrust::device_vector<float> d_objective_function_reflected(p);


	float * p_objective_function 		   = thrust::raw_pointer_cast(&d_objective_function[0]);
	float * p_simplex			 		   = thrust::raw_pointer_cast(&d_simplex[0]);
	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
	float * p_centroid 			 		   = thrust::raw_pointer_cast(&d_centroid[0]);
	uint  * p_indexes 				       = thrust::raw_pointer_cast(&d_indexes[0]);
	float * p_simplex_reflected	 		   = thrust::raw_pointer_cast(&d_simplex_reflected[0]);
	float * p_objective_function_reflected = thrust::raw_pointer_cast(&d_objective_function_reflected[0]);

	thrust::copy(start, start + n, d_start.begin());

	nelderMead_initialize<<< n + 1, n >>>(dimension, p_simplex, step, p_start);

	nelderMead_calculate<<< n + 1, psl - 2 >>>(psl, n, p_simplex, p_objective_function);
	
	thrust::sequence(d_indexes.begin(), d_indexes.end());
	thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	
	nelderMead_centroid<<< n, n + 1 - p >>>(p_centroid, p_simplex, p_indexes, dimension, p);
	
	nelderMead_reflection<<< p, n >>>(p_simplex_reflected, p_centroid, p_simplex, p_indexes, dimension, p, reflection_coef);
	
	nelderMead_calculate<<< p, psl - 2 >>>(psl, n, p_simplex_reflected, p_objective_function_reflected);
	
	nelderMead_update<<< p, 1 >>>(p_simplex_reflected, p_centroid, p_simplex, p_indexes, p_objective_function, p_objective_function_reflected, dimension, p, reflection_coef);



	/*
	thrust::host_vector<float> h_objective_function = d_objective_function;
	thrust::host_vector<float> h_objective_function_reflected = d_objective_function_reflected;

	thrust::host_vector<float> h_simplex = d_simplex;
	thrust::host_vector<float> h_simplex_reflected = d_simplex_reflected;
	thrust::host_vector<uint>  h_indexes = d_indexes;


	for(int i = 0; i < n + 1; i++){
		printf("%d. %5.5f\n", i + 1, h_objective_function[i]);

		for(int j = 0; j < dimension; j++){
			printf("%5.5f ", h_simplex[h_indexes[j]]);
		}
		printf("\n");
	}
	printf("\n");
	for(int i = 0; i < p; i++){
		printf("%d. %5.5f\n", i + 1, h_objective_function_reflected[i]);

		for(int j = 0; j < dimension; j++){
			printf("%5.5f ", h_simplex_reflected[j]);
		}
		printf("\n");
	}
	*/

}

/*
void nelmin ( float (*fn)(float*), int n, float start[], float xmin[], float *ynewlo, float reqmin, float step[], int konvge, int kcount, int *icount, int *numres, int *ifault )
{
	const float ccoeff = 0.5;
	const float ecoeff = 2.0;
	const float rcoeff = 1.0;
	int ihi,ilo,l,nn;
	float del,dn;
	float x,y2star,ylo,ystar,z;

	
	float dnn, rq;
	const float eps = 0.001;
	int jcount;
	

	//  Check the input parameters.
	if ( reqmin <= 0.0 ) { 
		*ifault = 1; 
		return; 
	}
	if ( n < 1 ) { 
		*ifault = 1; 
		return; 
	}
	if ( konvge < 1 ) { 
		*ifault = 1; 
		return; 
	}

	std::vector<float> p(n*(n+1));
	std::vector<float> pstar(n);
	std::vector<float> p2star(n);
	std::vector<float> pbar(n);
	std::vector<float> y(n+1);

	*icount = 0;
	*numres = 0;
	
	
	dn = ( float ) ( n );
	nn = n + 1;
	del = 1.0;

	jcount = konvge; 
	dnn = ( float ) ( nn );
	rq = reqmin * dn;
	//  Initial or restarted loop.
	for ( ; ; ) {
		
		for (int i = 0; i < n; i++ ) {
			p[i+n*n] = start[i]; 
		}
		
		y[n] = (*fn)( start );

		*icount = *icount + 1;

		for (int j = 0; j < n; j++ ) {
			x = start[j];
			start[j] = start[j] + step[j] * del;
			
			for (int i = 0; i < n; i++ ) { 
				p[i+j*n] = start[i]; 
			}

			y[j] = (*fn)( start );
			*icount = *icount + 1;
			start[j] = x;
		}
		//  The simplex construction is complete.
		//                    
		//  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
		//  the vertex of the simplex to be replaced.
		ylo = y[0];
		ilo = 0;

		for (int i = 1; i < nn; i++ ) {
			if ( y[i] < ylo ) { 
				ylo = y[i]; 
				ilo = i; 
			}
		}
		//  Inner loop.
		for ( ; ; ) {
			if ( kcount <= *icount ) { 
				break; 
			}

			*ynewlo = y[0];
			ihi = 0;
			
			for (int i = 1; i < nn; i++ ) {
				if ( *ynewlo < y[i] ) { 
					*ynewlo = y[i]; 
					ihi = i; 
				}
			}
			//  Calculate PBAR, the centroid of the simplex vertices
			//  excepting the vertex with Y value YNEWLO.
			for (int i = 0; i < n; i++ ) {
				z = 0.0;
				
				for (int j = 0; j < nn; j++ ) { 
					z = z + p[i+j*n]; 
				}

				z = z - p[i+ihi*n];  
				pbar[i] = z / dn;
			}
			//  Reflection through the centroid.
			for (int i = 0; i < n; i++ ) {
				pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
			}

			ystar = (*fn)( &pstar[0] );
			*icount = *icount + 1;
			//  Successful reflection, so extension.
			if ( ystar < ylo ) {
				for (int i = 0; i < n; i++ ) {
					p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
				}
				y2star = (*fn)( &p2star[0] );
				*icount = *icount + 1;
			//  Check extension.
				if ( ystar < y2star ) {
					for (int i = 0; i < n; i++ ) { 
						p[i+ihi*n] = pstar[i]; 
					}
					y[ihi] = ystar;
				} else { //  Retain extension or contraction.
					for (int i = 0; i < n; i++ ) { 
						p[i+ihi*n] = p2star[i]; 
					}
					y[ihi] = y2star;
				}
			} else { //  No extension.
				l = 0;
				for (int i = 0; i < nn; i++ ) {
					if ( ystar < y[i] ) {
						l += 1;
					}
				}

				if ( 1 < l ) {
					for (int i = 0; i < n; i++ ) { 
						p[i+ihi*n] = pstar[i]; 
					}
					y[ihi] = ystar;
				}
				//  Contraction on the Y(IHI) side of the centroid.
				else if ( l == 0 ) {
					for (int i = 0; i < n; i++ ) {
						p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
					}
					y2star = (*fn)( &p2star[0] );
					*icount = *icount + 1;
			//  Contract the whole simplex.
					if ( y[ihi] < y2star ) {
						for (int j = 0; j < nn; j++ ) {
							for (int i = 0; i < n; i++ ) {
								p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
								xmin[i] = p[i+j*n];
							}
							y[j] = (*fn)( xmin );
							*icount = *icount + 1;
						}
						ylo = y[0];
						ilo = 0;
					
						for (int i = 1; i < nn; i++ ) {
							if ( y[i] < ylo ) { ylo = y[i]; ilo = i; }
						}
						continue;
					}
			//  Retain contraction.
					else {
						for (int i = 0; i < n; i++ ) {
							p[i+ihi*n] = p2star[i];
						}
						y[ihi] = y2star;
					}
				}
			//  Contraction on the reflection side of the centroid.
				else if ( l == 1 ) {
					for (int i = 0; i < n; i++ ) {
						p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
					}
					y2star = (*fn)( &p2star[0] );
					*icount = *icount + 1;
			//
			//  Retain reflection?
			//
					if ( y2star <= ystar ) {
						for (int i = 0; i < n; i++ ) { 
							p[i+ihi*n] = p2star[i]; 
						}
						y[ihi] = y2star;
					}
					else {
						for (int i = 0; i < n; i++ ) { 
							p[i+ihi*n] = pstar[i]; 
						}
						y[ihi] = ystar;
					}
				}
			}
			//  Check if YLO improved.
			if ( y[ihi] < ylo ) { 
				ylo = y[ihi]; ilo = ihi; 
			}
			
			
			jcount = jcount - 1;
			
			if ( 0 < jcount ) { 
				continue; 
			}

			//  Check to see if minimum reached.
			if ( *icount <= kcount ) {
				jcount = konvge;

				z = 0.0;
				for (int i = 0; i < nn; i++ ) { 
					z = z + y[i]; 
				}
				x = z / dnn;

				z = 0.0;
				for (int i = 0; i < nn; i++ ) {
					z = z + pow ( y[i] - x, 2 );
				}

				if ( z <= rq ) {
					break;
				}
			}
			
		}

		//  Factorial tests to check that YNEWLO is a local minimum.
		for (int i = 0; i < n; i++ ) { 
			xmin[i] = p[i+ilo*n]; 
		}

		*ynewlo = y[ilo];

		if ( kcount < *icount ) { 
			*ifault = 2; 
			break; 
		}
		

		
		*ifault = 0;

		for (int i = 0; i < n; i++ ) {
			del = step[i] * eps;
			xmin[i] = xmin[i] + del;
			z = (*fn)( xmin );
			*icount = *icount + 1;
			
			if ( z < *ynewlo ) { 
				*ifault = 2; break; 
			}

			xmin[i] = xmin[i] - del - del;
			z = (*fn)( xmin );
			*icount = *icount + 1;
			
			if ( z < *ynewlo ) { 
				*ifault = 2; 
				break; 
			}

			xmin[i] = xmin[i] + del;
		}

		if ( *ifault == 0 ) { break; }
		//  Restart the procedure.
		for (int i = 0; i < n; i++ ) { 
			start[i] = xmin[i]; 
		}

		del = eps;
		*numres = *numres + 1;
		
	}
	return;
}
#endif
*/

#endif