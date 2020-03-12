#ifndef NELMIN_H
#define NELMIN_H

#include "util.h"

// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html

template <typename Real>
void nelmin ( Real (*fn)(Real*), int dimension, Real start[], Real xmin[], Real *worst, Real reqmin, Real step[], int konvge, int iterations_number, int *evaluations_used, int *numres, int *ifault )
{
	const Real contraction_coef = 0.5;
	const Real expansion_coef   = 2.0;
	const Real reflection_coef  = 1.0;

	int index_worst, index_best, l, nn;

	Real dn;
	Real x, y2star, best, obj_reflection, z;

	Real dnn, rq;
	const Real eps = 0.001;
	int jcount;
	
	std::vector<Real> simplex(dimension * (dimension + 1)); // p_simplex 
	std::vector<Real> centroid(dimension);

	std::vector<Real> reflection(dimension);
	std::vector<Real> p2star(dimension);

	std::vector<Real> obj_function(dimension + 1); // p_obj_function

	*evaluations_used = 0;
	

	//	jcount = konvge; 
	//  Initial or restarted loop.
	for ( ; ; ) {
		
		for (int i = 0; i < dimension; i++ ) {
			simplex[i + dimension * dimension] = start[i]; 
		}
		
		obj_function[dimension] = (*fn)( start );

		*evaluations_used = *evaluations_used + 1;

		for (int j = 0; j < dimension; j++ ) {
			x = start[j];
			start[j] = start[j] + step[j];
			
			for (int i = 0; i < dimension; i++ ) { 
				simplex[i + j * dimension] = start[i]; 
			}

			obj_function[j] = (*fn)( start );
			*evaluations_used = *evaluations_used + 1;
			start[j] = x;
		}
		//  The simplex construction is complete.
		//                    
		//  Find highest and lowest Y values.  worst = Y(index_worst) indicates
		//  the vertex of the simplex to be replaced.
		best = obj_function[0];
		index_best = 0;

		for (int i = 1; i < dimension + 1; i++ ) {
			if ( obj_function[i] < best ) { 
				best = obj_function[i]; 
				index_best = i; 
			}
		}
		//  Inner loop.
		for ( ; ; ) {
			//printf("%d - %d\n", iterations_number, *evaluations_used);
			if ( iterations_number <= *evaluations_used ) { 
				break; 
			}

			*worst = obj_function[0];
			index_worst = 0;
			
			for (int i = 1; i < dimension + 1; i++ ) {
				if ( obj_function[i]  > *worst ) { 
					*worst = obj_function[i]; 
					index_worst = i; 
				}
			}
			//  Calculate centroid, the centroid of the simplex vertices
			//  excepting the vertex with Y value worst.
			for (int i = 0; i < dimension; i++ ) {
				z = 0.0;
				
				for (int j = 0; j < dimension + 1; j++ ) { 
					z = z + simplex[i + j * dimension]; 
				}

				z = z - simplex[i + index_worst * dimension];  
				centroid[i] = z / dimension;
			}
			//  Reflection through the centroid.
			for (int i = 0; i < dimension; i++ ) {
				reflection[i] = centroid[i] + reflection_coef * ( centroid[i] - simplex[i + index_worst * dimension] );
			}

			obj_reflection = (*fn)( &reflection[0] );
			*evaluations_used = *evaluations_used + 1;
			//  Successful reflection, so extension.
			if ( obj_reflection < best ) {
				for (int i = 0; i < dimension; i++ ) {
					p2star[i] = centroid[i] + expansion_coef * ( reflection[i] - centroid[i] );
				}
				y2star = (*fn)( &p2star[0] );
				*evaluations_used = *evaluations_used + 1;
			//  Check extension.
				if ( y2star < obj_reflection ) {
					for (int i = 0; i < dimension; i++ ) { 
						simplex[i + index_worst * dimension] = p2star[i]; 
					}
					obj_function[index_worst] = y2star;
					
				} else { //  Retain extension or contraction.
					for (int i = 0; i < dimension; i++ ) { 
						simplex[i + index_worst * dimension] = reflection[i]; 
					}
					obj_function[index_worst] = obj_reflection;
				}
			} else { //  No extension.
				l = 0;
				for (int i = 0; i < dimension + 1; i++ ) {
					if ( obj_reflection < obj_function[i] ) {
						l += 1;
					}
				}

				if ( l > 1 ) {
					for (int i = 0; i < dimension; i++ ) { 
						simplex[i + index_worst * dimension] = reflection[i]; 
					}
					obj_function[index_worst] = obj_reflection;
				}
				//  Contraction on the Y(index_worst) side of the centroid.
				else if ( l == 0 ) {

					for (int i = 0; i < dimension; i++ ) {
						p2star[i] = centroid[i] + contraction_coef * ( simplex[i + index_worst * dimension] - centroid[i] );
					}
					y2star = (*fn)( &p2star[0] );
					*evaluations_used = *evaluations_used + 1;
			//  Contract the whole simplex.
					if ( obj_function[index_worst] < y2star ) {
						for (int j = 0; j < dimension + 1; j++ ) {
							for (int i = 0; i < dimension; i++ ) {
								simplex[i + j * dimension] = ( simplex[i + j * dimension] + simplex[i + index_best * dimension] ) * 0.5;
								xmin[i] = simplex[i + j * dimension];
							}
							obj_function[j] = (*fn)( xmin );
							*evaluations_used = *evaluations_used + 1;
						}
						best = obj_function[0];
						index_best = 0;
					
						for (int i = 1; i < dimension + 1; i++ ) {
							if ( obj_function[i] < best ) { 
								best = obj_function[i]; 
								index_best = i; 
							}
						}
						continue;
					}
			//  Retain contraction.
					else {
						for (int i = 0; i < dimension; i++ ) {
							simplex[i + index_worst * dimension] = p2star[i];
						}
						obj_function[index_worst] = y2star;
					}
				}
			//  Contraction on the reflection side of the centroid.
				else if ( l == 1 ) {
					for (int i = 0; i < dimension; i++ ) {
						p2star[i] = centroid[i] + contraction_coef * ( reflection[i] - centroid[i] );
					}
					y2star = (*fn)( &p2star[0] );
					*evaluations_used = *evaluations_used + 1;
			//
			//  Retain reflection?
			//
					if ( y2star <= obj_reflection ) {
						for (int i = 0; i < dimension; i++ ) { 
							simplex[i + index_worst * dimension] = p2star[i]; 
						}
						simplex[index_worst] = y2star;
					}
					else {
						for (int i = 0; i < dimension; i++ ) { 
							simplex[i + index_worst * dimension] = reflection[i]; 
						}
						simplex[index_worst] = obj_reflection;
					}
				}
			}
			//  Check if best improved.
			if ( simplex[index_worst] < best ) { 
				best = simplex[index_worst]; 
				index_best = index_worst; 
			}
			
			/*
			jcount = jcount - 1;
			
			if ( jcount > 0 ) { 
				continue; 
			}

			//  Check to see if minimum reached.
			if ( *evaluations_used <= iterations_number ) {
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
			*/
			
		}

		/*
		//  Factorial tests to check that worst is a local minimum.
		for (int i = 0; i < n; i++ ) { 
			xmin[i] = p[i+index_best*n]; 
		}

		*worst = y[index_best];
		*/

		if ( iterations_number < *evaluations_used ) { 
			*ifault = 2; 
			break; 
		}
		
		/*

		
		*ifault = 0;

		for (int i = 0; i < n; i++ ) {
			del = step[i] * eps;
			xmin[i] = xmin[i] + del;
			z = (*fn)( xmin );
			*evaluations_used = *evaluations_used + 1;
			
			if ( z < *worst ) { 
				*ifault = 2; break; 
			}

			xmin[i] = xmin[i] - del - del;
			z = (*fn)( xmin );
			*evaluations_used = *evaluations_used + 1;
			
			if ( z < *worst ) { 
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
		*/
		
	}
	return;
}
#endif
