#ifndef NELMIN_H
#define NELMIN_H

#include "util.hpp"
#include "print.hpp"


// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html



struct Calculate3DAB{
    int protein_length;

    float * p_vertex;
    float * aminoacid_position;

	const char * aminoacid_sequence;

    Calculate3DAB(float * _p_vertex, float * _aminoacid_position, const char * _aminoacid_sequence, int _protein_length){
        p_vertex = _p_vertex, 
		aminoacid_position = _aminoacid_position, 
		protein_length = _protein_length;
		aminoacid_sequence = _aminoacid_sequence;
	}
    
    float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d, dx, dy, dz;

		float s = 0.0f;

        sum += (1.0f - cosf(p_vertex[id])) / 4.0f;

        for(int i = id + 2; i < protein_length; i++){

			s = 0;

            if(aminoacid_sequence[id] == 'A' && aminoacid_sequence[i] == 'A')
                c = 1.0;
            else if(aminoacid_sequence[id] == 'B' && aminoacid_sequence[i] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = aminoacid_position[id] - aminoacid_position[i];
            dy = aminoacid_position[id + protein_length] - aminoacid_position[i + protein_length];
            dz = aminoacid_position[id + protein_length * 2] - aminoacid_position[i + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );

            s += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
			sum += s;
                
        }
        return sum;
    }
};


void calculateCoordinates(float * p_vertex, float * aminoacid_position, int protein_length){

	aminoacid_position[0] = 0.0f;
	aminoacid_position[0 + protein_length] = 0.0f;
	aminoacid_position[0 + protein_length * 2] = 0.0f;

	aminoacid_position[1] = 0.0f;
	aminoacid_position[1 + protein_length] = 1.0f; 
	aminoacid_position[1 + protein_length * 2] = 0.0f;

	aminoacid_position[2] = cosf(p_vertex[0]);
	aminoacid_position[2 + protein_length] = sinf(p_vertex[0]) + 1.0f;
	aminoacid_position[2 + protein_length * 2] = 0.0f;

	for(int i = 3; i < protein_length; i++){
		aminoacid_position[i] = aminoacid_position[i - 1] + cosf(p_vertex[i - 2]) * cosf(p_vertex[i + protein_length - 5]); // i - 3 + protein_length - 2
		aminoacid_position[i + protein_length] = aminoacid_position[i - 1 + protein_length] + sinf(p_vertex[i - 2]) * cosf(p_vertex[i + protein_length - 5]);
		aminoacid_position[i + protein_length * 2] = aminoacid_position[i - 1 + protein_length * 2] + sinf(p_vertex[i + protein_length - 5]);
	}
}

float calculate3DABOffLattice(float * p_vertex, void * problem_parameters){

	ABOffLattice * parametersAB = (ABOffLattice*)problem_parameters;

    int protein_length = (*parametersAB).protein_length;
	std::vector<float> aminoacid_position(protein_length * 3);
	
	calculateCoordinates(p_vertex, &aminoacid_position[0], protein_length);

	Calculate3DAB unary_op(p_vertex, &aminoacid_position[0], (*parametersAB).aminoacid_sequence, protein_length);

	float sum = 0;

	for(int i = 0; i < protein_length - 2; i++){
		sum += unary_op(i);
	}
	return sum;
}


void nelderMead_initialize(int dimension, float * p_simplex, float * p_start, float step){

	for(int i = 0; i < dimension + 1; i++){

		int stride = i * dimension;

		for(int j = 0; j < dimension; j++){

			if(i != j){
				p_simplex[stride + j] = p_start[j];
			}else{
				p_simplex[stride + j] = p_start[j] + step;
			}
		}
	}
}

void nelderMead_calculateSimplex(int dimension, int &evaluations_used, float * p_simplex, float * p_obj_function, void * problem_parameters = NULL){

	for(int i =  0; i < dimension + 1; i++){
		p_obj_function[i] = calculate3DABOffLattice(p_simplex + (i * dimension), problem_parameters);
		// p_obj_function[i] = (*fn)(p_simplex + (i * dimension));
		evaluations_used += 1;
	}
}

void nelderMead_calculateVertex(int &evaluations_used, float &obj, float * p_vertex, void * problem_parameters = NULL){
	
	obj = calculate3DABOffLattice(p_vertex, problem_parameters);
	evaluations_used = evaluations_used + 1;
	// obj = (*fn)( &p_vertex[0] );
}


void nelderMead_findBest(int dimension, float &best, int &index_best, float * p_obj_function){

	best = p_obj_function[0];
	index_best = 0;

	for(int i =  1; i < dimension + 1; i++){
		if(p_obj_function[i] < best){
			best = p_obj_function[i];
			index_best = i;
		}
	}
}

void nelderMead_findWorst(int dimension, float &worst, int &index_worst, float * p_obj_function){

	worst = p_obj_function[0];
	index_worst = 0;
	
	for (int i = 1; i < dimension + 1; i++ ) {
		if ( worst < p_obj_function[i] ) { 
			worst = p_obj_function[i]; 
			index_worst = i; 
		}
	}
}

void nelderMead_centroid(int dimension, int index_worst, float * p_simplex, float * p_centroid){

	float sum;
	for (int i = 0; i < dimension; i++ ) {
		sum = 0.0;
		
		for (int j = 0; j < dimension + 1; j++ ) { 
			sum += p_simplex[j *  dimension + i];
		}
		sum -= p_simplex[index_worst * dimension + i];
		p_centroid[i] = sum / dimension;
	}
}

void nelderMead_reflection(int dimension, int index_worst, float reflection_coef, float * p_simplex, float * p_centroid, float * p_reflection){

	for (int i = 0; i < dimension; i++ ) {
		p_reflection[i] = p_centroid[i] + reflection_coef * ( p_centroid[i] - p_simplex[i + index_worst * dimension] );
	}
}

void nelderMead_expansion(int dimension, float expansion_coef, float * p_centroid, float * p_reflection, float * p_expansion){

	for (int i = 0; i < dimension; i++ ) {
		p_expansion[i] = p_centroid[i] + expansion_coef * ( p_reflection[i] - p_centroid[i] );
	}
}

void nelderMead_replacement(int dimension, int index, float * p_simplex, float * p_vertex, float obj, float * p_obj_function){

	for (int i = 0; i < dimension; i++ ) { 
		p_simplex[index * dimension + i] = p_vertex[i]; 
	}
	p_obj_function[index] = obj;
}

void nelderMead_contraction(int dimension, float contraction_coef, float * p_centroid, int index, float * p_simplex, float * p_vertex){

	for (int i = 0; i < dimension; i++ ) {
	 	p_vertex[i] = p_centroid[i] + contraction_coef * ( p_simplex[index * dimension + i] - p_centroid[i] );
	}
}

void nelderMead_shrink(int dimension, int index_best, float * p_simplex){

	int stride_best = index_best * dimension;

	for(int i = 0; i < dimension + 1; i++){

		int stride = i * dimension;
		for(int j = 0; j < dimension; j++){
			 p_simplex[stride + j] = (p_simplex[stride + j] + p_simplex[index_best * dimension + j]) * 0.5f;
			// p_simplex[stride + j] = 0.5 * p_simplex[stride_best + j] + (1.0 - 0.5) * p_simplex[stride + j];
		}
	}
}

NelderMeadResult nelderMead (NelderMead &parameters, void * problem_parameters = NULL)
{

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 2.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;

	int index_worst, index_best;

	float best, worst;
	float obj_reflection, obj_vertex;

	
	std::vector<float> simplex(dimension * (dimension + 1)); // p_simplex 
	std::vector<float> centroid(dimension);

	std::vector<float> reflection(dimension);
	std::vector<float> vertex(dimension);

	std::vector<float> obj_function(dimension + 1); // p_obj_function

	float * p_simplex 		 = &simplex[0];
	float * p_centroid 		 = &centroid[0];
	float * p_reflection 	 = &reflection[0];
	float * p_vertex 		 = &vertex[0];
	float * p_obj_function	 = &obj_function[0];

	nelderMead_initialize(dimension, p_simplex, parameters.p_start, parameters.step);
	nelderMead_calculateSimplex(dimension, parameters.evaluations_used, p_simplex, p_obj_function, problem_parameters);

	// /*p*/printSimplex(dimension, p_simplex, "Initialize");
	// /*p*/printObjFunction(dimension, p_obj_function, "Obj Function Initialize");

	nelderMead_findBest(dimension, best, index_best, p_obj_function);

	// /*p*/printf("best: %.5f, index_best: %d\n\n", best, index_best);	

	for (int k = 0; k < parameters.iterations_number; k++) {

		// /*p*/printf("--------------------- STARTING ITERATION %d ---------------------\n\n", k);

		// /*p*/printObjFunction(dimension, p_obj_function, "Obj Function");

		nelderMead_findWorst(dimension, worst, index_worst, p_obj_function);
		
		// /*p*/printf("%d. worst: %.5f, index_worst: %d\n", k, worst, index_worst);

		nelderMead_centroid(dimension, index_worst, p_simplex, p_centroid);

		// /*p*/printVertex(dimension, p_centroid, "Centroid");

		nelderMead_reflection(dimension, index_worst, parameters.reflection_coef, p_simplex, p_centroid, p_reflection);
		nelderMead_calculateVertex(parameters.evaluations_used, obj_reflection, p_reflection, problem_parameters);
	
		// /*p*/printVertex(dimension, p_reflection, "Reflection");
		// /*p*/printSingleObjFunction(obj_reflection, "Obj Function Reflection");

		if(obj_reflection < best){
			nelderMead_expansion(dimension, parameters.expansion_coef, p_centroid, p_reflection, p_vertex);
			nelderMead_calculateVertex(parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters);

			// /*p*/printVertex(dimension, p_vertex, "Expansion");
			// /*p*/printSingleObjFunction(obj_vertex, "Obj Function Expansion");


			if(obj_vertex < best){
				nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);

				// /*p*/printSimplex(dimension, p_simplex, "Case 1a: Replacement Expansion");

			}else{
				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);

				// /*p*/printSimplex(dimension, p_simplex, "Case 1b: Replacement Reflection");
			}
		}else{
			int c = 0;
			for(int i = 0; i < dimension + 1; i++){
				if(obj_reflection < p_obj_function[i]){
					c++;
				}
			}
			// /*p*/printf("%d. c: %d\n", k, c);

			/* Se reflection melhor que segundo pior vértice (e pior) */
			if(c >= 2){
				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);

				// /*p*/printSimplex(dimension, p_simplex, "Case 2: Replacement Reflection");
			}else{

				if(obj_reflection < worst){
					nelderMead_contraction(dimension, parameters.contraction_coef, p_centroid, 0, p_reflection, p_vertex);
					
					// /*p*/printVertex(dimension, p_vertex, "Contraction (reflection better)");
				}else{
					nelderMead_contraction(dimension, parameters.contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);

					// /*p*/printVertex(dimension, p_vertex, "Contraction (worst better)");
				}
				nelderMead_calculateVertex(parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters);

				// /*p*/printSingleObjFunction(obj_vertex, "Obj Function Contraction");	

				 if(obj_vertex < obj_reflection and obj_vertex < worst){
					nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);

					// /*p*/printSimplex(dimension, p_simplex, "Case 3a: Replacement Contraction");
				 }else if(obj_reflection < worst){
					nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);

					// /*p*/printSimplex(dimension, p_simplex, "Case 3b: Replacement Reflection");
				}else{
					nelderMead_shrink(dimension, index_best, p_simplex);
					nelderMead_calculateSimplex(dimension, parameters.evaluations_used, p_simplex, p_obj_function, problem_parameters);

					// /*p*/printSimplex(dimension, p_simplex, "Case 3c: Shrink");
					// /*p*/printObjFunction(dimension, p_obj_function, "Obj Function Shrink");

					nelderMead_findBest(dimension, best, index_best, p_obj_function);

					// /*p*/printf("best: %.5f, index_best: %d\n\n", best, index_best);	
				}
			}
		}

		if (p_obj_function[index_worst] < best){ 
			best = p_obj_function[index_worst]; 
			index_best = index_worst; 
			// /*p*/printf("updating best at %d, now it is %.7f\n", k, best);
		}
	}
	
	NelderMeadResult result;

	result.best = best;
	result.best_vertex.resize(dimension);
	result.evaluations_used = parameters.evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = p_simplex[index_best * dimension + i];
	}

	return result;
}

float nelmin ( float (*fn)(float*), int dimension, float start[], float xmin[], float reqmin, int konvge, int iterations_number)
{

	float contraction_coef = 0.5f;
	float expansion_coef = 2.0f;
	float reflection_coef = 1.0f;

	float step = 1.0f;

	int index_worst, index_best;

	float best, worst;
	float obj_reflection, obj_vertex;

	int l;
	float x, z;

	
	std::vector<float> simplex(dimension * (dimension + 1)); // p_simplex 
	std::vector<float> centroid(dimension);

	std::vector<float> reflection(dimension);
	std::vector<float> vertex(dimension);

	std::vector<float> obj_function(dimension + 1); // p_obj_function

	float * p_simplex 		 = &simplex[0];
	float * p_centroid 		 = &centroid[0];
	float * p_reflection 	 = &reflection[0];
	float * p_vertex 		 = &vertex[0];
	float * p_obj_function	 = &obj_function[0];

	float * p_start = start;

	int evaluations_used = 0;	

	nelderMead_initialize(dimension, p_simplex, p_start, step);
	nelderMead_calculateSimplex(dimension, evaluations_used, p_simplex, p_obj_function);

	nelderMead_findBest(dimension, best, index_best, p_obj_function);

	//  Inner loop.
	for (int k = 0; k < iterations_number; k++) {

		nelderMead_findWorst(dimension, worst, index_worst, p_obj_function);
		
		//  Calculate centroid, the centroid of the simplex vertices
		//  excepting the vertex with Y value worst.
		nelderMead_centroid(dimension, index_worst, p_simplex, p_centroid);

		//  Reflection through the centroid.
		nelderMead_reflection(dimension, index_worst, reflection_coef, p_simplex, p_centroid, p_reflection);
		nelderMead_calculateVertex(evaluations_used, obj_reflection, p_reflection);

		//  Successful reflection, so extension.
		if ( obj_reflection < best ) {

			nelderMead_expansion(dimension, expansion_coef, p_centroid, p_reflection, p_vertex);
			nelderMead_calculateVertex(evaluations_used, obj_vertex, p_vertex);

		//  Check extension.
			if ( obj_vertex <  obj_reflection) {
				nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
			} else { //  Retain extension or contraction.
				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}
		} else { //  No extension.
			l = 0;
			
			/* ??? */
			for (int i = 0; i < dimension + 1; i++ ) {
				if ( obj_reflection < obj_function[i] ) {
					l += 1;
				}
			}

			if ( 1 < l ) {

				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}
			//  Contraction on the Y(index_worst) side of the centroid.
			else if ( l == 0 ) {

				nelderMead_contraction(dimension, contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);
				nelderMead_calculateVertex(evaluations_used, obj_vertex, p_vertex);

		//  Contract the whole simplex.
				if ( obj_function[index_worst] < obj_vertex ) {

					nelderMead_shrink(dimension, index_best, p_simplex);
					nelderMead_calculateSimplex(dimension, evaluations_used, p_simplex, p_obj_function);

					nelderMead_findBest(dimension, best, index_best, p_obj_function);

					continue;
				}
		//  Retain contraction.
				else {

					nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
				}
			}
		//  Contraction on the reflection side of the centroid.
			else if ( l == 1 ) {

				nelderMead_contraction(dimension, contraction_coef, p_centroid, 0, p_reflection, p_vertex);
				nelderMead_calculateVertex(evaluations_used, obj_vertex, p_vertex);
		//
		//  Retain reflection?
		//
				if ( obj_vertex <= obj_reflection ) {
					nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
				}
				else {
					nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
				}
			}
		}
		//  Check if best improved.
		if ( obj_function[index_worst] < best ) { 
			best = obj_function[index_worst]; 
			index_best = index_worst; 
		}
		
	}
	return best;
}
#endif
