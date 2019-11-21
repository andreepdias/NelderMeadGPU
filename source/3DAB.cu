#include "nelmin.h"
#include "util.h"

__constant__ char device_aminoacid_sequence[150];
char              host_aminoacid_sequence[150];


struct Calculate3DAB{
    const int dimension;
    const int protein_length;

    const float * angles;
    const float * aminoacids;


    Calculate3DAB(const float * _angles, const float * _aminoacids, const int _dimension, const int _protein_length)
        : angles(_angles), aminoacids(_aminoacids), dimension(_dimension), protein_length(_protein_length)
    {
    };
    
    #ifdef USE_HOST
    __host__ float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d, dx, dy, dz;

        sum += (1.0f - cosf(angles[id])) / 4.0f;

        for(unsigned int i = id + 2; i < protein_length; i++){

            if(host_aminoacid_sequence[id] == 'A' && host_aminoacid_sequence[i] == 'A')
                c = 1.0;
            else if(host_aminoacid_sequence[id] == 'B' && host_aminoacid_sequence[i] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = aminoacids[id] - aminoacids[i];
            dy = aminoacids[id + protein_length] - aminoacids[i + protein_length];
            dz = aminoacids[id + protein_length * 2] - aminoacids[i + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
            
            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
                
        }
        return sum;
    }
    
    #else
    __device__ float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d, dx, dy, dz;

        sum += (1.0f - cosf(angles[id])) / 4.0f;

        for(unsigned int i = id + 2; i < protein_length; i++){

            if(device_aminoacid_sequence[id] == 'A' && device_aminoacid_sequence[i] == 'A')
                c = 1.0;
            else if(device_aminoacid_sequence[id] == 'B' && device_aminoacid_sequence[i] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = aminoacids[id] - aminoacids[i];
            dy = aminoacids[id + protein_length] - aminoacids[i + protein_length];
            dz = aminoacids[id + protein_length * 2] - aminoacids[i + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
            
            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
                
        }
        return sum;
    }
    
    #endif
};


struct ObjectiveFunction
{
    
    thrust::device_vector<float> device_angles;
    thrust::device_vector<float> device_aminoacids_position;
    
    thrust::host_vector<float> host_angles;
    thrust::host_vector<float> host_aminoacids_position;
    
    const int dimension;
    const int protein_length;

    ObjectiveFunction(std::vector<float>& _host_angles, const int _dimension, const int _protein_length, std::string _aminoacid_sequence)
        : dimension(_dimension), protein_length(_protein_length)
    {
        host_angles = _host_angles;
        device_angles = _host_angles;
        
        device_aminoacids_position.resize(dimension * 3);
        host_aminoacids_position.resize(dimension * 3);

        char aa_sequence[150];
        memset(aa_sequence, 0, sizeof(char) * 150);
        strcpy(aa_sequence, _aminoacid_sequence.c_str());

        #ifdef USE_HOST
        strcpy(host_aminoacid_sequence, aa_sequence);

        #else
        cudaMemcpyToSymbol(device_aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

        #endif

    };

    void calculateCoordinates(float * angles, thrust::host_vector<float> &aminoacids, int protein_length){

        aminoacids[0] = 0.0f;
        aminoacids[0 + protein_length] = 0.0f;
        aminoacids[0 + protein_length * 2] = 0.0f;
    
        aminoacids[1] = 0.0f;
        aminoacids[1 + protein_length] = 1.0f; 
        aminoacids[1 + protein_length * 2] = 0.0f;
    
        aminoacids[2] = cosf(angles[0]);
        aminoacids[2 + protein_length] = sinf(angles[0]) + 1.0f;
        aminoacids[2 + protein_length * 2] = 0.0f;
    
        for(int i = 3; i < protein_length; i++){
            aminoacids[i] = aminoacids[i - 1] + cosf(angles[i - 2]) * cosf(angles[i + protein_length - 5]); // i - 3 + protein_length - 2
            aminoacids[i + protein_length] = aminoacids[i - 1 + protein_length] + sinf(angles[i - 2]) * cosf(angles[i + protein_length - 5]);
            aminoacids[i + protein_length * 2] = aminoacids[i - 1 + protein_length * 2] + sinf(angles[i + protein_length - 5]);
        }
    }

    #ifdef USE_HOST
    float calculate(float * angles){
        
        calculateCoordinates(angles, host_aminoacids_position, protein_length);

        Calculate3DAB unary_op(angles, &host_aminoacids_position[0], dimension, protein_length);

        float sum = 0;

        for(int i = 0; i < protein_length - 2; i++){
            sum += unary_op(i);
        }

        return sum;
    }

    #else
    float calculate(float * angles){
        
        calculateCoordinates(angles, host_aminoacids_position, protein_length);
        device_aminoacids_position = host_aminoacids_position;

        thrust::copy(angles, angles + dimension, device_angles.begin());

        float * p_angles = thrust::raw_pointer_cast(&device_angles[0]);
        float * p_aminoacids = thrust::raw_pointer_cast(&device_aminoacids_position[0]);

        Calculate3DAB unary_op(p_angles, p_aminoacids, dimension, protein_length);
        thrust::plus<float> binary_op;
    
        float result = thrust::transform_reduce(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(protein_length - 2), unary_op, 0.0f, binary_op);

        return result;
    }

    #endif
};


void * objectiveFunction_object = NULL;

float func(float * angles){
    if(objectiveFunction_object){
        return ((ObjectiveFunction*) objectiveFunction_object)->calculate(angles);
    }
    return(0.0f);
}

int main(void)
{
    
    /* Leitura do arquivo com nome da proteína, cadeia de aminoácidos e ângulos iniciais */
    std::ifstream input_file("input.txt");
    
    std::string protein_name, protein_chain;
    std::vector<float> angles;

    input_file >> protein_name;
    input_file >> protein_chain;

    float x;
    while(input_file >> x){
        angles.push_back(x * PI / 180.0f);
    }

    int protein_length = protein_chain.size();
    int dimension = angles.size();

    
    /* Declaração da classe que invoca GPU para calcular a função objetivo */
    ObjectiveFunction oo(angles, dimension, protein_length, protein_chain);

    
    /* Declaração parâmetros do Nelder-Mead */
    std::vector<float> start = angles;
    std::vector<float> step(dimension, 1.0f);
    std::vector<float> xmin(dimension);
    
    int icount, ifault, numres;
    
    float ynewlo = oo.calculate( &start[0] );
    float a = ynewlo;
    float reqmin = 1.0E-18;
    int kcount = 1000000;
    int konvge = 10;

    objectiveFunction_object = &oo;

    /* Execução do Nelder-Mead */
    double tini, tend;

    tini = stime();
    nelmin<float> (func, dimension, &start[0], &xmin[0], &ynewlo, reqmin, &step[0], konvge, kcount, &icount, &numres, &ifault);
    tend = stime();

    printf("Estado IFault = %d\nF(x) = %f (%f)\nNumero de iteracoes = %d\nNumero de restarts = %d\nTempo: %lf\n", ifault, ynewlo, a,  icount, numres, tend - tini);


    return 0;

}