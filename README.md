# NelderMeadGPU

Implementação do algortimo de otimização númerica por busca direta Nelder-Mead em CUDA C++.

A função de avaliação codificada é sobre o problema de predição de proteínas com o modelo ab-initio 3D AB Off-Lattice. 
Todas as rotinas são executadas em GPU mas o ganho de performance só é justificado para as instâncias maiores do problema (com proteíans com mais de 75 aminoácidos).
