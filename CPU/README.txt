*************************************************************************************
					Information
The program could be used to generate random Erdos-Renyi graphs. Three algorithms are 
implemented, which are the basic ER algorithm, the ZER algorithm and the PreZER algo-
rihtm. ZER introduces the skipping idea into ER. PreZER pre-computes the probabilties 
used in ZER.
*************************************************************************************




*************************************************************************************
					Usage
Compile "RandomGraph.cpp" and run it in the following way:

./YOUR_COMPILED_FILE -a ALGORITHM -n NUMBER_OF_VERTICES -p PROBABILITY -o OUTPUT_FILE




The parameters:
a: the algorithm, 1.ER; 2.ZER; 3.PreZER
n: the number of vertices in the generated graph
p: the probability for each edge to be generated
o: the name of the output file
*************************************************************************************
 
