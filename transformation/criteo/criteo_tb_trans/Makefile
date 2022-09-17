all: txt2svm txt2feat

txt2svm: txt2svm.cpp criteo.hpp
	g++ -O3 -fopenmp -std=c++11 -Wall txt2svm.cpp -o txt2svm

txt2feat: txt2feat.cpp criteo.hpp
	g++ -O2 -std=c++11 -Wall txt2feat.cpp -o txt2feat

clean:
	rm txt2svm txt2feat
