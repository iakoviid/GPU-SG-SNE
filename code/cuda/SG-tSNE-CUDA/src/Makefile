FLAGS = -std=c++14 -arch=sm_60 -lfftw3 -lcufft -lcusparse -lcusolver
SGFLAGS=-arch=sm_60  -std=c++14 -lcufft -lcusparse -lcusolver
BSRPADFLAGS= -arch=sm_60 -lcusparse -lcusolver

sg_test: graph_rescaling.o sparsematrix.o sparse_reorder.o
	nvcc testing_modules/sg_test.cu sgtsne.cu gradient_descend.cu  utils.cu sparse_reorder.o pq.cu matrix_converter.cpp   FrepNoReloc.cu  graph_rescaling.o  gridding1.cu non_periodic_convD.cu non_periodic_convF.cu  nuconv.cu   sparsematrix.o $(SGFLAGS) -w -o sg_test

sg_testStream: graph_rescaling.o sparsematrix.o sparse_reorder.o
	nvcc testing_modules/sg_test.cu sgtsne.cu gradient_descendStreams.cu  utils.cu sparse_reorder.o pqStreams.cu matrix_converter.cpp   FrepNoRelocStreams.cu  graph_rescaling.o  griddingStreams.cu non_periodic_convDStreams.cu non_periodic_convFStreams.cu  nuconvStreams.cu   sparsematrix.o $(SGFLAGS) --default-stream per-thread -w -o sg_testStream

sg_testReloc: graph_rescaling.o sparsematrix.o sparse_reorder.o
	nvcc testing_modules/sg_test.cu sgtsne.cu gradient_descend.cu  utils.cu sparse_reorder.o pq.cu matrix_converter.cpp mmio.cu  Frep.cu sort.cu scan.cu relocate_new0.cu relocateData.cpp  graph_rescaling.o  gridding.cu  non_periodic_convD.cu  nuconv.cu   sparsematrix.o $(SGFLAGS) -w -o sg_testReloc

kltest: graph_rescaling.o sparsematrix.o sparse_reorder.o
	nvcc testing_modules/kl_test.cu gradient_descend.cu timers.cpp utils.cu sparse_reorder.o pq.cu matrix_converter.cpp prepareMatrix.cu FrepNoReloc.cu Frep.cpp gradient_descendCPU.cpp graph_rescaling.o gridding.cpp gridding.cu non_periodic_conv.cpp non_periodic_convD.cu nuconv.cpp nuconv.cu pq.cpp  relocateData.cpp sparsematrix.o $(FLAGS) -o kltest

kltest0: graph_rescaling.o sparsematrix.o sparse_reorder.o
	nvcc testing_modules/kl_test.cu gradient_descend.cu timers.cpp utils.cu sparse_reorder.o pq.cu matrix_converter.cpp prepareMatrix.cu FrepNoReloc0.cu Frep.cpp gradient_descendCPU.cpp graph_rescaling.o gridding.cpp gridding.cu non_periodic_conv.cpp non_periodic_convD.cu nuconv.cpp nuconv.cu pq.cpp  relocateData.cpp sparsematrix.o $(FLAGS) -o kltest0

testbsrPadding: example_read.cu sparse_reorder.o sparsematrix.o
	nvcc example_read.cu  sparse_reorder.o sparsematrix.o $(BSRPADFLAGS) -o testbsrPadding

testbsrpq: testing_modules/testBSRpq.cu sparse_reorder.o sparsematrix.o
	nvcc testing_modules/testBSRpq.cu  sparse_reorder.o sparsematrix.o pq.cu prepareMatrix.cu matrix_converter.cpp $(BSRPADFLAGS) -o testbsrpq

pq_test: testing_modules/pq_test.cu matrix_converter.cpp prepareMatrix.cu sparse_reorder.o
	nvcc testing_modules/pq_test.cu  matrix_converter.cpp prepareMatrix.cu $(BSRPADFLAGS) -o pq_test

F_test:testing_modules/total_test.cu
	nvcc testing_modules/total_test.cu Frep.cpp FrepNoReloc.cu nuconv.cpp nuconv.cu non_periodic_convD.cu non_periodic_conv.cpp sort.cu scan.cu relocate_new0.cu relocateData.cpp gridding.cu gridding.cpp timers.cpp utils.cu -lcufft -lfftw3 -arch=sm_60 -o F_test -std=c++11

testRescaling:testing_modules/test_graph_rescaling.cu graph_rescaling.o sparsematrix.o graph_rescaling.cu
	nvcc testing_modules/test_graph_rescaling.cu graph_rescaling.cu graph_rescaling.o sparsematrix.o -lcusparse -o testRescaling

graph_rescaling.o: graph_rescaling.cpp graph_rescaling.hpp
	g++ -c graph_rescaling.cpp

sparsematrix.o: sparsematrix.cpp sparsematrix.hpp
	g++ -c sparsematrix.cpp

sparse_reorder.o: sparse_reorder.cu sparse_reorder.cuh
	nvcc -c sparse_reorder.cu -lcusolver -lcusparse

clean:
	rm -f *.o kl_test sg_test testRescaling
