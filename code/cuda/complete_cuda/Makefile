all:
	nvcc main.cu loadData.cu tsne.cu tsneCPU.cpp utils.cu gradient_descendCPU.cpp timers.cpp gradient_descend.cu Frep.cpp Frep.cu relocateData.cpp relocateData.cu  nuconv.cpp -lcudart -lfftw3 -lcufft -o fuck
clean:
	rm -f *.o
