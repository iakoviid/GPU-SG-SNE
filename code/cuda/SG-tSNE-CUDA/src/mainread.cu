#include <stdio.h>
#include"mmio.h"

int main(int argc, char **argv) {
  int N ;
  int M ;
int nz;
int *I, *J;
matval *val;
ReadMatrix(&M,&N,&nz,&I,&J, &val,argc,argv);
printf(" done\n" );
return 0;
}
