#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){
printf("IDX-----------------------------------\n");
int ng=atoi(argv[1]);
for(int s=0;s<2;s++){
for(int idual=0; idual<ng-3;idual+=6){
for(int ifine=0;ifine<4;ifine++){
int i=3*s+idual+ifine;
printf("i=%d  ifine=%d idual=%d s=%d\n",i,ifine,idual,s);
}
}
}



return 0;
}
