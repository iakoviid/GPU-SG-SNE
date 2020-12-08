#include "timers.hpp"
#include <cstddef>
struct timeval tsne_start_timer(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv;
}

double tsne_stop_timer(const char * event_name, timeval begin){
  struct timeval end;
  gettimeofday(&end, NULL);
  double stime = (end.tv_sec - begin.tv_sec) * 1000.0;    // sec to ms
  stime += (end.tv_usec - begin.tv_usec) / 1000.0; // us to ms

#ifdef PRINT_DEBUG_TIME
  printf("%-20s : %8.4lf s\n",event_name, stime);
#endif
  return(stime);
}
