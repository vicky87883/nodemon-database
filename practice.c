#include <stdio.h>

#define RESET   "\033[0m"    // Reset color
#define RED     "\033[31m"   // Red
#define GREEN   "\033[32m"   // Green
#define YELLOW  "\033[33m"   // Yellow
#define BLUE    "\033[34m"   // Blue
#define MAGENTA "\033[35m"   // Magenta
#define CYAN    "\033[36m"   // Cyan
#define WHITE   "\033[37m"   // White

int main() {
  printf(RED "This is red text\n" RESET);
  printf(GREEN "This is green text\n" RESET);
  printf(YELLOW "This is yellow text\n" RESET);
  printf(BLUE "This is blue text\n" RESET);
  printf(MAGENTA "This is magenta text\n" RESET);
  printf(CYAN "This is cyan text\n" RESET);
  printf(WHITE "This is white text\n" RESET);

  return 0;
}
